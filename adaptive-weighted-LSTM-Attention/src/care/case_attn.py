"""CaseAttn-LSTM (original C): LSTM + attention over retrieved past cases (features + actual prices).

Ported from train_02_bc.py with the same retrieval, initialization, training loop and RNG order,
so split2_2018 reproduces the original C. Case pool: training days strictly before each training
day; validation/test days use training days only.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baseline import Baseline, seed_all


def features(lstm, x, f, w):
    _, (h, _) = lstm(x)
    return torch.cat((h[-1], f.flatten(1), w), dim=1)


@torch.no_grad()
def search_features(model, tensors, device, batch_size):
    model.eval()
    return torch.cat([
        features(model.lstm, *(t[i:i + batch_size].to(device) for t in tensors[:3])).cpu()
        for i in range(0, len(tensors[0]), batch_size)
    ])


def neighbors(z, ntrain, k):
    mean = z[:ntrain].mean(0)
    std = z[:ntrain].std(0, unbiased=False).clamp_min(1e-6)
    normalized = nn.functional.normalize((z - mean) / std, dim=1)
    score = normalized @ normalized[:ntrain].T
    rows = torch.arange(len(z))[:, None]
    cols = torch.arange(ntrain)[None, :]
    allowed = cols < rows
    score.masked_fill_(~allowed, -torch.inf)
    ids = torch.argsort(score, dim=1, descending=True, stable=True)[:, :k]
    mask = torch.gather(allowed, 1, ids)
    assert ((ids < rows) | ~mask).all()
    assert (ids < ntrain).all()
    return ids, mask


def mean_prediction(ids, mask, targets):
    values = targets[ids]
    weights = mask.float()
    return (values * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1)[:, None]


def select_k(z, tensors, ntrain, nval, k_grid):
    """k chosen by the validation MSE of B (mean price of the k nearest days), as in train_02."""
    grid = sorted(set(min(k, ntrain) for k in k_grid))
    all_ids, all_mask = neighbors(z, ntrain, max(grid))
    val = slice(ntrain, ntrain + nval)
    trials = []
    for k in grid:
        pred = mean_prediction(all_ids[val, :k], all_mask[val, :k], tensors[3])
        trials.append({"k": k, "val_MSE_scaled": (pred - tensors[3][val]).square().mean().item()})
    k = min(trials, key=lambda x: (x["val_MSE_scaled"], x["k"]))["k"]
    return k, all_ids[:, :k].clone(), all_mask[:, :k].clone(), trials


class CaseModel(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.lstm = nn.LSTM(3, 64, num_layers=1, batch_first=True)
        self.query = nn.Linear(119, 32)
        self.key = nn.Linear(119, 32)
        self.value = nn.Sequential(nn.Linear(143, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(183, 64), nn.ReLU(), nn.Linear(64, 24))
        self.lstm.load_state_dict({k[5:]: v for k, v in initial.items() if k.startswith("lstm.")})
        with torch.no_grad():
            self.mlp[0].weight[:, :119].copy_(initial["mlp.0.weight"])
            self.mlp[0].weight[:, 119:].zero_()
            self.mlp[0].bias.copy_(initial["mlp.0.bias"])
            self.mlp[2].weight.copy_(initial["mlp.2.weight"])
            self.mlp[2].bias.copy_(initial["mlp.2.bias"])

    def forward(self, batch_ids, all_tensors, ids, mask, device):
        chosen = ids[batch_ids]
        valid = mask[batch_ids].to(device)
        unique, inverse = torch.unique(chosen[mask[batch_ids]], return_inverse=True)
        current = features(self.lstm, *(t[batch_ids].to(device) for t in all_tensors[:3]))
        context = torch.zeros(len(batch_ids), 64, device=device)
        weights = torch.zeros_like(valid, dtype=current.dtype)
        if unique.numel():
            past = features(self.lstm, *(t[unique].to(device) for t in all_tensors[:3]))
            labels = all_tensors[3][unique].to(device)
            keys = current.new_zeros(len(batch_ids), chosen.shape[1], 32)
            values = current.new_zeros(len(batch_ids), chosen.shape[1], 64)
            keys[valid] = self.key(past)[inverse.to(device)]
            values[valid] = self.value(torch.cat((past, labels), 1))[inverse.to(device)]
            scores = (self.query(current)[:, None, :] * keys).sum(-1) / (32 ** 0.5)
            scores = scores.masked_fill(~valid, -1e9)
            weights = scores.softmax(1) * valid
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            context = (weights[..., None] * values).sum(1)
        return self.mlp(torch.cat((current, context), 1)), weights


@torch.no_grad()
def evaluate(model, loader, tensors, ids, mask, device):
    model.eval()
    predictions, attention = [], []
    total, count = 0.0, 0
    for batch_ids in loader:
        output, weights = model(batch_ids, tensors, ids, mask, device)
        target = tensors[3][batch_ids].to(device)
        total += (output - target).square().sum().item()
        count += target.numel()
        predictions.append(output.cpu())
        attention.append(weights.cpu())
    return total / count, torch.cat(predictions).numpy().astype(np.float64), torch.cat(attention).numpy()


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def attention_entropy(weights, mask):
    """Attention entropy per day, normalized by log(number of valid cases) to [0, 1]."""
    w = np.clip(weights, 1e-12, 1.0)
    n = np.asarray(mask).sum(1)
    h = -(weights * np.log(w)).sum(1)
    return np.where(n > 1, h / np.log(np.maximum(n, 2)), 0.0)


def train(initial, seed, tensors, ids, mask, ntrain, nval, device, folder, *,
          epochs=100, patience=10, batch_size=32, lr=1e-3, log_prefix=""):
    """Returns scaled predictions and attention for validation and test days, and info.

    With nval > 0: early stopping on validation MSE (as train_02). With nval == 0 (fold models):
    train exactly `epochs` epochs and keep the last weights; validation outputs are empty.
    "Test" days are all days after ntrain + nval."""
    seed_all(seed)
    model = CaseModel(initial).to(device)
    original = Baseline().to(device)          # same RNG consumption and init check as train_02
    original.load_state_dict(initial)
    with torch.no_grad():
        probe_ids = torch.arange(min(4, ntrain))
        before = original(*(t[probe_ids].to(device) for t in tensors[:3]))
        after, _ = model(probe_ids, tensors, ids, mask, device)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    del original
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(_state(model), folder / "initial_state.pt")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(torch.arange(ntrain), batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(torch.arange(ntrain, ntrain + nval), batch_size)
    test_loader = DataLoader(torch.arange(ntrain + nval, len(tensors[0])), batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, stale, log = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_ids in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(batch_ids, tensors, ids, mask, device)
            target = tensors[3][batch_ids].to(device)
            loss = nn.functional.mse_loss(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * target.numel()
            count += target.numel()
        if nval == 0:
            log.append({"epoch": epoch, "train_MSE_scaled": total / count})
            pd.DataFrame(log).to_csv(folder / "history.csv", index=False)
            print(f"{log_prefix} epoch={epoch:03d} train={total / count:.6f}", flush=True)
            continue
        val_loss, _, _ = evaluate(model, val_loader, tensors, ids, mask, device)
        log.append({"epoch": epoch, "train_MSE_scaled": total / count, "val_MSE_scaled": val_loss})
        pd.DataFrame(log).to_csv(folder / "history.csv", index=False)
        print(f"{log_prefix} epoch={epoch:03d} train={total / count:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(_state(model), folder / "best_model.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    if nval == 0:
        best, best_epoch = None, epochs
        torch.save(_state(model), folder / "best_model.pt")
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    k = ids.shape[1]
    if nval:
        _, val_pred, val_weights = evaluate(model, val_loader, tensors, ids, mask, device)
    else:
        val_pred, val_weights = np.zeros((0, 24)), np.zeros((0, k), dtype=np.float32)
    _, test_pred, test_weights = evaluate(model, test_loader, tensors, ids, mask, device)
    return val_pred, test_pred, val_weights, test_weights, {"best_epoch": best_epoch, "best_val_MSE_scaled": best}
