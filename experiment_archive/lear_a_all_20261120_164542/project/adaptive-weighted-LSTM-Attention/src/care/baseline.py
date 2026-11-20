"""LSTM baseline (A): same architecture and training loop as train_01_baseline.py."""
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 64, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(119, 64), nn.ReLU(), nn.Linear(64, 24))

    def features(self, history, forecast, weekday):
        _, (h, _) = self.lstm(history)
        return torch.cat((h[-1], forecast.flatten(1), weekday), dim=1)

    def forward(self, history, forecast, weekday):
        return self.mlp(self.features(history, forecast, weekday))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(device, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; use --device cpu")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device(device)


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _subset(tensors, idx):
    return TensorDataset(*(t[idx] for t in tensors))


@torch.no_grad()
def predict(model, tensors, idx, device, batch_size=256):
    """Scaled predictions for day indices `idx`, shape (len(idx), 24)."""
    model.eval()
    out = []
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        out.append(model(*(t[b].to(device) for t in tensors[:3])).cpu())
    return torch.cat(out).numpy().astype(np.float64)


@torch.no_grad()
def encode(model, tensors, device, batch_size=256):
    """119-dim LSTM features (h + forecast + weekday) for every day."""
    model.eval()
    n = len(tensors[0])
    return torch.cat([
        model.features(*(t[i:i + batch_size].to(device) for t in tensors[:3])).cpu()
        for i in range(0, n, batch_size)
    ]).numpy().astype(np.float64)


def train(tensors, train_idx, seed, device, folder, *, val_idx=None, epochs=100,
          patience=10, batch_size=32, lr=1e-3, log_prefix=""):
    """Train A. With val_idx: early stopping on validation MSE (as train_01).
    Without val_idx: train exactly `epochs` epochs and keep the last weights."""
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(_subset(tensors, train_idx), batch_size, shuffle=True, generator=generator)
    model = Baseline().to(device)
    torch.save(_state(model), folder / "initial_state.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, stale, log = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for history, forecast, weekday, target in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(history.to(device), forecast.to(device), weekday.to(device))
            loss = nn.functional.mse_loss(pred, target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * target.numel()
            count += target.numel()
        row = {"epoch": epoch, "train_MSE_scaled": total / count}
        if val_idx is not None:
            val_pred = predict(model, tensors, val_idx, device)
            val_loss = float(np.mean((val_pred - tensors[3][val_idx].numpy()) ** 2))
            row["val_MSE_scaled"] = val_loss
            if val_loss < best:
                best, best_epoch, stale = val_loss, epoch, 0
                torch.save(_state(model), folder / "best_model.pt")
            else:
                stale += 1
        log.append(row)
        pd.DataFrame(log).to_csv(folder / "history.csv", index=False)
        print(f"{log_prefix} epoch={epoch:03d} " + " ".join(f"{k}={v:.5f}" for k, v in row.items() if k != "epoch"),
              flush=True)
        if val_idx is not None and stale >= patience:
            break
    if val_idx is None:
        best_epoch = epochs
        torch.save(_state(model), folder / "best_model.pt")
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    return model, {"best_epoch": best_epoch, "best_val_MSE_scaled": None if val_idx is None else best}
