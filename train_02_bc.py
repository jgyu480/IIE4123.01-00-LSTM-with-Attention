"""B: nearest-day mean. C: shared LSTM + retrieved-case attention."""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import train_01_baseline as A
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def features(lstm, x, f, w):
    _, (h, _) = lstm(x)
    return torch.cat((h[-1], f.flatten(1), w), dim=1)


@torch.no_grad()
def search_features(model, tensors, device, batch_size):
    model.eval()
    return torch.cat([
        features(model.lstm, *(t[i:i+batch_size].to(device) for t in tensors[:3])).cpu()
        for i in range(0, len(tensors[0]), batch_size)
    ])


def neighbors(z, ntrain, k):
    # 학습 입력 기준으로 특징 표준화 후 코사인 유사도 계산.
    mean = z[:ntrain].mean(0)
    std = z[:ntrain].std(0, unbiased=False).clamp_min(1e-6)
    normalized = nn.functional.normalize((z-mean)/std, dim=1)
    score = normalized @ normalized[:ntrain].T
    rows = torch.arange(len(z))[:, None]
    cols = torch.arange(ntrain)[None, :]

    # 학습: 이전 학습 날짜만. 검증/테스트: 학습 날짜만.
    allowed = cols < rows
    score.masked_fill_(~allowed, -torch.inf)
    ids = torch.argsort(score, dim=1, descending=True, stable=True)[:, :k]
    mask = torch.gather(allowed, 1, ids)
    assert ((ids < rows) | ~mask).all()
    assert (ids < ntrain).all()
    return ids, mask, {"mean": mean.tolist(), "std": std.tolist()}


def mean_prediction(ids, mask, targets):
    values = targets[ids]
    weights = mask.float()
    return (values*weights[..., None]).sum(1)/weights.sum(1).clamp_min(1)[:, None]


class CaseModel(nn.Module):
    def __init__(self, initial, variant):
        super().__init__()
        self.variant = variant
        self.lstm = nn.LSTM(3, 64, num_layers=1, batch_first=True)
        self.query = nn.Linear(119, 32)
        self.key = nn.Linear(119, 32)
        self.value = nn.Sequential(nn.Linear(143, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(183, 64), nn.ReLU(), nn.Linear(64, 24))

        self.lstm.load_state_dict({
            k[5:]: v for k, v in initial.items() if k.startswith("lstm.")
        })
        with torch.no_grad():
            self.mlp[0].weight[:, :119].copy_(initial["mlp.0.weight"])
            self.mlp[0].weight[:, 119:].zero_()
            self.mlp[0].bias.copy_(initial["mlp.0.bias"])
            self.mlp[2].weight.copy_(initial["mlp.2.weight"])
            self.mlp[2].bias.copy_(initial["mlp.2.bias"])

    def forward(self, batch_ids, all_tensors, ids, mask, device):
        chosen = ids[batch_ids]
        valid = mask[batch_ids].to(device)

        # 유효한 사례만 처리. 같은 배치의 중복 사례는 한 번만 인코딩.
        unique, inverse = torch.unique(
            chosen[mask[batch_ids]], return_inverse=True
        )
        current = features(
            self.lstm, *(t[batch_ids].to(device) for t in all_tensors[:3])
        )
        context = torch.zeros(len(batch_ids), 64, device=device)
        weights = torch.zeros_like(valid, dtype=current.dtype)

        if unique.numel():
            past = features(
                self.lstm, *(t[unique].to(device) for t in all_tensors[:3])
            )
            labels = all_tensors[3][unique].to(device)
            if self.variant == "no_price":
                labels = torch.zeros_like(labels)

            keys = current.new_zeros(len(batch_ids), chosen.shape[1], 32)
            values = current.new_zeros(len(batch_ids), chosen.shape[1], 64)
            keys[valid] = self.key(past)[inverse.to(device)]
            values[valid] = self.value(
                torch.cat((past, labels), 1)
            )[inverse.to(device)]

            if self.variant == "uniform":
                weights = valid.float()/valid.sum(1, keepdim=True).clamp_min(1)
            else:
                scores = (
                    self.query(current)[:, None, :]*keys
                ).sum(-1)/(32**0.5)
                scores = scores.masked_fill(~valid, -1e9)
                weights = scores.softmax(1)*valid
                weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-12)

            context = (weights[..., None]*values).sum(1)

        return self.mlp(torch.cat((current, context), 1)), weights


@torch.no_grad()
def evaluate(model, loader, tensors, ids, mask, device):
    model.eval()
    predictions, attention = [], []
    total, count = 0.0, 0
    for batch_ids in loader:
        output, weights = model(batch_ids, tensors, ids, mask, device)
        target = tensors[3][batch_ids].to(device)
        total += (output-target).square().sum().item()
        count += target.numel()
        predictions.append(output.cpu())
        attention.append(weights.cpu())
    return (
        total/count,
        torch.cat(predictions).numpy().astype(np.float64),
        torch.cat(attention).numpy(),
    )


def train_variant(initial, variant, seed, tensors, ids, mask,
                  ntrain, nval, cfg, device, folder):
    seed_all(seed)
    model = CaseModel(initial, variant).to(device)

    # 공통 초기 가중치와 초기 예측이 A와 일치하는지 확인.
    original = A.Baseline().to(device)
    original.load_state_dict(initial)
    with torch.no_grad():
        probe_ids = torch.arange(min(4, ntrain))
        probe = [t[probe_ids].to(device) for t in tensors[:3]]
        before = original(*probe)
        after, _ = model(probe_ids, tensors, ids, mask, device)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    del original

    folder.mkdir(parents=True, exist_ok=False)
    torch.save(state(model), folder/"initial_state.pt")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        torch.arange(ntrain), cfg["batch_size"],
        shuffle=True, generator=generator
    )
    val_loader = DataLoader(
        torch.arange(ntrain, ntrain+nval), cfg["batch_size"]
    )
    test_loader = DataLoader(
        torch.arange(ntrain+nval, len(tensors[0])), cfg["batch_size"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_epoch, stale, log = float("inf"), 0, 0, []

    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        total, count = 0.0, 0
        for batch_ids in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(batch_ids, tensors, ids, mask, device)
            target = tensors[3][batch_ids].to(device)
            loss = nn.functional.mse_loss(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            total += loss.item()*target.numel()
            count += target.numel()

        val_loss, _, _ = evaluate(
            model, val_loader, tensors, ids, mask, device
        )
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        log.append({
            "epoch": epoch,
            "train_MSE_scaled": total/count,
            "val_MSE_scaled": val_loss,
        })
        pd.DataFrame(log).to_csv(folder/"history.csv", index=False)
        print(
            f"{folder.parent.parent.name} seed={seed} {variant} "
            f"epoch={epoch:03d} train={total/count:.6f} val={val_loss:.6f}",
            flush=True
        )
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(state(model), folder/"best_model.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break

    model.load_state_dict(torch.load(
        folder/"best_model.pt", map_location="cpu", weights_only=True
    ))
    _, prediction, weights = evaluate(
        model, test_loader, tensors, ids, mask, device
    )
    return prediction, weights, {
        "best_epoch": best_epoch,
        "best_val_MSE_scaled": best,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-run", type=Path)
    parser.add_argument(
        "--k-grid", nargs="+", type=int, default=[3, 5, 10, 20]
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=["attention", "uniform", "no_price"],
        default=["attention", "uniform", "no_price"]
    )
    args = parser.parse_args()
    if not args.k_grid or min(args.k_grid) < 1:
        parser.error("k must be positive")

    required = [
        "initial_state.pt", "best_model.pt",
        "preprocessing.json", "metrics.json"
    ]

    def complete(run):
        try:
            cfg = read(run/"config.json")
            return bool(cfg["markets"] and cfg["seeds"]) and all(
                (run/market/f"seed_{seed}"/name).is_file()
                for market in cfg["markets"]
                for seed in cfg["seeds"]
                for name in required
            )
        except (OSError, KeyError, ValueError):
            return False

    if args.a_run:
        a_run = args.a_run.expanduser().resolve()
    else:
        runs = sorted(
            (ROOT/"results"/"baseline").glob("*"), reverse=True
        )
        if not runs:
            raise SystemExit(
                "A 결과가 없습니다. 먼저 train_01_baseline.py를 실행하세요."
            )
        a_run = runs[0]

    if not complete(a_run):
        raise SystemExit(
            f"A 실행이 아직 완료되지 않았거나 파일이 부족합니다: {a_run}"
        )

    cfg = read(a_run/"config.json")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("A와 동일한 CUDA 장치가 필요합니다.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.__version__ != cfg["torch"]:
        raise SystemExit(
            "A 실행 때와 PyTorch 버전이 다릅니다. 같은 환경을 사용하세요."
        )

    out = ROOT/"results"/"abc"/datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    A.write_json(out/"config.json", {
        "a_run": str(a_run),
        "a_config": cfg,
        "variants": args.variants,
        "k_grid": args.k_grid,
        "retrieval": (
            "Frozen A best encoder; train-feature-standardized cosine; "
            "candidates fixed for B/C."
        ),
        "case_pool": (
            "Train: strictly earlier training days. "
            "Validation/test: training days only."
        ),
        "C_training": (
            "From A initial_state, not trained A checkpoint. "
            "Shared LSTM trained on current and retrieved inputs."
        ),
        "selection": (
            "B k selected by validation MSE and shared across C variants. "
            "Early stopping uses validation MSE only."
        ),
        "empty_pool": (
            "First training day: zero context; all A training targets retained."
        ),
        "capacity": (
            "A/C common initial weights match; C has extra parameters. "
            "Uniform control isolates attention weighting."
        ),
    })
    print(f"A 결과: {a_run}\n출력: {out}", flush=True)
    rows = []

    for market in cfg["markets"]:
        datasets, info, dates, truth, naive = A.load_market(
            market, cfg["val_days"], cfg["test_days"]
        )
        ntrain, nval, ntest = map(len, datasets)
        tensors = tuple(
            torch.cat([ds.tensors[j] for ds in datasets])
            for j in range(4)
        )

        for seed in cfg["seeds"]:
            source = a_run/market/f"seed_{seed}"
            old = read(source/"preprocessing.json")
            for key in [
                "data_sha256", "columns", "mean", "std",
                "splits", "high_threshold"
            ]:
                if old[key] != info[key]:
                    raise RuntimeError(
                        f"A와 데이터/전처리가 다릅니다: {market}, {key}"
                    )

            folder = out/market/f"seed_{seed}"
            folder.mkdir(parents=True)
            A.write_json(folder/"preprocessing.json", info)

            model_a = A.Baseline().to(device)
            model_a.load_state_dict(torch.load(
                source/"best_model.pt",
                map_location="cpu", weights_only=True
            ))
            z = search_features(
                model_a, tensors, device, cfg["batch_size"]
            )
            grid = sorted(set(min(k, ntrain) for k in args.k_grid))
            all_ids, all_mask, scaling = neighbors(z, ntrain, max(grid))
            valid_slice = slice(ntrain, ntrain+nval)
            test_slice = slice(ntrain+nval, None)

            trials = []
            for k in grid:
                pred = mean_prediction(
                    all_ids[valid_slice, :k],
                    all_mask[valid_slice, :k],
                    tensors[3]
                )
                score = (
                    pred-tensors[3][valid_slice]
                ).square().mean().item()
                trials.append({"k": k, "val_MSE_scaled": score})
            k = min(
                trials, key=lambda x: (x["val_MSE_scaled"], x["k"])
            )["k"]
            ids = all_ids[:, :k].clone()
            mask = all_mask[:, :k].clone()

            A.write_json(folder/"retrieval.json", {
                "selected_k": k,
                "validation_trials": trials,
                "scaling": scaling,
            })
            np.savez_compressed(
                folder/"candidate_indices.npz",
                ids=ids.numpy(), mask=mask.numpy()
            )

            _, a_scaled = A.evaluate(
                model_a,
                DataLoader(datasets[2], cfg["batch_size"]),
                device
            )
            predictions = {
                "A": a_scaled*info["std"][0]+info["mean"][0],
                "previous_week": naive,
            }
            b_scaled = mean_prediction(
                ids[test_slice], mask[test_slice], tensors[3]
            ).numpy().astype(np.float64)
            predictions["B"] = (
                b_scaled*info["std"][0]+info["mean"][0]
            )

            a_scores = A.metrics(
                truth, predictions["A"], info["high_threshold"]
            )
            saved_scores = read(source/"metrics.json")["baseline"]
            for name, value in a_scores.items():
                if value is not None and not np.isclose(
                    value, saved_scores[name], rtol=1e-5, atol=1e-6
                ):
                    raise RuntimeError(
                        f"A 재평가 불일치: {market}, seed={seed}, {name}"
                    )

            initial = torch.load(
                source/"initial_state.pt",
                map_location="cpu", weights_only=True
            )
            details = {
                "A": read(source/"metrics.json"),
                "B": {"selected_k": k},
            }
            for variant in dict.fromkeys(args.variants):
                pred, weights, detail = train_variant(
                    initial, variant, seed, tensors, ids, mask,
                    ntrain, nval, cfg, device, folder/variant
                )
                name = {
                    "attention": "C",
                    "uniform": "C_uniform",
                    "no_price": "C_no_price",
                }[variant]
                predictions[name] = (
                    pred*info["std"][0]+info["mean"][0]
                )
                details[name] = detail
                np.savez_compressed(
                    folder/variant/"test_attention.npz",
                    weights=weights,
                    candidate_indices=ids[test_slice].numpy()
                )

            results = {}
            for name, pred in predictions.items():
                scores = A.metrics(
                    truth, pred, info["high_threshold"]
                )
                results[name] = {
                    **scores, **details.get(name, {})
                }
                rows.append({
                    "market": market, "seed": seed,
                    "model": name, **scores
                })
                print(
                    f"{market} seed={seed} {name}: "
                    f"MAE={scores['MAE']:.4f} "
                    f"RMSE={scores['RMSE']:.4f}",
                    flush=True
                )
            A.write_json(folder/"metrics.json", results)

            timestamps = [
                day+pd.Timedelta(hours=h)
                for day in dates for h in range(24)
            ]
            table = {"Date": timestamps, "actual": truth.ravel()}
            table.update({
                name: pred.ravel()
                for name, pred in predictions.items()
            })
            pd.DataFrame(table).to_csv(
                folder/"predictions.csv", index=False
            )
            scores_frame = pd.DataFrame(rows)
            scores_frame.to_csv(
                out/"metrics_all_seeds.csv", index=False
            )
            columns = [
                col for col in scores_frame.columns
                if col not in ["market", "seed", "model"]
            ]
            scores_frame.groupby(
                ["market", "model"]
            )[columns].agg(["mean", "std"]).to_csv(out/"summary.csv")

    print(f"완료: {out}", flush=True)


if __name__ == "__main__":
    main()
