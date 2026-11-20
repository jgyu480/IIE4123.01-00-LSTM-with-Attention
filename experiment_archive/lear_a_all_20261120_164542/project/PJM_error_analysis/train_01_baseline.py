"""EPF baseline: 168x3 -> LSTM(64) + forecast(48) + weekday(7) -> MLP -> 24."""
import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 64, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(119, 64), nn.ReLU(), nn.Linear(64, 24))

    def forward(self, history, forecast, weekday):
        _, (h, _) = self.lstm(history)
        return self.mlp(torch.cat((h[-1], forecast.flatten(1), weekday), dim=1))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_market(market, val_days, test_days):
    path = ROOT / "data" / f"{market}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {market} ...", flush=True)
        with urlopen(f"https://zenodo.org/records/4624805/files/{market}.csv", timeout=120) as response:
            contents = response.read()
        temporary = path.with_suffix(".csv.part")
        temporary.write_bytes(contents)
        temporary.replace(path)
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    if frame.shape[1] != 3:
        raise ValueError(f"{path}: expected price and two exogenous columns")
    expected = pd.date_range(frame.index[0], periods=len(frame), freq="h")
    if frame.index[0] != frame.index[0].normalize() or not frame.index.equals(expected) or len(frame) % 24:
        raise ValueError("CSV must have complete days and consecutive, unique hourly timestamps")
    raw = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("CSV contains missing or nonfinite values")
    test_start = len(raw) - test_days * 24
    val_start = test_start - val_days * 24
    if val_start <= 168:
        raise ValueError("Not enough training data")
    mean, std = raw[:val_start].mean(0), raw[:val_start].std(0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = ((raw - mean) / std).astype(np.float32)
    starts = np.arange(168, len(raw), 24)
    history = np.stack([scaled[t-168:t] for t in starts])
    forecast = np.stack([scaled[t:t+24, 1:] for t in starts])
    weekday = np.eye(7, dtype=np.float32)[frame.index[starts].dayofweek]
    targets = np.stack([scaled[t:t+24, 0] for t in starts])
    tensors = [torch.from_numpy(x) for x in (history, forecast, weekday, targets)]
    masks = (starts < val_start, (starts >= val_start) & (starts < test_start), starts >= test_start)
    datasets = [TensorDataset(*(x[mask] for x in tensors)) for mask in masks]
    test_positions = starts[masks[2]]
    truth = np.stack([raw[t:t+24, 0] for t in test_positions])
    naive = np.stack([raw[t-168:t-144, 0] for t in test_positions])
    metadata = {
        "market": market, "columns": list(frame.columns),
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mean": mean.tolist(), "std": std.tolist(),
        "high_threshold": float(np.quantile(raw[:val_start, 0], 0.95)),
        "splits": {name: {"first_target": str(frame.index[starts[mask][0]]),
                          "last_target": str(frame.index[starts[mask][-1]]),
                          "days": int(mask.sum())}
                   for name, mask in zip(("train", "val", "test"), masks)},
        "protocol": "Frozen model; daily rolling inputs with previous days' observed prices; no refit on validation/test.",
    }
    return datasets, metadata, frame.index[test_positions], truth, naive


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    outputs, squared_error, count = [], 0.0, 0
    for history, forecast, weekday, target in loader:
        prediction = model(history.to(device), forecast.to(device), weekday.to(device))
        difference = prediction - target.to(device)
        squared_error += difference.square().sum().item()
        count += target.numel()
        outputs.append(prediction.cpu().numpy())
    return squared_error / count, np.concatenate(outputs).astype(np.float64)


def metrics(truth, prediction, threshold):
    high, predicted_high = truth > threshold, prediction > threshold
    error = prediction - truth
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "high_RMSE": float(np.sqrt(np.square(error[high]).mean())) if high.any() else None,
        "total_exceedance_MAE": float(np.abs(np.maximum(prediction-threshold, 0).sum(1)
                                               - np.maximum(truth-threshold, 0).sum(1)).mean()),
        "peak_hour_MAE": float(np.abs(prediction.argmax(1) - truth.argmax(1)).mean()),
        "high_recall": float(predicted_high[high].mean()) if high.any() else None,
        "false_positive_rate": float(predicted_high[~high].mean()) if (~high).any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=["PJM", "FR"], default=["PJM", "FR"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-days", type=int, default=364)
    parser.add_argument("--test-days", type=int, default=364)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for key in ("epochs", "patience", "batch_size", "val_days", "test_days", "threads"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.lr <= 0:
        parser.error("lr must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; use --device cpu")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    run = ROOT / "results" / "baseline" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not args.check_only:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", {**vars(args), "torch": torch.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(),
                   "architecture": "LSTM(3,64,1) + forecast(48) + weekday(7); MLP(119,64,24); MSE"})
    records = []
    for market in args.markets:
        (train_ds, val_ds, test_ds), info, dates, truth, naive = load_market(market, args.val_days, args.test_days)
        print(f"{market}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} days", flush=True)
        if args.check_only:
            batch = next(iter(DataLoader(train_ds, batch_size=args.batch_size)))
            model = Baseline().to(device)
            prediction = model(*(x.to(device) for x in batch[:3]))
            assert batch[0].shape[1:] == (168, 3) and batch[1].shape[1:] == (24, 2)
            assert batch[2].shape[1:] == (7,) and prediction.shape == batch[3].shape
            loss = nn.functional.mse_loss(prediction, batch[3].to(device))
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            print(f"CHECK OK: history={tuple(batch[0].shape)}, forecast={tuple(batch[1].shape)}, "
                  f"weekday={tuple(batch[2].shape)}, output={tuple(prediction.shape)}; backward OK", flush=True)
            continue
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            folder = run / market / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            write_json(folder / "preprocessing.json", info)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(val_ds, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, args.batch_size, shuffle=False)
            model = Baseline().to(device)
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, folder / "initial_state.pt")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best, best_epoch, stale, log = float("inf"), 0, 0, []
            for epoch in range(1, args.epochs + 1):
                model.train()
                total, count = 0.0, 0
                for history, forecast, weekday, target in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(history.to(device), forecast.to(device), weekday.to(device))
                    loss = nn.functional.mse_loss(prediction, target.to(device))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * target.numel()
                    count += target.numel()
                val_loss, _ = evaluate(model, val_loader, device)
                if not np.isfinite(val_loss):
                    raise RuntimeError("Nonfinite validation loss")
                log.append({"epoch": epoch, "train_MSE_scaled": total/count, "val_MSE_scaled": val_loss})
                pd.DataFrame(log).to_csv(folder / "history.csv", index=False)
                print(f"{market} seed={seed} epoch={epoch:03d} train={total/count:.6f} val={val_loss:.6f}", flush=True)
                if val_loss < best:
                    best, best_epoch, stale = val_loss, epoch, 0
                    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, folder / "best_model.pt")
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
            _, prediction = evaluate(model, test_loader, device)
            prediction = prediction * info["std"][0] + info["mean"][0]
            scores = metrics(truth, prediction, info["high_threshold"])
            write_json(folder / "metrics.json", {"best_epoch": best_epoch, "best_val_MSE_scaled": best,
                       "baseline": scores, "previous_week": metrics(truth, naive, info["high_threshold"])})
            timestamps = [day + pd.Timedelta(hours=h) for day in dates for h in range(24)]
            pd.DataFrame({"Date": timestamps, "actual": truth.ravel(), "prediction": prediction.ravel(),
                          "previous_week": naive.ravel()}).to_csv(folder / "predictions.csv", index=False)
            records.append({"market": market, "seed": seed, "best_epoch": best_epoch, **scores})
            pd.DataFrame(records).to_csv(run / "metrics_all_seeds.csv", index=False)
            print(json.dumps({"market": market, "seed": seed, **scores}, ensure_ascii=False), flush=True)
    if records:
        table = pd.DataFrame(records)
        names = [key for key in records[0] if key not in ("market", "seed", "best_epoch")]
        table.groupby("market")[names].agg(["mean", "std"]).to_csv(run / "summary.csv")
        print(f"Saved: {run}", flush=True)


if __name__ == "__main__":
    main()
