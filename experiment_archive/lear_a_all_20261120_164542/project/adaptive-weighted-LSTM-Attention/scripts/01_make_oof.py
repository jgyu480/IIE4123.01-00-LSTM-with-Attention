"""Step 1a: train the LSTM baseline (A) and build out-of-fold (OOF) residuals.

For each split and seed:
  * main LSTM: trained on all training days, early stopping on validation (same as train_01).
    Its predictions give the residuals for validation and test days.
  * fold LSTMs: expanding window over 364-day blocks at the end of the training period.
    Fold j is trained only on days before block j (fixed epochs = main best epoch) and
    predicts block j, so every training-period residual is out-of-sample.

Residual r = actual - prediction ($/MWh). Output: results/<tag>/<split>/seed_<s>/residuals.npz
"""
import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from care import baseline, data


def blocks_for(ntrain, block_days, min_train_days):
    """(start, end) day-index blocks counted back from the end of the training period."""
    out, end = [], ntrain
    while end - block_days >= min_train_days:
        out.insert(0, (end - block_days, end))
        end -= block_days
    if not out:
        raise SystemExit("training period too short for any OOF block")
    return out


def run(split, seed, args, device):
    folder = args.out / split / f"seed_{seed}"
    if (folder / "residuals.npz").exists():
        print(f"skip (done): {folder}", flush=True)
        return
    m = data.load(split, args.csv)
    n = len(m.days)
    truth = m.truth
    mean, std = m.scaling(m.ntrain)
    T = m.tensors(mean, std)
    train_idx = np.arange(m.ntrain)
    val_idx = np.arange(m.ntrain, m.ntrain + m.nval)
    all_idx = np.arange(n)

    model, info = baseline.train(
        T, train_idx, seed, device, folder / "lstm_main", val_idx=val_idx,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr,
        log_prefix=f"{split} seed={seed} main")
    pred_main = baseline.predict(model, T, all_idx, device) * std[0] + mean[0]
    features = baseline.encode(model, T, device)

    pred = np.full((n, 24), np.nan)
    source = np.full(n, -1, dtype=int)          # -1 none, 0 main LSTM, j>=1 fold j
    blocks = blocks_for(m.ntrain, args.block_days, args.min_train_days)
    fold_info = []
    for j, (bs, be) in enumerate(blocks, start=1):
        fmean, fstd = m.scaling(bs)             # fold scaling uses only hours before the block
        FT = m.tensors(fmean, fstd)
        fit_idx = np.arange(bs)
        assert fit_idx.max() < bs, "fold model must not see its own block"
        fmodel, _ = baseline.train(
            FT, fit_idx, seed, device, folder / f"lstm_fold{j}", val_idx=None,
            epochs=info["best_epoch"], batch_size=args.batch_size, lr=args.lr,
            log_prefix=f"{split} seed={seed} fold{j}")
        pred[bs:be] = baseline.predict(fmodel, FT, np.arange(bs, be), device) * fstd[0] + fmean[0]
        source[bs:be] = j
        fold_info.append({"fold": j, "fit_days": int(bs),
                          "fit_first": str(m.days[0].date()), "fit_last": str(m.days[bs - 1].date()),
                          "block_first": str(m.days[bs].date()), "block_last": str(m.days[be - 1].date())})
    pred[m.ntrain:] = pred_main[m.ntrain:]
    source[m.ntrain:] = 0
    residual = truth - pred

    np.savez_compressed(
        folder / "residuals.npz", days=np.array(m.days.strftime("%Y-%m-%d"), dtype="U10"), part=m.part(), truth=truth,
        pred_lstm=pred, pred_main=pred_main, residual=residual, source=source, features=features)
    high_threshold = float(np.quantile(m.raw[:m.starts[m.ntrain], 0], 0.95))
    test = slice(m.ntrain + m.nval, None)
    err = pred_main[test] - truth[test]
    meta = {
        "split": split, "seed": seed, "data_sha256": m.sha256,
        "days": {"train": m.ntrain, "val": m.nval, "test": m.ntest,
                 "train_first": str(m.days[0].date()), "val_first": str(m.days[m.ntrain].date()),
                 "test_first": str(m.days[m.ntrain + m.nval].date()), "test_last": str(m.days[-1].date())},
        "main": info, "folds": fold_info, "high_threshold": high_threshold,
        "scaling_main": {"mean": mean.tolist(), "std": std.tolist()},
        "test_LSTM": {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean()))},
        "residual_definition": "actual - prediction, $/MWh; train-period residuals from fold models (OOF)",
        "args": {k: str(v) for k, v in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "python": platform.python_version()},
        "device": str(device),
    }
    (folder / "oof_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{split} seed={seed} LSTM test MAE={meta['test_LSTM']['MAE']:.4f} "
          f"RMSE={meta['test_LSTM']['RMSE']:.4f} -> {folder}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--block-days", type=int, default=364)
    p.add_argument("--min-train-days", type=int, default=300)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    device = baseline.setup(args.device, args.threads)
    for split in args.splits:
        for seed in args.seeds:
            run(split, seed, args, device)


if __name__ == "__main__":
    main()
