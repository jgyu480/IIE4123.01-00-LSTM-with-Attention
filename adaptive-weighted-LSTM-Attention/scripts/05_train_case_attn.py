"""Step 2a: train CaseAttn-LSTM (original C) for each split and seed, reusing the LSTM from step 1a.

Retrieval uses the trained main LSTM's features; C starts from the main LSTM's initial weights
(same protocol as train_02_bc.py). Saves validation and test predictions in $/MWh:
results/<tag>/<split>/seed_<s>/case_attn/predictions.npz
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from care import baseline, case_attn, data


def run(split, seed, args, device):
    folder = args.out / split / f"seed_{seed}"
    out = folder / "case_attn"
    if (out / "predictions.npz").exists():
        print(f"skip (done): {out}", flush=True)
        return
    meta = json.loads((folder / "oof_meta.json").read_text(encoding="utf-8"))
    m = data.load(split, args.csv)
    mean, std = np.array(meta["scaling_main"]["mean"]), np.array(meta["scaling_main"]["std"])
    T = m.tensors(mean, std)
    model_a = baseline.Baseline().to(device)
    model_a.load_state_dict(torch.load(folder / "lstm_main" / "best_model.pt", map_location="cpu", weights_only=True))
    z = case_attn.search_features(model_a, T, device, args.batch_size)
    k, ids, mask, trials = case_attn.select_k(z, T, m.ntrain, m.nval, args.k_grid)
    initial = torch.load(folder / "lstm_main" / "initial_state.pt", map_location="cpu", weights_only=True)
    val_pred, test_pred, weights, info = case_attn.train(
        initial, seed, T, ids, mask, m.ntrain, m.nval, device, out,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr,
        log_prefix=f"{split} seed={seed} C")
    to_raw = lambda p: p * std[0] + mean[0]
    np.savez_compressed(out / "predictions.npz", val=to_raw(val_pred), test=to_raw(test_pred),
                        test_attention=weights, candidate_ids=ids.numpy(), candidate_mask=mask.numpy())
    truth = m.truth[m.ntrain + m.nval:]
    err = to_raw(test_pred) - truth
    info.update({"k": k, "k_trials": trials,
                 "test_MAE": float(np.abs(err).mean()), "test_RMSE": float(np.sqrt((err ** 2).mean()))})
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{split} seed={seed} C k={k} test MAE={info['test_MAE']:.4f} RMSE={info['test_RMSE']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    p.add_argument("--k-grid", nargs="+", type=int, default=[3, 5, 10, 20])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
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
