"""Step 2a: train CaseAttn-LSTM (original C) for each split and seed, reusing the LSTM from step 1a.

Main model: retrieval uses the trained main LSTM's features; C starts from the main LSTM's initial
weights (same protocol as train_02_bc.py). Saves validation and test predictions in $/MWh:
results/<tag>/<split>/seed_<s>/case_attn/predictions.npz

Fold models (out-of-fold C predictions for the learned gate, 07_learned_gate.py): for each
training-period block of 01_make_oof.py, C is trained only on days before the block, with that
fold's scaling, the fold LSTM's features for retrieval (case pool = days before the block), the fold
LSTM's initial weights, the main C's k and a fixed number of epochs (= main C best epoch).
Saves results/<tag>/<split>/seed_<s>/case_attn/oof.npz:
  pred     (days, 24)  fold predictions on blocks, main predictions on validation/test, NaN elsewhere
  source   (days,)     same as residuals.npz (-1 none, 0 main, j fold j)
  entropy  (days,)     attention entropy / log(k) of the model that made `pred`
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from care import baseline, case_attn, data


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def retrieval(model_folder, tensors, device, batch_size):
    model_a = baseline.Baseline().to(device)
    model_a.load_state_dict(load_state(model_folder / "best_model.pt"))
    return case_attn.search_features(model_a, tensors, device, batch_size)


def train_main(folder, out, m, T, mean, std, seed, args, device):
    z = retrieval(folder / "lstm_main", T, device, args.batch_size)
    k, ids, mask, trials = case_attn.select_k(z, T, m.ntrain, m.nval, args.k_grid)
    if (out / "predictions.npz").exists():
        info = json.loads((out / "info.json").read_text(encoding="utf-8"))
        if info["k"] != k:
            raise SystemExit(f"{out}: saved k={info['k']} but retrieval gives k={k}; delete {out} and rerun")
        print(f"main C done, reusing: {out}", flush=True)
        model = case_attn.CaseModel(load_state(folder / "lstm_main" / "initial_state.pt")).to(device)
        model.load_state_dict(load_state(out / "best_model.pt"))
        loader = lambda a, b: DataLoader(torch.arange(a, b), args.batch_size)
        n0, nv = m.ntrain, m.nval
        _, val_pred, val_w = case_attn.evaluate(model, loader(n0, n0 + nv), T, ids, mask, device)
        _, test_pred, test_w = case_attn.evaluate(model, loader(n0 + nv, len(T[0])), T, ids, mask, device)
        return info, val_pred * std[0] + mean[0], test_pred * std[0] + mean[0], val_w, test_w, mask.numpy()
    initial = load_state(folder / "lstm_main" / "initial_state.pt")
    val_pred, test_pred, val_w, test_w, info = case_attn.train(
        initial, seed, T, ids, mask, m.ntrain, m.nval, device, out,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr,
        log_prefix=f"{m.split} seed={seed} C")
    to_raw = lambda p: p * std[0] + mean[0]
    np.savez_compressed(out / "predictions.npz", val=to_raw(val_pred), test=to_raw(test_pred),
                        val_attention=val_w, test_attention=test_w,
                        candidate_ids=ids.numpy(), candidate_mask=mask.numpy())
    truth = m.truth[m.ntrain + m.nval:]
    err = to_raw(test_pred) - truth
    info.update({"k": k, "k_trials": trials,
                 "test_MAE": float(np.abs(err).mean()), "test_RMSE": float(np.sqrt((err ** 2).mean()))})
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{m.split} seed={seed} C k={k} test MAE={info['test_MAE']:.4f} RMSE={info['test_RMSE']:.4f}", flush=True)
    return info, to_raw(val_pred), to_raw(test_pred), val_w, test_w, mask.numpy()


def train_fold(folder, out, m, j, bs, be, k, epochs, seed, args, device):
    """C fold j: fit on days [0, bs), predict block [bs, be). Returns $/MWh predictions and entropy."""
    fmean, fstd = m.scaling(bs)                       # hours before the block only
    FT = tuple(t[:be] for t in m.tensors(fmean, fstd))
    z = retrieval(folder / f"lstm_fold{j}", FT, device, args.batch_size)
    ids, mask = case_attn.neighbors(z, bs, min(k, bs))
    assert (ids < bs).all(), "fold C must not retrieve from its own block"
    _, pred, _, weights, _ = case_attn.train(
        load_state(folder / f"lstm_fold{j}" / "initial_state.pt"), seed, FT, ids, mask, bs, 0, device,
        out / f"fold{j}", epochs=epochs, batch_size=args.batch_size, lr=args.lr,
        log_prefix=f"{m.split} seed={seed} C fold{j}")
    return pred * fstd[0] + fmean[0], case_attn.attention_entropy(weights, mask[bs:].numpy())


def run(split, seed, args, device):
    folder = args.out / split / f"seed_{seed}"
    out = folder / "case_attn"
    if (out / "oof.npz").exists():
        print(f"skip (done): {out}", flush=True)
        return
    meta = json.loads((folder / "oof_meta.json").read_text(encoding="utf-8"))
    m = data.load(split, args.csv)
    if m.sha256 != meta["data_sha256"]:
        raise SystemExit(f"{folder}: CSV differs from the one used in 01_make_oof.py")
    mean, std = np.array(meta["scaling_main"]["mean"]), np.array(meta["scaling_main"]["std"])
    T = m.tensors(mean, std)
    info, val_pred, test_pred, val_w, test_w, mask = train_main(folder, out, m, T, mean, std, seed, args, device)

    n, n0, nv = len(m.days), m.ntrain, m.nval
    source = np.load(folder / "residuals.npz")["source"]
    pred = np.full((n, 24), np.nan)
    entropy = np.full(n, np.nan)
    pred[n0:n0 + nv], pred[n0 + nv:] = val_pred, test_pred
    entropy[n0:n0 + nv] = case_attn.attention_entropy(val_w, mask[n0:n0 + nv])
    entropy[n0 + nv:] = case_attn.attention_entropy(test_w, mask[n0 + nv:])
    folds = []
    for j in sorted(set(source[source >= 1].tolist())):
        block = np.flatnonzero(source == j)
        bs, be = int(block[0]), int(block[-1]) + 1
        assert len(block) == be - bs
        pred[bs:be], entropy[bs:be] = train_fold(folder, out, m, j, bs, be, info["k"], info["best_epoch"],
                                                 seed, args, device)
        err = pred[bs:be] - m.truth[bs:be]
        folds.append({"fold": j, "fit_days": bs, "block_days": be - bs, "epochs": info["best_epoch"],
                      "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean()))})
        print(f"{split} seed={seed} C fold{j} block MAE={folds[-1]['MAE']:.4f} RMSE={folds[-1]['RMSE']:.4f}",
              flush=True)
    np.savez_compressed(out / "oof.npz", pred=pred, source=source, entropy=entropy)
    (out / "oof_info.json").write_text(json.dumps({"k": info["k"], "folds": folds}, indent=2), encoding="utf-8")


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
