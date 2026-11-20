"""Step 2c: learned (logistic) gate between LSTM (A) and CaseAttn-LSTM (C).

    alpha_d = sigmoid(w . x_d + b)
    prediction_d = alpha_d * A_d + (1 - alpha_d) * C_d        (one alpha per target day)

w, b minimize the squared error of the combined prediction (+ L2 on w) on days where both
A and C predictions are out-of-sample:
  * training-period blocks: fold models (A: 01_make_oof.py, C: 05_train_case_attn.py), each
    trained only on days before its block
  * validation days: main models (used only for their early stopping)
Test days are never used for fitting; the gate is applied to the main models' test predictions.

Gate inputs x_d (known at forecast time; standardized on the fitting days):
  log_prev_vol      log(1 + std of D-1 prices)
  level_shift       signed log of (D-1 mean - last 7 days mean)
  sys_change_pct    system load forecast change, D vs D-1 (%)
  zonal_change_pct  zonal load forecast change, D vs D-1 (%)
  log_disagree      log(1 + mean |A - C| over 24 h)
  attn_entropy      CaseAttn attention entropy / log(k)

Models written (stage gate_learn), next to the step 2b models for comparison:
  Gate-learn        fit on OOF blocks + validation days (main)
  Gate-learn-val    fit on validation days only (sensitivity: fold models differ from main models)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from care import data, metrics

FEATURES = ["log_prev_vol", "level_shift", "sys_change_pct", "zonal_change_pct", "log_disagree", "attn_entropy"]
KEEP = ["LSTM", "CaseAttn", "Blend-val", "Gate-rule"]          # step 2b models carried over for comparison


def gate_inputs(m, pred_a, pred_c, entropy):
    regime = data.regime_features(m)
    col = {name: regime[:, i] for i, name in enumerate(data.REGIME_COLUMNS)}
    return np.column_stack([
        np.log1p(col["prev_vol"]),
        np.sign(col["level_shift"]) * np.log1p(np.abs(col["level_shift"])),
        col["sys_change_pct"],
        col["zonal_change_pct"],
        np.log1p(np.abs(pred_a - pred_c).mean(1)),
        entropy,
    ])


def fit_gate(x, a, c, y, l2, scale):
    """Full-batch L-BFGS on mean squared error (in units of `scale`) + l2 * |w|^2. Deterministic."""
    X, A, C, Y = (torch.from_numpy(v).double() for v in (x, a / scale, c / scale, y / scale))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        alpha = torch.sigmoid(X @ w + b)[:, None]
        loss = ((alpha * A + (1 - alpha) * C - Y) ** 2).mean() + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        alpha = torch.sigmoid(X @ w + b)[:, None]
        mse = float(((alpha * A + (1 - alpha) * C - Y) ** 2).mean()) * scale ** 2
    return w.detach().numpy(), float(b.detach()), mse


def gate_alpha(x, w, b):
    return 1.0 / (1.0 + np.exp(-(x @ w + b)))


def run(split, seed, args):
    folder = args.out / split / f"seed_{seed}"
    z = np.load(folder / "residuals.npz")
    c = np.load(folder / "case_attn" / "oof.npz")
    step2b = json.loads((folder / "gate" / "metrics.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(folder / "gate" / "predictions.csv")
    meta = json.loads((folder / "oof_meta.json").read_text(encoding="utf-8"))
    m = data.load(split, args.csv)
    n0, nv = m.ntrain, m.nval
    val, test = np.arange(n0, n0 + nv), np.arange(n0 + nv, len(m.days))
    truth = z["truth"]
    if not np.array_equal(z["source"], c["source"]):
        raise SystemExit(f"{folder}: A and C OOF blocks differ; rerun 05_train_case_attn.py")

    # A and C predictions that are out-of-sample for every fitting day; main models on val/test.
    pred_a = z["pred_lstm"].copy()
    pred_a[test] = z["pred_main"][test]
    pred_c = c["pred"]
    x_all = gate_inputs(m, pred_a, pred_c, c["entropy"])
    oof_days = np.flatnonzero(z["source"] >= 1)
    fits = {"Gate-learn": np.concatenate([oof_days, val]), "Gate-learn-val": val}

    masks = {k: v[test] for k, v in data.eval_masks(m, meta["high_threshold"]).items()}
    scale = float(truth[:n0].std())
    preds, gates = {}, {}
    for name, idx in fits.items():
        assert idx.max() < n0 + nv, "gate must not see test days"
        assert np.isfinite(pred_a[idx]).all() and np.isfinite(pred_c[idx]).all()
        mu, sd = x_all[idx].mean(0), x_all[idx].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        x = (x_all - mu) / sd
        w, b, fit_mse = fit_gate(x[idx], pred_a[idx], pred_c[idx], truth[idx], args.l2, scale)
        alpha = gate_alpha(x[test], w, b)[:, None]
        preds[name] = alpha * pred_a[test] + (1 - alpha) * pred_c[test]
        gates[name] = {
            "fit_days": int(len(idx)), "fit_MSE": fit_mse, "l2": args.l2, "bias": b,
            "weights": dict(zip(FEATURES, w.tolist())),
            "feature_mean": dict(zip(FEATURES, mu.tolist())), "feature_std": dict(zip(FEATURES, sd.tolist())),
            "fit_alpha_mean": float(gate_alpha(x[idx], w, b).mean()),
            "test_alpha": {"mean": float(alpha.mean()), "min": float(alpha.min()), "max": float(alpha.max()),
                           "days_below_0.25": int((alpha < 0.25).sum()),
                           "crash_mean": float(alpha[masks["crash"]].mean()) if masks["crash"].any() else None},
        }

    a_test = frame["LSTM"].to_numpy().reshape(-1, 24)
    if not np.allclose(a_test, pred_a[test]):
        raise SystemExit(f"{folder}: gate/predictions.csv does not match residuals.npz; rerun 06_gate.py")
    all_preds = {name: frame[name].to_numpy().reshape(-1, 24) for name in KEEP}
    all_preds.update(preds)
    taus = step2b["taus"]
    table = {name: metrics.scores(truth[test], p, high_threshold=meta["high_threshold"], taus=taus, masks=masks)
             for name, p in all_preds.items()}

    out = folder / "gate_learn"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"taus": taus, "features": FEATURES, "gates": gates, "test": table}, indent=2), encoding="utf-8")
    result = frame[["Date", "actual"]].copy()
    for name, p in all_preds.items():
        result[name] = p.ravel()
    result.to_csv(out / "predictions.csv", index=False)
    g = gates["Gate-learn"]
    print(f"{split} seed={seed} Gate-learn fit_days={g['fit_days']} test alpha mean={g['test_alpha']['mean']:.3f} "
          f"MAE={table['Gate-learn']['MAE']:.4f} RMSE={table['Gate-learn']['RMSE']:.4f} "
          f"(rule MAE={table['Gate-rule']['MAE']:.4f} RMSE={table['Gate-rule']['RMSE']:.4f})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    p.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on gate weights (fixed, not tuned)")
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    for split in args.splits:
        for seed in args.seeds:
            run(split, seed, args)


if __name__ == "__main__":
    main()
