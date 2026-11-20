"""Step 2b: combine LSTM (A) and CaseAttn-LSTM (C).

    prediction = alpha * LSTM + (1 - alpha) * CaseAttn

Models:
  Blend-0.5   alpha = 0.5 fixed
  Blend-val   constant alpha fitted on validation days (least squares, clipped to [0, 1])
  Gate-rule   alpha = 0.5 on ordinary days, alpha = 0 (CaseAttn only) when the previous day's
              price volatility exceeds the training-period 90% quantile (fixed in advance)
Sensitivity (reported, not selected): quantile 0.75 / 0.95, ordinary-day alpha 0.25 / 0.75.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from care import data, metrics

MAIN = {"q": 0.90, "alpha_calm": 0.5}
SENSITIVITY = [{"q": 0.75, "alpha_calm": 0.5}, {"q": 0.95, "alpha_calm": 0.5},
               {"q": 0.90, "alpha_calm": 0.25}, {"q": 0.90, "alpha_calm": 0.75}]


def gate_alpha(prev_vol, threshold, alpha_calm):
    return np.where(prev_vol > threshold, 0.0, alpha_calm)[:, None]


def run(split, seed, args):
    folder = args.out / split / f"seed_{seed}"
    z = np.load(folder / "residuals.npz")
    c = np.load(folder / "case_attn" / "predictions.npz")
    meta = json.loads((folder / "oof_meta.json").read_text(encoding="utf-8"))
    m = data.load(split, args.csv)
    n0 = m.ntrain
    val, test = np.arange(n0, n0 + m.nval), np.arange(n0 + m.nval, len(m.days))
    truth, a = z["truth"], z["pred_main"]
    c_val, c_test = c["val"], c["test"]
    prev_vol = data.regime_features(m)[:, 1]

    d = (a[val] - c_val).ravel()
    alpha_val = float(np.clip(((truth[val] - c_val).ravel() @ d) / (d @ d), 0.0, 1.0))
    preds = {"LSTM": a[test], "CaseAttn": c_test,
             "Blend-0.5": 0.5 * a[test] + 0.5 * c_test,
             "Blend-val": alpha_val * a[test] + (1 - alpha_val) * c_test}
    gates = {}
    for cfg in [MAIN] + SENSITIVITY:
        name = "Gate-rule" if cfg is MAIN else f"Gate-q{cfg['q']}-a{cfg['alpha_calm']}"
        threshold = float(np.quantile(prev_vol[:n0], cfg["q"]))      # training days only
        w = gate_alpha(prev_vol[test], threshold, cfg["alpha_calm"])
        preds[name] = w * a[test] + (1 - w) * c_test
        gates[name] = {**cfg, "threshold": threshold, "test_days_case_only": int((w[:, 0] == 0).sum())}

    oof = z["source"] >= 1
    abs_oof = np.abs(z["residual"][oof])
    taus = {"p95": float(np.quantile(abs_oof, 0.95)), "p99": float(np.quantile(abs_oof, 0.99))}
    masks = {k: v[test] for k, v in data.eval_masks(m, meta["high_threshold"]).items()}
    table = {name: metrics.scores(truth[test], p, high_threshold=meta["high_threshold"], taus=taus, masks=masks)
             for name, p in preds.items()}

    out = folder / "gate"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"taus": taus, "alpha_val": alpha_val, "gates": gates, "test": table}, indent=2), encoding="utf-8")
    days = pd.to_datetime(z["days"])
    stamps = (days.values[test][:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    frame = pd.DataFrame({"Date": stamps, "actual": truth[test].ravel()})
    for name, p in preds.items():
        frame[name] = p.ravel()
    frame.to_csv(out / "predictions.csv", index=False)
    print(f"{split} seed={seed} alpha_val={alpha_val:.3f} gate days(C only)={gates['Gate-rule']['test_days_case_only']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    for split in args.splits:
        for seed in args.seeds:
            run(split, seed, args)


if __name__ == "__main__":
    main()
