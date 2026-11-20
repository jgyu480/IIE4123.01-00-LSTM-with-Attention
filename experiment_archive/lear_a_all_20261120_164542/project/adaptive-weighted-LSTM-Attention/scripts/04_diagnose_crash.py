"""Diagnostic (training period only, test untouched): does the mean residual of similar days
point the right way, with the right size, on past crash days?

Uses the OOF residuals of split2_2018 (blocks 2014, 2015, 2016). Days whose previous day was
high-priced (D-1 mean > train 95% quantile) are grouped by what actually happened next
(evaluation-only labels):
  crash   : D mean < 0.6 x D-1 mean
  partial : 0.6 - 0.85
  sustain : >= 0.85
  normal  : D-1 not high-priced

Two pool modes:
  causal    : only earlier days (what a forecast could use; early 2014 has a tiny pool)
  leave-out : any OOF day more than 7 days away (diagnostic upper bound: is there signal at all?)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from care import data
from care.retrieval import neighbors, standardize

SPLIT = "split2_2018"
K_GRID = [1, 5, 10, 40]


def groups(m, threshold):
    truth = m.truth
    prev_mean = data.regime_features(m)[:, 0]
    ratio = truth.mean(1) / prev_mean
    high = prev_mean > threshold
    g = np.where(~high, "normal", np.where(ratio < 0.6, "crash", np.where(ratio < 0.85, "partial", "sustain")))
    return g


def run_seed(folder, m, args):
    z = np.load(folder / "residuals.npz")
    meta = json.loads((folder / "oof_meta.json").read_text(encoding="utf-8"))
    residual, source, truth = z["residual"], z["source"], z["truth"]
    oof = source >= 1
    g = groups(m, meta["high_threshold"])
    keys = {"lstm": (standardize(z["features"], m.ntrain), "cosine"),
            "regime": (standardize(data.regime_features(m), m.ntrain), "euclidean")}
    pool_before = np.concatenate([[0], np.cumsum(oof)[:-1]])   # OOF days strictly before d
    rows = []
    for key_name, (key, metric) in keys.items():
        for mode in ["causal", "leave-out"]:
            for k in K_GRID:
                q = np.flatnonzero(oof & ((pool_before >= k) if mode == "causal" else True))
                ids = neighbors(key, metric, q, oof, k, causal=(mode == "causal"), gap=args.gap)
                corr = residual[ids].mean(1)                     # (q, 24)
                actual = residual[q]
                for name in ["crash", "partial", "sustain", "normal"]:
                    sel = g[q] == name
                    if not sel.any():
                        continue
                    a, c = actual[sel].mean(1), corr[sel].mean(1)   # daily mean residual
                    before = np.abs(actual[sel]).mean()
                    rows.append({
                        "key": key_name, "mode": mode, "k": k, "group": name, "days": int(sel.sum()),
                        "actual_resid": a.mean(), "pred_corr": c.mean(),
                        "sign_agree": float(np.mean(np.sign(a) == np.sign(c))),
                        "MAE_before": before,
                        "MAE_a0.5": np.abs(actual[sel] - 0.5 * corr[sel]).mean(),
                        "MAE_a1": np.abs(actual[sel] - corr[sel]).mean(),
                    })
                hi = np.isin(g[q], ["crash", "partial", "sustain"])
                rows.append({"key": key_name, "mode": mode, "k": k, "group": "high_prev:spearman",
                             "days": int(hi.sum()),
                             "actual_resid": pd.Series(actual[hi].mean(1)).rank().corr(
                                 pd.Series(corr[hi].mean(1)).rank())})   # Spearman via ranks
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    p.add_argument("--gap", type=int, default=7)
    args = p.parse_args()
    base = ROOT / "results" / args.tag / SPLIT
    m = data.load(SPLIT, args.csv)
    seeds = sorted(p for p in base.glob("seed_*") if (p / "residuals.npz").exists())
    rows = []
    for s in seeds:
        rows += [{"seed": s.name, **r} for r in run_seed(s, m, args)]
    df = pd.DataFrame(rows)
    df.to_csv(base / "diagnose_crash_per_seed.csv", index=False)
    cols = ["days", "actual_resid", "pred_corr", "sign_agree", "MAE_before", "MAE_a0.5", "MAE_a1"]
    summary = df.groupby(["key", "mode", "k", "group"], sort=False)[cols].mean()
    summary.to_csv(base / "diagnose_crash.csv")

    g = groups(m, json.loads((seeds[0] / "oof_meta.json").read_text())["high_threshold"])
    oof = np.load(seeds[0] / "residuals.npz")["source"] >= 1
    crash_days = [str(d.date()) for d in m.days[oof & (g == "crash")]]
    pd.set_option("display.width", 220)
    print(f"{len(seeds)} seeds, OOF days={int(oof.sum())}, crash days in OOF period: {crash_days}\n")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
