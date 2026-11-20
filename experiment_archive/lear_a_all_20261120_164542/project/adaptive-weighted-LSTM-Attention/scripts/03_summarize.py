"""Summarize a stage (res_knn, gate or gate_learn) across seeds.

Writes to results/<tag>/<split>/<stage>_*.csv:
  per_seed    per-seed test metrics
  summary     mean/std over seeds
  seed_mean   metrics of the seed-averaged prediction
  bootstrap   paired weekly block bootstrap vs each reference model (seed-averaged predictions)
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

SHOW = ["MAE", "RMSE", "high_RMSE", "CVaR5_day_MAE", "worst_day_MAE", "n_large_p95", "calm_MAE", "crash_MAE"]
LOSSES = [("day_MAE", lambda e: np.abs(e).mean(1)), ("day_MSE", lambda e: (e ** 2).mean(1))]


def run(split, args):
    base = args.out / split
    stage = args.stage
    seeds = sorted(p for p in base.glob("seed_*") if (p / stage / "metrics.json").exists())
    if not seeds:
        print(f"{split}: no finished seeds for {stage}")
        return
    rows, preds, taus = [], [], []
    for s in seeds:
        res = json.loads((s / stage / "metrics.json").read_text(encoding="utf-8"))
        taus.append(res["taus"])
        for model, sc in res["test"].items():
            rows.append({"seed": s.name, "model": model, **sc})
        preds.append(pd.read_csv(s / stage / "predictions.csv"))
    per_seed = pd.DataFrame(rows)
    order = list(dict.fromkeys(per_seed.model))
    summary = per_seed.groupby("model")[SHOW].agg(["mean", "std"]).loc[order]
    per_seed.to_csv(base / f"{stage}_per_seed.csv", index=False)
    summary.to_csv(base / f"{stage}_summary.csv")

    m = data.load(split, args.csv)
    meta = json.loads((seeds[0] / "oof_meta.json").read_text(encoding="utf-8"))
    test = slice(m.ntrain + m.nval, None)
    truth = m.truth[test]
    masks = {k: v[test] for k, v in data.eval_masks(m, meta["high_threshold"]).items()}
    tau = {k: float(np.mean([t[k] for t in taus])) for k in taus[0]}
    mean_pred = {name: np.mean([p[name].to_numpy() for p in preds], 0).reshape(-1, 24) for name in order}
    seed_mean = pd.DataFrame({name: metrics.scores(truth, p, high_threshold=meta["high_threshold"], taus=tau,
                                                   masks=masks) for name, p in mean_pred.items()}).T
    seed_mean.to_csv(base / f"{stage}_seed_mean.csv")

    boot = []
    for ref in args.references:
        for name in order:
            if name == ref:
                continue
            for loss, f in LOSSES:
                boot.append({"model": name, "vs": ref, "loss": loss, **metrics.block_bootstrap(
                    f(mean_pred[name] - truth), f(mean_pred[ref] - truth), n_boot=args.n_boot)})
    boot = pd.DataFrame(boot)
    boot.to_csv(base / f"{stage}_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== {stage} / {split}: {len(seeds)} seeds, test {m.days[test][0].date()} ~ {m.days[-1].date()}, "
          f"crash days={int(masks['crash'].sum())}, calm days={int(masks['calm'].sum())} ===")
    print("\n[mean over seeds]")
    print(summary.xs("mean", axis=1, level=1).round(3).to_string())
    print("\n[seed-averaged prediction]")
    print(seed_mean[SHOW].round(3).to_string())
    print("\n[paired weekly block bootstrap, seed-averaged; diff < 0 = model better than reference]")
    print(boot.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="res_knn", choices=["res_knn", "gate", "gate_learn"])
    p.add_argument("--references", nargs="+", default=["LSTM"])
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    for split in args.splits:
        run(split, args)


if __name__ == "__main__":
    main()
