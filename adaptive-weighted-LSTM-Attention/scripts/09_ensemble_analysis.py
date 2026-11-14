"""Step 3b: error diversity and ensembles of A (LSTM), B (case mean), C (CaseAttn) and LEAR.

All on seed-averaged predictions; ensemble weights only from validation days (grid step 0.05).
  1. A/B/C diversity on test: hourly error correlation, equal averages, validation-weighted A+B+C.
  2. LEAR screening on validation (criteria fixed before looking at test): hourly error correlation
     with A and with C <= 0.85 and validation MAE within 5% of the better of A and C.
  3. Test comparison: A, B, C, Gate-rule(A,C), LEAR and LEAR ensembles.
     Control: LEAR+A (no attention) vs LEAR+C (attention) isolates the case-attention module.
  4. Paired weekly block bootstrap (diff < 0 = model better than reference), also outside the
     January window for split2 (test start 2017-12-26 .. 2018-01-31).

B = mean actual price of the k retrieved days saved by 05 (candidate_ids / candidate_mask), as in
train_02_bc.py. LEAR from 08_lear.py (main LEAR = average of its windows).
Writes results/<tag>/<split>/ensemble_{diversity,lear_screen,test,bootstrap}.csv
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from care import data, metrics

LOSSES = {"day_MAE": lambda e: np.abs(e).mean(1), "day_MSE": lambda e: (e ** 2).mean(1)}


def simplex(n, step=0.05):
    g = np.round(np.arange(0, 1 + 1e-9, step), 4)
    return [w for w in itertools.product(g, repeat=n) if abs(sum(w) - 1) < 1e-9]


def val_weights(parts, y):
    """Constant weights (sum 1, >= 0) minimizing validation MSE."""
    return min(simplex(len(parts)), key=lambda w: ((sum(wi * p for wi, p in zip(w, parts)) - y) ** 2).mean())


def combine(w, parts):
    return sum(wi * p for wi, p in zip(w, parts))


def corr(e1, e2):
    return float(np.corrcoef(e1.ravel(), e2.ravel())[0, 1])


def load(split, out):
    """Seed-averaged val/test predictions of A, B, C, Gate-rule(A,C) and LEAR, plus truth and days."""
    seeds = sorted(p for p in (out / split).glob("seed_*") if (p / "gate" / "predictions.csv").exists())
    if not seeds:
        raise SystemExit(f"{split}: no seeds with 06_gate.py output")
    z0 = np.load(seeds[0] / "residuals.npz")
    part = z0["part"]
    masks = {"val": part == "val", "test": part == "test"}
    P = {k: {"A": [], "B": [], "C": []} for k in masks}
    rule = []
    for s in seeds:
        z = np.load(s / "residuals.npz")
        c = np.load(s / "case_attn" / "predictions.npz")
        ids, mask = c["candidate_ids"], c["candidate_mask"]
        b = (z["truth"][ids] * mask[..., None]).sum(1) / np.maximum(mask.sum(1), 1)[:, None]
        for k, sel in masks.items():
            P[k]["A"].append(z["pred_main"][sel])
            P[k]["B"].append(b[sel])
            P[k]["C"].append(c[k])
        rule.append(pd.read_csv(s / "gate" / "predictions.csv")["Gate-rule"].to_numpy().reshape(-1, 24))
    P = {k: {name: np.mean(v, 0) for name, v in d.items()} for k, d in P.items()}
    P["test"]["Gate-rule(A,C)"] = np.mean(rule, 0)
    truth = {k: z0["truth"][sel] for k, sel in masks.items()}
    for k, sel in masks.items():
        f = out / split / "lear" / f"{k}.npz"
        if not f.exists():
            raise SystemExit(f"{f} missing; run 08_lear.py")
        lear = np.load(f)
        assert np.array_equal(lear["idx"], np.flatnonzero(sel)) and np.allclose(lear["truth"], truth[k])
        P[k]["LEAR"] = lear["ens"]
    days = pd.to_datetime(z0["days"][masks["test"]])
    return P, truth, days, len(seeds)


def run(split, args):
    base = args.out / split
    P, truth, days, nseeds = load(split, args.out)
    V, T, yv, y = P["val"], P["test"], truth["val"], truth["test"]
    E = {k: p - y for k, p in T.items()}
    jan = (days <= "2018-01-31") if split == "split2_2018" else np.zeros(len(days), bool)
    print(f"\n========== {split}: {nseeds} seeds, test {days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d} ==========")

    # 1. A/B/C diversity (test)
    abc = list("ABC")
    div = pd.DataFrame([[corr(E[a], E[b]) for b in abc] for a in abc], abc, abc)
    if jan.any():
        div_nj = pd.DataFrame([[corr(E[a][~jan], E[b][~jan]) for b in abc] for a in abc], abc, abc)
        div = pd.concat({"all": div, "outside_Jan": div_nj}, axis=1)
    div.to_csv(base / "ensemble_diversity.csv")
    print("\n[1] hourly error correlation, test\n" + div.round(3).to_string())

    # 2. LEAR screening (validation)
    ev = {k: p - yv for k, p in V.items()}
    best = min(np.abs(ev["A"]).mean(), np.abs(ev["C"]).mean())
    screen = pd.DataFrame({k: {"val_MAE": np.abs(e).mean(), "val_RMSE": np.sqrt((e ** 2).mean()),
                               "corr_A": corr(e, ev["A"]), "corr_C": corr(e, ev["C"])} for k, e in ev.items()}).T
    screen["MAE_vs_bestAC_%"] = 100 * (screen["val_MAE"] / best - 1)
    passed = (screen[["corr_A", "corr_C"]].max(axis=1) <= 0.85) & (screen["MAE_vs_bestAC_%"] <= 5)
    screen["pass"] = ["-" if k in ("A", "C") else ("yes" if v else "no") for k, v in passed.items()]
    screen.to_csv(base / "ensemble_lear_screen.csv")
    print("\n[2] screening on validation (corr <= 0.85 with A and C, MAE within 5% of best of A, C)\n"
          + screen.round(3).to_string())

    # 3. test comparison
    models, weights = {k: T[k] for k in ["A", "B", "C", "Gate-rule(A,C)", "LEAR"]}, {}
    for members in [("A", "B", "C"), ("LEAR", "A"), ("LEAR", "B"), ("LEAR", "C"), ("A", "LEAR", "C")]:
        w = val_weights([V[k] for k in members], yv)
        name = "+".join(members)
        models[name] = combine(w, [T[k] for k in members])
        weights[name] = dict(zip(members, map(float, w)))
    rows = {}
    for name, p in models.items():
        e = p - y
        r = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
             "corr_LEAR": corr(e, E["LEAR"]), "corr_C": corr(e, E["C"])}
        if jan.any():
            r.update({"Jan_MAE": np.abs(e[jan]).mean(), "Jan_RMSE": np.sqrt((e[jan] ** 2).mean()),
                      "rest_MAE": np.abs(e[~jan]).mean(), "rest_RMSE": np.sqrt((e[~jan] ** 2).mean())})
        rows[name] = r
    table = pd.DataFrame(rows).T
    table.to_csv(base / "ensemble_test.csv")
    (base / "ensemble_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print("\n[3] test (ensemble weights from validation: " + json.dumps(weights) + ")\n" + table.round(3).to_string())

    # 4. bootstrap
    pairs = [("LEAR+C", "LEAR+A"), ("LEAR", "A"), ("LEAR", "C"),
             ("LEAR+A", "LEAR"), ("LEAR+C", "LEAR"), ("LEAR+B", "LEAR"), ("LEAR+C", "C"), ("A+B+C", "C")]
    out = []
    for model, ref in pairs:
        for loss, f in LOSSES.items():
            r = metrics.block_bootstrap(f(models[model] - y), f(models[ref] - y), n_boot=args.n_boot)
            row = {"model": model, "vs": ref, "loss": loss, **r}
            if jan.any():
                r2 = metrics.block_bootstrap(f(models[model][~jan] - y[~jan]), f(models[ref][~jan] - y[~jan]),
                                             n_boot=args.n_boot)
                row.update({"noJan_diff": r2["diff"], "noJan_ci_low": r2["ci_low"], "noJan_ci_high": r2["ci_high"]})
            out.append(row)
    boot = pd.DataFrame(out)
    boot.to_csv(base / "ensemble_bootstrap.csv", index=False)
    print("\n[4] paired weekly block bootstrap (diff < 0 = model better than reference)\n"
          + boot.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--tag", default="step1")
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    pd.set_option("display.width", 220)
    for split in args.splits:
        run(split, args)


if __name__ == "__main__":
    main()
