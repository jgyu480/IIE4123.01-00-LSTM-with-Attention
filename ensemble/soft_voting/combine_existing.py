"""Ensembles of the models that already exist: A, B, C and LEAR (incl. short / long windows).

Input: the export tables of adaptive-weighted-LSTM-Attention/scripts/10_export_predictions.py
  <results>/<split>/export/predictions_{val,test}.csv
Pool of base models:
  A, B, C            seed-averaged LSTM / case mean / LSTM + case attention
  LEAR               average of the four LEAR windows
  LEAR_S, LEAR_L     average of windows 56+84 (short) and 1092+1456 (long); LEAR = (LEAR_S + LEAR_L) / 2,
                     so LEAR is never combined with LEAR_S or LEAR_L
Every subset of 1..max-size models is combined two ways:
  avg   equal weights
  valw  weights >= 0 summing to 1 that minimize validation MSE (grid step 0.05)
Selection uses validation only: per method, the combination with the lowest validation RMSE is the
"selected" one. Test numbers are reported for every combination, but only the selected ones are
honest estimates; picking the best test row would overfit the test year.

Writes ensemble/results/combine/<market>_<split>_{corr_val,corr_test,combos}.csv and summary.csv
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AWLA = HERE.parents[1] / "adaptive-weighted-LSTM-Attention"
sys.path.insert(0, str(AWLA / "src"))
from care import metrics  # noqa: E402

RUNS = {"PJM": AWLA / "results" / "step1", "DE": AWLA / "results" / "de"}
POOL = ["A", "B", "C", "LEAR", "LEAR_S", "LEAR_L"]
LOSSES = {"day_MSE": lambda e: (e ** 2).mean(1), "day_MAE": lambda e: np.abs(e).mean(1)}


def load(folder, part):
    f = pd.read_csv(folder / "export" / f"predictions_{part}.csv")
    f["LEAR_S"] = (f["LEAR_w56"] + f["LEAR_w84"]) / 2
    f["LEAR_L"] = (f["LEAR_w1092"] + f["LEAR_w1456"]) / 2
    n = len(f) // 24
    pred = {k: f[k].to_numpy().reshape(n, 24) for k in POOL}
    flags = {k: f[k].to_numpy().reshape(n, 24)[:, 0].astype(bool) for k in ["extreme_window", "volatile_prev_day"]}
    return pred, f["actual"].to_numpy().reshape(n, 24), flags, pd.to_datetime(f["Date"]).dt.year.mode()[0]


def simplex(n, step):
    g = np.round(np.arange(0, 1 + 1e-9, step), 4)
    return np.array([w for w in itertools.product(g, repeat=n) if abs(sum(w) - 1) < 1e-9])


def valid_combo(c):
    return not ("LEAR" in c and ("LEAR_S" in c or "LEAR_L" in c))


def rmse(p, y):
    return float(np.sqrt(((p - y) ** 2).mean()))


def run(market, split, args):
    folder = RUNS[market] / split
    V, yv, _, _ = load(folder, "val")
    T, y, flags, year = load(folder, "test")
    ext = flags["extreme_window"]
    tag = f"{market}_{split}"
    for part, P, truth in [("val", V, yv), ("test", T, y)]:
        err = np.array([(P[k] - truth).ravel() for k in POOL])
        pd.DataFrame(np.corrcoef(err), POOL, POOL).round(3).to_csv(args.out / f"{tag}_corr_{part}.csv")

    rows, preds = [], {}
    grids = {n: simplex(n, args.step) for n in range(2, args.max_size + 1)}
    for size in range(1, args.max_size + 1):
        for combo in itertools.combinations(POOL, size):
            if not valid_combo(combo):
                continue
            Xv = np.stack([V[k] for k in combo])
            Xt = np.stack([T[k] for k in combo])
            methods = {"single": np.ones(1)} if size == 1 else {"avg": np.full(size, 1 / size)}
            if size > 1:
                mse = (((np.tensordot(grids[size], Xv, 1) - yv) ** 2).mean(axis=(1, 2)))
                methods["valw"] = grids[size][int(np.argmin(mse))]
            for method, w in methods.items():
                pv, pt = np.tensordot(w, Xv, 1), np.tensordot(w, Xt, 1)
                name = "+".join(combo) if size == 1 else f"{method}({'+'.join(combo)})"
                preds[name] = pt
                e = pt - y
                rows.append({"market": market, "split": split, "test_year": year, "model": name, "method": method,
                             "size": size, "weights": " ".join(f"{k}={x:.2f}" for k, x in zip(combo, w)),
                             "val_RMSE": rmse(pv, yv), "MAE": float(np.abs(e).mean()), "RMSE": rmse(pt, y),
                             "ext_MAE": float(np.abs(e[ext]).mean()) if ext.any() else np.nan,
                             "rest_MAE": float(np.abs(e[~ext]).mean())})
    table = pd.DataFrame(rows)
    for method in ["single", "avg", "valw"]:
        sel = table[table.method == method]
        table.loc[sel["val_RMSE"].idxmin(), "selected"] = method
    refs = {"LEAR": "LEAR", "valw(LEAR+A)": "valw(A+LEAR)"}
    for i in table.index[table["selected"].notna()]:
        name = table.at[i, "model"]
        for label, ref in refs.items():
            if name == ref:
                continue
            for loss, f in LOSSES.items():
                r = metrics.block_bootstrap(f(preds[name] - y), f(preds[ref] - y), n_boot=args.n_boot)
                table.at[i, f"{loss}_vs_{label}"] = f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
    table = table.sort_values("val_RMSE")
    table.to_csv(args.out / f"{tag}_combos.csv", index=False)
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"])
    p.add_argument("--max-size", type=int, default=4)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    args.out = HERE / "results" / "combine"
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)
    tables = [run(m, s, args) for m in args.markets for s in args.splits]
    allt = pd.concat(tables)
    allt.to_csv(args.out / "summary.csv", index=False)
    sel = allt[allt["selected"].notna()]
    cols = ["market", "test_year", "selected", "model", "weights", "val_RMSE", "RMSE", "MAE", "ext_MAE",
            "day_MSE_vs_LEAR", "day_MSE_vs_valw(LEAR+A)"]
    print("\n[combinations selected on validation; test numbers]\n" + sel[cols].to_string(index=False))
    ref = allt[allt.model.isin(["LEAR", "valw(A+LEAR)", "valw(A+LEAR_S+LEAR_L)"])]
    print("\n[reference rows]\n" + ref[["market", "test_year", "model", "val_RMSE", "RMSE", "MAE", "ext_MAE"]].to_string(index=False))
    for m in args.markets:
        for s in args.splits:
            c = pd.read_csv(args.out / f"{m}_{s}_corr_test.csv", index_col=0)
            print(f"\n[test hourly error correlation] {m} {s}\n{c.round(2).to_string()}")


if __name__ == "__main__":
    main()
