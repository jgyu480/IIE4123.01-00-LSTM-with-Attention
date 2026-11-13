"""Step 1b: LSTM + mean residual of similar past days (Res-kNN), plus controls.

    prediction = LSTM prediction + alpha * mean(OOF residuals of k most similar earlier days)

Models (k and alpha chosen on validation RMSE; alpha=0 equals the plain LSTM):
  Res-kNN-lstm     similar days by the LSTM's 119-dim features (cosine), as in the original B/C
  Res-kNN-regime   similar days by forecast-time regime features (standardized, Euclidean)
  Res-random       k random earlier days (control: is the similarity search doing anything?)
  Res-Ridge        ridge regression from regime features to the 24-hour residual
Inference-only ablation (same k/alpha as its base model, pool changed at test time):
  *-no2014         2014-01-01..2014-03-31 removed from the case pool

Case pool for target day d: days strictly before d that have an out-of-sample residual
(OOF training blocks, validation days, earlier test days). Residuals of D-1 are known at
forecast time under the existing protocol (prices up to D-1 are observed).
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
from care.retrieval import neighbors, standardize

K_GRID = [5, 10, 20, 40]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
RIDGE_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
EXCLUDE_2014 = ("2014-01-01", "2014-03-31")


def ridge_fit(x, y, lam):
    mx, my = x.mean(0), y.mean(0)
    xc = x - mx
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (y - my))
    return lambda z: (z - mx) @ w + my


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def select(base_val, corrections_val, truth_val):
    """corrections_val: {setting: (val_days,24)}. Returns best setting, alpha and full grid."""
    grid = []
    for setting, corr in corrections_val.items():
        for alpha in ALPHA_GRID:
            grid.append({"setting": setting, "alpha": alpha,
                         "val_RMSE": rmse(base_val + alpha * corr, truth_val)})
    best = min(grid, key=lambda g: (round(g["val_RMSE"], 10), g["alpha"], g["setting"]))
    return best, grid


def run(split, seed, args):
    folder = args.out / split / f"seed_{seed}"
    z = np.load(folder / "residuals.npz")
    meta = json.loads((folder / "oof_meta.json").read_text(encoding="utf-8"))
    m = data.load(split, args.csv)
    truth, base, residual, source = z["truth"], z["pred_main"], z["residual"], z["source"]
    n, ntrain, nval = len(m.days), m.ntrain, m.nval
    val = np.arange(ntrain, ntrain + nval)
    test = np.arange(ntrain + nval, n)
    pool = source >= 0
    oof = source >= 1
    assert np.isfinite(residual[pool]).all()

    keys = {
        "lstm": (standardize(z["features"], ntrain), "cosine"),
        "regime": (standardize(data.regime_features(m), ntrain), "euclidean"),
        "random": (None, "random"),
    }
    days = pd.to_datetime(z["days"])
    exclude = (days >= EXCLUDE_2014[0]) & (days <= EXCLUDE_2014[1])

    test_pred, selection = {"LSTM": base[test]}, {}
    for name, (key, metric) in keys.items():
        rng = np.random.default_rng(seed)
        ids_val = neighbors(key, metric, val, pool, max(K_GRID), rng=rng)
        corr_val = {f"k={k}": residual[ids_val[:, :k]].mean(1) for k in K_GRID}
        best, grid = select(base[val], corr_val, truth[val])
        k = int(best["setting"].split("=")[1])
        ids_test = neighbors(key, metric, test, pool, k, rng=rng)
        label = f"Res-kNN-{name}" if name != "random" else "Res-random"
        test_pred[label] = base[test] + best["alpha"] * residual[ids_test].mean(1)
        selection[label] = {"k": k, "alpha": best["alpha"], "val_RMSE": best["val_RMSE"], "grid": grid}
        if name != "random":
            ids_ex = neighbors(key, metric, test, pool, k, exclude=exclude)
            test_pred[f"{label}-no2014"] = base[test] + best["alpha"] * residual[ids_ex].mean(1)

    # Ridge: fit on OOF days for validation; refit on OOF + validation days for test.
    x = standardize(data.regime_features(m), ntrain)
    corr_val = {f"lambda={lam}": ridge_fit(x[oof], residual[oof], lam)(x[val]) for lam in RIDGE_GRID}
    best, grid = select(base[val], corr_val, truth[val])
    lam = float(best["setting"].split("=")[1])
    fit = oof.copy()
    fit[val] = True
    test_pred["Res-Ridge"] = base[test] + best["alpha"] * ridge_fit(x[fit], residual[fit], lam)(x[test])
    selection["Res-Ridge"] = {"lambda": lam, "alpha": best["alpha"], "val_RMSE": best["val_RMSE"], "grid": grid}

    abs_oof = np.abs(residual[oof])
    taus = {"p95": float(np.quantile(abs_oof, 0.95)), "p99": float(np.quantile(abs_oof, 0.99))}
    masks = {k: v[test] for k, v in data.eval_masks(m, meta["high_threshold"]).items()}
    table = {name: metrics.scores(truth[test], p, high_threshold=meta["high_threshold"], taus=taus, masks=masks)
             for name, p in test_pred.items()}

    out = folder / "res_knn"
    out.mkdir(exist_ok=True)
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"taus": taus, "test": table}, indent=2), encoding="utf-8")
    stamps = (days.values[test][:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    frame = pd.DataFrame({"Date": stamps, "actual": truth[test].ravel()})
    for name, p in test_pred.items():
        frame[name] = p.ravel()
    frame.to_csv(out / "predictions.csv", index=False)
    print(f"\n{split} seed={seed}  (tau p95={taus['p95']:.2f})")
    print(pd.DataFrame(table).T[["MAE", "RMSE", "CVaR5_day_MAE", "worst_day_MAE", "calm_MAE", "crash_MAE"]]
          .round(3).to_string())
    print({k: {kk: v[kk] for kk in v if kk != "grid"} for k, v in selection.items()})


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
