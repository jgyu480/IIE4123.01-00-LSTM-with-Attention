"""Step 3a: LEAR benchmark (Lago et al. 2021, epftoolbox), re-implemented with scikit-learn.

For target day D and each hour h, a separate Lasso model with inputs
  prices of D-1, D-2, D-3, D-7 (4 x 24)
  exogenous 1 and 2 (load forecasts) of D, D-1, D-7 (2 x 3 x 24)
  day-of-week dummies of D (7)
Inputs and target: median/MAD ("invariant") scaling + asinh, fitted on the calibration window.
alpha: LassoLarsIC(aic) on the window, then Lasso(alpha) refit, as in epftoolbox.
noise_variance = var(y): what LassoLarsIC used before scikit-learn 1.1 (epftoolbox era); also needed
for windows shorter than the number of features (56 / 84 days).
Recalibrated every day on the `window` days immediately before D (information known at D only),
so a day's prediction does not depend on the split. Windows 56/84/1092/1456 days; the main LEAR is
their average ("ens", epftoolbox standard). Not on GPU: 24 small Lasso fits per day, parallel on CPU.

Output: results/<tag>/<split>/lear/<part>.npz (idx = project day index, truth, w<window>, ens) in $/MWh.
Runtime on 15 CPU cores: about 15 minutes per split and part.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import median_abs_deviation
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC

from care import data

OFFSET = 7          # project day index i (data.load) = calendar day i + 7


def build(raw, dayofweek):
    """Features X (calendar days, 247) and prices Y (calendar days, 24); rows < 7 are invalid."""
    P, X1, X2 = (raw[:, j].reshape(-1, 24) for j in range(3))
    X = np.full((len(P), 247), np.nan)
    for D in range(7, len(P)):
        X[D] = np.concatenate([P[D - 1], P[D - 2], P[D - 3], P[D - 7],
                               X1[D], X1[D - 1], X1[D - 7], X2[D], X2[D - 1], X2[D - 7],
                               np.eye(7)[dayofweek[D]]])
    return X, P


def invariant(train):
    med = np.median(train, 0)
    mad = median_abs_deviation(train, axis=0, scale="normal")
    return med, np.where(mad < 1e-8, 1.0, mad)


def forecast_day(X, Y, D, window):
    lo = max(7, D - window)
    Xt, Yt = X[lo:D].copy(), Y[lo:D]
    med_x, mad_x = invariant(Xt[:, :-7])                  # day-of-week dummies are not scaled
    Xt[:, :-7] = np.arcsinh((Xt[:, :-7] - med_x) / mad_x)
    x = X[D].copy()
    x[:-7] = np.arcsinh((x[:-7] - med_x) / mad_x)
    out = np.empty(24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for h in range(24):
            med_y, mad_y = invariant(Yt[:, h:h + 1])
            yt = np.arcsinh((Yt[:, h] - med_y[0]) / mad_y[0])
            alpha = LassoLarsIC(criterion="aic", max_iter=2500, noise_variance=float(np.var(yt))).fit(Xt, yt).alpha_
            model = Lasso(alpha=alpha, max_iter=2500).fit(Xt, yt)
            out[h] = np.sinh(model.predict(x[None])[0]) * mad_y[0] + med_y[0]
    return out, D - lo


def run(split, part, args):
    out = args.out / split / "lear" / f"{part}.npz"
    if out.exists() and not args.limit:
        print(f"skip (done): {out}", flush=True)
        return
    m = data.load(split, args.csv)
    calendar = m.hours[::24]
    assert calendar[OFFSET] == m.days[0]
    X, Y = build(m.raw, calendar.dayofweek.to_numpy())
    idx = (np.arange(m.ntrain, m.ntrain + m.nval) if part == "val"
           else np.arange(m.ntrain + m.nval, len(m.days)))
    if args.limit:
        idx = idx[:args.limit]
    cal = idx + OFFSET
    preds = {}
    for w in args.windows:
        t = time.time()
        res = Parallel(n_jobs=args.jobs)(delayed(forecast_day)(X, Y, D, w) for D in cal)
        preds[f"w{w}"] = np.array([r[0] for r in res])
        err = preds[f"w{w}"] - Y[cal]
        print(f"{split} {part} window={w} (min days used={min(r[1] for r in res)}) "
              f"MAE={np.abs(err).mean():.4f} RMSE={np.sqrt((err ** 2).mean()):.4f} [{time.time() - t:.0f}s]", flush=True)
    ens = np.mean(list(preds.values()), 0)
    err = ens - Y[cal]
    print(f"{split} {part} LEAR-ens MAE={np.abs(err).mean():.4f} RMSE={np.sqrt((err ** 2).mean()):.4f}", flush=True)
    if args.limit:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, idx=idx, truth=Y[cal], windows=np.array(args.windows), ens=ens, **preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--parts", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    p.add_argument("--windows", nargs="+", type=int, default=[56, 84, 1092, 1456])
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0, help="only the first N days, nothing saved (quick check)")
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    for split in args.splits:
        for part in args.parts:
            run(split, part, args)


if __name__ == "__main__":
    main()
