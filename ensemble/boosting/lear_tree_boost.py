"""Boosting, stage 2 with trees: LEAR (stage 1) + gradient-boosted trees fitted to LEAR's residuals.

    final(d, h) = LEAR(d, h) + c * tree(features(d, h))

Stage 1  LEAR forecasts of every day (ensemble/lear_history.py). LEAR is recalibrated daily on past
         days only, so its residuals r = actual - LEAR are out-of-sample on every day, training days
         included. No fold models are needed (unlike the residual corrections tried before).
Stage 2  scikit-learn HistGradientBoostingRegressor (the XGBoost / LightGBM family) on r.
         One row per (day, hour). Features, all known before the target day starts:
           calendar   hour, weekday, month
           LEAR       LEAR at the hour, LEAR day mean, short-minus-long window gap at the hour
                      (short = 56/84-day windows, long = 1092/1456-day windows), LEAR minus D-1 price
           prices     D-1 and D-7 price at the hour, D-1 mean / std / max / min, previous 7-day mean
           exogenous  both load forecasts at the hour and their D mean, change of D mean vs D-1 mean (%)
Protocol (same split as the project; test days never used for any choice):
  fit on training days -> on validation days choose the number of trees (staged predictions) and the
  shrinkage c in {0, 0.1, ..., 1} -> apply to test days. Two losses: squared_error and
  absolute_error; the one with the lower validation RMSE is the main model ("boost").
Reported against LEAR and against soft voting valw(A + LEAR_S + LEAR_L) (weights from validation).

Output: ensemble/boosting/results/<market>/<split>/predictions_{val,test}.csv, summary.json
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
ENS = HERE.parent
AWLA = ENS.parent / "adaptive-weighted-LSTM-Attention"
sys.path.insert(0, str(AWLA / "src"))
from care import data, metrics  # noqa: E402

CSV = {"PJM": data.DEFAULT_CSV, **{m: AWLA / "data" / "epftoolbox" / f"{m}.csv" for m in ["NP", "BE", "FR", "DE"]}}
EXPORT_TAG = {"PJM": "step1", "DE": "de", "NP": "np", "BE": "be", "FR": "fr"}
SHRINK = np.round(np.arange(0, 1.0001, 0.1), 2)
LOSSES = {"day_MSE": lambda e: (e ** 2).mean(1), "day_MAE": lambda e: np.abs(e).mean(1)}
FEATURES = ["hour", "weekday", "month", "lear", "lear_mean", "lear_gap_SL", "lear_minus_p1",
            "p1", "p7", "p1_mean", "p1_std", "p1_max", "p1_min", "p7d_mean",
            "x1", "x2", "x1_mean", "x2_mean", "x1_chg", "x2_chg"]


def features(m, lear, days):
    """Rows (len(days) * 24, len(FEATURES)) in day-major, hour-minor order."""
    p, x1, x2 = m.raw[:, 0], m.raw[:, 1], m.raw[:, 2]
    short = (lear["w56"] + lear["w84"]) / 2
    long_ = (lear["w1092"] + lear["w1456"]) / 2
    rows = []
    for j, d in enumerate(days):
        t = m.starts[d]
        prev, week = p[t - 24:t], p[t - 168:t]
        day_feat = [m.days[d].dayofweek, m.days[d].month, lear["ens"][j].mean(),
                    prev.mean(), prev.std(), prev.max(), prev.min(), week.mean(),
                    x1[t:t + 24].mean(), x2[t:t + 24].mean(),
                    100 * (x1[t:t + 24].mean() / x1[t - 24:t].mean() - 1),
                    100 * (x2[t:t + 24].mean() / x2[t - 24:t].mean() - 1)]
        for h in range(24):
            rows.append([h, day_feat[0], day_feat[1], lear["ens"][j, h], day_feat[2],
                         short[j, h] - long_[j, h], lear["ens"][j, h] - p[t - 24 + h],
                         p[t - 24 + h], p[t - 168 + h], *day_feat[3:8],
                         x1[t + h], x2[t + h], *day_feat[8:]])
    return np.asarray(rows, dtype=np.float64)


def fit_one(loss, X, r, Xv, rv, Xt, args):
    model = HistGradientBoostingRegressor(
        loss=loss, learning_rate=args.lr, max_iter=args.max_iter, max_leaf_nodes=31,
        min_samples_leaf=args.min_leaf, l2_regularization=1.0, early_stopping=False, random_state=0)
    model.fit(X, r)
    best_it, best_mse, best_pv = 0, np.inf, None
    for it, pv in enumerate(model.staged_predict(Xv), start=1):     # trees chosen on validation only
        mse = ((rv - pv) ** 2).mean()
        if mse < best_mse:
            best_it, best_mse, best_pv = it, mse, pv
    pt = next(itertools.islice(model.staged_predict(Xt), best_it - 1, None))
    c = min(SHRINK, key=lambda s: ((rv - s * best_pv) ** 2).mean())
    return best_pv, pt, best_it, float(c)


def soft_vote(market, split, val_days, test_days, m):
    """valw(A + LEAR_S + LEAR_L) from the 10_export_predictions tables, aligned to our days."""
    folder = AWLA / "results" / EXPORT_TAG[market] / split / "export"
    if not folder.exists():
        return None
    out = []
    for part, days in [("val", val_days), ("test", test_days)]:
        f = pd.read_csv(folder / f"predictions_{part}.csv")
        f["S"] = (f["LEAR_w56"] + f["LEAR_w84"]) / 2
        f["L"] = (f["LEAR_w1092"] + f["LEAR_w1456"]) / 2
        n = len(f) // 24
        assert pd.Timestamp(f["Date"].iloc[0]) == m.days[days[0]] and n == len(days)
        out.append({k: f[k].to_numpy().reshape(n, 24) for k in ["A", "S", "L", "actual"]})
    V, T = out
    grid = [w for w in itertools.product(SHRINK, repeat=3) if abs(sum(w) - 1) < 1e-9]
    w = min(grid, key=lambda w: ((w[0] * V["A"] + w[1] * V["S"] + w[2] * V["L"] - V["actual"]) ** 2).mean())
    return w[0] * T["A"] + w[1] * T["S"] + w[2] * T["L"], dict(zip(["A", "LEAR_S", "LEAR_L"], map(float, w)))


def run(market, split, args):
    t0 = time.time()
    m = data.load(split, CSV[market])
    z = np.load(ENS / "results" / market / "lear_all.npz")
    keep = z["day"] < len(m.days)
    day = z["day"][keep]
    lear = {k: z[k][keep] for k in ["ens", "w56", "w84", "w1092", "w1456"]}
    truth = m.truth[day]
    assert np.allclose(truth, z["truth"][keep])
    part = np.where(day < m.ntrain, "train", np.where(day < m.ntrain + m.nval, "val", "test"))
    X = features(m, lear, day)
    r = (truth - lear["ens"]).ravel()
    hour_part = np.repeat(part, 24)
    tr, va, te = (hour_part == k for k in ["train", "val", "test"])

    runs = {}
    for loss in ["squared_error", "absolute_error"]:
        pv, pt, n_trees, c = fit_one(loss, X[tr], r[tr], X[va], r[va], X[te], args)
        runs[loss] = {"pv": pv, "pt": pt, "trees": n_trees, "c": c,
                      "val_RMSE": float(np.sqrt(((r[va] - c * pv) ** 2).mean()))}
    main = min(runs, key=lambda k: runs[k]["val_RMSE"])

    nv, nt = int((part == "val").sum()), int((part == "test").sum())
    L_val, L_test = lear["ens"][part == "val"], lear["ens"][part == "test"]
    y_val, y_test = truth[part == "val"], truth[part == "test"]
    P = {"LEAR": L_test}
    for loss, rr in runs.items():
        tag = "sq" if loss == "squared_error" else "abs"
        P[f"boost_{tag}"] = L_test + rr["c"] * rr["pt"].reshape(nt, 24)
        P[f"boost_{tag}_c1"] = L_test + rr["pt"].reshape(nt, 24)
    P["boost"] = P["boost_sq" if main == "squared_error" else "boost_abs"]
    sv = soft_vote(market, split, day[part == "val"], day[part == "test"], m)
    if sv is not None:
        P["soft_vote"], sv_w = sv
    flagged = data.extreme_days(m)[0][day[part == "test"]]
    ext = np.zeros(nt, bool)
    for a, b in data.events(flagged):
        ext[a:b + 1] = True

    table = {}
    for k, p in P.items():
        e = p - y_test
        table[k] = {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                    "ext_MAE": float(np.abs(e[ext]).mean()) if ext.any() else None,
                    "rest_MAE": float(np.abs(e[~ext]).mean())}
    boots = []
    for model in ["boost", "boost_sq", "boost_abs"]:
        for ref in ["LEAR", "soft_vote"]:
            if ref not in P:
                continue
            for loss, f in LOSSES.items():
                b = metrics.block_bootstrap(f(P[model] - y_test), f(P[ref] - y_test), n_boot=args.n_boot)
                boots.append({"model": model, "vs": ref, "loss": loss, **b})
    if "soft_vote" in P:
        for loss, f in LOSSES.items():
            b = metrics.block_bootstrap(f(P["soft_vote"] - y_test), f(P["LEAR"] - y_test), n_boot=args.n_boot)
            boots.append({"model": "soft_vote", "vs": "LEAR", "loss": loss, **b})

    out = HERE / "results" / market / split
    out.mkdir(parents=True, exist_ok=True)
    for name, days_p, base, extra in [("val", day[part == "val"], L_val, None), ("test", day[part == "test"], L_test, P)]:
        stamps = (m.days.values[days_p][:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
        f = pd.DataFrame({"Date": stamps, "actual": truth[np.isin(day, days_p)].ravel(), "LEAR": base.ravel()})
        for loss, rr in runs.items():
            tag = "sq" if loss == "squared_error" else "abs"
            f[f"tree_{tag}"] = (rr["pv"] if name == "val" else rr["pt"])      # raw residual prediction
        if extra is not None:
            for k, p in extra.items():
                if k != "LEAR":
                    f[k] = p.ravel()
        f.to_csv(out / f"predictions_{name}.csv", index=False, float_format="%.4f")
    summary = {"market": market, "split": split, "test_first": str(m.days[day[part == "test"][0]].date()),
               "train_days": int((part == "train").sum()), "train_first": str(m.days[day[0]].date()),
               "val_days": nv, "test_days": nt, "extreme_window_days": int(ext.sum()),
               "main_loss": main, "features": FEATURES,
               "runs": {k: {kk: v for kk, v in rr.items() if kk in ("trees", "c", "val_RMSE")} for k, rr in runs.items()},
               "soft_vote_weights": sv_w if sv is not None else None, "test": table, "bootstrap": boots}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n===== {market} {split}: test {summary['test_first']}~, train {summary['train_days']} days from "
          f"{summary['train_first']}, {time.time() - t0:.0f}s =====")
    for k, rr in summary["runs"].items():
        print(f"  {k:15s} trees={rr['trees']:4d} shrink c={rr['c']:.1f} val_RMSE(residual)={rr['val_RMSE']:.3f}"
              + ("  <- main" if k == main else ""))
    if sv is not None:
        print(f"  soft vote weights: {sv_w}")
    print(pd.DataFrame(table).T.round(3).to_string())
    print(pd.DataFrame(boots).round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=["PJM"], choices=list(CSV))
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--min-leaf", type=int, default=100)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    pd.set_option("display.width", 200)
    for market in args.markets:
        for split in args.splits:
            run(market, split, args)


if __name__ == "__main__":
    main()
