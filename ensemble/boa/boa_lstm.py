"""BOA online aggregation (team lead's pipeline) combined with LSTM boosting, two ways.

Port of the lead's lear_a_simple_improvements/run.py (branch results/all-lear-a-experiments-20261120_164542,
experiment_archive/.../results/lear_a_simple_improvements/run.py). Every step and every number of that
script is kept; the base model is now a parameter and the reporting is extended.

Pipeline on a base model X (validation + test days only, all online: day t uses actual prices of days < t):
  TrainingRecent  X + beta * (A - X), beta per hour from an expanding and a 14-day least-squares fit on
                  past days (ridge towards 0.5)
  4 correction families on TrainingRecent, each configured on validation days 60..181:
    LevelShape  recent daily-level and hourly-shape bias
    Reconcile   independent daily / 6-hour ridge forecasts (weekly refit on past days)
    BiasMix     rolling bias + A-vs-X mixture shift
    MeanError28_half  half of the 28-day mean error
  BOA             Bernstein Online Aggregation of the members, starting at validation day 182; loss
                  (MAE / MSE) and hour groups (1 / 4) chosen on validation days 182..272; days 273..363
                  are a validation holdout; test days are never used for a choice.
Variants:
  BOA5     X = LEAR, the lead's 5 members (reproduces the lead's saved predictions exactly)
  BOA6     BOA5 members + LSTM boosting as a sixth expert
  BOA5LB   X = LSTM boosting instead of LEAR, everything else unchanged
LSTM boosting = ensemble/boosting/lear_lstm_boost.py --select-frac 0.5: its choices used validation days
  0..181 only, so it is out-of-sample from day 182 on (all BOA days and the test year). Caveat for BOA5LB:
  the correction families are configured on days 60..181, where LSTM boosting is slightly in-sample
  (its early stopping and c), so those configurations may trust it a little too much. Test is clean.
A / LEAR inputs (--a-source):
  repo  adaptive-weighted-LSTM-Attention/results/<tag>/<split>/export (all 4 test years)
  lead  the export the lead used, copied from the branch to results/PJM/split2_2018/lead_export/
        (PJM 2018 only; its A differs from the repo export: test MAE 4.003 vs 4.028)
Not ported: OriginalRecent (a reference read from another experiment's files, not used by BOA).

Run:    python ensemble/boa/boa_lstm.py                                    (4 test years, repo A)
        python ensemble/boa/boa_lstm.py --a-source lead --markets PJM --splits split2_2018
Output: ensemble/boa/results/<market>/<split>/a_<source>/{scores.csv, bootstrap.csv, boa_weights.csv,
        boa_selection_trials.csv, predictions_test.csv, selection.json}
        ensemble/boa/results/summary_a_<source>.csv (one column per test year)
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ENS = HERE.parent
sys.path.insert(0, str(ENS / "boosting"))
import lear_tree_boost as tree  # noqa: E402  (also puts adaptive-weighted-LSTM-Attention/src on the path)
from care import data, metrics  # noqa: E402

FAMILIES = ["LevelShape", "Reconcile", "BiasMix"]


def log(message):
    print(message, flush=True)


def score(actual, prediction):
    e = prediction - actual
    mean_e = e.mean(axis=1)
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MSE": float(np.mean(e ** 2)),
            "level_MSE": float(np.mean(mean_e ** 2)), "shape_MSE": float(np.mean((e - mean_e[:, None]) ** 2))}


def daily_table(folder, columns):
    frames = []
    for part in ["val", "test"]:
        t = pd.read_csv(folder / f"predictions_{part}.csv", parse_dates=["Date"])
        t["part"] = part
        frames.append(t)
    frame = pd.concat(frames, ignore_index=True)
    assert frame["Date"].is_monotonic_increasing and not frame["Date"].duplicated().any()
    groups = list(frame.groupby(frame["Date"].dt.normalize(), sort=True))
    for day, g in groups:
        assert len(g) == 24 and g["Date"].dt.hour.tolist() == list(range(24)) and g["part"].nunique() == 1, day
    dates = pd.DatetimeIndex([d for d, _ in groups])
    parts = np.array([g["part"].iloc[0] for _, g in groups])
    return dates, parts, {c: np.stack([g[c].to_numpy(dtype=float) for _, g in groups]) for c in columns}


def aggregate_forecasts(raw_table, positions, dates, out):
    """Independent daily / 6-hour ridge forecasts (Reconcile input); they do not depend on the base model."""
    raw = raw_table.to_numpy(dtype=float)
    calendar_raw = raw.reshape(-1, 24, 3)
    calendar_dates = pd.DatetimeIndex(raw_table.index[::24])
    DAYS, N = len(calendar_raw), len(positions)
    S = np.zeros((5, 24))
    S[0] = 1 / 24
    for j in range(4):
        S[j + 1, 6 * j:6 * (j + 1)] = 1 / 6
    aggregate_y = calendar_raw[:, :, 0] @ S.T
    aggregate_load = np.stack([calendar_raw[:, :, c] @ S.T for c in [1, 2]], axis=-1)
    aggregate_x = None
    for dd in range(7, DAYS):
        prev = calendar_raw[dd - 1, :, 0]
        vector = np.concatenate([aggregate_y[dd - 1], aggregate_y[dd - 2], aggregate_y[dd - 7],
                                 aggregate_load[dd].ravel(), (aggregate_load[dd] - aggregate_load[dd - 1]).ravel(),
                                 [prev.min(), prev.max(), prev.std(), prev[-1]], np.eye(7)[calendar_dates[dd].dayofweek]])
        if aggregate_x is None:
            aggregate_x = np.full((DAYS, len(vector)), np.nan)
        aggregate_x[dd] = vector
    calendar_ids = positions // 24
    raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    aggs = {}
    for window, ridge_alpha in itertools.product([365, 1092], [10.0, 100.0]):
        cache = out / f"aggregate_w{window}_a{ridge_alpha:g}.npz"
        identity = f"{raw_hash}:{window}:{ridge_alpha}:{dates[0].date()}:{N}"
        agg = None
        if cache.is_file():
            with np.load(cache) as z:
                if str(z["identity"].item()) == identity:
                    agg = z["prediction"].copy()
        if agg is None:
            agg = np.empty((N, 5))
            model = None
            for i, dd in enumerate(calendar_ids):
                if model is None or i % 7 == 0:                      # weekly refit on days before dd
                    model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
                    model.fit(aggregate_x[max(7, dd - window):dd], aggregate_y[max(7, dd - window):dd])
                agg[i] = model.predict(aggregate_x[dd:dd + 1])[0]
            np.savez_compressed(cache, prediction=agg, identity=identity)
        aggs[(window, ridge_alpha)] = agg
    return aggs, S


def online_members(Y, A, X, warm, choose, scale, aggs, S):
    """The lead's TrainingRecent + correction families on base X. Returns (predictions, selected configs)."""
    N = len(Y)
    difference = A - X
    csn = np.vstack([np.zeros(24), np.cumsum(difference * (Y - X), axis=0)])
    csd = np.vstack([np.zeros(24), np.cumsum(difference ** 2, axis=0)])
    beta = np.full_like(Y, 0.5)
    for t in range(60, N):
        energy = np.maximum(csd[t] / t, 1e-6)
        ridge = 10 * energy
        hourly = np.clip((csn[t] + 0.5 * ridge) / (csd[t] + ridge), 0, 1)
        start = max(0, t - 14)
        recent = np.clip((csn[t] - csn[start] + ridge * hourly) / (csd[t] - csd[start] + ridge), 0, 1)
        beta[t] = 0.5 * hourly + 0.5 * recent
    BASE = X + beta * difference
    ERR = Y - BASE
    predictions, selected = {"TrainingRecent": BASE}, {"TrainingRecent_beta_test_mean": None}

    def consider(family, config, prediction, best):
        assert np.isfinite(prediction[60:]).all(), family
        values = score(Y[choose], prediction[choose])
        for objective in ["MAE", "RMSE"]:
            cur = best.get(objective)
            if cur is None or values[objective] < cur[0] - 1e-10:
                best[objective] = (values[objective], config.copy(), prediction.copy())

    def register(family, best):
        for objective, (_, config, prediction) in best.items():
            predictions[f"{family}_{objective}"] = prediction
            selected[f"{family}_{objective}"] = config

    # 1. LevelShape
    clip_value = max(float(np.quantile(np.abs(ERR[warm]), 0.95)), 1e-6)
    best = {}
    for robust, level_window, shape_window in itertools.product([False, True], [3, 7, 14], [14, 28, 56]):
        observed = np.clip(ERR, -clip_value, clip_value) if robust else ERR
        obs_level = observed.mean(axis=1)
        obs_shape = observed - obs_level[:, None]
        estimate_level, estimate_shape = np.zeros(N), np.zeros_like(Y)
        for t in range(60, N):
            estimate_level[t] = obs_level[max(0, t - level_window):t].mean()
            v = obs_shape[max(0, t - shape_window):t].mean(axis=0)
            padded = np.pad(v, (1, 1), mode="edge")
            v = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
            estimate_shape[t] = v - v.mean()
        for alpha, gamma in itertools.product([0.0, 0.25, 0.5], repeat=2):
            consider("LevelShape", {"robust": robust, "level_window": level_window, "shape_window": shape_window,
                                    "level_strength": alpha, "shape_strength": gamma},
                     BASE + alpha * estimate_level[:, None] + gamma * estimate_shape, best)
    register("LevelShape", best)

    # 2. Reconcile
    best = {}
    for (window, ridge_alpha), agg in aggs.items():
        daily_gap = agg[:, 0] - BASE.mean(axis=1)
        for strength in [0.0, 0.25, 0.5, 1.0]:
            consider("Reconcile", {"mode": "daily_mean", "window": window, "ridge_alpha": ridge_alpha,
                                   "strength": strength}, BASE + strength * daily_gap[:, None], best)
        hour_variance = max(float(np.mean((BASE[warm] - Y[warm]) ** 2)), 1e-6)
        agg_variance = np.mean((agg[warm] - Y[warm] @ S.T) ** 2, axis=0)
        W = np.diag(hour_variance / np.maximum(agg_variance, 1e-6))
        for rho in [0.1, 1.0, 10.0]:
            pr = np.linalg.solve(np.eye(24) + rho * S.T @ W @ S, (BASE + rho * agg @ W @ S).T).T
            consider("Reconcile", {"mode": "daily_and_6hour", "window": window, "ridge_alpha": ridge_alpha,
                                   "rho": rho}, pr, best)
    register("Reconcile", best)

    # 3. BiasMix
    best = {}
    for window, regularization, groups in itertools.product([14, 28, 56], [0.1, 1.0, 10.0], [1, 4]):
        pr = BASE.copy()
        width = 24 // groups
        for t in range(60, N):
            start = max(0, t - window)
            day_weights = np.exp(-np.log(2) * (t - 1 - np.arange(start, t)) / (window / 2))
            for g in range(groups):
                hours = slice(g * width, (g + 1) * width)
                xgap = difference[start:t, hours].ravel() / scale
                target = ERR[start:t, hours].ravel() / scale
                weights = np.repeat(day_weights, width)
                weights = weights / weights.sum()
                Xm = np.column_stack([np.ones(len(xgap)), xgap])
                matrix = Xm.T @ (weights[:, None] * Xm) + regularization * np.diag([1.0, 4.0])
                theta = np.linalg.solve(matrix, Xm.T @ (weights * target))
                pr[t, hours] += scale * theta[0] + np.clip(theta[1], -0.25, 0.25) * difference[t, hours]
        consider("BiasMix", {"window": window, "regularization": regularization, "hour_groups": groups}, pr, best)
    register("BiasMix", best)

    mean_correction = np.zeros_like(Y)
    for t in range(60, N):
        mean_correction[t] = ERR[max(0, t - 28):t].mean(axis=0)
    predictions["MeanError28_half"] = BASE + 0.5 * mean_correction
    return predictions, selected, beta


def boa(Y, experts, loss, group_count, start_boa, scale):
    """Bernstein Online Aggregation, unchanged from the lead's script. experts: (days, hours, experts)."""
    N = len(Y)
    count = experts.shape[-1]
    width = 24 // group_count
    initial = np.full((group_count, count), 1 / count)
    weight = initial.copy()
    bound, variance, regret, rate = (np.zeros_like(weight) for _ in range(4))
    output = np.full_like(Y, np.nan)
    saved = np.full((N, group_count, count), np.nan)
    for t in range(start_boa, N):
        saved[t] = weight                       # forecast with weights from previous days, then update
        for g in range(group_count):
            hours = slice(g * width, (g + 1) * width)
            pe = experts[t, hours] / scale
            forecast = pe @ weight[g]
            output[t, hours] = forecast * scale
            truth = Y[t, hours] / scale
            gradient = 2 * (forecast - truth) if loss == "MSE" else np.sign(forecast - truth)
            excess = np.mean(gradient[:, None] * (pe - forecast[:, None]), axis=0)
            bound[g] = np.maximum(bound[g], np.abs(excess))
            variance[g] += excess ** 2
            new_rate = np.minimum(np.sqrt(-np.log(initial[g]) / np.maximum(variance[g], 1e-12)),
                                  1 / (2 * np.maximum(bound[g], 1e-12)))
            regret[g] += 0.5 * excess * (1 + new_rate * excess) + bound[g] * (2 * rate[g] * excess > 1)
            rate[g] = new_rate
            weight[g] = softmax(np.log(initial[g]) + np.log(np.maximum(new_rate, 1e-12)) - new_rate * regret[g])
    return output, saved


def mark(b):
    return "○" if b["ci_high"] < 0 else ("×" if b["ci_low"] > 0 else "−")


def run(market, split, a_source, args):
    base_dir = HERE / "results" / market / split
    if a_source == "lead":
        inputs = base_dir / "lead_export"
        if not inputs.exists():
            log(f"skip {market} {split}: no lead export")
            return None
    else:
        inputs = tree.AWLA / "results" / tree.EXPORT_TAG[market] / split / "export"
    OUT = base_dir / f"a_{a_source}"
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"\n===== {market} {split}, A from {a_source} ({inputs.relative_to(ENS.parent)}) =====")
    dates, parts, d = daily_table(inputs, ["actual", "A", "LEAR"])
    Y, A, L = d["actual"], d["A"], d["LEAR"]
    ldates, _, lb = daily_table(ENS / "boosting" / "results" / market / split / args.lstm, ["actual", "lstm_boost", "LEAR"])
    assert ldates.equals(dates) and np.allclose(lb["actual"], Y, atol=2e-3) and np.allclose(lb["LEAR"], L, atol=2e-3)
    LB = lb["lstm_boost"]
    val = np.flatnonzero(parts == "val")
    test = np.flatnonzero(parts == "test")
    assert len(val) >= 300

    raw_table = pd.read_csv(tree.CSV[market], index_col=0)
    raw_table.index = pd.to_datetime(raw_table.index)
    positions = raw_table.index.get_indexer(dates)
    assert (positions >= 168).all()
    raw = raw_table.to_numpy(dtype=float)
    assert np.allclose(np.stack([raw[t:t + 24, 0] for t in positions]), Y, atol=0.002, rtol=0)

    warm = val[:60]
    cut1, cut2 = len(val) // 2, 3 * len(val) // 4
    choose, boa_choose, hold = val[60:cut1], val[cut1:cut2], val[cut2:]
    scale = max(float(Y[warm].std()), 1e-6)
    start_boa = int(boa_choose[0])

    aggs, S = aggregate_forecasts(raw_table, positions, dates, base_dir)
    predictions = {"LEAR": L, "A": A, "LSTM_boost": LB}
    selected, betas = {}, {}
    for prefix, X in [("", L), ("LB:", LB)]:
        preds, sel, beta = online_members(Y, A, X, warm, choose, scale, aggs, S)
        predictions.update({prefix + k: v for k, v in preds.items()})
        selected.update({prefix + k: v for k, v in sel.items() if v is not None})
        betas[prefix or "LEAR:"] = float(beta[test].mean())
    log(f"mean beta (weight on A) in test: base LEAR {betas['LEAR:']:.3f}, base LSTM boosting {betas['LB:']:.3f}")

    variants = {"BOA5": ("", []), "BOA6": ("", ["LSTM_boost"]), "BOA5LB": ("LB:", [])}
    weight_rows, boa_trials = [], []
    for tag, (prefix, extra) in variants.items():
        for objective in ["MAE", "RMSE"]:
            members = [prefix + n for n in ["TrainingRecent", "MeanError28_half"]
                       + [f"{f}_{objective}" for f in FAMILIES]] + extra
            experts = np.stack([predictions[n] for n in members], axis=-1)
            options = []
            for loss, gc in itertools.product(["MAE", "MSE"], [1, 4]):
                out, w = boa(Y, experts, loss, gc, start_boa, scale)
                values = score(Y[boa_choose], out[boa_choose])
                boa_trials.append({"model": f"{tag}_{objective}", "loss": loss, "hour_groups": gc, **values})
                options.append((values[objective], {"loss": loss, "hour_groups": gc, "members": members}, out, w))
            _, config, out, w = min(options, key=lambda o: o[0])
            name = f"{tag}_{objective}"
            predictions[name], selected[name] = out, config
            for part_name, ids in [("boa_choose", boa_choose), ("holdout", hold), ("test", test)]:
                mean_w = np.nanmean(w[ids], axis=(0, 1))
                weight_rows.append({"model": name, "days": part_name,
                                    **{n.replace(f"_{objective}", ""): v for n, v in zip(members, mean_w.round(4))}})

    # Reporting (test never used above).
    repro = None
    if a_source == "lead":
        lead = pd.read_csv(inputs / "lead_boa_predictions_test.csv")
        repro = float(np.abs(predictions["BOA5_RMSE"][test] - lead["BOA_RMSE"].to_numpy().reshape(-1, 24)).max())
        log(f"BOA5 reproduction vs the lead's saved BOA_RMSE test predictions: max abs diff {repro:.2e}")

    m = data.load(split, tree.CSV[market])
    test_days = np.arange(m.ntrain + m.nval, len(m.days))
    assert m.days[test_days[0]] == dates[test[0]] and len(test_days) == len(test)
    ext = np.zeros(len(test), bool)
    for a, b in data.events(data.extreme_days(m)[0][test_days]):
        ext[a:b + 1] = True

    show = ["LEAR", "A", "LSTM_boost", "TrainingRecent", "LB:TrainingRecent", "LevelShape_RMSE", "LB:LevelShape_RMSE",
            "BOA5_RMSE", "BOA6_RMSE", "BOA5LB_RMSE", "BOA5_MAE", "BOA6_MAE", "BOA5LB_MAE"]
    rows = []
    for name in show:
        pr = predictions[name]
        e = pr[test] - Y[test]
        rows.append({"method": name, **{f"holdout_{k}": v for k, v in score(Y[hold], pr[hold]).items() if k in ("MAE", "RMSE")},
                     "test_MAE": float(np.abs(e).mean()), "test_RMSE": float(np.sqrt((e ** 2).mean())),
                     "test_ext_MAE": float(np.abs(e[ext]).mean()), "test_rest_MAE": float(np.abs(e[~ext]).mean())})
    scores = pd.DataFrame(rows)
    LOSSES = {"day_MSE": lambda e: (e ** 2).mean(1), "day_MAE": lambda e: np.abs(e).mean(1)}
    pairs = [("BOA5LB_RMSE", "BOA5_RMSE"), ("BOA6_RMSE", "BOA5_RMSE"), ("BOA5LB_RMSE", "BOA6_RMSE"),
             ("BOA5_RMSE", "LSTM_boost"), ("BOA6_RMSE", "LSTM_boost"), ("BOA5LB_RMSE", "LSTM_boost"),
             ("BOA5LB_RMSE", "LB:TrainingRecent"), ("BOA5_RMSE", "TrainingRecent"), ("LSTM_boost", "LEAR")]
    boots = []
    for a, b in pairs:
        for loss, f in LOSSES.items():
            r = metrics.block_bootstrap(f(predictions[a][test] - Y[test]), f(predictions[b][test] - Y[test]), n_boot=args.n_boot)
            boots.append({"model": a, "vs": b, "loss": loss, **r})
    boots = pd.DataFrame(boots)
    weights = pd.DataFrame(weight_rows)

    scores.to_csv(OUT / "scores.csv", index=False)
    boots.to_csv(OUT / "bootstrap.csv", index=False)
    weights.to_csv(OUT / "boa_weights.csv", index=False)
    pd.DataFrame(boa_trials).to_csv(OUT / "boa_selection_trials.csv", index=False)
    stamps = (dates.values[test][:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    pd.DataFrame({"Date": stamps, "actual": Y[test].ravel(), **{k: predictions[k][test].ravel() for k in show}}) \
        .to_csv(OUT / "predictions_test.csv", index=False, float_format="%.4f")
    (OUT / "selection.json").write_text(json.dumps({"selected": selected, "lstm_folder": args.lstm, "a_source": a_source,
                                                    "mean_beta_test": betas, "boa5_reproduction_max_abs_diff": repro,
                                                    "extreme_window_days": int(ext.sum())}, indent=2), encoding="utf-8")
    log(scores.round(3).to_string(index=False))
    log(boots.round(3).to_string(index=False))
    test_w = weights[weights["days"] == "test"].set_index("model")
    log("mean BOA weights in test\n" + test_w.drop(columns="days").dropna(axis=1, how="all").to_string())

    year = f"{market} {m.days[test_days[0]].year + (1 if m.days[test_days[0]].month == 12 else 0)}"
    summary = {"test_year": year, "A_source": a_source, "beta_A_base_LEAR": betas["LEAR:"], "beta_A_base_LB": betas["LB:"],
               "LB_weight_in_BOA6": float(test_w.loc["BOA6_RMSE", "LSTM_boost"])}
    for name in ["LEAR", "LSTM_boost", "BOA5_RMSE", "BOA6_RMSE", "BOA5LB_RMSE"]:
        r = scores.set_index("method").loc[name]
        summary[f"{name}_RMSE"], summary[f"{name}_MAE"], summary[f"{name}_extMAE"] = \
            r["test_RMSE"], r["test_MAE"], r["test_ext_MAE"]
    for (a, b), g in boots.groupby(["model", "vs"], sort=False):
        summary[f"{a} vs {b}"] = " / ".join(mark(row) for _, row in g.iterrows())   # day_MSE / day_MAE
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=["PJM", "DE"], choices=["PJM", "DE"])
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--a-source", default="repo", choices=["repo", "lead"],
                   help="repo: A from adaptive-weighted-LSTM-Attention/results/<tag>/<split>/export; "
                        "lead: the lead's A copied to results/<market>/<split>/lead_export (PJM split2_2018 only)")
    p.add_argument("--lstm", default="lstm_val0.5", help="LSTM boosting folder under boosting/results/<market>/<split>/")
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    pd.set_option("display.width", 250)
    rows = [r for mk in args.markets for sp in args.splits if (r := run(mk, sp, args.a_source, args)) is not None]
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "results" / f"summary_a_{args.a_source}.csv", index=False)
    log("\n===== summary (significance: day_MSE / day_MAE, ○ better, × worse, − no difference) =====")
    log(table.round(3).T.to_string())


if __name__ == "__main__":
    main()
