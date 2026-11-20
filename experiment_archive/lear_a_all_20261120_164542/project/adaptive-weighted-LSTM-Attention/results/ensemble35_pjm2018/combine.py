from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

source, output = map(Path, sys.argv[1:])
val = pd.read_csv(source / "predictions_val.csv", parse_dates=["Date"])
test = pd.read_csv(source / "predictions_test.csv", parse_dates=["Date"])
models = ["LEAR", "A", "B", "C"]

for frame in (val, test):
    required = [*models, "actual"]
    if frame["Date"].duplicated().any() or not np.isfinite(frame[required]).all().all():
        raise ValueError("Duplicate timestamps or invalid predictions")
if val["Date"].max() >= test["Date"].min():
    raise ValueError("Validation must end before test begins")

actual = val["actual"].to_numpy()
scale = max(float(actual.std()), 1e-8)

def fixed_weights(pred, target, anchor, penalty=0.0):
    errors = (pred - target[:, None]) / scale
    moment = errors.T @ errors / len(errors)

    def objective(w):
        diff = w - anchor
        return w @ moment @ w + penalty * (diff @ diff)

    result = minimize(
        objective, anchor, method="SLSQP", bounds=[(0, 1)] * len(anchor),
        jac=lambda w: 2 * (moment @ w + penalty * (w - anchor)),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()

def group_weights(pred, target, groups, levels, anchor):
    return np.stack([
        fixed_weights(pred[groups == g], target[groups == g], anchor, 0.02)
        if np.any(groups == g) else anchor
        for g in levels
    ])

def attention_features(frame, pred):
    angle = 2 * np.pi * frame["Date"].dt.hour.to_numpy() / 24
    volatile = frame["volatile_prev_day"].to_numpy(dtype=float)
    spread = np.ptp(pred, axis=1) / scale
    return np.column_stack([
        np.ones(len(frame)), np.sin(angle), np.cos(angle),
        volatile, np.clip(spread, 0, 5),
    ])

def learn_attention(pred, target, x, anchor):
    offset = np.log(np.maximum(anchor, 0.02))
    target = target / scale
    pred = pred / scale
    penalty = 0.01

    def objective(flat):
        coef = flat.reshape(x.shape[1], pred.shape[1])
        weight = softmax(x @ coef + offset, axis=1)
        forecast = (weight * pred).sum(axis=1)
        error = forecast - target
        gradient = weight * (pred - forecast[:, None]) * error[:, None]
        gradient = 2 * x.T @ gradient / len(x) + 2 * penalty * coef
        loss = np.mean(error ** 2) + penalty * np.sum(coef ** 2)
        return loss, gradient.ravel()

    result = minimize(
        objective, np.zeros(x.shape[1] * pred.shape[1]),
        jac=True, method="L-BFGS-B", options={"maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.reshape(x.shape[1], pred.shape[1]), offset

def metrics(frame, forecast):
    error = forecast - frame["actual"].to_numpy()
    extreme = frame["extreme_window"].to_numpy(dtype=bool)
    daily_mse = pd.Series(error ** 2).groupby(
        frame["Date"].dt.normalize()
    ).mean().mean()
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "daily_MSE": daily_mse,
        "extreme_MAE": np.abs(error[extreme]).mean() if extreme.any() else np.nan,
    }

rows, params, forecasts = [], {}, {}
val_hour, test_hour = (f["Date"].dt.hour.to_numpy() for f in (val, test))
val_vol, test_vol = (f["volatile_prev_day"].to_numpy(dtype=int) for f in (val, test))

# Fit all 35 configurations on validation predictions.
for count in range(1, 4):
    for members in combinations(["A", "B", "C"], count):
        names = ["LEAR", *members]
        key = "+".join(names)
        pv, pt = (f[names].to_numpy() for f in (val, test))
        equal = np.full(len(names), 1 / len(names))
        static = fixed_weights(pv, actual, equal)
        hourly = group_weights(pv, actual, val_hour, range(24), static)
        regime = group_weights(pv, actual, val_vol, range(2), static)
        xv, xt = attention_features(val, pv), attention_features(test, pt)
        coef, offset = learn_attention(pv, actual, xv, static)
        weights = {
            "equal": (equal, equal),
            "static": (static, static),
            "hourly": (hourly[val_hour], hourly[test_hour]),
            "volatility": (regime[val_vol], regime[test_vol]),
            "attention": (softmax(xv @ coef + offset, axis=1),
                          softmax(xt @ coef + offset, axis=1)),
        }
        params[key] = {
            "models": names, "static": static.tolist(),
            "hourly": hourly.tolist(), "volatility": regime.tolist(),
            "attention_coef": coef.tolist(), "attention_offset": offset.tolist(),
        }
        for method, (wv, wt) in weights.items():
            forecasts[key, method] = ((pv * wv).sum(axis=1), (pt * wt).sum(axis=1))

assert len(forecasts) == 35
val_predictions = val[["Date", "actual"]].copy()
test_predictions = test[["Date", "actual"]].copy()

for (combo, method), (val_pred, test_pred) in forecasts.items():
    row = {"combo": combo, "method": method}
    row.update({f"val_{k}": v for k, v in metrics(val, val_pred).items()})
    row.update({f"test_{k}": v for k, v in metrics(test, test_pred).items()})
    rows.append(row)
    val_predictions[f"{combo}__{method}"] = val_pred
    test_predictions[f"{combo}__{method}"] = test_pred

scores = pd.DataFrame(rows).sort_values("val_daily_MSE", kind="stable")
scores.to_csv(output / "ensemble35_scores.csv", index=False)
val_predictions.to_csv(output / "ensemble35_val_predictions.csv", index=False)
test_predictions.to_csv(output / "ensemble35_test_predictions.csv", index=False)
(output / "weights.json").write_text(json.dumps(params, indent=2))
print(scores.head(10).to_string(index=False))
print(f"완료: 35개 조합 → {output / 'ensemble35_scores.csv'}")
