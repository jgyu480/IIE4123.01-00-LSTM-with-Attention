import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/step1/split2_2018/export"

# 1. Load and validate the existing predictions.
frames = []
for part in ["val", "test"]:
    path = SOURCE / f"predictions_{part}.csv"
    if not path.is_file():
        raise SystemExit(f"예측 파일이 없습니다: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame["part"] = part
    frames.append(frame)

frame = pd.concat(frames, ignore_index=True)
required = ["actual", "LEAR", "A", "C"]
if not np.isfinite(frame[required].to_numpy()).all():
    raise ValueError("예측 또는 실제값에 NaN/Inf가 있습니다.")
if frame["Date"].duplicated().any():
    raise ValueError("중복 시각이 있습니다.")
if not frame["Date"].is_monotonic_increasing:
    raise ValueError("예측 파일의 날짜 순서를 확인하세요.")

groups = list(frame.groupby(frame["Date"].dt.normalize(), sort=True))
for day, group in groups:
    if len(group) != 24 or group["Date"].dt.hour.tolist() != list(range(24)):
        raise ValueError(f"24시간 구성이 잘못되었습니다: {day}")
    if group["part"].nunique() != 1:
        raise ValueError("하루 안에 val/test가 섞여 있습니다.")

dates = pd.DatetimeIndex([day for day, _ in groups])
if not np.all(np.diff(dates.values) == np.timedelta64(1, "D")):
    raise ValueError("날짜 사이에 빠진 날이 있습니다.")

def daily(column):
    return np.stack([group[column].to_numpy() for _, group in groups])

Y, L, A, C = [daily(column) for column in required]
D = A - L
DELTA = C - A
parts = np.array([group["part"].iloc[0] for _, group in groups])
nv = int((parts == "val").sum())
n = len(Y)
if not (parts[:nv] == "val").all() or not (parts[nv:] == "test").all():
    raise ValueError("val 다음 test 순서여야 합니다.")
if nv < 240 or n <= nv:
    raise ValueError("검증 기간 240일 이상과 test 예측이 필요합니다.")

warmup = 60
cut = nv // 2
train_days = np.arange(warmup, cut)
tune_days = np.arange(cut, nv)
test_days = np.arange(nv, n)
scale = max(float(Y[:warmup].std()), 1e-6)

# Forecast-time features. Target-day actual prices are never included.
previous_mean = np.r_[Y[0].mean(), Y[:-1].mean(axis=1)]
previous_std = np.r_[Y[0].std(), Y[:-1].std(axis=1)]
weekday_angle = 2 * np.pi * dates.dayofweek.to_numpy() / 7
raw = np.column_stack([
    L.mean(axis=1),
    A.mean(axis=1),
    np.abs(D).mean(axis=1),
    previous_mean,
    previous_std,
    np.sin(weekday_angle),
    np.cos(weekday_angle),
])
mean = raw[1:warmup].mean(axis=0)
std = np.maximum(raw[1:warmup].std(axis=0), 1e-6)
X = (raw - mean) / std

# 2. Causal Hourly and Recent reference weights.
# Every day's weights use actuals from strictly earlier days.
numerator = D * (Y - L)
denominator = D ** 2
num_sum = np.vstack([np.zeros(24), np.cumsum(numerator, axis=0)])
den_sum = np.vstack([np.zeros(24), np.cumsum(denominator, axis=0)])
hourly_beta = np.full_like(Y, 0.5)
recent_beta = np.full_like(Y, 0.5)

for t in range(warmup, n):
    # Stabilize the expanding hourly estimate with ten pseudo-observations.
    energy = np.maximum(den_sum[t] / t, 1e-6)
    ridge = 10 * energy
    hourly_beta[t] = np.clip(
        (num_sum[t] + ridge * 0.5) / (den_sum[t] + ridge), 0, 1
    )
    start = max(0, t - 14)
    local_beta = np.clip(
        (num_sum[t] - num_sum[start] + ridge * hourly_beta[t])
        / (den_sum[t] - den_sum[start] + ridge), 0, 1
    )
    recent_beta[t] = 0.5 * hourly_beta[t] + 0.5 * local_beta

hourly = L + hourly_beta * D
recent = L + recent_beta * D
recent_residual = Y - recent

# Search only days whose actuals are already observed.
# Use a common pool of previous 365 days for every retrieval method.
pools = {
    t: np.arange(max(warmup, t - 365), t)
    for t in range(warmup + 8, n)
}
# Meta-fitting starts after enough earlier reference days exist.
train_days = train_days[train_days >= warmup + 8]

def retrieve(t, k, log_metric, uniform=False):
    pool = pools[t]
    metric = np.exp(log_metric)
    distance = np.mean((X[pool] - X[t]) ** 2 * metric, axis=1)
    order = np.argsort(distance, kind="stable")[:min(k, len(pool))]
    ids = pool[order]
    if uniform:
        weight = np.full(len(ids), 1 / len(ids))
    else:
        weight = softmax(-distance[order])
    return ids, weight

def case_prediction(t, k, strength, log_metric, method):
    ids, weight = retrieve(
        t, k, log_metric, uniform=(method == "EqualCase")
    )
    w = weight[:, None]
    beta = recent_beta[t]

    if method == "CDelta":
        # Transfer the historically useful part of C-A.
        delta = DELTA[ids]
        energy = (w * delta ** 2).sum(axis=0)
        eta = np.clip(
            (w * delta * recent_residual[ids]).sum(axis=0)
            / (1.5 * energy + 1e-6),
            0, beta,
        )
        return recent[t] + strength * eta * DELTA[t]

    # Estimate the LEAR/A mix directly from retrieved joint errors.
    energy = (w * D[ids] ** 2).sum(axis=0)
    ridge = 0.5 * energy + 1e-6
    local_beta = np.clip(
        ((w * D[ids] * (Y[ids] - L[ids])).sum(axis=0) + ridge * beta)
        / (energy + ridge),
        0, 1,
    )
    blended_beta = (1 - strength) * beta + strength * local_beta
    return L[t] + blended_beta * D[t]

# 3. Learn a small attention metric on validation first-half only.
# Seven feature weights, regularized toward the original distance.
zero_metric = np.zeros(X.shape[1])

def attention_loss(log_metric):
    prediction = np.stack([
        case_prediction(t, 16, 0.5, log_metric, "LearnedCase")
        for t in train_days
    ])
    error = (prediction - Y[train_days]) / scale
    return float(np.mean(error ** 2) + 0.01 * np.mean(log_metric ** 2))

print("[1/3] 사례 어텐션 거리 가중치 학습", flush=True)
initial_loss = attention_loss(zero_metric)
fit = minimize(
    attention_loss,
    zero_metric,
    method="Powell",
    bounds=[(-2, 2)] * len(zero_metric),
    options={"maxiter": 60, "maxfev": 1500, "ftol": 1e-5},
)
if np.isfinite(fit.fun) and fit.fun <= initial_loss:
    learned_metric = fit.x
else:
    learned_metric = zero_metric.copy()
print(
    f"metric loss {initial_loss:.6f} -> "
    f"{attention_loss(learned_metric):.6f}; optimizer={fit.message}",
    flush=True,
)

# 4. Select k and correction strength on validation second-half.
# strength=0 permits retaining Recent when retrieval is unhelpful.
print("[2/3] 검증 후반에서 설정 선택", flush=True)
configurations = [
    ("EqualCase", zero_metric),
    ("KernelCase", zero_metric),
    ("LearnedCase", learned_metric),
    ("CDelta", zero_metric),
]
selected = {}
predictions = {"Recent": recent.copy()}
trials = []

for method, metric in configurations:
    best = None
    for k in [8, 16, 32]:
        for strength in [0.0, 0.25, 0.5, 1.0]:
            pred = np.stack([
                case_prediction(t, k, strength, metric, method)
                for t in tune_days
            ])
            mse = float(np.mean((pred - Y[tune_days]) ** 2))
            row = {
                "method": method, "k": k, "strength": strength,
                "val_MSE": mse,
                "val_MAE": float(np.abs(pred - Y[tune_days]).mean()),
            }
            trials.append(row)
            rank = (mse, strength, k)
            if best is None or rank < best[0]:
                best = (rank, k, strength)
    _, k, strength = best
    selected[method] = {"k": k, "strength": strength}
    p = np.full_like(Y, np.nan)
    for t in range(cut, n):
        p[t] = case_prediction(t, k, strength, metric, method)
    predictions[method] = p
    print(f"{method}: k={k}, strength={strength}", flush=True)

# 5. Evaluate after all settings have been selected.
print("[3/3] 테스트 평가 및 저장", flush=True)

if "extreme_window" in frame:
    extreme = np.array([
        str(group["extreme_window"].iloc[0]).lower() in ("true", "1", "1.0")
        for _, group in groups
    ])
else:
    extreme = np.zeros(n, dtype=bool)

def score(pred, ids):
    error = pred[ids] - Y[ids]
    ext = extreme[ids]
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "extreme_MAE": float(np.abs(error[ext]).mean()) if ext.any() else None,
        "rest_MAE": float(np.abs(error[~ext]).mean()) if (~ext).any() else None,
    }

all_predictions = {"LEAR": L, "A": A, "C": C, "Hourly": hourly, **predictions}
rows = []
for method, pred in all_predictions.items():
    row = {"method": method}
    row.update({f"val_{k}": v for k, v in score(pred, tune_days).items()})
    row.update({f"test_{k}": v for k, v in score(pred, test_days).items()})
    rows.append(row)

scores = pd.DataFrame(rows)
reference = scores.loc[scores["method"] == "Recent"].iloc[0]
for metric in ["MAE", "RMSE"]:
    scores[f"gain_vs_Recent_{metric}_pct"] = 100 * (
        1 - scores[f"test_{metric}"] / reference[f"test_{metric}"]
    )

# Validation rank determines the displayed order.
scores = scores.sort_values("val_RMSE", kind="stable")
scores.to_csv(OUT / "scores.csv", index=False)
pd.DataFrame(trials).to_csv(OUT / "validation_trials.csv", index=False)

for part, ids in [("val", tune_days), ("test", test_days)]:
    stamps = (
        dates.values[ids, None]
        + np.arange(24)[None, :] * np.timedelta64(1, "h")
    ).ravel()
    saved = pd.DataFrame({"Date": stamps, "actual": Y[ids].ravel()})
    for method, pred in all_predictions.items():
        saved[method] = pred[ids].ravel()
    saved.to_csv(OUT / f"predictions_{part}.csv", index=False)

metadata = {
    "selected": selected,
    "learned_metric": np.exp(learned_metric).tolist(),
    "optimizer_message": str(fit.message),
    "warmup_days": warmup,
    "metric_training_end": str(dates[cut - 1]),
    "selection_start": str(dates[cut]),
    "test_start": str(dates[nv]),
    "protocol": "Previous complete days only; rolling observed-outcome memory",
    "recent_definition": "14-day hourly weight shrunk toward expanding hourly weight, mix=0.5",
}
(OUT / "settings.json").write_text(json.dumps(metadata, indent=2))

print(scores.to_string(index=False), flush=True)
print(f"\n완료. 결과 저장 위치: {OUT}", flush=True)
(OUT / "complete.ok").write_text("complete\n")
