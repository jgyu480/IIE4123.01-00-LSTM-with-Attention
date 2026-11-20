import fcntl
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
guard = (OUT / "run.lock").open("w")
try:
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("이 실험이 이미 실행 중입니다.")

# Reuse only the previous diagnosis data loader.
loader_path = Path("results/lear_a_diagnosis/analyze.py").resolve()
source = loader_path.read_text()
marker = "# 1. Error concentration"
if marker not in source:
    raise SystemExit("기존 진단 코드의 데이터 읽기 구간을 찾지 못했습니다.")
prefix = source.split(marker, 1)[0]
ns = {"__file__": str(loader_path), "__name__": "__loader__"}
exec(compile(prefix, str(loader_path), "exec"), ns)
(OUT / "loader_snapshot.py").write_text(prefix)

Y, A, L = [ns[k] for k in ["Y", "A", "L"]]
original = ns["B"]
dates, parts = ns["dates"], ns["parts"]
raw, positions = ns["raw"], ns["positions"]
raw_table = ns["raw_table"]
N = len(Y)
val = np.flatnonzero(parts == "val")
test = np.flatnonzero(parts == "test")

if len(val) < 300:
    raise ValueError("이 실험에는 검증 날짜 300일 이상이 필요합니다.")

warm = val[:60]
cut1 = len(val) // 2
cut2 = 3 * len(val) // 4
choose = val[60:cut1]
boa_choose = val[cut1:cut2]
hold = val[cut2:]
scale = max(float(Y[warm].std()), 1e-6)

def log(message):
    print(message, flush=True)

def score(actual, prediction):
    e = prediction - actual
    mean_e = e.mean(axis=1)
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MSE": float(np.mean(e**2)),
        "level_MSE": float(np.mean(mean_e**2)),
        "shape_MSE": float(np.mean((e-mean_e[:, None])**2)),
    }

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))

# Common causal Recent: same definition as the earlier TrainingRecent.
difference = A - L
num = difference * (Y - L)
den = difference**2
csn = np.vstack([np.zeros(24), np.cumsum(num, axis=0)])
csd = np.vstack([np.zeros(24), np.cumsum(den, axis=0)])
beta = np.full_like(Y, 0.5)

for t in range(60, N):
    energy = np.maximum(csd[t] / t, 1e-6)
    ridge = 10 * energy
    hourly = np.clip((csn[t] + 0.5*ridge)/(csd[t] + ridge), 0, 1)
    start = max(0, t-14)
    recent = np.clip(
        (csn[t]-csn[start] + ridge*hourly)
        / (csd[t]-csd[start] + ridge), 0, 1
    )
    beta[t] = 0.5*hourly + 0.5*recent

BASE = L + beta*difference
ERR = Y - BASE

log(f"設定選択 {len(choose)}日 / BOA設定選択 {len(boa_choose)}日 / "
    f"検証holdout {len(hold)}日 / test {len(test)}日")
log("공통 기준은 재현 Recent, 별도 비교 기준은 OriginalRecent입니다.")

all_trials = []
selected = {}
predictions = {
    "TrainingRecent": BASE,
    "OriginalRecent": original,
    "A": A,
    "LEAR": L,
}

def consider(family, config, prediction, best):
    if not np.isfinite(prediction[60:]).all():
        raise ValueError(f"{family}: nonfinite prediction")
    values = score(Y[choose], prediction[choose])
    all_trials.append({
        "family": family, "config": json.dumps(config),
        **{f"selection_{k}": v for k, v in values.items()},
    })
    for objective in ["MAE", "RMSE"]:
        current = best.get(objective)
        if current is None or values[objective] < current[0]-1e-10:
            best[objective] = (values[objective], config.copy(), prediction.copy())

def register(family, best):
    for objective, (_, config, prediction) in best.items():
        name = f"{family}_{objective}"
        predictions[name] = prediction
        selected[name] = config
        log(f"[선택] {name}: {config}")

# 1. Separate level and shape bias correction.
log("\n[1/4] 하루 수준·시간대별 모양 보정")
level = ERR.mean(axis=1)
shape = ERR - level[:, None]
clip_value = max(float(np.quantile(np.abs(ERR[warm]), 0.95)), 1e-6)
best = {}

for robust, level_window, shape_window in itertools.product(
    [False, True], [3, 7, 14], [14, 28, 56]
):
    observed = np.clip(ERR, -clip_value, clip_value) if robust else ERR
    obs_level = observed.mean(axis=1)
    obs_shape = observed - obs_level[:, None]
    estimate_level = np.zeros(N)
    estimate_shape = np.zeros_like(Y)

    for t in range(60, N):
        estimate_level[t] = obs_level[max(0, t-level_window):t].mean()
        v = obs_shape[max(0, t-shape_window):t].mean(axis=0)
        # Smooth adjacent hours without joining 23:00 to 00:00.
        padded = np.pad(v, (1, 1), mode="edge")
        v = (padded[:-2]+2*padded[1:-1]+padded[2:])/4
        estimate_shape[t] = v-v.mean()

    for alpha, gamma in itertools.product([0.0, 0.25, 0.5], repeat=2):
        p = BASE + alpha*estimate_level[:, None] + gamma*estimate_shape
        consider("LevelShape", {
            "robust": robust, "level_window": level_window,
            "shape_window": shape_window,
            "level_strength": alpha, "shape_strength": gamma,
        }, p, best)

register("LevelShape", best)

# 2. Independent daily / six-hour aggregate forecasts.
# Uses original historical price/load inputs, including training-period days.
log("\n[2/4] 하루·6시간 평균 회귀와 곡선 조정")
calendar_raw = raw.reshape(-1, 24, 3)
calendar_dates = pd.DatetimeIndex(raw_table.index[::24])
DAYS = len(calendar_raw)
S = np.zeros((5, 24))
S[0] = 1/24
for j in range(4):
    S[j+1, 6*j:6*(j+1)] = 1/6

aggregate_y = calendar_raw[:, :, 0] @ S.T
aggregate_load = np.stack([
    calendar_raw[:, :, c] @ S.T for c in [1, 2]
], axis=-1)

aggregate_x = None
for d in range(7, DAYS):
    previous_price = calendar_raw[d-1, :, 0]
    vector = np.concatenate([
        aggregate_y[d-1], aggregate_y[d-2], aggregate_y[d-7],
        aggregate_load[d].ravel(),
        (aggregate_load[d]-aggregate_load[d-1]).ravel(),
        [
            previous_price.min(), previous_price.max(),
            previous_price.std(), previous_price[-1],
        ],
        np.eye(7)[calendar_dates[d].dayofweek],
    ])
    if aggregate_x is None:
        aggregate_x = np.full((DAYS, len(vector)), np.nan)
    aggregate_x[d] = vector

calendar_ids = positions // 24
raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
best = {}

for window, ridge_alpha in itertools.product([365, 1092], [10.0, 100.0]):
    cache = OUT / f"aggregate_w{window}_a{ridge_alpha:g}.npz"
    identity = f"{raw_hash}:{code_hash}:{window}:{ridge_alpha}"
    aggregate_prediction = None
    if cache.is_file():
        with np.load(cache) as z:
            if str(z["identity"].item()) == identity:
                aggregate_prediction = z["prediction"].copy()

    if aggregate_prediction is None:
        aggregate_prediction = np.empty((N, 5))
        model = None
        for i, d in enumerate(calendar_ids):
            # Weekly refit, using only days strictly before the refit date.
            if model is None or i % 7 == 0:
                start = max(7, d-window)
                model = make_pipeline(
                    StandardScaler(), Ridge(alpha=ridge_alpha)
                )
                model.fit(aggregate_x[start:d], aggregate_y[start:d])
            aggregate_prediction[i] = model.predict(aggregate_x[d:d+1])[0]
            if i % 120 == 0:
                log(f"평균 회귀 window={window} alpha={ridge_alpha:g}: {i}/{N}")
        np.savez_compressed(
            cache, prediction=aggregate_prediction, identity=identity
        )

    # Simple independent daily-mean adjustment.
    daily_gap = aggregate_prediction[:, 0] - BASE.mean(axis=1)
    for strength in [0.0, 0.25, 0.5, 1.0]:
        p = BASE + strength*daily_gap[:, None]
        consider("Reconcile", {
            "mode": "daily_mean", "window": window,
            "ridge_alpha": ridge_alpha, "strength": strength,
        }, p, best)

    # Reconcile all five aggregates, weighted by warm-up forecast errors.
    hour_variance = max(float(np.mean((BASE[warm]-Y[warm])**2)), 1e-6)
    agg_variance = np.mean(
        (aggregate_prediction[warm]-Y[warm]@S.T)**2, axis=0
    )
    precision = hour_variance / np.maximum(agg_variance, 1e-6)
    W = np.diag(precision)
    for rho in [0.1, 1.0, 10.0]:
        matrix = np.eye(24) + rho*S.T@W@S
        rhs = BASE + rho*aggregate_prediction@W@S
        p = np.linalg.solve(matrix, rhs.T).T
        consider("Reconcile", {
            "mode": "daily_and_6hour", "window": window,
            "ridge_alpha": ridge_alpha, "rho": rho,
        }, p, best)

register("Reconcile", best)

# 3. Rolling regularized bias and mixture correction.
log("\n[3/4] 편향·결합 비율 동시 보정")
best = {}
for window, regularization, groups in itertools.product(
    [14, 28, 56], [0.1, 1.0, 10.0], [1, 4]
):
    p = BASE.copy()
    width = 24 // groups
    for t in range(60, N):
        start = max(0, t-window)
        day_weights = np.exp(
            -np.log(2)*(t-1-np.arange(start, t))/(window/2)
        )
        for g in range(groups):
            hours = slice(g*width, (g+1)*width)
            xgap = difference[start:t, hours].ravel()/scale
            target = ERR[start:t, hours].ravel()/scale
            weights = np.repeat(day_weights, width)
            weights = weights/weights.sum()
            X = np.column_stack([np.ones(len(xgap)), xgap])
            matrix = X.T@(weights[:, None]*X)
            matrix += regularization*np.diag([1.0, 4.0])
            theta = np.linalg.solve(matrix, X.T@(weights*target))
            # Conservative maximum change in the A-vs-LEAR mixture.
            shift = np.clip(theta[1], -0.25, 0.25)
            p[t, hours] += scale*theta[0]+shift*difference[t, hours]

    consider("BiasMix", {
        "window": window, "regularization": regularization,
        "hour_groups": groups,
    }, p, best)

register("BiasMix", best)

# Simple control with the same baseline and causal memory.
mean_correction = np.zeros_like(Y)
for t in range(60, N):
    mean_correction[t] = ERR[max(0, t-28):t].mean(axis=0)
predictions["MeanError28_half"] = BASE+0.5*mean_correction

# 4. Adaptive Bernstein Online Aggregation.
# Components are fixed using only the earlier selection segment.
# Start BOA after that segment, before its own selection and holdout.
log("\n[4/4] BOA 온라인 결합")
start_boa = int(boa_choose[0])

def boa(experts, loss, group_count):
    # experts: (days, hours, experts)
    count = experts.shape[-1]
    width = 24//group_count
    initial = np.full((group_count, count), 1/count)
    weight = initial.copy()
    bound = np.zeros_like(weight)
    variance = np.zeros_like(weight)
    regret = np.zeros_like(weight)
    rate = np.zeros_like(weight)
    output = np.full_like(Y, np.nan)
    saved_weights = np.full((N, group_count, count), np.nan)

    for t in range(start_boa, N):
        # Forecast all hours first; update only after the complete day is observed.
        saved_weights[t] = weight
        for g in range(group_count):
            hours = slice(g*width, (g+1)*width)
            p = experts[t, hours]/scale
            forecast = p@weight[g]
            output[t, hours] = forecast*scale
            truth = Y[t, hours]/scale
            gradient = (
                2*(forecast-truth)
                if loss == "MSE" else np.sign(forecast-truth)
            )
            excess = np.mean(gradient[:, None]*(p-forecast[:, None]), axis=0)
            bound[g] = np.maximum(bound[g], np.abs(excess))
            variance[g] += excess**2
            new_rate = np.minimum(
                np.sqrt(-np.log(initial[g])/np.maximum(variance[g], 1e-12)),
                1/(2*np.maximum(bound[g], 1e-12)),
            )
            regret[g] += (
                0.5*excess*(1+new_rate*excess)
                + bound[g]*(2*rate[g]*excess > 1)
            )
            rate[g] = new_rate
            logits = (
                np.log(initial[g])+np.log(np.maximum(new_rate, 1e-12))
                - new_rate*regret[g]
            )
            weight[g] = softmax(logits)
    return output, saved_weights

boa_trials = []
for objective in ["MAE", "RMSE"]:
    members = [
        "TrainingRecent", "MeanError28_half",
        f"LevelShape_{objective}",
        f"Reconcile_{objective}",
        f"BiasMix_{objective}",
    ]
    experts = np.stack([predictions[name] for name in members], axis=-1)
    options = []
    for loss, group_count in itertools.product(["MAE", "MSE"], [1, 4]):
        p, weights = boa(experts, loss, group_count)
        values = score(Y[boa_choose], p[boa_choose])
        config = {"loss": loss, "hour_groups": group_count, "members": members}
        boa_trials.append({
            "objective": objective, "config": json.dumps(config),
            **values,
        })
        options.append((values[objective], config, p, weights))

    _, config, p, weights = min(options, key=lambda item: item[0])
    name = f"BOA_{objective}"
    predictions[name] = p
    selected[name] = config
    np.savez_compressed(
        OUT/f"{name}_weights.npz",
        dates=dates.astype(str).to_numpy(), weights=weights,
        members=np.array(members),
    )
    log(f"[선택] {name}: {config}")

# Final metrics. No test metrics were used in any parameter selection.
rows = []
reference = score(Y[test], original[test])
for name, p in predictions.items():
    row = {"method": name}
    for label, ids in [("holdout", hold), ("test", test)]:
        if np.isfinite(p[ids]).all():
            row.update({
                f"{label}_{k}": v for k, v in score(Y[ids], p[ids]).items()
            })
    for objective in ["MAE", "RMSE"]:
        row[f"gain_vs_OriginalRecent_{objective}_pct"] = (
            100*(1-row[f"test_{objective}"]/reference[objective])
        )
    rows.append(row)

scores = pd.DataFrame(rows).sort_values("holdout_RMSE", kind="stable")
scores.to_csv(OUT/"scores.csv", index=False)
pd.DataFrame(all_trials).to_csv(OUT/"method_selection_trials.csv", index=False)
pd.DataFrame(boa_trials).to_csv(OUT/"boa_selection_trials.csv", index=False)

for label, ids in [("holdout", hold), ("test", test)]:
    timestamps = (
        dates.values[ids, None]
        + np.arange(24)[None, :]*np.timedelta64(1, "h")
    ).ravel()
    table = pd.DataFrame({"Date": timestamps, "actual": Y[ids].ravel()})
    for name, p in predictions.items():
        if np.isfinite(p[ids]).all():
            table[name] = p[ids].ravel()
    table.to_csv(OUT/f"predictions_{label}.csv", index=False)

# Exploratory weekly-block uncertainty intervals.
rng = np.random.default_rng(42)
bootstrap_rows = []
for name in selected:
    for reference_name in ["TrainingRecent", "OriginalRecent"]:
        p = predictions[name][test]
        reference_prediction = predictions[reference_name][test]
        for loss in ["MAE", "MSE"]:
            difference_loss = (
                np.abs(p-Y[test]).mean(axis=1)
                - np.abs(reference_prediction-Y[test]).mean(axis=1)
                if loss == "MAE" else
                ((p-Y[test])**2).mean(axis=1)
                - ((reference_prediction-Y[test])**2).mean(axis=1)
            )
            draws = []
            for _ in range(2000):
                starts = rng.integers(0, len(test), int(np.ceil(len(test)/7)))
                index = (
                    starts[:, None]+np.arange(7)[None, :]
                ).ravel()[:len(test)] % len(test)
                draws.append(difference_loss[index].mean())
            lower, upper = np.quantile(draws, [0.025, 0.975])
            bootstrap_rows.append({
                "method": name, "reference": reference_name, "loss": loss,
                "mean_difference": float(difference_loss.mean()),
                "CI_low": float(lower), "CI_high": float(upper),
                "note": "negative favors method; exploratory, unadjusted",
            })
pd.DataFrame(bootstrap_rows).to_csv(OUT/"bootstrap.csv", index=False)

settings = {
    "selected": selected,
    "baseline": "Causal 14-day/expanding-hourly blend, same as prior TrainingRecent",
    "original_recent_validation_available": bool(np.isfinite(original[val]).all()),
    "original_recent_test": reference,
    "method_selection_dates": dates[choose].astype(str).tolist(),
    "boa_selection_dates": dates[boa_choose].astype(str).tolist(),
    "holdout_dates": dates[hold].astype(str).tolist(),
    "aggregate_forecaster": "Independent daily and 6-hour Ridge targets; weekly refit",
    "test_update": "Only previous complete observed days",
    "loader_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
    "raw_sha256": raw_hash,
    "code_sha256": code_hash,
}
save_json(OUT/"settings.json", settings)
log("\n"+scores.to_string(index=False))
(OUT/"complete.ok").write_text("complete\n")
log(f"\n완료. 결과 위치: {OUT}")
