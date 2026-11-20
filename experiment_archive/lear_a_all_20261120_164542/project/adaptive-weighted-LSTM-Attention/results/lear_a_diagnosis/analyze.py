import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/step1/split2_2018/export"
RECENT = ROOT / "results/lear_a_context_attention"
report = []

def say(text):
    print(text, flush=True)
    report.append(str(text))

def save(name, rows):
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(OUT / name, index=False)
    return table

def metric(actual, pred):
    error = pred - actual
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias_pred_minus_actual": float(error.mean()),
    }

# Load hourly predictions and verify daily alignment.
frames = []
for part in ["val", "test"]:
    table = pd.read_csv(
        SOURCE / f"predictions_{part}.csv", parse_dates=["Date"]
    )
    table["part"] = part
    frames.append(table)
frame = pd.concat(frames, ignore_index=True)

if frame["Date"].duplicated().any():
    raise ValueError("예측 시각이 중복됩니다.")
if not frame["Date"].is_monotonic_increasing:
    raise ValueError("예측 날짜 순서가 잘못되었습니다.")

groups = list(frame.groupby(frame["Date"].dt.normalize(), sort=True))
for day, group in groups:
    if len(group) != 24 or group["Date"].dt.hour.tolist() != list(range(24)):
        raise ValueError(f"24시간 구성이 잘못되었습니다: {day}")
    if group["part"].nunique() != 1:
        raise ValueError("하루에 val/test가 섞여 있습니다.")

dates = pd.DatetimeIndex([day for day, _ in groups])
parts = np.array([group["part"].iloc[0] for _, group in groups])

def daily(column):
    return np.stack([
        group[column].to_numpy(dtype=float) for _, group in groups
    ])

Y, A, L = [daily(name) for name in ["actual", "A", "LEAR"]]
val = np.flatnonzero(parts == "val")
test = np.flatnonzero(parts == "test")
if dates[val[-1]] >= dates[test[0]]:
    raise ValueError("검증/테스트 날짜가 겹칩니다.")

# Recover exactly the saved OriginalRecent test predictions.
recent = pd.Series(np.nan, index=pd.DatetimeIndex(frame["Date"]))
used = []
for path in sorted(RECENT.rglob("*.csv")):
    try:
        table = pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        continue
    names = {str(c).lower().strip(): c for c in table.columns}
    date_col = next(
        (names[c] for c in ["date", "datetime", "timestamp"] if c in names),
        None,
    )
    pred_col = next(
        (names[c] for c in ["recent", "pred_recent", "recent_pred"] if c in names),
        None,
    )
    if date_col is None or pred_col is None:
        continue
    stamps = pd.to_datetime(table[date_col], errors="coerce")
    values = pd.to_numeric(table[pred_col], errors="coerce")
    incoming = pd.Series(values.to_numpy(), index=stamps)
    incoming = incoming[~incoming.index.isna()].dropna()
    if incoming.index.duplicated().any():
        raise ValueError(f"Recent 날짜 중복: {path}")
    common = recent.index.intersection(incoming.index)
    overlap = common[recent.loc[common].notna().to_numpy()]
    if len(overlap) and not np.allclose(
        recent.loc[overlap], incoming.loc[overlap], atol=0.002, rtol=0
    ):
        raise ValueError(f"Recent 파일 간 예측값 불일치: {path}")
    recent.loc[common] = incoming.loc[common]
    used.append(str(path))

B = recent.to_numpy().reshape(-1, 24)
if not np.isfinite(B[test]).all():
    raise ValueError("원래 Recent의 test 예측을 모두 찾지 못했습니다.")
for array in [Y, A, L]:
    if not np.isfinite(array).all():
        raise ValueError("원본 예측/실제값에 NaN 또는 Inf가 있습니다.")

# Read raw history and future-known load forecasts.
raw_table = pd.read_csv(
    ROOT.parent / "PJM_error_analysis/data/PJM.csv", index_col=0
)
raw_table.index = pd.to_datetime(raw_table.index)
positions = raw_table.index.get_indexer(dates)
if (positions < 168).any():
    raise ValueError("원본 데이터 날짜 또는 168시간 이력이 없습니다.")
raw = raw_table.to_numpy(dtype=float)
H = np.stack([raw[t-168:t] for t in positions])
F = np.stack([raw[t:t+24, 1:3] for t in positions])
truth_check = np.stack([raw[t:t+24, 0] for t in positions])
if not np.allclose(truth_check, Y, atol=0.002, rtol=0):
    raise ValueError("원본 데이터와 예측 CSV의 실제 가격이 다릅니다.")

y, a, l, b = [array[test] for array in [Y, A, L, B]]
error = b-y
daily_mse = np.mean(error**2, axis=1)
daily_mae = np.abs(error).mean(axis=1)

say("기준: 원래 저장된 OriginalRecent")
say(json.dumps(metric(y, b), indent=2))
say(f"분석 날짜: {dates[test[0]].date()} ~ {dates[test[-1]].date()}")

# 1. Error concentration and exact daily level/shape MSE decomposition.
mean_error = error.mean(axis=1)
shape_error = error-mean_error[:, None]
level_mse = float(np.mean(mean_error**2))
shape_mse = float(np.mean(shape_error**2))
total_mse = float(np.mean(error**2))
assert np.isclose(level_mse+shape_mse, total_mse)

decomposition = {
    "total_MSE": total_mse,
    "daily_level_MSE": level_mse,
    "daily_shape_MSE": shape_mse,
    "daily_level_MSE_share_pct": 100*level_mse/total_mse,
    "daily_shape_MSE_share_pct": 100*shape_mse/total_mse,
}
save("01_error_decomposition.csv", [decomposition])
say("\n[하루 평균 수준 / 하루 안의 모양]")
say(json.dumps(decomposition, indent=2))

concentration = []
for fraction in [0.01, 0.05, 0.10]:
    count = max(1, int(np.ceil(len(test)*fraction)))
    ids = np.argsort(daily_mse)[-count:]
    concentration.append({
        "worst_day_fraction": fraction, "days": count,
        "share_of_squared_error_pct": 100*daily_mse[ids].sum()/daily_mse.sum(),
        "share_of_absolute_error_pct": 100*daily_mae[ids].sum()/daily_mae.sum(),
    })
save("02_error_concentration.csv", concentration)
say(pd.DataFrame(concentration).to_string(index=False))

daily_rows = pd.DataFrame({
    "date": dates[test].astype(str),
    "MAE": daily_mae,
    "RMSE": np.sqrt(daily_mse),
    "mean_error": mean_error,
    "shape_RMSE": np.sqrt(np.mean(shape_error**2, axis=1)),
    "A_MAE": np.abs(a-y).mean(axis=1),
    "LEAR_MAE": np.abs(l-y).mean(axis=1),
})
save("03_daily_errors.csv", daily_rows.sort_values("RMSE", ascending=False))

hour_rows = []
for h in range(24):
    hour_rows.append({
        "hour": h, **metric(y[:, h], b[:, h]),
        "squared_error_share_pct": 100*(error[:, h]**2).sum()/(error**2).sum(),
        "A_MAE": np.abs(a[:, h]-y[:, h]).mean(),
        "LEAR_MAE": np.abs(l[:, h]-y[:, h]).mean(),
    })
hour_table = save("04_hourly_errors.csv", hour_rows)
say("\n[오차가 큰 시간대]")
say(hour_table.sort_values("RMSE", ascending=False).head(6).to_string(index=False))

# 2. Retrospective oracle bounds. These require actuals and are NOT forecasts.
low = np.minimum(a, l)
high = np.maximum(a, l)
oracle_pair = np.clip(y, low, high)
ea, el = a-y, l-y
same_side = ea*el > 0
same_share = 100*(error[same_side]**2).sum()/(error**2).sum()

oracle_rows = [
    {"method": "OriginalRecent", **metric(y, b)},
    {"method": "A", **metric(y, a)},
    {"method": "LEAR", **metric(y, l)},
    {"method": "ORACLE_A_LEAR_convex_bound", **metric(y, oracle_pair)},
]
window_names = ["LEAR_w56", "LEAR_w84", "LEAR_w1092", "LEAR_w1456"]
expert_names = ["A", *window_names]
if all(name in frame for name in window_names):
    experts = np.stack([daily(name)[test] for name in expert_names], axis=-1)
    oracle_experts = np.clip(y, experts.min(axis=-1), experts.max(axis=-1))
    oracle_rows.append({
        "method": "ORACLE_A_four_LEAR_convex_bound",
        **metric(y, oracle_experts),
    })
    window_rows = []
    for j, name in enumerate(expert_names):
        window_rows.append({
            "method": name, **metric(y, experts[:, :, j]),
            "closest_expert_hour_share_pct": 100*np.mean(
                np.argmin(np.abs(experts-y[..., None]), axis=-1) == j
            ),
        })
    save("06_window_experts.csv", window_rows)
    window_error = (experts-y[..., None]).reshape(-1, len(expert_names))
    pd.DataFrame(
        np.corrcoef(window_error.T), index=expert_names, columns=expert_names
    ).to_csv(OUT / "06_window_error_correlations.csv")

save("05_oracle_bounds.csv", oracle_rows)
say("\n[가중치만 바꿀 때의 사후적 한계: oracle은 실제값을 본 진단값]")
say(pd.DataFrame(oracle_rows).to_string(index=False))
say(f"A와 LEAR가 같은 방향으로 틀린 시간 비율: {100*same_side.mean():.2f}%")
say(f"그 시간대가 Recent 제곱오차에서 차지하는 비율: {same_share:.2f}%")

# 3. Forecast-origin conditions. Thresholds come from validation first-half.
prev = H[:, -24:, 0]
signals = {
    "previous_price_level": prev.mean(axis=1),
    "previous_price_volatility": prev.std(axis=1),
    "system_load_level": F[:, :, 0].mean(axis=1),
    "zonal_load_level": F[:, :, 1].mean(axis=1),
    "system_load_change": F[:, :, 0].mean(axis=1)-H[:, -24:, 1].mean(axis=1),
    "system_load_max_hourly_ramp": np.abs(np.diff(F[:, :, 0], axis=1)).max(axis=1),
    "A_LEAR_disagreement": np.abs(A-L).mean(axis=1),
}
threshold_ids = val[:len(val)//2]
regime_rows = []
for name, values in signals.items():
    q1, q2 = np.quantile(values[threshold_ids], [1/3, 2/3])
    labels = np.digitize(values[test], [q1, q2])
    for group in range(3):
        mask = labels == group
        if not mask.any():
            continue
        regime_rows.append({
            "condition": name, "group": ["low", "middle", "high"][group],
            "days": int(mask.sum()),
            "Recent_MAE": np.abs(error[mask]).mean(),
            "Recent_RMSE": np.sqrt(np.mean(error[mask]**2)),
            "A_MAE": np.abs(a[mask]-y[mask]).mean(),
            "LEAR_MAE": np.abs(l[mask]-y[mask]).mean(),
            "A_better_than_LEAR_hour_pct": 100*np.mean(
                np.abs(a[mask]-y[mask]) < np.abs(l[mask]-y[mask])
            ),
        })
save("07_condition_errors.csv", regime_rows)

# 4. Cross-hour residual structure. Correlation does not prove predictability.
for name, residual in [
    ("raw", error), ("demeaned", shape_error)
]:
    corr = pd.DataFrame(residual).corr()
    corr.to_csv(OUT / f"08_hour_error_corr_{name}.csv")
    pairs = [
        {"hour1": i, "hour2": j, "correlation": corr.iloc[i, j]}
        for i in range(24) for j in range(i+1, 24)
        if np.isfinite(corr.iloc[i, j])
    ]
    save(
        f"08_top_hour_error_pairs_{name}.csv",
        sorted(pairs, key=lambda row: abs(row["correlation"]), reverse=True)[:20],
    )

# 5. Predictability probes using chronological validation splits.
# OriginalRecent validation is unavailable, so use a separate fixed A+LEAR baseline.
# Fit baseline weights on warm-up days only, then freeze them.
warm = val[:60]
available = val[60:]
nfit = int(len(available)*0.60)
ntune = int(len(available)*0.20)
fit = available[:nfit]
tune = available[nfit:nfit+ntune]
hold = available[nfit+ntune:]

weights = []
for h in range(24):
    result = minimize_scalar(
        lambda w: np.mean(
            (L[warm, h]+w*(A[warm, h]-L[warm, h])-Y[warm, h])**2
        ),
        bounds=(0, 1), method="bounded",
    )
    weights.append(float(result.x))
weights = np.asarray(weights)
P = L+(A-L)*weights
residual = Y-P

# Each probe predicts the 24-hour residual vector.
week = np.eye(7)[dates.dayofweek]
summary = np.column_stack(list(signals.values()))
basic = np.column_stack([
    P, A-L, week, summary,
])
future_curve = np.column_stack([
    F.reshape(len(Y), -1),
    np.diff(F, axis=1).reshape(len(Y), -1),
])
history_curve = H.reshape(len(Y), 7, 4, 6, 3).mean(axis=3).reshape(len(Y), -1)

feature_sets = {
    "basic": basic,
    "basic_plus_future_load_curve": np.column_stack([basic, future_curve]),
    "basic_plus_past_6hour_patches": np.column_stack([basic, history_curve]),
    "all_context": np.column_stack([basic, future_curve, history_curve]),
}

probe_rows = []
probe_predictions = {}
for name, x in feature_sets.items():
    best = None
    for alpha in [1, 10, 100, 1000, 10000]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[fit], residual[fit])
        correction = model.predict(x[tune])
        for strength in [0, 0.25, 0.5, 1.0]:
            mse = np.mean(
                (P[tune]+strength*correction-Y[tune])**2
            )
            rank = (float(mse), strength, -alpha)
            if best is None or rank < best[0]:
                best = (rank, alpha, strength)

    _, alpha, strength = best
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    # Refit on the already observed fit+tune segment.
    refit = np.r_[fit, tune]
    model.fit(x[refit], residual[refit])
    row = {"probe": name, "alpha": alpha, "strength": strength}
    for label, ids in [("validation_holdout", hold), ("test", test)]:
        prediction = P[ids]+strength*model.predict(x[ids])
        row.update({
            f"{label}_{key}": value
            for key, value in metric(Y[ids], prediction).items()
        })
        if label == "test":
            probe_predictions[name] = prediction
    probe_rows.append(row)

for name, prediction in [
    ("probe_fixed_baseline", P[test]), ("OriginalRecent", B[test])
]:
    row = {"probe": name}
    row.update({
        f"test_{key}": value for key, value in metric(y, prediction).items()
    })
    if name == "probe_fixed_baseline":
        row.update({
            f"validation_holdout_{key}": value
            for key, value in metric(Y[hold], P[hold]).items()
        })
    probe_rows.append(row)

probe_table = save("09_predictability_probes.csv", probe_rows)
say("\n[입력의 추가 가치 검사]")
say("진단용 고정 A+LEAR를 기준으로 학습. OriginalRecent는 test 비교만 표시.")
say(probe_table.to_string(index=False))

# Weekly block bootstrap for probe MSE differences.
# Exploratory intervals, with no multiple-comparison correction.
rng = np.random.default_rng(42)
bootstrap_rows = []
for name, prediction in probe_predictions.items():
    for reference_name, reference in [
        ("probe_fixed_baseline", P[test]),
        ("OriginalRecent", B[test]),
    ]:
        diff = (
            np.mean((prediction-y)**2, axis=1)
            - np.mean((reference-y)**2, axis=1)
        )
        samples = []
        block = 7
        blocks_needed = int(np.ceil(len(diff)/block))
        for _ in range(2000):
            starts = rng.integers(0, len(diff), blocks_needed)
            indices = (
                starts[:, None]+np.arange(block)[None, :]
            ).ravel() % len(diff)
            samples.append(diff[indices[:len(diff)]].mean())
        lo, hi = np.quantile(samples, [0.025, 0.975])
        bootstrap_rows.append({
            "probe": name, "reference": reference_name,
            "daily_MSE_diff": diff.mean(),
            "CI_low": lo, "CI_high": hi,
            "interpretation": "negative favors probe; exploratory, unadjusted",
        })
save("10_probe_bootstrap.csv", bootstrap_rows)

metadata = {
    "original_recent_files": used,
    "original_recent_test": metric(y, b),
    "probe_baseline": "Hourly fixed weights fitted on first 60 validation days",
    "probe_original_recent_validation_used": False,
    "probe_fit_dates": dates[fit].astype(str).tolist(),
    "probe_tune_dates": dates[tune].astype(str).tolist(),
    "probe_holdout_dates": dates[hold].astype(str).tolist(),
    "causal_assumption": "Previous day prices observed, matching repository protocol",
    "oracle_warning": "Oracle uses target actuals; not achievable forecast scores",
}
(OUT / "settings.json").write_text(json.dumps(metadata, indent=2))
(OUT / "summary.txt").write_text("\n".join(report), encoding="utf-8")
say(f"\n완료. 결과 위치: {OUT}")
