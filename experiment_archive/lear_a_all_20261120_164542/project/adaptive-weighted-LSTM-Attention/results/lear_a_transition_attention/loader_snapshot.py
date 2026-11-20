import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/step1/split2_2018/export"
RECENT_DIR = ROOT / "results/lear_a_context_attention"
SEEDS = [42, 43, 44]
METHODS = ["MLP", "HorizonAttention", "WindowAttention", "HistoryAttention"]
DEVICE = torch.device("cpu")

def log(message):
    print(message, flush=True)

# Load the common forecast table.
frames = []
for part in ["val", "test"]:
    path = SOURCE / f"predictions_{part}.csv"
    table = pd.read_csv(path, parse_dates=["Date"])
    table["part"] = part
    frames.append(table)

frame = pd.concat(frames, ignore_index=True)
if frame["Date"].duplicated().any():
    raise ValueError("예측 시각이 중복됩니다.")
if not frame["Date"].is_monotonic_increasing:
    raise ValueError("예측 파일의 날짜 순서가 잘못되었습니다.")

groups = list(frame.groupby(frame["Date"].dt.normalize(), sort=True))
for day, group in groups:
    if len(group) != 24 or group["Date"].dt.hour.tolist() != list(range(24)):
        raise ValueError(f"24시간 구성이 잘못되었습니다: {day}")
    if group["part"].nunique() != 1:
        raise ValueError("하루에 val/test가 섞여 있습니다.")

dates = pd.DatetimeIndex([day for day, _ in groups])
parts = np.array([group["part"].iloc[0] for _, group in groups])

def daily(column):
    return np.stack([group[column].to_numpy(dtype=float) for _, group in groups])

Y, A, L = [daily(name) for name in ["actual", "A", "LEAR"]]
E = np.stack([
    daily(name)
    for name in ["A", "LEAR_w56", "LEAR_w84", "LEAR_w1092", "LEAR_w1456"]
], axis=-1)

# Recover the original Recent predictions without changing their definition.
recent = pd.Series(np.nan, index=pd.DatetimeIndex(frame["Date"]))
used = []
for path in sorted(RECENT_DIR.rglob("*.csv")):
    try:
        table = pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        continue

    names = {str(c).lower().strip(): c for c in table.columns}
    time_col = next(
        (names[c] for c in ["date", "datetime", "timestamp"] if c in names), None
    )
    pred_col = next(
        (names[c] for c in ["recent", "pred_recent", "recent_pred"] if c in names),
        None,
    )
    if time_col is None or pred_col is None:
        continue

    index = pd.to_datetime(table[time_col], errors="coerce")
    values = pd.to_numeric(table[pred_col], errors="coerce")
    incoming = pd.Series(values.to_numpy(), index=index)
    incoming = incoming[~incoming.index.isna()].dropna()
    if incoming.index.duplicated().any():
        raise ValueError(f"Recent 파일의 시각이 중복됩니다: {path}")

    common = recent.index.intersection(incoming.index)
    overlap = common[recent.loc[common].notna().to_numpy()]
    if len(overlap) and not np.allclose(
        recent.loc[overlap], incoming.loc[overlap], atol=0.002, rtol=0
    ):
        raise ValueError(f"Recent 예측 파일 간 값이 다릅니다: {path}")
    recent.loc[common] = incoming.loc[common]
    used.append(str(path))

if not used:
    files = "\n".join(str(p) for p in RECENT_DIR.rglob("*") if p.is_file())
    raise SystemExit(
        "원래 Recent의 시간별 예측 CSV를 찾지 못했습니다.\n"
        "Date와 Recent 열이 있는 CSV가 필요합니다.\n"
        f"현재 파일:\n{files}"
    )

# VALIDATION_RECENT_FIX
original_recent = recent.to_numpy().reshape(-1, 24)
test = np.flatnonzero(parts == "test")
if not np.isfinite(original_recent[test]).all():
    raise ValueError("원래 Recent의 test 예측이 일부 없습니다.")

# Construct a consistent causal baseline for both meta-training and test.
# A day uses only outcomes from strictly earlier complete days.
difference = A - L
numerator = difference * (Y - L)
denominator = difference ** 2
num_sum = np.vstack([np.zeros(24), np.cumsum(numerator, axis=0)])
den_sum = np.vstack([np.zeros(24), np.cumsum(denominator, axis=0)])
beta = np.full_like(Y, 0.5)

warmup = 60
for t in range(warmup, len(Y)):
    energy = np.maximum(den_sum[t] / t, 1e-6)
    ridge = 10 * energy
    hourly_weight = np.clip(
        (num_sum[t] + 0.5 * ridge) / (den_sum[t] + ridge),
        0, 1,
    )
    first = max(0, t - 14)
    recent_weight = np.clip(
        (num_sum[t] - num_sum[first] + ridge * hourly_weight)
        / (den_sum[t] - den_sum[first] + ridge),
        0, 1,
    )
    beta[t] = 0.5 * hourly_weight + 0.5 * recent_weight

B = L + beta * difference
val = np.flatnonzero((parts == "val") & (np.arange(len(Y)) >= warmup))
if len(val) < 90:
    raise ValueError(f"검증 날짜가 부족합니다: {len(val)}일")
if dates[val[-1]] >= dates[test[0]]:
    raise ValueError("검증/테스트 날짜가 겹칩니다.")

log("학습 기준: 과거 14일과 누적 시간대별 비중을 결합한 Recent")
log("최종 비교 기준: 원래 파일에 저장된 OriginalRecent")
original_error = original_recent[test] - Y[test]
log(
    f"OriginalRecent test MAE={np.abs(original_error).mean():.5f}, "
    f"RMSE={np.sqrt(np.mean(original_error**2)):.5f}"
)

# Chronological meta-training / early stopping / selection split.
nfit = int(len(val) * 0.70)
nstop = int(len(val) * 0.15)
fit_ids = val[:nfit]
stop_ids = val[nfit:nfit + nstop]
select_ids = val[nfit + nstop:]

log(f"Recent 파일: {used}")
log(f"날짜 수: 학습 {len(fit_ids)}, 조기종료 {len(stop_ids)}, "
    f"방법선택 {len(select_ids)}, 테스트 {len(test)}")

# Original PJM inputs, matched by timestamp rather than row position.
raw_path = ROOT.parent / "PJM_error_analysis/data/PJM.csv"
raw_frame = pd.read_csv(raw_path, index_col=0)
raw_frame.index = pd.to_datetime(raw_frame.index)
raw = raw_frame.to_numpy(dtype=float)
positions = raw_frame.index.get_indexer(dates)
if (positions < 168).any():
    raise ValueError("PJM 입력 날짜를 찾지 못했거나 과거 168시간이 부족합니다.")

history = np.stack([raw[t-168:t] for t in positions])
future = np.stack([raw[t:t+24, 1:3] for t in positions])
previous = history[:, -24:, :]
if not np.allclose(
    np.stack([raw[t:t+24, 0] for t in positions]), Y, atol=0.002, rtol=0
):
    raise ValueError("원본 PJM과 저장된 예측 파일의 실제 가격이 다릅니다.")

hour = 2 * np.pi * np.arange(24) / 24
hour_features = np.broadcast_to(
    np.column_stack([np.sin(hour), np.cos(hour)]),
    (len(Y), 24, 2),
)
week = 2 * np.pi * dates.dayofweek.to_numpy() / 7
week_features = np.broadcast_to(
    np.column_stack([np.sin(week), np.cos(week)])[:, None, :],
    (len(Y), 24, 2),
)
load_ramp = np.diff(
    np.concatenate([history[:, -1:, 1:3], future], axis=1), axis=1
)
recent_stats = np.broadcast_to(
    np.column_stack([
        history[:, -24:, 0].mean(axis=1),
        history[:, -24:, 0].std(axis=1),
    ])[:, None, :],
    (len(Y), 24, 2),
)

# All token features are available at the forecast origin.
features = np.concatenate([
    A[..., None], L[..., None], B[..., None], (A-L)[..., None],
    future, load_ramp, previous[:, :, 0:1],
    future - previous[:, :, 1:3],
    hour_features, week_features, recent_stats,
], axis=-1)

feature_mean = features[fit_ids].mean(axis=(0, 1))
feature_std = np.maximum(features[fit_ids].std(axis=(0, 1)), 1e-5)
features = np.clip((features-feature_mean)/feature_std, -12, 12)

history_mean = history[fit_ids].mean(axis=(0, 1))
history_std = np.maximum(history[fit_ids].std(axis=(0, 1)), 1e-5)
memory = np.clip((history-history_mean)/history_std, -12, 12)
memory = memory.reshape(len(Y), 28, 18)  # 6-hour patches.

scale = max(float(Y[fit_ids].std()), 1e-5)
B_scaled, E_scaled = B / scale, E / scale
target = (Y-B) / scale

usable = np.r_[val, test]
for name, array in [
    ("features", features), ("memory", memory),
    ("experts", E_scaled), ("baseline", B_scaled),
]:
    if not np.isfinite(array[usable]).all():
        raise ValueError(f"{name}에 NaN/Inf가 있습니다.")

