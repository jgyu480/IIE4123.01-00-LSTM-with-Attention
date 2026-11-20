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

