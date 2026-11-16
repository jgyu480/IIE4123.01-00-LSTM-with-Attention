"""Volatility-gate 공용 모듈: 시간순 창(window) 로더, 예측 시점 특징, Gating MLP, 지표.

규칙
- 목표일 D의 시작 시각 인덱스를 t라 하면, 모든 gate 특징은 price[t-168:t] (D-1 23시까지)만 사용한다.
- 선택적 확장 특징 load_chg 만 목표일의 수요 "예보"(모델 A도 입력으로 쓰는 값)를 사용한다.
- 기존 PJM_error_analysis/train_01_baseline.py, train_02_bc.py 는 수정하지 않고 import 해서 쓴다.
  이 모듈을 먼저 import 하면 그 폴더가 sys.path 에 추가된다.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent              # volatility-gate/ : 이 실험의 코드와 결과
BASE = ROOT.parent / "PJM_error_analysis"           # 기존 A·B·C 코드, 데이터, 저장된 결과 (읽기 전용)
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
DEFAULT_CSV = BASE / "data" / "PJM.csv"
BLOCK_DAYS = 364

# ---------------------------------------------------------------------------
# 1. 데이터: train_01_baseline.load_market 과 같은 규칙을, 끝에서 trim_days 를 잘라낸 창에 적용
# ---------------------------------------------------------------------------


def read_frame(csv=DEFAULT_CSV):
    frame = pd.read_csv(csv, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    expected = pd.date_range(frame.index[0], periods=len(frame), freq="h")
    if frame.shape[1] != 3 or not frame.index.equals(expected) or len(frame) % 24:
        raise ValueError("CSV must have 3 columns and complete, consecutive hourly days")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError("CSV contains missing or nonfinite values")
    return frame


def load_window(trim_days=0, val_days=BLOCK_DAYS, test_days=BLOCK_DAYS, csv=DEFAULT_CSV):
    """끝에서 trim_days 일을 버린 뒤 train_01_baseline.load_market 과 동일하게 분할·표준화.

    trim_days=0 이면 기존 실험(test = 마지막 364일)과 같은 텐서가 나온다(테스트로 검증).
    표준화 통계는 해당 창의 train 구간 시간만 사용한다. 잘려 나간 미래 행은 읽지 않는다.
    """
    frame = read_frame(csv)
    if trim_days:
        frame = frame.iloc[:-trim_days * 24]
    raw = frame.to_numpy(dtype=np.float64)
    test_start = len(raw) - test_days * 24
    val_start = test_start - val_days * 24
    if val_start <= 168:
        raise ValueError("Not enough training data")
    mean, std = raw[:val_start].mean(0), raw[:val_start].std(0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = ((raw - mean) / std).astype(np.float32)
    starts = np.arange(168, len(raw), 24)
    history = np.stack([scaled[t - 168:t] for t in starts])
    forecast = np.stack([scaled[t:t + 24, 1:] for t in starts])
    weekday = np.eye(7, dtype=np.float32)[frame.index[starts].dayofweek]
    targets = np.stack([scaled[t:t + 24, 0] for t in starts])
    tensors = [torch.from_numpy(x) for x in (history, forecast, weekday, targets)]
    masks = (starts < val_start, (starts >= val_start) & (starts < test_start), starts >= test_start)
    datasets = [TensorDataset(*(x[m] for x in tensors)) for m in masks]
    test_pos = starts[masks[2]]
    info = {
        "market": "PJM", "columns": list(frame.columns), "trim_days": int(trim_days),
        "data_sha256": hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
        "mean": mean.tolist(), "std": std.tolist(),
        "high_threshold": float(np.quantile(raw[:val_start, 0], 0.95)),
        "splits": {name: {"first_target": str(frame.index[starts[m][0]]),
                          "last_target": str(frame.index[starts[m][-1]]), "days": int(m.sum())}
                   for name, m in zip(("train", "val", "test"), masks)},
    }
    return {
        "datasets": datasets, "info": info, "raw": raw, "index": frame.index, "starts": starts,
        "masks": masks, "test_dates": frame.index[test_pos],
        "truth": np.stack([raw[t:t + 24, 0] for t in test_pos]),
    }


def origin_name(trim_days, csv=DEFAULT_CSV):
    """평가 블록의 마지막 날짜 연도로 이름을 붙인다 (trim 0 -> o2018)."""
    frame = read_frame(csv)
    last = frame.index[len(frame) - trim_days * 24 - 1]
    return f"o{last.year}"


# ---------------------------------------------------------------------------
# 2. Gate 특징 (예측 시점에 알 수 있는 값만)
# ---------------------------------------------------------------------------

# name -> (개념, 변환)  변환: "slog" = sign(x)*log1p(|x|), "none"
FEATURES = {
    "vol_std":     ("산포: D-1 24시간 가격 표준편차", "slog"),
    "lvl_mean":    ("수준: D-1 평균 가격", "slog"),
    "range":       ("산포: D-1 최대-최소", "slog"),
    "trend_dd":    ("추세(일간): D-1 평균 - D-2 평균", "slog"),
    "trend_intra": ("추세(일중): D-1 마지막 시간 - 첫 시간", "slog"),
    "osc_mad":     ("진동: D-1 인접 시간 변화량 절댓값 평균", "slog"),
    "wk_mean":     ("수준: 최근 7일 평균", "slog"),
    "wk_std":      ("산포: 최근 7일 표준편차", "slog"),
    # --- 추가 제안 특징 ---
    "osc_chop":    ("진동(무차원): sum|dp| / (max-min). 단조=1, 봉우리 1개=2, 반복 등락일수록 큼", "none"),
    "vol_rel":     ("산포 충격: D-1 표준편차 / 직전 6일 일별 표준편차 평균", "slog"),
    "load_chg":    ("선택 확장: 목표일 계통 수요예보 평균 / D-1 수요예보 평균 - 1", "none"),
}
USER_FEATURES = ["vol_std", "lvl_mean", "range", "trend_dd", "trend_intra", "osc_mad", "wk_mean", "wk_std"]


def gate_features(raw, starts):
    """raw: (hours, 3) [price, system load forecast, zonal load forecast]. 반환: DataFrame(len(starts))."""
    price, sys_load = raw[:, 0], raw[:, 1]
    rows = []
    for t in starts:
        if t < 168:
            raise ValueError("need 168 hours of history")
        d1, d2, wk = price[t - 24:t], price[t - 48:t - 24], price[t - 168:t]
        diff = np.diff(d1)
        rng = d1.max() - d1.min()
        earlier_std = wk[:144].reshape(6, 24).std(1).mean()
        rows.append({
            "vol_std": d1.std(), "lvl_mean": d1.mean(), "range": rng,
            "trend_dd": d1.mean() - d2.mean(), "trend_intra": d1[-1] - d1[0],
            "osc_mad": np.abs(diff).mean(), "wk_mean": wk.mean(), "wk_std": wk.std(),
            "osc_chop": np.abs(diff).sum() / (rng + 1e-6),
            "vol_rel": d1.std() / (earlier_std + 1e-6),
            # 목표일 구간은 "예보" 열만 읽는다 (가격 열은 절대 읽지 않음)
            "load_chg": sys_load[t:t + 24].mean() / sys_load[t - 24:t].mean() - 1.0,
        })
    return pd.DataFrame(rows)


def transform(frame, names):
    cols = []
    for name in names:
        x = frame[name].to_numpy(dtype=np.float64)
        cols.append(np.sign(x) * np.log1p(np.abs(x)) if FEATURES[name][1] == "slog" else x)
    return np.column_stack(cols)


def fit_scaler(x):
    mean, std = x.mean(0), x.std(0)
    return mean, np.where(std < 1e-8, 1.0, std)


# ---------------------------------------------------------------------------
# 3. Gating network
# ---------------------------------------------------------------------------


def blend(alpha, pred_a, pred_c):
    """alpha = C의 비중. alpha: (N,1) 또는 (N,24)."""
    return (1.0 - alpha) * pred_a + alpha * pred_c


class GateMLP(nn.Module):
    """market_features -> Linear(in,hidden) -> ReLU -> Linear(hidden,out) -> (+anchor) -> sigmoid.

    out_dim=1: 하루에 alpha 하나. out_dim=4: 6시간 블록별 alpha(24시간으로 반복).
    anchor_logit 이 주어지면 마지막 층을 0으로 초기화해 학습 시작 시 alpha == sigmoid(anchor) (고정 가중치)
    가 되도록 한다. 즉 "고정 가중치에서 출발해 데이터가 지지할 때만 벗어나는" 게이트다.
    """

    def __init__(self, in_dim, hidden=8, out_dim=1, anchor_logit=None):
        super().__init__()
        if 24 % out_dim:
            raise ValueError("out_dim must divide 24")
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.register_buffer("anchor", torch.tensor(0.0 if anchor_logit is None else float(anchor_logit)))
        if anchor_logit is not None:
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)

    def alpha(self, x):
        a = torch.sigmoid(self.anchor + self.net(x))
        return a.repeat_interleave(24 // self.out_dim, dim=1) if self.out_dim > 1 else a

    def forward(self, x, pred_a, pred_c):
        a = self.alpha(x)
        return blend(a, pred_a, pred_c), a


def fixed_alpha(pred_a, pred_c, truth):
    """min_alpha mean((1-a)A + aC - y)^2 의 닫힌 해, [0,1]로 자름."""
    d = (pred_c - pred_a).ravel()
    return float(np.clip(((truth - pred_a).ravel() @ d) / max(d @ d, 1e-12), 0.0, 1.0))


# ---------------------------------------------------------------------------
# 4. 지표
# ---------------------------------------------------------------------------


def mae(truth, pred):
    return float(np.abs(pred - truth).mean())


def rmse(truth, pred):
    return float(np.sqrt(np.square(pred - truth).mean()))


def block_bootstrap_diff(loss_a, loss_b, block=7, n_boot=5000, seed=0):
    """일 손실 차이(a-b) 평균의 블록 부트스트랩 95% 구간. 음수면 a가 더 좋다."""
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(diff)
    blocks = [diff[i:i + block] for i in range(0, n, block)]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    means = np.array([np.concatenate([blocks[j] for j in row]).mean() for row in picks])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)
