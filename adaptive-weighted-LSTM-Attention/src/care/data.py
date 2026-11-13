"""PJM data loading, time-ordered splits, scaling, and forecast-time regime features.

Day/hour convention matches train_01_baseline.py: target day d starts at hour
index starts[d]; inputs use the previous 168 hours (prices up to D-1 are known)
and the target day's two load forecasts.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO / "PJM_error_analysis" / "data" / "PJM.csv"

# Number of days removed from the end of the data before splitting.
# split2_2018 reproduces the original experiment (test = last 364 days).
SPLITS = {"split2_2018": 0, "split1_2017": 364}


@dataclass
class Market:
    raw: np.ndarray          # (hours, 3): price, system load forecast, zonal load forecast
    hours: pd.DatetimeIndex  # hourly timestamps
    starts: np.ndarray       # hour index where each target day starts
    days: pd.DatetimeIndex   # target day dates
    ntrain: int
    nval: int
    ntest: int
    sha256: str
    split: str

    @property
    def truth(self):
        """Actual prices per target day, shape (days, 24)."""
        return np.stack([self.raw[t:t + 24, 0] for t in self.starts])

    def part(self):
        """'train' / 'val' / 'test' label per target day."""
        return np.array(["train"] * self.ntrain + ["val"] * self.nval + ["test"] * self.ntest)

    def scaling(self, first_day):
        """Mean/std over all hours before target day `first_day` (train-side only)."""
        block = self.raw[:self.starts[first_day]]
        mean, std = block.mean(0), block.std(0)
        return mean, np.where(std < 1e-8, 1.0, std)

    def tensors(self, mean, std):
        """history (days,168,3), forecast (days,24,2), weekday (days,7), target (days,24), all scaled."""
        scaled = ((self.raw - mean) / std).astype(np.float32)
        history = np.stack([scaled[t - 168:t] for t in self.starts])
        forecast = np.stack([scaled[t:t + 24, 1:] for t in self.starts])
        weekday = np.eye(7, dtype=np.float32)[self.days.dayofweek]
        target = np.stack([scaled[t:t + 24, 0] for t in self.starts])
        return tuple(torch.from_numpy(x) for x in (history, forecast, weekday, target))


def load(split, csv=DEFAULT_CSV, val_days=364, test_days=364):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}; choose from {list(SPLITS)}")
    csv = Path(csv)
    frame = pd.read_csv(csv, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    expected = pd.date_range(frame.index[0], periods=len(frame), freq="h")
    if frame.shape[1] != 3 or not frame.index.equals(expected) or len(frame) % 24:
        raise ValueError("CSV must have 3 columns and complete, consecutive hourly days")
    drop = SPLITS[split] * 24
    if drop:
        frame = frame.iloc[:-drop]
    raw = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite values in CSV")
    test_start = len(raw) - test_days * 24
    val_start = test_start - val_days * 24
    starts = np.arange(168, len(raw), 24)
    ntrain = int((starts < val_start).sum())
    nval = int(((starts >= val_start) & (starts < test_start)).sum())
    ntest = int((starts >= test_start).sum())
    return Market(raw, frame.index, starts, frame.index[starts], ntrain, nval, ntest,
                  hashlib.sha256(csv.read_bytes()).hexdigest(), split)


REGIME_COLUMNS = [
    "prev_mean", "prev_vol", "level_shift", "eve_minus_mean",
    "prev_trend", "sys_change_pct", "zonal_change_pct", "weekend",
]


def regime_features(m: Market):
    """Features known at forecast time for each target day (prices up to D-1, load forecasts for D)."""
    rows = []
    p, sysl, zl = m.raw[:, 0], m.raw[:, 1], m.raw[:, 2]
    for d, t in enumerate(m.starts):
        prev = p[t - 24:t]
        prev_mean = prev.mean()
        rows.append([
            prev_mean,
            prev.std(ddof=1),
            prev_mean - p[t - 168:t].mean(),          # D-1 vs last 7 days
            p[t - 6:t].mean() - prev_mean,            # D-1 evening (18-23h) vs D-1 mean
            prev_mean - p[t - 48:t - 24].mean(),      # D-1 vs D-2
            (sysl[t:t + 24].mean() / sysl[t - 24:t].mean() - 1) * 100,
            (zl[t:t + 24].mean() / zl[t - 24:t].mean() - 1) * 100,
            float(m.days[d].dayofweek >= 5),
        ])
    return np.asarray(rows, dtype=np.float64)


def eval_masks(m: Market, high_threshold):
    """Per-day evaluation groups. `crash` uses the actual target-day price: evaluation only."""
    truth = m.truth
    feats = regime_features(m)
    prev_mean, prev_vol = feats[:, 0], feats[:, 1]
    val = slice(m.ntrain, m.ntrain + m.nval)
    calm = prev_vol <= np.median(prev_vol[val])          # threshold from validation days
    crash = (prev_mean > high_threshold) & (truth.mean(1) < 0.6 * prev_mean)
    return {"calm": calm, "crash": crash}
