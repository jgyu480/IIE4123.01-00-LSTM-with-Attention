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

class Corrector(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        width = 16
        self.embed = nn.Linear(features.shape[-1], width)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(0.1)
        )
        self.attention = None
        if method in ["HorizonAttention", "HistoryAttention", "WindowAttention"]:
            self.attention = nn.MultiheadAttention(
                width, num_heads=2, dropout=0.1, batch_first=True
            )
        if method == "HistoryAttention":
            self.memory_embed = nn.Linear(18, width)
            self.memory_position = nn.Parameter(torch.zeros(1, 28, width))
        if method == "WindowAttention":
            self.expert_embed = nn.Linear(2, width)
            self.expert_id = nn.Parameter(torch.randn(1, 5, width) * 0.02)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, mem, experts, baseline):
        z = torch.nn.functional.gelu(self.embed(x))
        if self.method == "HorizonAttention":
            context, _ = self.attention(z, z, z, need_weights=False)
            z = self.norm(z + context)
        elif self.method == "HistoryAttention":
            kv = self.memory_embed(mem) + self.memory_position
            context, _ = self.attention(z, kv, kv, need_weights=False)
            z = self.norm(z + context)
        elif self.method == "WindowAttention":
            # Each hour queries five expert tokens.
            n, hours, count = experts.shape
            expert_input = torch.stack([
                experts, experts-baseline[..., None]
            ], dim=-1).reshape(n*hours, count, 2)
            kv = self.expert_embed(expert_input) + self.expert_id
            query = z.reshape(n*hours, 1, -1)
            _, weights = self.attention(query, kv, kv)
            local = (
                weights[:, 0] * experts.reshape(n*hours, count)
            ).sum(-1).reshape(n, hours)
            # Start exactly from the existing Recent prediction.
            residual = self.output(self.mlp(z)).squeeze(-1)
            return residual * (local-baseline)
        return self.output(self.mlp(z)).squeeze(-1)

T = [
    torch.tensor(a, dtype=torch.float32)
    for a in [features, memory, E_scaled, B_scaled, target]
]

def loader(ids, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*(tensor[ids] for tensor in T)),
        batch_size=16, shuffle=shuffle, generator=generator,
    )

@torch.no_grad()
def predict(model, ids):
    model.eval()
    out = []
    for x, mem, experts, baseline, _ in loader(ids):
        out.append(model(
            x.to(DEVICE), mem.to(DEVICE),
            experts.to(DEVICE), baseline.to(DEVICE)
        ).cpu().numpy())
    return np.concatenate(out)

def train(method, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Corrector(method).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    best_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    training_log = []

    for epoch in range(1, 201):
        model.train()
        for x, mem, experts, baseline, y in loader(fit_ids, True, seed+epoch):
            optimizer.zero_grad(set_to_none=True)
            correction = model(
                x.to(DEVICE), mem.to(DEVICE),
                experts.to(DEVICE), baseline.to(DEVICE)
            )
            loss = (
                (correction-y.to(DEVICE)).square().mean()
                + 0.02 * correction.square().mean()
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method} seed={seed}: nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        correction = predict(model, stop_ids)
        stop_loss = float(np.mean((correction-target[stop_ids]) ** 2))
        training_log.append({"epoch": epoch, "stop_MSE_scaled": stop_loss})
        if stop_loss < best_loss-1e-7:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            log(f"{method} seed={seed} epoch={epoch} stop={stop_loss:.6f}")
        if stale >= 15:
            break

    model.load_state_dict(best_state)
    folder = OUT / method / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, folder / "best.pt")
    pd.DataFrame(training_log).to_csv(folder / "history.csv", index=False)
    log(f"{method} seed={seed} best_epoch={best_epoch}")

    return (
        predict(model, select_ids) * scale,
        predict(model, test) * scale,
    )

selected = {}
corrections = {}
selection_rows = []

# No test metrics are used during fitting or method selection.
for method in METHODS:
    log(f"\n[학습 시작] {method}")
    seed_outputs = [train(method, seed) for seed in SEEDS]
    cv = np.mean([item[0] for item in seed_outputs], axis=0)
    ct = np.mean([item[1] for item in seed_outputs], axis=0)
    choices = []
    for strength in [0.0, 0.25, 0.5, 1.0]:
        error = B[select_ids] + strength*cv - Y[select_ids]
        row = {
            "method": method, "strength": strength,
            "val_MAE": float(np.abs(error).mean()),
            "val_RMSE": float(np.sqrt(np.mean(error**2))),
        }
        choices.append(row)
        selection_rows.append(row)
    best = min(choices, key=lambda row: (row["val_RMSE"], row["strength"]))
    selected[method] = best
    corrections[method] = (cv, ct)
    log(f"[검증 선택] {best}")

pred_val = {"Recent": B[select_ids]}
pred_test = {
    "OriginalRecent": original_recent[test],
    "Recent": B[test],
    "A": A[test],
    "LEAR": L[test],
}
for method, (cv, ct) in corrections.items():
    strength = selected[method]["strength"]
    pred_val[method] = B[select_ids] + strength*cv
    pred_test[method] = B[test] + strength*ct

def metrics(pred, ids):
    error = pred-Y[ids]
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }

reference = metrics(original_recent[test], test)
rows = []
for method, pred in pred_test.items():
    row = {"method": method}
    if method in pred_val:
        row.update({
            f"val_{key}": value
            for key, value in metrics(pred_val[method], select_ids).items()
        })
    row.update({
        f"test_{key}": value
        for key, value in metrics(pred, test).items()
    })
    for metric in ["MAE", "RMSE"]:
        row[f"gain_{metric}_pct"] = (
            100 * (1-row[f"test_{metric}"]/reference[metric])
        )
    rows.append(row)

scores = pd.DataFrame(rows).sort_values("val_RMSE", kind="stable")
scores.to_csv(OUT / "scores.csv", index=False)
pd.DataFrame(selection_rows).to_csv(OUT / "selection.csv", index=False)

for part, ids, predictions in [
    ("val", select_ids, pred_val), ("test", test, pred_test)
]:
    timestamps = (
        dates.values[ids, None]
        + np.arange(24)[None, :] * np.timedelta64(1, "h")
    ).ravel()
    saved = pd.DataFrame({"Date": timestamps, "actual": Y[ids].ravel()})
    for method, pred in predictions.items():
        saved[method] = pred.ravel()
    saved.to_csv(OUT / f"predictions_{part}.csv", index=False)

settings = {
    "recent_files": used, "seeds": SEEDS,
    "comparison_reference": "OriginalRecent",
    "training_baseline": "Causal 14-day and expanding hourly blend",
    "training_baseline_equals_original": False,
    "selected": selected, "scale": scale,
    "feature_mean": feature_mean.tolist(),
    "feature_std": feature_std.tolist(),
    "history_mean": history_mean.tolist(),
    "history_std": history_std.tolist(),
    "fit_dates": dates[fit_ids].astype(str).tolist(),
    "stop_dates": dates[stop_ids].astype(str).tolist(),
    "selection_dates": dates[select_ids].astype(str).tolist(),
}
(OUT / "settings.json").write_text(json.dumps(settings, indent=2))
log("\n" + scores.to_string(index=False))
(OUT / "complete.ok").write_text("complete\n")
log(f"\n완료. 결과 저장 위치: {OUT}")
