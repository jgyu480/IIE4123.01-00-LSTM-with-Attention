import fcntl
import hashlib
import json
from pathlib import Path

RUN_OUT = Path(__file__).resolve().parent
run_guard = (RUN_OUT / "run.lock").open("w")
try:
    fcntl.flock(run_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("같은 실험이 이미 실행 중입니다.")

# Reuse the same validated data loader, causal Recent, and splits.
source_path = Path("results/lear_a_neural_attention/train.py").resolve()
source = source_path.read_text()
marker = "class Corrector(nn.Module):"
if marker not in source or "# VALIDATION_RECENT_FIX" not in source:
    raise SystemExit("앞서 수정한 lear_a_neural_attention/train.py가 필요합니다.")

prefix = source.split(marker, 1)[0]
namespace = {"__file__": str(source_path), "__name__": "__loader__"}
exec(compile(prefix, str(source_path), "exec"), namespace)

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)
DEVICE = torch.device("cpu")

Y, A, L, B = [namespace[k] for k in ["Y", "A", "L", "B"]]
features, memory = [namespace[k] for k in ["features", "memory"]]
history, future = [namespace[k] for k in ["history", "future"]]
original_recent = namespace["original_recent"]
fit_ids, stop_ids, select_ids, test = [
    namespace[k] for k in ["fit_ids", "stop_ids", "select_ids", "test"]
]
dates = namespace["dates"]
scale = namespace["scale"]
OUT = RUN_OUT
SEEDS = [42, 43, 44]
METHODS = [
    "MLP_Transition",
    "Horizon_Plain",
    "Horizon_Transition",
    "History_Transition",
    "ErrorMemory_Transition",
]

def log(message):
    print(message, flush=True)

# Forecast-time state, including model forecasts and load changes.
previous = history[:, -24:, 0]
state_raw = np.column_stack([
    previous.mean(axis=1),
    previous.std(axis=1),
    previous.mean(axis=1)-history[:, -48:-24, 0].mean(axis=1),
    history[:, -6:, 0].mean(axis=1)-previous.mean(axis=1),
    future.mean(axis=1),
    (future-history[:, -24:, 1:3]).mean(axis=1),
    A.mean(axis=1)-previous.mean(axis=1),
    L.mean(axis=1)-previous.mean(axis=1),
    np.abs(A-L).mean(axis=1),
    (A[:, 16:21]-L[:, 16:21]).mean(axis=1),
])
state_mean = state_raw[fit_ids].mean(axis=0)
state_std = np.maximum(state_raw[fit_ids].std(axis=0), 1e-6)
state = np.clip((state_raw-state_mean)/state_std, -12, 12)

# Training labels only. Actual target-day prices never enter the inputs.
day_change = Y.mean(axis=1)-previous.mean(axis=1)
lower = min(float(np.quantile(day_change[fit_ids], 0.10)), -1e-6)
upper = max(float(np.quantile(day_change[fit_ids], 0.90)), 1e-6)
classes = np.where(day_change < lower, 0, np.where(day_change > upper, 2, 1))
counts = np.bincount(classes[fit_ids], minlength=3)
class_weights = len(fit_ids)/(3*np.maximum(counts, 1))
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Store causally generated Recent residuals as a rolling memory.
residual = (Y-B)/scale
N = len(Y)
memory_ids = np.zeros((N, 28), dtype=int)
memory_mask = np.zeros((N, 28), dtype=bool)
memory_age = np.zeros((N, 28), dtype=np.float32)
for t in range(N):
    # Include only prior days after the baseline's warm-up period.
    ids = np.arange(max(60, t-28), t)
    if len(ids):
        memory_ids[t, :len(ids)] = ids
        memory_mask[t, :len(ids)] = True
        memory_age[t, :len(ids)] = (t-ids)/28

case_states = state[memory_ids]
case_values = residual[memory_ids]
case_values = np.where(memory_mask[..., None], case_values, 0)
valid_rows = np.r_[fit_ids, stop_ids, select_ids, test]

for name, array in [
    ("features", features), ("state", state), ("history", memory),
    ("residual", residual), ("case_states", case_states),
    ("case_values", case_values),
]:
    if not np.isfinite(array[valid_rows]).all():
        raise ValueError(f"{name}에 NaN/Inf가 있습니다.")

T = [
    torch.tensor(x, dtype=torch.float32)
    for x in [
        features, state, memory, case_states,
        case_values, memory_mask.astype(float), memory_age, residual,
    ]
]
labels = torch.tensor(classes, dtype=torch.long)

def loader(ids, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*(x[ids] for x in T), labels[ids]),
        batch_size=16, shuffle=shuffle, generator=generator,
    )

class Model(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        d = 16
        self.embed = nn.Linear(features.shape[-1], d)
        self.state_embed = nn.Linear(state.shape[-1], d)
        self.norm = nn.LayerNorm(d)
        self.feedforward = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1)
        )
        self.mean_head = nn.Linear(d, 1)
        self.shape_head = nn.Linear(d, 1)
        self.transition_head = nn.Linear(d, 3)

        self.attention = nn.MultiheadAttention(
            d, num_heads=2, dropout=0.1, batch_first=True
        )
        self.history_embed = nn.Linear(18, d)
        self.history_position = nn.Parameter(torch.zeros(1, 28, d))
        self.case_key = nn.Linear(state.shape[-1], d)
        self.age_penalty = nn.Parameter(torch.tensor(0.0))
        self.case_gain = nn.Parameter(torch.tensor(0.0))

        # All variants start from the same Recent prediction.
        for head in [self.mean_head, self.shape_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, state_x, hist, keys, values, mask, age):
        z = torch.nn.functional.gelu(self.embed(x))
        state_z = torch.nn.functional.gelu(self.state_embed(state_x))
        z = z+state_z[:, None, :]

        if self.method.startswith("Horizon"):
            context, _ = self.attention(z, z, z, need_weights=False)
            z = self.norm(z+context)

        if self.method == "History_Transition":
            kv = self.history_embed(hist)+self.history_position
            context, _ = self.attention(z, kv, kv, need_weights=False)
            z = self.norm(z+context)

        case_correction = None
        if self.method == "ErrorMemory_Transition":
            k = self.case_key(keys)
            scores = (state_z[:, None, :]*k).sum(-1)/(k.shape[-1]**0.5)
            scores = scores-torch.nn.functional.softplus(self.age_penalty)*age
            scores = scores.masked_fill(mask == 0, -1e4)
            weights = scores.softmax(-1)*mask
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-8)
            context = (weights[..., None]*k).sum(1)
            z = self.norm(z+context[:, None, :])
            case_correction = (weights[..., None]*values).sum(1)

        h = self.feedforward(z)
        daily = self.mean_head(h.mean(1))
        shape = self.shape_head(h).squeeze(-1)
        shape = shape-shape.mean(1, keepdim=True)
        correction = daily+shape
        if case_correction is not None:
            correction = correction+torch.tanh(self.case_gain)*case_correction

        logits = self.transition_head(z.mean(1))
        return correction, logits

@torch.no_grad()
def predict(model, ids):
    model.eval()
    corrections, probabilities = [], []
    for batch in loader(ids):
        x, state_x, hist, keys, values, mask, age, _, _ = batch
        correction, logits = model(
            x, state_x, hist, keys, values, mask, age
        )
        corrections.append(correction.numpy())
        probabilities.append(logits.softmax(-1).numpy())
    return np.concatenate(corrections), np.concatenate(probabilities)

def train(method, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Model(method)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.02
    )
    ce = nn.CrossEntropyLoss(weight=class_weights)
    auxiliary_weight = 0.0 if method == "Horizon_Plain" else 0.05

    best_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    rows = []

    for epoch in range(1, 201):
        model.train()
        for batch in loader(fit_ids, True, seed+epoch):
            x, state_x, hist, keys, values, mask, age, target, label = batch
            optimizer.zero_grad(set_to_none=True)
            correction, logits = model(
                x, state_x, hist, keys, values, mask, age
            )
            forecast_loss = (correction-target).square().mean()
            loss = (
                forecast_loss
                + 0.02*correction.square().mean()
                + auxiliary_weight*ce(logits, label)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}: nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        correction, _ = predict(model, stop_ids)
        stop_loss = float(np.mean((correction-residual[stop_ids])**2))
        rows.append({"epoch": epoch, "stop_MSE_scaled": stop_loss})
        if stop_loss < best_loss-1e-7:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1

        if epoch == 1 or epoch % 20 == 0:
            log(f"{method} seed={seed} epoch={epoch} stop={stop_loss:.6f}")
        if stale >= 20:
            break

    model.load_state_dict(best_state)
    folder = OUT/method/f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, folder/"best.pt")
    pd.DataFrame(rows).to_csv(folder/"history.csv", index=False)
    log(f"{method} seed={seed} best_epoch={best_epoch}")

    cv, pv = predict(model, select_ids)
    ct, pt = predict(model, test)
    return cv*scale, ct*scale, pv, pt

log(f"\n학습 {len(fit_ids)}일 / 조기종료 {len(stop_ids)}일 / "
    f"방법선택 {len(select_ids)}일 / 테스트 {len(test)}일")
log(f"전환 보조 학습 기준: 하락 {lower:.3f}, 상승 {upper:.3f}")
log(f"학습 클래스 수: {counts.tolist()}")

# Preserve the source that defines the common baseline and inputs.
(OUT/"loader_snapshot.py").write_text(prefix)
results = {}
probabilities = {}
selection = {}
trial_rows = []

for method in METHODS:
    log(f"\n[학습 시작] {method}")
    outputs = [train(method, seed) for seed in SEEDS]
    results[method] = (
        np.mean([x[0] for x in outputs], axis=0),
        np.mean([x[1] for x in outputs], axis=0),
    )
    probabilities[method] = np.mean([x[3] for x in outputs], axis=0)

# Control: uniform average of the same prior residual memory.
uniform = (
    case_values*memory_mask[..., None]
).sum(1)/np.maximum(memory_mask.sum(1), 1)[:, None]
results["MeanErrorMemory"] = (
    uniform[select_ids]*scale, uniform[test]*scale
)

# Select correction strength exclusively on the selection segment.
for method, (cv, ct) in results.items():
    options = []
    for strength in [0.0, 0.25, 0.5, 1.0]:
        err = B[select_ids]+strength*cv-Y[select_ids]
        row = {
            "method": method, "strength": strength,
            "val_MAE": float(np.abs(err).mean()),
            "val_RMSE": float(np.sqrt(np.mean(err**2))),
        }
        options.append(row)
        trial_rows.append(row)
    best = min(options, key=lambda x: (x["val_RMSE"], x["strength"]))
    selection[method] = best
    log(f"[검증 선택] {best}")

def metrics(pred, ids):
    error = pred-Y[ids]
    mean_error = error.mean(axis=1)
    centered = error-mean_error[:, None]
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "level_MSE": float(np.mean(mean_error**2)),
        "shape_MSE": float(np.mean(centered**2)),
    }

pred_val = {"TrainingRecent": B[select_ids]}
pred_test = {
    "OriginalRecent": original_recent[test],
    "TrainingRecent": B[test],
    "A": A[test], "LEAR": L[test],
}
for method, (cv, ct) in results.items():
    strength = selection[method]["strength"]
    pred_val[method] = B[select_ids]+strength*cv
    pred_test[method] = B[test]+strength*ct

reference = metrics(original_recent[test], test)
rows = []
for method, prediction in pred_test.items():
    row = {"method": method}
    if method in pred_val:
        row.update({
            f"val_{k}": v for k, v in metrics(pred_val[method], select_ids).items()
        })
    row.update({
        f"test_{k}": v for k, v in metrics(prediction, test).items()
    })
    for key in ["MAE", "RMSE"]:
        row[f"gain_vs_OriginalRecent_{key}_pct"] = (
            100*(1-row[f"test_{key}"]/reference[key])
        )
    rows.append(row)

scores = pd.DataFrame(rows).sort_values("val_RMSE", kind="stable")
scores.to_csv(OUT/"scores.csv", index=False)
pd.DataFrame(trial_rows).to_csv(OUT/"selection.csv", index=False)

for part, ids, predictions in [
    ("val", select_ids, pred_val), ("test", test, pred_test)
]:
    stamps = (
        dates.values[ids, None]
        + np.arange(24)[None, :]*np.timedelta64(1, "h")
    ).ravel()
    saved = pd.DataFrame({"Date": stamps, "actual": Y[ids].ravel()})
    for method, prediction in predictions.items():
        saved[method] = prediction.ravel()
    saved.to_csv(OUT/f"predictions_{part}.csv", index=False)

probability_table = pd.DataFrame({
    "date": dates[test].astype(str), "actual_transition": classes[test]
})
for method, p in probabilities.items():
    for j, name in enumerate(["drop", "middle", "rise"]):
        probability_table[f"{method}_{name}"] = p[:, j]
probability_table.to_csv(OUT/"transition_probabilities_test.csv", index=False)

settings = {
    "seeds": SEEDS, "selection": selection,
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "thresholds": {"drop": lower, "rise": upper},
    "fit_dates": dates[fit_ids].astype(str).tolist(),
    "stop_dates": dates[stop_ids].astype(str).tolist(),
    "selection_dates": dates[select_ids].astype(str).tolist(),
    "original_recent_test": reference,
    "baseline": "Same causal TrainingRecent as patched neural-attention experiment",
    "original_recent_used_as_training_baseline": False,
    "memory": "Previous complete days only; rolling 28-day residual memory",
    "label_order": ["drop", "middle", "rise"],
}
(OUT/"settings.json").write_text(json.dumps(settings, indent=2))
log("\n"+scores.to_string(index=False))
(OUT/"complete.ok").write_text("complete\n")
log(f"\n완료. 결과 위치: {OUT}")
