"""Evaluation metrics and paired weekly block bootstrap. All errors in $/MWh, error = prediction - actual."""
import numpy as np


def scores(truth, pred, *, high_threshold, taus, masks):
    """truth/pred: (days, 24). taus: {"p95": value, ...} large-error thresholds on |actual - prediction|.
    masks: {"calm": bool(days), "crash": bool(days)}."""
    err = pred - truth
    abs_err = np.abs(err)
    day_mae = abs_err.mean(1)
    high = truth > high_threshold
    n_tail = max(1, int(np.ceil(0.05 * len(day_mae))))
    out = {
        "MAE": abs_err.mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "high_RMSE": np.sqrt((err[high] ** 2).mean()) if high.any() else np.nan,
        "CVaR5_day_MAE": np.sort(day_mae)[::-1][:n_tail].mean(),
        "worst_day_MAE": day_mae.max(),
        "bias": err.mean(),
    }
    for name, tau in taus.items():
        out[f"n_large_{name}"] = int((abs_err > tau).sum())
    for name, mask in masks.items():
        out[f"{name}_MAE"] = abs_err[mask].mean() if mask.any() else np.nan
        out[f"{name}_days"] = int(mask.sum())
    return {k: float(v) for k, v in out.items()}


def block_bootstrap(loss_model, loss_base, block=7, n_boot=5000, seed=0):
    """Paired daily loss difference (model - base). Returns mean diff, 95% CI, win rate."""
    diff = np.asarray(loss_model) - np.asarray(loss_base)
    nb = len(diff) // block
    blocks = diff[:nb * block].reshape(nb, block).mean(1)
    rng = np.random.default_rng(seed)
    boots = blocks[rng.integers(0, nb, size=(n_boot, nb))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "win_rate": float((diff < 0).mean())}
