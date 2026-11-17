"""Export ensemble ingredients: every model's hourly predictions on validation and test days, in one table.

Models (A, B, C seed-averaged; $/MWh or EUR/MWh):
  A            LSTM (01_make_oof.py, main model)
  B            mean actual price of the k retrieved past days (candidates saved by 05)
  C            LSTM + case attention (05_train_case_attn.py, main model)
  LEAR         average of the four LEAR windows (08_lear.py)
  LEAR_w<N>    LEAR with an N-day calibration window (56, 84, 1092, 1456)
Flags per row (same for the 24 hours of a day):
  volatile_prev_day  previous day's price std > training-day 90% quantile (known at forecast time)
  extreme_window     inside an extreme-event window (data.extreme_days; uses actual prices,
                     for evaluation only, never as a model input)

Writes results/<tag>/<split>/export/
  predictions_val.csv, predictions_test.csv   Date, part, actual, model columns, flags
  error_corr_val.csv, error_corr_test.csv     hourly error correlation between models
  scores.csv                                  MAE / RMSE per model and part (+ extreme / rest on test)
Ensemble weights must come from the validation file only; compare ensembles against LEAR.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from care import data


def windows_mask(flagged):
    """Extreme-event windows: first..last flagged day of each event (events within 7 days merge)."""
    out = np.zeros(len(flagged), bool)
    for a, b in data.events(flagged):
        out[a:b + 1] = True
    return out


def run(split, args):
    base = args.out / split
    seeds = sorted(p for p in base.glob("seed_*") if (p / "case_attn" / "predictions.npz").exists())
    if not seeds:
        raise SystemExit(f"{base}: no seeds with 05_train_case_attn.py output")
    m = data.load(split, args.csv)
    z0 = np.load(seeds[0] / "residuals.npz")
    part = z0["part"]
    vol = data.regime_features(m)[:, 1]
    volatile = vol > np.quantile(vol[:m.ntrain], 0.90)
    flagged = data.extreme_days(m)[0]
    out_dir = base / "export"
    out_dir.mkdir(exist_ok=True)
    scores = []
    for name in ["val", "test"]:
        sel = part == name
        idx = np.flatnonzero(sel)
        A, B, C = [], [], []
        for s in seeds:
            z = np.load(s / "residuals.npz")
            c = np.load(s / "case_attn" / "predictions.npz")
            ids, mask = c["candidate_ids"], c["candidate_mask"]
            b = (z["truth"][ids] * mask[..., None]).sum(1) / np.maximum(mask.sum(1), 1)[:, None]
            A.append(z["pred_main"][sel]); B.append(b[sel]); C.append(c[name])
        P = {"A": np.mean(A, 0), "B": np.mean(B, 0), "C": np.mean(C, 0)}
        lear = np.load(base / "lear" / f"{name}.npz")
        assert np.array_equal(lear["idx"], idx), f"{split} {name}: LEAR days do not match"
        P["LEAR"] = lear["ens"]
        for w in lear["windows"]:
            P[f"LEAR_w{w}"] = lear[f"w{w}"]
        truth = z0["truth"][sel]
        assert np.allclose(lear["truth"], truth)

        stamps = (m.days.values[idx][:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
        frame = pd.DataFrame({"Date": stamps, "part": name, "actual": truth.ravel()})
        for k, p in P.items():
            frame[k] = p.ravel()
        ext = windows_mask(flagged[idx])
        frame["volatile_prev_day"] = np.repeat(volatile[idx], 24)
        frame["extreme_window"] = np.repeat(ext, 24)
        frame.to_csv(out_dir / f"predictions_{name}.csv", index=False, float_format="%.4f")

        names = list(P)
        err = np.array([(P[k] - truth).ravel() for k in names])
        pd.DataFrame(np.corrcoef(err), names, names).round(4).to_csv(out_dir / f"error_corr_{name}.csv")
        for k, p in P.items():
            e = p - truth
            row = {"part": name, "model": k, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean())}
            if ext.any():
                row.update({"extreme_MAE": np.abs(e[ext]).mean(), "rest_MAE": np.abs(e[~ext]).mean()})
            scores.append(row)
        print(f"{split} {name}: {len(idx)} days, {len(seeds)} seeds, models={names}, "
              f"extreme-window days={int(ext.sum())}, volatile days={int(volatile[idx].sum())}", flush=True)
    pd.DataFrame(scores).round(4).to_csv(out_dir / "scores.csv", index=False)
    print(f"-> {out_dir}")
    print(pd.read_csv(out_dir / "error_corr_test.csv", index_col=0).round(2).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--tag", default="step1")
    p.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    args = p.parse_args()
    args.out = ROOT / "results" / args.tag
    pd.set_option("display.width", 200)
    for split in args.splits:
        run(split, args)


if __name__ == "__main__":
    main()
