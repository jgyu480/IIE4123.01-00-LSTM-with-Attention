"""Stage 1 of the LEAR -> LSTM boosting ensemble: LEAR forecasts for every day of a market.

LEAR is recalibrated every day on the days strictly before it (see
adaptive-weighted-LSTM-Attention/scripts/08_lear.py, whose functions are reused unchanged), so each
forecast is out-of-sample and does not depend on any train/validation/test split. One file per
market therefore serves every split, and the residuals (actual - LEAR) on training days are clean
targets for the stage-2 residual LSTM (no fold models needed).

Days: from the first day with at least --min-history days before it, to the end of the data.
Windows 56/84/1092/1456 days; while less history exists than a window, the window uses all
available days (an expanding window). The main LEAR is the average of the four windows.

Output: ensemble/results/<market>/lear_all.npz
  day      project day index (data.load: day i = calendar day i + 7)
  date     YYYY-MM-DD
  history  days of history available before the day
  truth    actual prices (days, 24)
  w<N>     LEAR forecast with an N-day window (days, 24)
  ens      average of the windows (days, 24)
Runtime (15 CPU cores, PJM): about 75 minutes. Finished days are kept in chunk files, so an
interrupted run resumes where it stopped.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
AWLA = REPO / "adaptive-weighted-LSTM-Attention"
sys.path.insert(0, str(AWLA / "src"))
from care import data  # noqa: E402

_spec = importlib.util.spec_from_file_location("lear08", AWLA / "scripts" / "08_lear.py")
lear08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lear08)

MARKETS = {"PJM": data.DEFAULT_CSV, **{m: AWLA / "data" / "epftoolbox" / f"{m}.csv" for m in ["NP", "BE", "FR", "DE"]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="PJM", choices=list(MARKETS))
    p.add_argument("--windows", nargs="+", type=int, default=[56, 84, 1092, 1456])
    p.add_argument("--min-history", type=int, default=364, help="first forecast day needs this many past days")
    p.add_argument("--chunk", type=int, default=60, help="days per saved chunk (resume unit)")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0, help="only the first N days, nothing saved (quick check)")
    args = p.parse_args()

    m = data.load("split2_2018", MARKETS[args.market])            # split2 = full data, no days removed
    calendar = m.hours[::24]
    X, Y = lear08.build(m.raw, calendar.dayofweek.to_numpy())
    offset = lear08.OFFSET
    first = max(0, args.min_history - offset)                      # project day index of the first forecast
    days = np.arange(first, len(m.days))
    if args.limit:
        days = days[:args.limit]
    cal = days + offset

    out_dir = HERE / "results" / args.market
    parts = out_dir / "chunks"
    parts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for c0 in range(0, len(days), args.chunk):
        chunk_days = days[c0:c0 + args.chunk]
        f = parts / f"days_{chunk_days[0]:05d}_{chunk_days[-1]:05d}.npz"
        if f.exists() and not args.limit:
            continue
        D = chunk_days + offset
        preds = {}
        for w in args.windows:
            res = Parallel(n_jobs=args.jobs)(delayed(lear08.forecast_day)(X, Y, d, w) for d in D)
            preds[f"w{w}"] = np.array([r[0] for r in res])
        err = np.mean(list(preds.values()), 0) - Y[D]
        print(f"{args.market} days {m.days[chunk_days[0]]:%Y-%m-%d}..{m.days[chunk_days[-1]]:%Y-%m-%d} "
              f"LEAR MAE={np.abs(err).mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
        if not args.limit:
            np.savez_compressed(f, day=chunk_days, **preds)
    if args.limit:
        return

    files = sorted(parts.glob("days_*.npz"))
    loaded = [np.load(f) for f in files]
    day = np.concatenate([z["day"] for z in loaded])
    assert np.array_equal(day, days), "chunk files do not cover the requested days; delete chunks and rerun"
    preds = {f"w{w}": np.concatenate([z[f"w{w}"] for z in loaded]) for w in args.windows}
    ens = np.mean(list(preds.values()), 0)
    np.savez_compressed(out_dir / "lear_all.npz", day=day, date=np.array(m.days[day].strftime("%Y-%m-%d")),
                        history=day + offset, truth=Y[day + offset], windows=np.array(args.windows), ens=ens, **preds)
    err = ens - Y[day + offset]
    print(f"{args.market}: {len(day)} days {m.days[day[0]]:%Y-%m-%d}..{m.days[day[-1]]:%Y-%m-%d}, "
          f"LEAR MAE={np.abs(err).mean():.3f} RMSE={np.sqrt((err ** 2).mean()):.3f} -> {out_dir / 'lear_all.npz'}")


if __name__ == "__main__":
    main()
