"""Boosting, stage 2 with an LSTM: LEAR (stage 1) + an LSTM fitted to LEAR's residuals.

    final(d, :) = LEAR(d, :) + c * LSTM(inputs(d))          (24 hours at once)

Stage 1  LEAR forecasts of every day (ensemble/lear_history.py). LEAR is recalibrated daily on past
         days only, so its residuals r = actual - LEAR are out-of-sample on every day, training days
         included. No fold models are needed.
Stage 2  The LSTM of model A (care.baseline: LSTM(3, 64) on the previous 168 hours of price and both
         load forecasts, then an MLP with the target day's load forecasts and weekday), with two more
         inputs for the target day: LEAR's 24-hour forecast and the short-minus-long window gap
         (short = 56/84-day windows, long = 1092/1456-day windows). Output = the 24-hour residual.
         The last layer starts at zero, so training starts from "LEAR unchanged".
         Everything is scaled with the price mean/std of the hours before the validation year (as A).
Protocol (same split as the project; test days never used for any choice):
  train on training days with early stopping on validation residual MSE (as A: Adam 1e-3, batch 32,
  at most 100 epochs, patience 10), 5 seeds, average the seeds -> on validation days choose the
  shrinkage c in {0, 0.1, ..., 1} -> apply to test days. Two training losses: MSE and L1; the one with
  the lower validation RMSE is the main model ("lstm_boost").
Reported against LEAR, soft voting valw(A + LEAR_S + LEAR_L) and tree boosting (lear_tree_boost.py).

Output: ensemble/boosting/results/<market>/<split>/lstm/predictions_{val,test}.csv, summary.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ENS = HERE.parent
sys.path.insert(0, str(HERE))
import lear_tree_boost as tree  # noqa: E402  (also puts adaptive-weighted-LSTM-Attention/src on the path)
from care import baseline, data, metrics  # noqa: E402

LOSS_FN = {"mse": nn.functional.mse_loss, "l1": nn.functional.l1_loss}


class ResidualLSTM(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden + 48 + 7 + 48, 64), nn.ReLU(), nn.Linear(64, 24))
        nn.init.zeros_(self.mlp[-1].weight)          # start from residual 0 = plain LEAR
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, history, forecast, weekday, lear):
        _, (h, _) = self.lstm(history)
        return self.mlp(torch.cat((h[-1], forecast.flatten(1), weekday, lear.flatten(1)), dim=1))


@torch.no_grad()
def predict(model, T, idx, device, batch_size=256):
    model.eval()
    out = [model(*(t[idx[i:i + batch_size]].to(device) for t in T[:4])).cpu() for i in range(0, len(idx), batch_size)]
    return torch.cat(out).numpy().astype(np.float64)


def train_one(T, tr, va, seed, loss, device, args):
    baseline.seed_all(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*(t[tr] for t in T)), args.batch, shuffle=True, generator=g)
    model = ResidualLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    f = LOSS_FN[loss]
    rv = T[4][va].numpy()
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for *x, y in loader:
            opt.zero_grad(set_to_none=True)
            l = f(model(*(t.to(device) for t in x)), y.to(device))
            if not torch.isfinite(l):
                raise RuntimeError("nonfinite training loss")
            l.backward()
            opt.step()
        v = float(((predict(model, T, va, device) - rv) ** 2).mean())    # early stopping on validation MSE
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "val_MSE_scaled": best}


def run(market, split, args, device):
    t0 = time.time()
    m = data.load(split, tree.CSV[market])
    z = np.load(ENS / "results" / market / "lear_all.npz")
    keep = z["day"] < len(m.days)
    day = z["day"][keep]
    lear = {k: z[k][keep] for k in ["ens", "w56", "w84", "w1092", "w1456"]}
    truth = m.truth[day]
    assert np.allclose(truth, z["truth"][keep])
    part = np.where(day < m.ntrain, "train", np.where(day < m.ntrain + m.nval, "val", "test"))

    mean, std = m.scaling(m.ntrain)                           # hours before the validation year
    pm, ps = mean[0], std[0]
    history, forecast, weekday, _ = m.tensors(mean, std)
    L = np.zeros((len(m.days), 24, 2), np.float32)            # LEAR inputs; days without LEAR stay unused
    gap = (lear["w56"] + lear["w84"]) / 2 - (lear["w1092"] + lear["w1456"]) / 2
    L[day, :, 0] = (lear["ens"] - pm) / ps
    L[day, :, 1] = gap / ps
    R = np.zeros((len(m.days), 24), np.float32)
    R[day] = (truth - lear["ens"]) / ps                       # residual target, scaled
    T = (history, forecast, weekday, torch.from_numpy(L), torch.from_numpy(R))
    tr, va, te = (torch.from_numpy(day[part == k]) for k in ["train", "val", "test"])
    nv, nt = len(va), len(te)
    rv = (truth - lear["ens"])[part == "val"]

    runs = {}
    for loss in args.losses:
        pv, pt, info = [], [], []
        for seed in range(args.seeds):
            model, inf = train_one(T, tr, va, seed, loss, device, args)
            pv.append(predict(model, T, va, device) * ps)
            pt.append(predict(model, T, te, device) * ps)
            info.append(inf)
            print(f"  {market} {split} {loss} seed={seed} best_epoch={inf['best_epoch']} "
                  f"val_RMSE(residual)={np.sqrt(((rv - pv[-1]) ** 2).mean()):.3f} [{time.time() - t0:.0f}s]", flush=True)
        pv, pt = np.mean(pv, 0), np.mean(pt, 0)               # seed average
        c = float(min(tree.SHRINK, key=lambda s: ((rv - s * pv) ** 2).mean()))
        runs[loss] = {"pv": pv, "pt": pt, "c": c, "seeds": info,
                      "val_RMSE": float(np.sqrt(((rv - c * pv) ** 2).mean()))}
    main = min(runs, key=lambda k: runs[k]["val_RMSE"])

    L_val, L_test = lear["ens"][part == "val"], lear["ens"][part == "test"]
    y_test = truth[part == "test"]
    P = {"LEAR": L_test}
    for loss, rr in runs.items():
        P[f"lstm_{loss}"] = L_test + rr["c"] * rr["pt"]
        P[f"lstm_{loss}_c1"] = L_test + rr["pt"]
    P["lstm_boost"] = P[f"lstm_{main}"]
    sv = tree.soft_vote(market, split, day[part == "val"], day[part == "test"], m)
    if sv is not None:
        P["soft_vote"], sv_w = sv
    tree_csv = HERE / "results" / market / split / "predictions_test.csv"
    if tree_csv.exists():
        f = pd.read_csv(tree_csv)
        assert pd.Timestamp(f["Date"].iloc[0]) == m.days[day[part == "test"][0]] and len(f) == nt * 24
        P["tree_boost"] = f["boost"].to_numpy().reshape(nt, 24)
    exp = tree.AWLA / "results" / tree.EXPORT_TAG[market] / split / "export" / "predictions_test.csv"
    if exp.exists():
        P["A"] = pd.read_csv(exp)["A"].to_numpy().reshape(nt, 24)               # reference only
    flagged = data.extreme_days(m)[0][day[part == "test"]]
    ext = np.zeros(nt, bool)
    for a, b in data.events(flagged):
        ext[a:b + 1] = True

    table = {}
    for k, p in P.items():
        e = p - y_test
        table[k] = {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                    "ext_MAE": float(np.abs(e[ext]).mean()) if ext.any() else None,
                    "rest_MAE": float(np.abs(e[~ext]).mean())}
    boots = []
    for ref in ["LEAR", "soft_vote", "tree_boost"]:
        if ref not in P:
            continue
        for loss, f in tree.LOSSES.items():
            b = metrics.block_bootstrap(f(P["lstm_boost"] - y_test), f(P[ref] - y_test), n_boot=args.n_boot)
            boots.append({"model": "lstm_boost", "vs": ref, "loss": loss, **b})

    out = HERE / "results" / market / split / "lstm"
    out.mkdir(parents=True, exist_ok=True)
    for name, days_p, base in [("val", day[part == "val"], L_val), ("test", day[part == "test"], L_test)]:
        stamps = (m.days.values[days_p][:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
        f = pd.DataFrame({"Date": stamps, "actual": truth[np.isin(day, days_p)].ravel(), "LEAR": base.ravel()})
        for loss, rr in runs.items():
            raw = rr["pv"] if name == "val" else rr["pt"]
            f[f"resid_{loss}"] = raw.ravel()                                  # raw residual prediction
            f[f"lstm_{loss}"] = (base + rr["c"] * raw).ravel()
        f["lstm_boost"] = f[f"lstm_{main}"]
        f.to_csv(out / f"predictions_{name}.csv", index=False, float_format="%.4f")
    summary = {"market": market, "split": split, "test_first": str(m.days[day[part == "test"][0]].date()),
               "train_days": int((part == "train").sum()), "train_first": str(m.days[day[0]].date()),
               "val_days": nv, "test_days": nt, "extreme_window_days": int(ext.sum()), "main_loss": main,
               "device": str(device), "settings": {k: v for k, v in vars(args).items() if k not in ("markets", "splits")},
               "runs": {k: {kk: v for kk, v in rr.items() if kk in ("c", "val_RMSE", "seeds")} for k, rr in runs.items()},
               "soft_vote_weights": sv_w if sv is not None else None, "test": table, "bootstrap": boots}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n===== {market} {split}: test {summary['test_first']}~, train {summary['train_days']} days from "
          f"{summary['train_first']}, {time.time() - t0:.0f}s =====")
    for k, rr in runs.items():
        print(f"  {k:4s} shrink c={rr['c']:.1f} val_RMSE(residual)={rr['val_RMSE']:.3f} "
              f"epochs={[s['best_epoch'] for s in rr['seeds']]}" + ("  <- main" if k == main else ""))
    print(pd.DataFrame(table).T.round(3).to_string())
    print(pd.DataFrame(boots).round(3).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+", default=["PJM"], choices=list(tree.CSV))
    p.add_argument("--splits", nargs="+", default=["split2_2018", "split1_2017"], choices=list(data.SPLITS))
    p.add_argument("--losses", nargs="+", default=["mse", "l1"], choices=list(LOSS_FN))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    pd.set_option("display.width", 200)
    device = baseline.setup(args.device, args.threads)
    for market in args.markets:
        for split in args.splits:
            run(market, split, args, device)


if __name__ == "__main__":
    main()
