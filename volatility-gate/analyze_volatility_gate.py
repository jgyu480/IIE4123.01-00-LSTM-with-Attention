"""Volatility gate 분석.

  diagnostic : 기존 PJM_error_analysis/results/abc/<run> 의 테스트 예측만 읽어 '변동성 vs A·C 상대 성능'을 점검한다.
               아무것도 학습하지 않는다. 여기 나오는 oracle 값은 테스트 정답을 쓴 사후 상한이며 성능이 아니다.
  evaluate   : train_volatility_gate.py 결과(results/vgate/<run>)를 평가한다.

구간 기준값(변동성·가격 분위수)은 모두 평가 origin 의 train 구간에서 계산한다.
'ex-ante' 구간은 예측 시점에 알 수 있는 값으로, 'ex-post' 구간은 목표일 실제 가격으로 정의한 사후 진단이다.

실행:  python analyze_volatility_gate.py diagnostic
       python analyze_volatility_gate.py evaluate --run results/vgate/<run>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vgate_lib as V

ROOT = Path(__file__).resolve().parent
TRIMS = [0, 364, 728, 1092]


def thresholds(origin):
    """평가 origin 의 train 목표일에서만 계산한 기준값."""
    trim = {V.origin_name(t): t for t in TRIMS}[origin]
    w = V.load_window(trim)
    train_starts = w["starts"][w["masks"][0]]
    f = V.gate_features(w["raw"], train_starts)
    daily_mean = np.stack([w["raw"][t:t + 24, 0] for t in train_starts]).mean(1)
    return {"origin": origin, "train_first": w["info"]["splits"]["train"]["first_target"],
            "train_last": w["info"]["splits"]["train"]["last_target"],
            "vol_q50": float(np.quantile(f["vol_std"], 0.5)), "vol_q90": float(np.quantile(f["vol_std"], 0.9)),
            "price_low_q10": float(np.quantile(daily_mean, 0.1)), "price_high_q90": float(np.quantile(daily_mean, 0.9))}


def day_context(dates, raw, index):
    starts = index.get_indexer(dates)
    f = V.gate_features(raw, starts)
    return pd.DataFrame({"date": dates, "prev_std": f["vol_std"].to_numpy(), "prev_mean": f["lvl_mean"].to_numpy(),
                         "actual_mean": np.stack([raw[t:t + 24, 0] for t in starts]).mean(1)})


def slices(ctx, th):
    hi, lo = th["price_high_q90"], th["price_low_q10"]
    prev_high = ctx["prev_mean"] > hi
    jan = (ctx["date"].dt.year == 2018) & (ctx["date"].dt.month == 1)
    out = {
        "all": ("-", np.ones(len(ctx), bool)),
        "vol_low(<=q50)": ("ex-ante", ctx["prev_std"] <= th["vol_q50"]),
        "vol_mid(q50-q90)": ("ex-ante", (ctx["prev_std"] > th["vol_q50"]) & (ctx["prev_std"] <= th["vol_q90"])),
        "vol_high(>q90)": ("ex-ante", ctx["prev_std"] > th["vol_q90"]),
        "prev_high_price": ("ex-ante", prev_high),
        "prev_low_price": ("ex-ante", ctx["prev_mean"] < lo),
        "high->drop": ("ex-post", prev_high & (ctx["actual_mean"] <= hi)),
        "high->persist": ("ex-post", prev_high & (ctx["actual_mean"] > hi)),
        "actual_low_price": ("ex-post", ctx["actual_mean"] < lo),
    }
    if jan.any():
        out["2018-01"] = ("calendar", jan)
        out["excl_2018-01"] = ("calendar", ~jan)
    return {k: (kind, np.asarray(m)) for k, (kind, m) in out.items()}


def slice_scores(truth, preds, masks):
    rows = []
    for name, (kind, m) in masks.items():
        row = {"slice": name, "kind": kind, "days": int(m.sum())}
        for model, p in preds.items():
            row[f"MAE_{model}"] = V.mae(truth[m], p[m]) if m.any() else np.nan
            row[f"RMSE_{model}"] = V.rmse(truth[m], p[m]) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(frame, keys):
    num = [c for c in frame.columns if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(frame[c])]
    g = frame.groupby(keys, sort=False)[num].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def md(frame, digits=3):
    f = frame.copy()
    for c in f.columns:
        if pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    lines = ["| " + " | ".join(map(str, f.columns)) + " |", "| " + " | ".join("---" for _ in f.columns) + " |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in f.to_numpy()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def diagnostic(args):
    run = args.abc or sorted(d for d in (V.BASE / "results" / "abc").glob("*") if d.is_dir())[-1]
    out = ROOT / "results" / "vgate_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    frame_raw = V.read_frame()
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    th = thresholds("o2018")
    reported = pd.read_csv(run / "metrics_all_seeds.csv")
    seeds = json.loads((run / "config.json").read_text(encoding="utf-8"))["a_config"]["seeds"]
    per_seed, assoc, conc, days_all = [], [], [], []
    for seed in seeds:
        f = pd.read_csv(run / "PJM" / f"seed_{seed}" / "predictions.csv", parse_dates=["Date"])
        dates = pd.DatetimeIndex(f["Date"].iloc[::24])
        assert np.allclose(f["actual"], raw[index.get_indexer(f["Date"]), 0], atol=1e-9), "actual != PJM.csv"
        truth = f["actual"].to_numpy().reshape(-1, 24)
        a, c = f["A"].to_numpy().reshape(-1, 24), f["C"].to_numpy().reshape(-1, 24)
        for name, p in (("A", a), ("C", c)):            # 원래 보고된 수치와 일치하는지 확인
            rep = reported[(reported.market == "PJM") & (reported.seed == seed) & (reported.model == name)].iloc[0]
            assert np.isclose(V.mae(truth, p), rep.MAE, atol=1e-6) and np.isclose(V.rmse(truth, p), rep.RMSE, atol=1e-6)
        ctx = day_context(pd.Series(dates), raw, index)
        masks = slices(ctx, th)
        s = slice_scores(truth, {"A": a, "C": c, "E50": 0.5 * a + 0.5 * c}, masks)
        mse_a, mse_c = ((a - truth) ** 2).mean(1), ((c - truth) ** 2).mean(1)
        gain = mse_a - mse_c                              # 양수 = C 가 그날 더 좋음
        s["C_win_rate"] = [float((gain[m] > 0).mean()) if m.any() else np.nan for _, m in masks.values()]
        s["share_of_C_gain"] = [float(gain[m].sum() / gain.sum()) for _, m in masks.values()]
        # 사후 oracle (테스트 정답 사용 -> 성능 아님, 구조적 상한 파악용)
        s["oracle_alpha_expost"] = [V.fixed_alpha(a[m], c[m], truth[m]) if m.any() else np.nan for _, m in masks.values()]
        s.insert(0, "seed", seed)
        per_seed.append(s)
        ea, ec = (a - truth).ravel(), (c - truth).ravel()
        notjan = np.repeat(masks["excl_2018-01"][1], 24)
        rank = lambda v: pd.Series(v).rank().to_numpy()
        assoc.append({
            "seed": seed,
            "spearman(prev_std, MSE_A-MSE_C)": float(np.corrcoef(rank(ctx["prev_std"]), rank(gain))[0, 1]),
            "spearman_excl_jan": float(np.corrcoef(rank(ctx["prev_std"][masks["excl_2018-01"][1]]),
                                                   rank(gain[masks["excl_2018-01"][1]]))[0, 1]),
            "error_corr_A_C": float(np.corrcoef(ea, ec)[0, 1]),
            "error_corr_A_C_excl_jan": float(np.corrcoef(ea[notjan], ec[notjan])[0, 1]),
            "oracle_const_alpha_expost": V.fixed_alpha(a, c, truth),
            "RMSE_oracle_const_expost": V.rmse(truth, V.blend(V.fixed_alpha(a, c, truth), a, c)),
        })
        top = np.argsort(-gain)[:10]
        conc.append(pd.DataFrame({"seed": seed, "date": dates[top].strftime("%Y-%m-%d"), "MSE_A-MSE_C": gain[top],
                                  "share_of_total_gain": gain[top] / gain.sum(), "prev_std": ctx["prev_std"].to_numpy()[top]}))
        days_all.append(pd.DataFrame({"seed": seed, "date": dates, "prev_std": ctx["prev_std"], "gain": gain}))
    per_seed = pd.concat(per_seed, ignore_index=True)
    per_seed.to_csv(out / "slices_per_seed.csv", index=False)
    summary = mean_std(per_seed, ["slice", "kind"])
    summary.to_csv(out / "slices_summary.csv", index=False)
    pd.DataFrame(assoc).to_csv(out / "association.csv", index=False)
    pd.concat(conc).to_csv(out / "top_gain_days.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(th, indent=2), encoding="utf-8")

    d = pd.concat(days_all)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for seed, g in d.groupby("seed"):
        ax[0].scatter(g["prev_std"], np.sign(g["gain"]) * np.log1p(np.abs(g["gain"])), s=6, alpha=0.5, label=f"seed {seed}")
    ax[0].axvline(th["vol_q90"], color="k", ls="--", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("previous-day price std"); ax[0].legend()
    ax[0].set_ylabel("signed log(1+|MSE_A - MSE_C|)  (>0: C better)")
    ax[0].set_title("Saved 2018 test: daily C gain vs previous-day volatility")
    keep = ["vol_low(<=q50)", "vol_mid(q50-q90)", "vol_high(>q90)", "2018-01", "excl_2018-01"]
    s = summary.set_index("slice").loc[keep]
    x = np.arange(len(keep))
    ax[1].bar(x - 0.2, s["RMSE_A_mean"], 0.4, yerr=s["RMSE_A_std"], label="A")
    ax[1].bar(x + 0.2, s["RMSE_C_mean"], 0.4, yerr=s["RMSE_C_std"], label="C")
    ax[1].set_xticks(x); ax[1].set_xticklabels(keep, rotation=20, fontsize=8); ax[1].set_ylabel("RMSE"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "diagnostic.png", dpi=130); plt.close(fig)

    cols = ["slice", "kind", "days_mean", "MAE_A_mean", "MAE_C_mean", "RMSE_A_mean", "RMSE_C_mean",
            "C_win_rate_mean", "share_of_C_gain_mean", "oracle_alpha_expost_mean"]
    print(json.dumps(th, indent=1))
    print(md(summary[cols]))
    print(md(pd.DataFrame(assoc)))
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
MODELS = {"A": "pred_A", "C": "pred_C", "E50": "pred_E50", "Efixed": "pred_Efixed", "gate": "pred_gate"}


def evaluate(args):
    run = args.run if args.run.is_absolute() else (ROOT / args.run if (ROOT / args.run).exists() else args.run.resolve())
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    frame_raw = V.read_frame()
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    overall, sliced, alpha_rows, boots, daily, corr = [], [], [], [], [], []
    th_cache = {}
    for path in sorted(run.glob("predictions_*.csv")):
        gate = path.stem.replace("predictions_", "")
        full = pd.read_csv(path, parse_dates=["Date"])
        test = full[full["split"] == "test"]
        for (origin, source), g0 in test.groupby(["origin", "source"], sort=False):
            th = th_cache.setdefault(origin, thresholds(origin))
            day_loss = {m: [] for m in MODELS}
            day_abs = {m: [] for m in MODELS}
            for seed, g in g0.groupby("seed"):
                dates = pd.Series(pd.DatetimeIndex(g["Date"].iloc[::24]))
                truth = g["actual"].to_numpy().reshape(-1, 24)
                preds = {m: g[c].to_numpy().reshape(-1, 24) for m, c in MODELS.items()}
                alpha = g["alpha"].to_numpy().reshape(-1, 24).mean(1)
                ctx = day_context(dates, raw, index)
                assert np.allclose(ctx["prev_std"], g["volatility_previous_day"].to_numpy()[::24])
                masks = slices(ctx, th)
                key = {"gate": gate, "origin": origin, "source": source, "seed": seed}
                overall.append({**key, **{f"{k}_{m}": fn(truth, p) for m, p in preds.items()
                                          for k, fn in (("MAE", V.mae), ("RMSE", V.rmse))}})
                s = slice_scores(truth, preds, masks)
                s["alpha_mean"] = [float(alpha[m].mean()) if m.any() else np.nan for _, m in masks.values()]
                for k, v in key.items():
                    s.insert(0, k, v)
                sliced.append(s)
                month = dates.dt.strftime("%Y-%m")
                alpha_rows.append(pd.DataFrame({**key, "date": dates, "month": month, "alpha": alpha,
                                                "prev_std": ctx["prev_std"]}))
                for m, p in preds.items():
                    day_loss[m].append(((p - truth) ** 2).mean(1))
                    day_abs[m].append(np.abs(p - truth).mean(1))
                d = pd.DataFrame({**key, "date": dates, "alpha": alpha, "prev_std": ctx["prev_std"]})
                for m, p in preds.items():
                    d[f"MAE_{m}"] = np.abs(p - truth).mean(1)
                    d[f"RMSE_{m}"] = np.sqrt(((p - truth) ** 2).mean(1))
                daily.append(d)
                if gate == "gate_vol":
                    ea, ec = (preds["A"] - truth).ravel(), (preds["C"] - truth).ravel()
                    corr.append({"origin": origin, "source": source, "seed": seed,
                                 "error_corr_A_C": float(np.corrcoef(ea, ec)[0, 1]),
                                 "daily_mean_error_corr": float(np.corrcoef((preds["A"] - truth).mean(1),
                                                                            (preds["C"] - truth).mean(1))[0, 1])})
            # seed 평균 일 손실 차이에 대한 주 단위 블록 부트스트랩
            for ref in ("Efixed", "E50", "A", "C"):
                for metric, store in (("MSE", day_loss), ("MAE", day_abs)):
                    mean, lo, hi = V.block_bootstrap_diff(np.mean(store["gate"], 0), np.mean(store[ref], 0))
                    boots.append({"gate": gate, "origin": origin, "source": source, "vs": ref, "metric": metric,
                                  "gate_minus_ref": mean, "ci_lo": lo, "ci_hi": hi,
                                  "verdict": "gate better" if hi < 0 else "gate worse" if lo > 0 else "no sig. diff"})
    overall = pd.DataFrame(overall)
    overall.to_csv(out / "overall_per_seed.csv", index=False)
    osum = mean_std(overall, ["gate", "origin", "source"])
    osum.to_csv(out / "overall_summary.csv", index=False)
    sliced = pd.concat(sliced, ignore_index=True)
    sliced.to_csv(out / "slices_per_seed.csv", index=False)
    ssum = mean_std(sliced, ["gate", "origin", "source", "slice", "kind"])
    ssum.to_csv(out / "slices_summary.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(out / "daily.csv", index=False)
    alpha = pd.concat(alpha_rows, ignore_index=True)
    alpha.groupby(["gate", "origin", "source", "month"])["alpha"].agg(["mean", "min", "max"]).reset_index().to_csv(
        out / "alpha_by_month.csv", index=False)
    sat = alpha.groupby(["gate", "origin", "source"])["alpha"].agg(
        mean="mean", std="std", min="min", max="max",
        frac_lt_005=lambda v: (v < 0.05).mean(), frac_gt_095=lambda v: (v > 0.95).mean()).reset_index()
    sat.to_csv(out / "alpha_saturation.csv", index=False)
    boots = pd.DataFrame(boots)
    boots.to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame(corr).to_csv(out / "error_correlation.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(th_cache, indent=2), encoding="utf-8")

    abl = pd.read_csv(run / "ablation.csv")
    asum = mean_std(abl.drop(columns=["features", "anchor", "out_dim"]), ["gate", "origin", "source"])
    asum.to_csv(out / "ablation_summary.csv", index=False)

    # ---- 그래프 ----
    for (origin, source), g in osum.groupby(["origin", "source"]):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for j, metric in enumerate(("MAE", "RMSE")):
            names = ["A", "C", "E50", "Efixed"] + list(g["gate"])
            vals = [g[f"{metric}_{m}_mean"].iloc[0] for m in ("A", "C", "E50", "Efixed")] + list(g[f"{metric}_gate_mean"])
            errs = [g[f"{metric}_{m}_std"].iloc[0] for m in ("A", "C", "E50", "Efixed")] + list(g[f"{metric}_gate_std"])
            ax[j].bar(names, vals, yerr=errs, color=["C0", "C1", "C7", "C7"] + ["C2"] * len(g))
            ax[j].set_ylim(min(vals) * 0.9, max(vals) * 1.03); ax[j].set_title(f"{origin} ({source}) {metric}")
            ax[j].tick_params(axis="x", labelsize=7)
        fig.tight_layout(); fig.savefig(out / f"overall_{origin}_{source}.png", dpi=130); plt.close(fig)
    for (origin, source), g in alpha.groupby(["origin", "source"]):
        gates = list(g["gate"].unique())
        fig, ax = plt.subplots(2, len(gates), figsize=(4.2 * len(gates), 6.5), squeeze=False)
        for j, gate in enumerate(gates):
            for seed, s in g[g["gate"] == gate].groupby("seed"):
                ax[0, j].scatter(s["prev_std"], s["alpha"], s=5, alpha=0.5, label=f"seed {seed}")
                ax[1, j].plot(s["date"], s["alpha"], lw=0.7)
            ax[0, j].set_xscale("log"); ax[0, j].set_ylim(-0.02, 1.02); ax[0, j].set_title(gate)
            ax[0, j].set_xlabel("previous-day price std"); ax[0, j].set_ylabel("alpha (weight on C)")
            ax[1, j].set_ylim(-0.02, 1.02); ax[1, j].tick_params(axis="x", labelsize=6, rotation=30)
        ax[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / f"alpha_{origin}_{source}.png", dpi=130); plt.close(fig)

    show = ["gate", "origin", "source"] + [f"{k}_{m}_mean" for k in ("MAE", "RMSE") for m in MODELS]
    print(md(osum[show]))
    print(md(boots[boots["vs"].isin(["Efixed"])]))
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnostic")
    d.add_argument("--abc", type=Path, default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    diagnostic(args) if args.cmd == "diagnostic" else evaluate(args)


if __name__ == "__main__":
    main()
