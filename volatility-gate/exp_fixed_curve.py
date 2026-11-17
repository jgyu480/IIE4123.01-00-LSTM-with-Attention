"""고정 상승 곡선 검증 (세션 내 점검용). 곡선·평가 방법·판정 기준을 새 seed 결과가 나오기 전에 고정한다.

규칙 (학습 없음)
    alpha(D) = seed 42 로 초기화한 학습 전 GateMLP(1, 8, 1) 의 출력
    입력 z = (slog(전날 24시간 가격 표준편차) - 2.035498726243355) / 0.4240097685083492
    대략 표준편차 2 -> 0.28, 8 -> 0.46, 20 -> 0.71, 50 -> 0.88 (변동성이 클수록 C 비중이 커진다)

이 곡선의 출처와 이미 본 것
    - 기존 gate_vol(o2018, seed 42)이 학습되지 않은 채 채택됐던 무작위 초기값이다. 2018 결과를 본 뒤에 골랐다.
    - A·C seed 42~44 로 2015~2018 에 적용한 결과도 이미 봤다 (2016·2018 저장분에서 유의하게 좋음, 나머지 차이 없음).
    - 그 이득이 거의 seed 42 의 A·C 에서만 나왔다.

이번에 확인하는 것
    A·C seed 45~51 (이 스크립트를 쓴 시점에는 아직 만들지 않음) 에서도 같은 결과가 나오는가.
    연도(2015~2018) 자체는 이미 본 구간이므로, 이 검증은 "A·C seed 에 대한 강건성"만 답한다.

판정 (새 seed 만 사용, 연도별 seed 평균 일 MSE 차이 = 곡선 - E50, 주 단위 블록 부트스트랩 95% 구간)
    지지      : 유의하게 나쁜 연도가 없고, 2018 을 포함해 2개 연도 이상에서 유의하게 좋다
    기각      : 유의하게 나쁜 연도가 있거나, 유의하게 좋은 연도가 하나도 없다
    판단 보류 : 그 밖의 경우
보조 지표 (판정에는 쓰지 않음): seed 별 RMSE 차이의 부호, 변동성 구간별 차이, 곡선을 2-매개변수 로지스틱으로
    근사한 보조 곡선 (예측 결과가 아니라 곡선 모양에만 맞춘다).

실행:  python exp_fixed_curve.py --seeds 45 46 47 48 49 50 51      (volatility-gate/ 안에서)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import vgate_lib as V
import analyze_volatility_gate as AN
import train_volatility_gate as G

ROOT = Path(__file__).resolve().parent
Z_MEAN, Z_STD = 2.035498726243355, 0.4240097685083492       # 2015~2017 목표일의 slog(전날 표준편차) 평균·표준편차
CURVE_SEED = 42


def slog(v):
    v = np.asarray(v, dtype=np.float64)
    return np.sign(v) * np.log1p(np.abs(v))


def primary_curve():
    G.seed_all(CURVE_SEED)
    model = V.GateMLP(1, 8, 1)
    model.eval()

    def alpha(vol):
        z = torch.tensor(((slog(vol) - Z_MEAN) / Z_STD).reshape(-1, 1)).float()
        with torch.no_grad():
            return model.alpha(z).double().numpy().ravel()
    return alpha


def logistic_approx(curve):
    """보조 곡선: logit(alpha) 를 z 의 1차식으로 근사. 곡선 모양에만 맞춘다 (가격·예측을 쓰지 않는다)."""
    vol = np.exp(np.linspace(np.log(1.0), np.log(50.0), 200))
    z = (slog(vol) - Z_MEAN) / Z_STD
    a = curve(vol)
    b, a0 = np.polyfit(z, np.log(a / (1 - a)), 1)
    return (lambda v: 1 / (1 + np.exp(-(a0 + b * (slog(v) - Z_MEAN) / Z_STD)))), float(a0), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--oof-tag", default="main")
    p.add_argument("--label", default="new")
    args = p.parse_args()
    oof = ROOT / "results" / "oof_ac" / args.oof_tag
    order = sorted(d.name for d in oof.glob("o*") if d.is_dir())
    out = ROOT / "results" / "vgate_fix"
    out.mkdir(parents=True, exist_ok=True)
    frame_raw = V.read_frame()
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    main_curve = primary_curve()
    approx, a0, b = logistic_approx(main_curve)
    grid = np.array([2, 5, 8, 12, 20, 30, 50.0])
    print("curve    :", dict(zip(grid.astype(int).tolist(), np.round(main_curve(grid), 2).tolist())))
    print(f"logistic : a={a0:+.3f} b={b:+.3f}", dict(zip(grid.astype(int).tolist(), np.round(approx(grid), 2).tolist())))
    curves = {"fixed_curve": main_curve, "logistic_approx": approx}

    per_seed, summary, slices = [], [], []
    for origin in order:
        th = AN.thresholds(origin)
        blocks = {s: G.load_oof(oof / origin / f"seed_{s}", raw, index) for s in args.seeds}
        vol = V.gate_features(raw, blocks[args.seeds[0]]["starts"])["vol_std"].to_numpy()
        masks = {"all": np.ones(len(vol), bool), "vol_low(<=q50)": vol <= th["vol_q50"],
                 "vol_mid(q50-q90)": (vol > th["vol_q50"]) & (vol <= th["vol_q90"]), "vol_high(>q90)": vol > th["vol_q90"]}
        for cname, curve in curves.items():
            alpha = curve(vol).reshape(-1, 1)
            assert alpha.min() >= 0 and alpha.max() <= 1
            day_g, day_e = [], []
            for s, b_ in blocks.items():
                assert np.array_equal(b_["starts"], blocks[args.seeds[0]]["starts"])
                gate, e50 = V.blend(alpha, b_["A"], b_["C"]), V.blend(0.5, b_["A"], b_["C"])
                day_g.append(((gate - b_["truth"]) ** 2).mean(1))
                day_e.append(((e50 - b_["truth"]) ** 2).mean(1))
                per_seed.append({"curve": cname, "origin": origin, "seed": s,
                                 "RMSE_A": V.rmse(b_["truth"], b_["A"]), "RMSE_C": V.rmse(b_["truth"], b_["C"]),
                                 "RMSE_E50": V.rmse(b_["truth"], e50), "RMSE_gate": V.rmse(b_["truth"], gate),
                                 "MAE_E50": V.mae(b_["truth"], e50), "MAE_gate": V.mae(b_["truth"], gate)})
                for name, m in masks.items():
                    slices.append({"curve": cname, "origin": origin, "seed": s, "slice": name, "days": int(m.sum()),
                                   "RMSE_A": V.rmse(b_["truth"][m], b_["A"][m]), "RMSE_C": V.rmse(b_["truth"][m], b_["C"][m]),
                                   "dRMSE": V.rmse(b_["truth"][m], gate[m]) - V.rmse(b_["truth"][m], e50[m]),
                                   "alpha_mean": float(alpha[m].mean())})
            mean, lo, hi = V.block_bootstrap_diff(np.mean(day_g, 0), np.mean(day_e, 0))
            summary.append({"curve": cname, "origin": origin, "MSE_diff": mean, "ci_lo": lo, "ci_hi": hi,
                            "verdict": "better" if hi < 0 else "worse" if lo > 0 else "no sig. diff"})
    per_seed = pd.DataFrame(per_seed)
    per_seed["dRMSE"] = per_seed["RMSE_gate"] - per_seed["RMSE_E50"]
    per_seed["dMAE"] = per_seed["MAE_gate"] - per_seed["MAE_E50"]
    agg = per_seed.groupby(["curve", "origin"], sort=False).agg(
        seeds=("seed", "size"), RMSE_A=("RMSE_A", "mean"), RMSE_C=("RMSE_C", "mean"), RMSE_E50=("RMSE_E50", "mean"),
        RMSE_gate=("RMSE_gate", "mean"), dRMSE_mean=("dRMSE", "mean"), dRMSE_std=("dRMSE", "std"),
        dRMSE_min=("dRMSE", "min"), dRMSE_max=("dRMSE", "max"), dMAE_mean=("dMAE", "mean"),
        seeds_better=("dRMSE", lambda v: int((v < 0).sum()))).reset_index()
    summary = agg.merge(pd.DataFrame(summary), on=["curve", "origin"])
    slices = pd.DataFrame(slices).groupby(["curve", "origin", "slice", "days"], sort=False)[
        ["RMSE_A", "RMSE_C", "dRMSE", "alpha_mean"]].mean().reset_index()
    per_seed.to_csv(out / f"fixed_curve_{args.label}_per_seed.csv", index=False)
    summary.to_csv(out / f"fixed_curve_{args.label}_summary.csv", index=False)
    slices.to_csv(out / f"fixed_curve_{args.label}_slices.csv", index=False)

    main_rows = summary[summary.curve == "fixed_curve"].set_index("origin")
    better = [o for o in main_rows.index if main_rows.loc[o, "verdict"] == "better"]
    worse = [o for o in main_rows.index if main_rows.loc[o, "verdict"] == "worse"]
    if worse or not better:
        decision = "기각"
    elif "o2018" in better and len(better) >= 2:
        decision = "지지"
    else:
        decision = "판단 보류"
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(f"\n판정 ({args.label}, seeds {args.seeds}): {decision}  | 유의하게 좋음 {better} | 유의하게 나쁨 {worse}")


if __name__ == "__main__":
    main()
