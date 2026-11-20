"""A·B·C 를 seed 평균한 뒤 결합하는 실험 (세션 내 점검용). 재학습 없이 저장된 OOF 예측만 쓴다.

비교 (결과를 보기 전에 고정)
    single : seed 하나짜리 모델의 RMSE 를 seed 10개에 대해 평균 (지금까지 보고한 방식)
    avg10  : 예측을 seed 10개에 대해 평균한 뒤 RMSE (A, B, C, E50, ABC)
    크기별 : seed n개(1, 2, 3, 5, 10) 평균. n < 10 은 무작위 부분집합 200개의 평균
판정: 일 MSE 차이의 주 단위 블록 부트스트랩 95% 구간.
    (1) avg10 ABC - avg10 E50      (2) avg10 E50 - single E50 (seed 평균의 효과)      (3) avg10 ABC - single E50
주의: 2017·2018 은 이미 여러 번 본 구간이다.

실행:  python exp_seed_average.py            (volatility-gate/ 안에서)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import vgate_lib as V
import train_volatility_gate as G
import exp_add_b as XB

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(42, 52))
COMBOS = {k: np.array(w, float) for k, w in XB.EQUAL.items() if k in ("A", "B", "C", "E50", "ABC")}


def main():
    oof = ROOT / "results" / "oof_ac" / "main"
    order = sorted(d.name for d in oof.glob("o*") if d.is_dir())
    out = ROOT / "results" / "vgate_fix"
    frame_raw = V.read_frame()
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    rows, boots, sizes = [], [], []
    rng = np.random.default_rng(0)
    for origin in order:
        blocks = [XB.load(oof / origin / f"seed_{s}", raw, index) for s in SEEDS]
        assert all(np.array_equal(b["starts"], blocks[0]["starts"]) for b in blocks)
        truth = blocks[0]["truth"]
        P = np.stack([b["P"] for b in blocks])                       # (seeds, days, 24, 3)
        day = lambda pred: ((pred - truth) ** 2).mean(1)
        single_day, avg_day = {}, {}
        for k, w in COMBOS.items():
            single = P @ w                                           # (seeds, days, 24)
            avg = single.mean(0)
            single_day[k] = np.mean([day(p) for p in single], 0)     # seed 평균 일 손실
            avg_day[k] = day(avg)
            rows.append({"origin": origin, "model": k,
                         "single_RMSE": float(np.mean([V.rmse(truth, p) for p in single])),
                         "single_std": float(np.std([V.rmse(truth, p) for p in single], ddof=1)),
                         "avg10_RMSE": V.rmse(truth, avg),
                         "single_MAE": float(np.mean([V.mae(truth, p) for p in single])), "avg10_MAE": V.mae(truth, avg)})
            for n in (1, 2, 3, 5, 10):
                picks = [rng.choice(10, n, replace=False) for _ in range(200)] if n < 10 else [np.arange(10)]
                r = [V.rmse(truth, single[idx].mean(0)) for idx in picks]
                sizes.append({"origin": origin, "model": k, "n_seeds": n, "RMSE": float(np.mean(r)), "RMSE_std": float(np.std(r))})
        for label, a, b in (("avg10 ABC - avg10 E50", avg_day["ABC"], avg_day["E50"]),
                            ("avg10 E50 - single E50", avg_day["E50"], single_day["E50"]),
                            ("avg10 ABC - single ABC", avg_day["ABC"], single_day["ABC"]),
                            ("avg10 ABC - single E50", avg_day["ABC"], single_day["E50"]),
                            ("avg10 C - avg10 A", avg_day["C"], avg_day["A"])):
            mean, lo, hi = V.block_bootstrap_diff(a, b)
            boots.append({"origin": origin, "comparison": label, "MSE_diff": mean, "ci_lo": lo, "ci_hi": hi,
                          "verdict": "better" if hi < 0 else "worse" if lo > 0 else "no sig. diff"})
    rows, boots, sizes = pd.DataFrame(rows), pd.DataFrame(boots), pd.DataFrame(sizes)
    rows["gain"] = rows["avg10_RMSE"] - rows["single_RMSE"]
    rows.to_csv(out / "seed_avg_summary.csv", index=False)
    boots.to_csv(out / "seed_avg_bootstrap.csv", index=False)
    sizes.to_csv(out / "seed_avg_sizes.csv", index=False)
    pd.set_option("display.width", 250)
    for col in ("single_RMSE", "avg10_RMSE", "gain", "single_MAE", "avg10_MAE"):
        print(f"\n{col}")
        print(rows.pivot(index="model", columns="origin", values=col).loc[list(COMBOS)].round(3).to_string())
    print("\nbootstrap")
    print(boots.round(3).to_string(index=False))
    print("\nRMSE by number of averaged seeds")
    print(sizes[sizes.model.isin(["E50", "ABC", "C"])].pivot(index=["model", "n_seeds"], columns="origin", values="RMSE").round(3).to_string())


if __name__ == "__main__":
    main()
