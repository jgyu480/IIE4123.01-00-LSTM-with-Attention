"""B(유사일 평균)를 결합에 추가하는 실험 (세션 내 점검용). 재학습 없이 저장된 OOF 예측의 A·B·C 열만 쓴다.

비교 대상 (결과를 보기 전에 고정)
    단일      : A, B, C
    단순 평균 : E50 = (A+C)/2  [기준],  AB = (A+B)/2,  BC = (B+C)/2,  ABC = (A+B+C)/3
    고정 가중 : W3 = 평가 연도보다 앞선 모든 OOF 연도에서 MSE 를 최소화하는 비음수·합 1 가중치 (seed 별)
    게이트    : G3 = softmax(a + b * z) 로 세 모델의 비중을 정한다. z = 표준화한 전날 변동성, a = b = 0 에서 출발
                (학습 전에는 정확히 ABC). 학습·early stopping·refit 규칙은 기존 게이트와 같다.

판정: 연도별 seed 평균 일 MSE 차이(후보 - E50)의 주 단위 블록 부트스트랩 95% 구간. 주 관심은 ABC vs E50.
주의: 2017·2018 은 이미 여러 번 본 구간이다. B 의 k 는 각 origin 의 검증 연도로 고른 값이라 평가 블록은 쓰지 않았다.

실행:  python exp_add_b.py            (volatility-gate/ 안에서)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import vgate_lib as V
import train_volatility_gate as G

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(42, 52))
CFG = {"epochs": 100, "patience": 10, "batch_size": 32, "lr": 0.001}
EQUAL = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1], "E50": [.5, 0, .5], "AB": [.5, .5, 0], "BC": [0, .5, .5],
         "ABC": [1 / 3, 1 / 3, 1 / 3]}
SIMPLEX = np.array([[i, j, 50 - i - j] for i in range(51) for j in range(51 - i)]) / 50.0     # 0.02 간격


def load(folder, raw, index):
    f = pd.read_csv(folder / "predictions.csv", parse_dates=["Date"])
    b = G.check_frame(f, raw, index, folder)
    assert np.isfinite(f["B"].to_numpy()).all()
    b["P"] = np.stack([b["A"], f["B"].to_numpy().reshape(-1, 24), b["C"]], 2)          # (days, 24, 3)
    b["x"] = V.transform(V.gate_features(raw, b["starts"]), ["vol_std"])
    return b


def best_weights(blocks):
    err = np.concatenate([(b["P"] - b["truth"][:, :, None]).reshape(-1, 3) for b in blocks])
    gram = err.T @ err
    return SIMPLEX[int(np.einsum("ni,ij,nj->n", SIMPLEX, gram, SIMPLEX).argmin())]


class Soft3(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3))
        self.b = nn.Parameter(torch.zeros(3))

    def weights(self, z):
        return torch.softmax(self.a + self.b * z, dim=1)                                 # (days, 3)

    def forward(self, z, p):
        return (p * self.weights(z)[:, None, :]).sum(2)


def train(z, p, y, seed, val=None, fixed_epochs=None):
    G.seed_all(seed)
    model = Soft3()
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    gen = torch.Generator().manual_seed(seed)
    loss_of = lambda zz, pp, yy: nn.functional.mse_loss(model(zz, pp), yy).item()
    copy = lambda: {k: v.clone() for k, v in model.state_dict().items()}
    best, best_epoch, state, stale = float("inf"), 0, copy(), 0
    if val is not None:
        with torch.no_grad():
            best = loss_of(*val)
    for epoch in range(1, (CFG["epochs"] if fixed_epochs is None else fixed_epochs) + 1):
        order = torch.randperm(len(z), generator=gen)
        for i in range(0, len(z), CFG["batch_size"]):
            idx = order[i:i + CFG["batch_size"]]
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(z[idx], p[idx]), y[idx]).backward()
            opt.step()
        if val is not None:
            with torch.no_grad():
                v = loss_of(*val)
            if v < best:
                best, best_epoch, state, stale = v, epoch, copy(), 0
            else:
                stale += 1
            if stale >= CFG["patience"]:
                break
    if val is not None:
        model.load_state_dict(state)
    else:
        best_epoch = fixed_epochs
    return model, best_epoch


def fit_gate(tr, va, seed, mu, sd):
    """평가 블록을 받지 않는다."""
    t = lambda a: torch.from_numpy(a).float()
    sc = lambda v: t((v - mu) / sd)
    cat = lambda bs, k: np.concatenate([b[k] for b in bs])
    m1, s1 = V.fit_scaler(cat(tr, "x"))
    _, best_epoch = train(t((cat(tr, "x") - m1) / s1), sc(cat(tr, "P")), sc(cat(tr, "truth")), seed,
                          val=(t((va["x"] - m1) / s1), sc(va["P"]), sc(va["truth"])))
    fit = tr + [va]
    m2, s2 = V.fit_scaler(cat(fit, "x"))
    model, _ = train(t((cat(fit, "x") - m2) / s2), sc(cat(fit, "P")), sc(cat(fit, "truth")), seed, fixed_epochs=best_epoch)
    return (lambda x: model.weights(t((x - m2) / s2)).detach().double().numpy()), best_epoch


def main():
    oof = ROOT / "results" / "oof_ac" / "main"
    order = sorted(d.name for d in oof.glob("o*") if d.is_dir())
    out = ROOT / "results" / "vgate_fix"
    frame_raw = V.read_frame()
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    B = {(o, s): load(oof / o / f"seed_{s}", raw, index) for o in order for s in SEEDS}
    rows, boots, wrows, corr = [], [], [], []
    for origin in order:
        earlier = order[:order.index(origin)]
        day = {}
        for s in SEEDS:
            b = B[(origin, s)]
            assert all(B[(e, s)]["starts"].max() < b["starts"].min() for e in earlier)
            preds = {k: b["P"] @ np.array(w, float) for k, w in EQUAL.items()}
            if earlier:
                w3 = best_weights([B[(e, s)] for e in earlier])
                preds["W3"] = b["P"] @ w3
                wrows.append({"origin": origin, "seed": s, "kind": "W3", "wA": w3[0], "wB": w3[1], "wC": w3[2], "best_epoch": np.nan})
            if len(earlier) >= 2:
                meta = json.loads((oof / origin / f"seed_{s}" / "metrics.json").read_text(encoding="utf-8"))
                wf, ep = fit_gate([B[(e, s)] for e in earlier[:-1]], B[(earlier[-1], s)], s,
                                  meta["scaling"]["mean"][0], meta["scaling"]["std"][0])
                w = wf(b["x"])
                assert np.allclose(w.sum(1), 1) and w.min() >= 0
                preds["G3"] = (b["P"] * w[:, None, :]).sum(2)
                wrows.append({"origin": origin, "seed": s, "kind": "G3", "wA": w[:, 0].mean(), "wB": w[:, 1].mean(),
                              "wC": w[:, 2].mean(), "best_epoch": ep})
            for k, p in preds.items():
                rows.append({"origin": origin, "seed": s, "model": k, "RMSE": V.rmse(b["truth"], p), "MAE": V.mae(b["truth"], p)})
                day.setdefault(k, {"MSE": [], "MAE": []})
                day[k]["MSE"].append(((p - b["truth"]) ** 2).mean(1))
                day[k]["MAE"].append(np.abs(p - b["truth"]).mean(1))
            e = (b["P"] - b["truth"][:, :, None]).reshape(-1, 3)
            c = np.corrcoef(e.T)
            corr.append({"origin": origin, "seed": s, "A-B": c[0, 1], "A-C": c[0, 2], "B-C": c[1, 2]})
        for k in day:
            if k == "E50":
                continue
            for metric in ("MSE", "MAE"):
                mean, lo, hi = V.block_bootstrap_diff(np.mean(day[k][metric], 0), np.mean(day["E50"][metric], 0))
                boots.append({"origin": origin, "model": k, "metric": metric, "minus_E50": mean, "ci_lo": lo, "ci_hi": hi,
                              "verdict": "better" if hi < 0 else "worse" if lo > 0 else "no sig. diff"})
    rows = pd.DataFrame(rows)
    e50 = rows[rows.model == "E50"].set_index(["origin", "seed"])["RMSE"]
    rows["dRMSE_vs_E50"] = rows["RMSE"] - e50.reindex(pd.MultiIndex.from_frame(rows[["origin", "seed"]])).to_numpy()
    summary = rows.groupby(["origin", "model"], sort=False).agg(
        RMSE=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE=("MAE", "mean"), dRMSE=("dRMSE_vs_E50", "mean"),
        seeds_better=("dRMSE_vs_E50", lambda v: int((v < 0).sum())), worst=("dRMSE_vs_E50", "max")).reset_index()
    rows.to_csv(out / "add_b_per_seed.csv", index=False)
    summary.to_csv(out / "add_b_summary.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "add_b_bootstrap.csv", index=False)
    pd.DataFrame(wrows).to_csv(out / "add_b_weights.csv", index=False)
    pd.DataFrame(corr).to_csv(out / "add_b_error_corr.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.pivot(index="model", columns="origin", values="RMSE").loc[list(EQUAL) + ["W3", "G3"]].round(3).to_string())
    print("\ndRMSE vs E50 / seeds better (of 10)")
    print(summary.pivot(index="model", columns="origin", values="dRMSE").loc[list(EQUAL) + ["W3", "G3"]].round(3).to_string())
    print(summary.pivot(index="model", columns="origin", values="seeds_better").loc[list(EQUAL) + ["W3", "G3"]].to_string())
    print(summary.pivot(index="model", columns="origin", values="RMSE_std").loc[list(EQUAL) + ["W3", "G3"]].round(3).to_string())
    print("\nbootstrap vs E50")
    print(pd.DataFrame(boots).round(3).to_string(index=False))
    w = pd.DataFrame(wrows)
    print("\nweights (mean, std over seeds)")
    print(w.groupby(["origin", "kind"])[["wA", "wB", "wC", "best_epoch"]].agg(["mean", "std"]).round(2).to_string())
    print("\nerror correlation (mean over seeds)")
    print(pd.DataFrame(corr).groupby("origin")[["A-B", "A-C", "B-C"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
