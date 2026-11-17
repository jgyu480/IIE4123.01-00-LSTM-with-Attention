"""보완 실험 (세션 내 점검용): 게이트 절차의 허점 세 가지를 고쳐 다시 평가한다.

  fix 1  E50 되돌리기 : early stopping 이 한 번도 개선을 못 보면(best_epoch == 0) alpha 를 0.5 로 고정한다.
                        기존 절차에서는 이때 "학습 안 된 무작위 초기 게이트"가 최종 모델이 됐다.
  fix 2  0.5 anchor   : 마지막 층을 0 으로 두고 alpha = 0.5 (= E50) 에서 출발한다. 기존 gate_anchor 는
                        seed 마다 흔들리는 Efixed 에서 출발했다.
  fix 3  2-매개변수   : alpha = sigmoid(a + b * z),  z = 표준화한 전날 변동성.  a = b = 0 에서 출발한다.

A·C 예측(OOF), 분할, 표준화, 손실, optimizer, early stopping, refit 규칙은 train_volatility_gate.py 와 같다.
게이트 학습과 "되돌리기" 판단에는 gate_train / gate_val 블록만 쓰고, 평가 블록은 예측에만 쓴다.

주의: 이 수정들은 2018·2017 평가 결과를 본 뒤에 정한 것이다. 같은 평가 블록에서 다시 잰 수치는
      독립적인 일반화 증거가 아니다.

실행:  python exp_gate_fixes.py            (volatility-gate/ 안에서)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import vgate_lib as V
import analyze_volatility_gate as AN
import train_volatility_gate as G

ROOT = Path(__file__).resolve().parent
U = V.USER_FEATURES

# kind: mlp(25~) / logit2(2).  anchor: None(무작위 초기화) / "efixed" / 0.5.  fallback: best_epoch==0 이면 alpha=0.5
GATES = {
    # --- 기존 3종 (재현, 비교 기준) ---
    "gate_vol":    {"kind": "mlp", "features": ["vol_std"], "anchor": None, "fallback": False},
    "gate_struct": {"kind": "mlp", "features": U, "anchor": None, "fallback": False},
    "gate_anchor": {"kind": "mlp", "features": U, "anchor": "efixed", "fallback": False},
    # --- fix 1: E50 되돌리기 ---
    "vol_fb":      {"kind": "mlp", "features": ["vol_std"], "anchor": None, "fallback": True},
    "struct_fb":   {"kind": "mlp", "features": U, "anchor": None, "fallback": True},
    # --- fix 2: 0.5 anchor ---
    "vol_a50":     {"kind": "mlp", "features": ["vol_std"], "anchor": 0.5, "fallback": False},
    "struct_a50":  {"kind": "mlp", "features": U, "anchor": 0.5, "fallback": False},
    # --- fix 3: 2-매개변수 (0.5 에서 출발하므로 fix 1·2 를 함께 만족) ---
    "logit2":      {"kind": "logit2", "features": ["vol_std"], "anchor": 0.5, "fallback": False},
}
REFERENCE = ["gate_vol", "gate_struct", "gate_anchor"]
SPREAD_GATES = ["gate_vol", "vol_fb", "vol_a50", "logit2"]
PAIRS = [("vol_fb", "gate_vol"), ("vol_a50", "gate_vol"), ("logit2", "gate_vol"),
         ("struct_fb", "gate_struct"), ("struct_a50", "gate_struct"), ("struct_a50", "gate_anchor")]


class Logit2(nn.Module):
    """alpha = sigmoid(a + b * z). a = b = 0 으로 시작하므로 학습 전에는 정확히 E50 이다."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def alpha(self, x):
        return torch.sigmoid(self.a + self.b * x[:, :1])

    def forward(self, x, pred_a, pred_c):
        a = self.alpha(x)
        return V.blend(a, pred_a, pred_c), a


def make_model(spec, in_dim, cfg, efixed):
    if spec["kind"] == "logit2":
        return Logit2()
    anchor = efixed if spec["anchor"] == "efixed" else spec["anchor"]
    return V.GateMLP(in_dim, cfg["hidden"], 1, None if anchor is None else G.logit(anchor))


def train_any(factory, x, pa, pc, y, seed, cfg, val=None, fixed_epochs=None):
    """G.train_gate 와 같은 루프. 모델 생성만 factory 로 바꿨다."""
    G.seed_all(seed)
    model = factory(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(seed)

    def loss_of(xx, aa, cc, yy):
        with torch.no_grad():
            return nn.functional.mse_loss(model(xx, aa, cc)[0], yy).item()

    copy = lambda: {k: v.clone() for k, v in model.state_dict().items()}
    best, best_epoch, best_state, stale = float("inf"), 0, copy(), 0
    if val is not None:
        best = loss_of(*val)
    epochs = cfg["epochs"] if fixed_epochs is None else fixed_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for i in range(0, len(x), cfg["batch_size"]):
            b = order[i:i + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[b], pa[b], pc[b])[0], y[b])
            loss.backward()
            optimizer.step()
        if val is not None:
            v = loss_of(*val)
            if v < best:
                best, best_epoch, best_state, stale = v, epoch, copy(), 0
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
    if val is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch, best


def prepare(test_name, order, seed, frame_raw, oof_dir, external):
    """G.run_origin 의 데이터 준비와 같다. 반환값에서 학습용(train/val/fit)과 평가용(test*)을 분리한다."""
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    blocks = {n: G.load_oof(oof_dir / n / f"seed_{seed}", raw, index) for n in order}
    earlier = order[:order.index(test_name)]
    if len(earlier) < 2:
        raise SystemExit(f"{test_name}: 앞선 OOF 블록이 2개 이상 필요합니다")
    keys = ("starts", "truth", "A", "C")
    tr = {k: G.stack([blocks[n] for n in earlier[:-1]], k) for k in keys}
    va = {k: blocks[earlier[-1]][k] for k in keys}
    tests = {"retrained": {k: blocks[test_name][k] for k in keys}}
    if external is not None:
        tests["saved_abc"] = {k: external[k] for k in keys}
    assert tr["starts"].max() < va["starts"].min()
    assert va["starts"].max() + 24 <= min(t["starts"].min() for t in tests.values())
    meta = json.loads((oof_dir / test_name / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
    fit = {k: np.concatenate([tr[k], va[k]]) for k in keys}
    learn = {"tr": tr, "va": va, "fit": fit,
             "f_tr": V.gate_features(raw, tr["starts"]), "f_va": V.gate_features(raw, va["starts"]),
             "mu": meta["scaling"]["mean"][0], "sd": meta["scaling"]["std"][0],
             "efixed_tr": V.fixed_alpha(tr["A"], tr["C"], tr["truth"]),
             "efixed": V.fixed_alpha(fit["A"], fit["C"], fit["truth"])}
    learn["f_fit"] = pd.concat([learn["f_tr"], learn["f_va"]], ignore_index=True)
    for t in tests.values():
        t["feats"] = V.gate_features(raw, t["starts"])
        t["dates"] = pd.Series(index[t["starts"]])
    return learn, tests


def fit_gate(spec, learn, gate_seed, cfg):
    """평가 블록을 받지 않는다: 학습·early stopping·되돌리기 판단은 gate_train / gate_val 만으로 이루어진다."""
    mu, sd = learn["mu"], learn["sd"]
    sc = lambda v: torch.from_numpy((v - mu) / sd).float()
    tx = lambda a, m, s: torch.from_numpy((a - m) / s).float()
    names = spec["features"]
    tr, va, fit = learn["tr"], learn["va"], learn["fit"]
    x_tr, x_va = V.transform(learn["f_tr"], names), V.transform(learn["f_va"], names)
    m1, s1 = V.fit_scaler(x_tr)
    val = (tx(x_va, m1, s1), sc(va["A"]), sc(va["C"]), sc(va["truth"]))
    _, best_epoch, best_val = train_any(lambda d: make_model(spec, d, cfg, learn["efixed_tr"]),
                                        tx(x_tr, m1, s1), sc(tr["A"]), sc(tr["C"]), sc(tr["truth"]),
                                        gate_seed, cfg, val=val)
    val_e50 = nn.functional.mse_loss(V.blend(0.5, val[1], val[2]), val[3]).item()
    info = {"best_epoch": best_epoch, "val_MSE_gate": best_val, "val_MSE_E50": val_e50,
            "reverted_to_E50": bool(spec["fallback"] and best_epoch == 0)}
    if info["reverted_to_E50"]:
        return (lambda feats: np.full((len(feats), 1), 0.5)), info
    x_fit = V.transform(learn["f_fit"], names)
    m2, s2 = V.fit_scaler(x_fit)
    model, _, _ = train_any(lambda d: make_model(spec, d, cfg, learn["efixed"]),
                            tx(x_fit, m2, s2), sc(fit["A"]), sc(fit["C"]), sc(fit["truth"]),
                            gate_seed, cfg, val=None, fixed_epochs=best_epoch)
    if spec["kind"] == "logit2":
        info["a"], info["b"] = model.a.item(), model.b.item()

    def alpha_of(feats):
        with torch.no_grad():
            return model.alpha(tx(V.transform(feats, names), m2, s2)).double().numpy()
    return alpha_of, info


def score(test, alpha):
    pred = V.blend(np.broadcast_to(alpha, test["A"].shape), test["A"], test["C"])
    return pred, {"MAE": V.mae(test["truth"], pred), "RMSE": V.rmse(test["truth"], pred)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-tag", default="main")
    p.add_argument("--test-origins", nargs="+", default=["o2018", "o2017"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--gate-seeds", type=int, default=20, help="같은 A·C 예측에 게이트 초기화만 바꿔 반복하는 횟수")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "vgate_fix")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = {"hidden": 8, "epochs": 100, "patience": 10, "batch_size": 32, "lr": 0.001}   # 기존과 동일
    oof_dir = ROOT / "results" / "oof_ac" / args.oof_tag
    order = sorted(d.name for d in oof_dir.glob("o*") if d.is_dir())
    args.out.mkdir(parents=True, exist_ok=True)
    frame_raw = V.read_frame()
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    abc = sorted(d for d in (V.BASE / "results" / "abc").glob("*") if d.is_dir())[-1]

    runs, spread, daily = [], [], {}
    for origin in args.test_origins:
        th = AN.thresholds(origin)
        for seed in args.seeds:
            external = None
            if origin == "o2018":
                ext = pd.read_csv(abc / "PJM" / f"seed_{seed}" / "predictions.csv", parse_dates=["Date"])
                external = G.check_frame(ext, raw, index, "saved abc")
            learn, tests = prepare(origin, order, seed, frame_raw, oof_dir, external)
            for gname, spec in GATES.items():
                alpha_of, info = fit_gate(spec, learn, seed, cfg)          # 게이트 seed = base seed (기존과 동일)
                for source, test in tests.items():
                    alpha = alpha_of(test["feats"])
                    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
                    pred, s = score(test, alpha)
                    e50, s50 = score(test, 0.5)
                    runs.append({"gate": gname, "origin": origin, "source": source, "seed": seed, **info,
                                 "alpha_mean": float(alpha.mean()), "alpha_min": float(alpha.min()),
                                 "alpha_max": float(alpha.max()),
                                 "MAE_A": V.mae(test["truth"], test["A"]), "RMSE_A": V.rmse(test["truth"], test["A"]),
                                 "MAE_C": V.mae(test["truth"], test["C"]), "RMSE_C": V.rmse(test["truth"], test["C"]),
                                 "MAE_E50": s50["MAE"], "RMSE_E50": s50["RMSE"],
                                 "MAE_gate": s["MAE"], "RMSE_gate": s["RMSE"]})
                    ctx = AN.day_context(test["dates"], raw, index)
                    daily.setdefault((gname, origin, source), []).append({
                        "truth": test["truth"], "gate": pred, "E50": e50, "A": test["A"], "C": test["C"],
                        "alpha": alpha.ravel(), "masks": AN.slices(ctx, th)})
            # 게이트 초기화만 바꿔 반복: "초기화의 우연"이 결과를 얼마나 흔드는지
            for gname in SPREAD_GATES:
                for gs in range(args.gate_seeds):
                    alpha_of, info = fit_gate(GATES[gname], learn, 1000 + gs, cfg)
                    for source, test in tests.items():
                        _, s = score(test, alpha_of(test["feats"]))
                        _, s50 = score(test, 0.5)
                        spread.append({"gate": gname, "origin": origin, "source": source, "seed": seed,
                                       "gate_seed": 1000 + gs, "best_epoch": info["best_epoch"],
                                       "reverted_to_E50": info["reverted_to_E50"],
                                       "dRMSE": s["RMSE"] - s50["RMSE"], "dMAE": s["MAE"] - s50["MAE"]})
            print(f"done {origin} seed={seed}", flush=True)

    runs = pd.DataFrame(runs)
    runs["dRMSE_vs_E50"] = runs["RMSE_gate"] - runs["RMSE_E50"]
    runs["dMAE_vs_E50"] = runs["MAE_gate"] - runs["MAE_E50"]
    runs.to_csv(args.out / "runs.csv", index=False)

    # 기존 결과 재현 확인 (같은 루프임을 수치로 확인)
    old = pd.read_csv(ROOT / "results" / "vgate" / "main" / "ablation.csv")
    chk = runs[runs.gate.isin(REFERENCE)].merge(old, on=["gate", "origin", "source", "seed"], suffixes=("", "_old"))
    assert len(chk) == len(runs[runs.gate.isin(REFERENCE)]) and np.allclose(chk.RMSE_gate, chk.RMSE_gate_old, atol=1e-9)
    assert (chk.best_epoch == chk.best_epoch_old).all()
    # 0.5 에서 출발하는 게이트는 학습이 거부되면 정확히 E50 이어야 한다
    z = runs[(runs.best_epoch == 0) & runs.gate.isin(["vol_fb", "struct_fb", "vol_a50", "struct_a50", "logit2"])]
    assert np.allclose(z.RMSE_gate, z.RMSE_E50, atol=1e-12) and np.allclose(z.alpha_min, 0.5) and np.allclose(z.alpha_max, 0.5)
    print("checks passed: reference gates reproduce saved results; untrained fixed gates equal E50", flush=True)

    summary = runs.groupby(["origin", "source", "gate"], sort=False).agg(
        RMSE_A=("RMSE_A", "mean"), RMSE_C=("RMSE_C", "mean"), RMSE_E50=("RMSE_E50", "mean"),
        RMSE_gate=("RMSE_gate", "mean"), MAE_E50=("MAE_E50", "mean"), MAE_gate=("MAE_gate", "mean"),
        dRMSE_mean=("dRMSE_vs_E50", "mean"), dRMSE_std=("dRMSE_vs_E50", "std"),
        dRMSE_worst=("dRMSE_vs_E50", "max"), dMAE_mean=("dMAE_vs_E50", "mean"),
        trained_runs=("best_epoch", lambda v: int((v > 0).sum())), n=("seed", "size"),
        alpha_min=("alpha_min", "min"), alpha_max=("alpha_max", "max")).reset_index()
    summary.to_csv(args.out / "summary.csv", index=False)

    boots, sl, day = [], [], {}
    for (gname, origin, source), items in daily.items():
        loss = lambda key, fn: np.mean([fn(it[key] - it["truth"]).mean(1) for it in items], 0)   # seed 평균 일 손실
        day[(gname, origin, source)] = {"MSE": loss("gate", np.square), "MAE": loss("gate", np.abs)}
        for ref in ("E50", "A"):
            for metric, fn in (("MSE", np.square), ("MAE", np.abs)):
                mean, lo, hi = V.block_bootstrap_diff(loss("gate", fn), loss(ref, fn))
                boots.append({"gate": gname, "origin": origin, "source": source, "vs": ref, "metric": metric,
                              "gate_minus_ref": mean, "ci_lo": lo, "ci_hi": hi,
                              "verdict": "gate better" if hi < 0 else "gate worse" if lo > 0 else "no sig. diff"})
        for name, (kind, _) in items[0]["masks"].items():
            row = {"gate": gname, "origin": origin, "source": source, "slice": name, "kind": kind}
            for key in ("gate", "E50", "A", "C"):
                row[f"RMSE_{key}"] = float(np.mean([V.rmse(it["truth"][it["masks"][name][1]], it[key][it["masks"][name][1]])
                                                    for it in items if it["masks"][name][1].any()] or [np.nan]))
            row["days"] = int(items[0]["masks"][name][1].sum())
            row["alpha_mean"] = float(np.mean([it["alpha"][it["masks"][name][1]].mean()
                                               for it in items if it["masks"][name][1].any()] or [np.nan]))
            sl.append(row)
    pd.DataFrame(boots).to_csv(args.out / "bootstrap.csv", index=False)
    pairs = []                                               # 수정 전후를 직접 비교 (new - old, 음수면 수정 후가 좋다)
    for new, old in PAIRS:
        for (g, origin, source) in [k for k in day if k[0] == new]:
            for metric in ("MSE", "MAE"):
                mean, lo, hi = V.block_bootstrap_diff(day[(new, origin, source)][metric], day[(old, origin, source)][metric])
                pairs.append({"new": new, "old": old, "origin": origin, "source": source, "metric": metric,
                              "new_minus_old": mean, "ci_lo": lo, "ci_hi": hi,
                              "verdict": "new better" if hi < 0 else "new worse" if lo > 0 else "no sig. diff"})
    pd.DataFrame(pairs).to_csv(args.out / "pairs.csv", index=False)
    sl = pd.DataFrame(sl)
    sl["dRMSE_vs_E50"] = sl["RMSE_gate"] - sl["RMSE_E50"]
    sl.to_csv(args.out / "slices.csv", index=False)

    spread = pd.DataFrame(spread)
    spread.to_csv(args.out / "gate_seed_runs.csv", index=False)
    sp = spread.groupby(["origin", "source", "gate"], sort=False).agg(
        dRMSE_mean=("dRMSE", "mean"), dRMSE_std=("dRMSE", "std"), dRMSE_min=("dRMSE", "min"),
        dRMSE_max=("dRMSE", "max"), share_worse_than_E50=("dRMSE", lambda v: float((v > 1e-12).mean())),
        share_better_than_E50=("dRMSE", lambda v: float((v < -1e-12).mean())),
        share_untrained=("best_epoch", lambda v: float((v == 0).mean())), n=("dRMSE", "size")).reset_index()
    # 초기화에 따른 흔들림만 분리: A·C seed 안에서의 표준편차를 평균
    within = spread.groupby(["origin", "source", "gate", "seed"])["dRMSE"].std().groupby(level=[0, 1, 2]).mean()
    sp = sp.merge(within.rename("dRMSE_std_within_AC_seed").reset_index(), on=["origin", "source", "gate"])
    sp.to_csv(args.out / "gate_seed_spread.csv", index=False)
    print(f"Saved: {args.out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
