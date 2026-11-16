"""Volatility gate: Prediction = (1 - alpha) * A + alpha * C,  alpha = sigmoid(MLP(시장 특징)).

A·C 는 고정(frozen)이고 gate 만 학습한다. gate 학습 데이터는 make_oof_predictions.py 가 만든
rolling-origin OOF 예측뿐이다. 평가 origin T 에 대해

    gate_train = T 보다 앞선 OOF 블록 중 마지막 블록을 뺀 나머지
    gate_val   = T 직전 OOF 블록 (early stopping 전용)
    test       = origin T 의 블록 (학습·선택·표준화 어디에도 쓰지 않음)

절차: (1) gate_train 으로 학습, gate_val 로 early stopping -> best_epoch
      (2) gate_train+gate_val 로 best_epoch 만큼 다시 학습(refit) -> test 예측 (채택 모델)
Efixed 의 고정 가중치도 같은 gate_train+gate_val 에서 최소제곱으로 정한다.

실행:  python train_volatility_gate.py            (volatility-gate/ 안에서)
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import vgate_lib as V

ROOT = Path(__file__).resolve().parent
U = V.USER_FEATURES

# 주 비교 대상 gate 3종 (실험 전에 고정)
MAIN_GATES = {
    "gate_vol":    {"features": ["vol_std"], "anchor": False, "out_dim": 1},   # 최소: 전날 표준편차만
    "gate_struct": {"features": U, "anchor": False, "out_dim": 1},             # 요청한 특징 8개 전체
    "gate_anchor": {"features": U, "anchor": True, "out_dim": 1},              # 고정 가중치에서 출발하는 gate
}


def ablation_gates():
    g = {}
    for i in range(1, len(U) + 1):                                   # 누적 추가
        g[f"add{i}_{U[i - 1]}"] = {"features": U[:i], "anchor": False, "out_dim": 1}
    extra = U
    for name in ["osc_chop", "vol_rel", "load_chg"]:                 # 추가 제안 특징 누적
        extra = extra + [name]
        g[f"add{len(extra)}_{name}"] = {"features": extra, "anchor": False, "out_dim": 1}
    for name in [f for f in V.FEATURES if f != "vol_std"]:           # vol_std + 한 개씩
        g[f"pair_{name}"] = {"features": ["vol_std", name], "anchor": False, "out_dim": 1}
    concepts = {"dispersion": ["vol_std", "range", "wk_std"], "trend": ["trend_dd", "trend_intra"],
                "oscillation": ["osc_mad", "osc_chop"], "level": ["lvl_mean", "wk_mean"]}
    for name, feats in concepts.items():                             # 개념별 단독
        g[f"only_{name}"] = {"features": feats, "anchor": False, "out_dim": 1}
    g["anchor_vol"] = {"features": ["vol_std"], "anchor": True, "out_dim": 1}
    g["anchor_concepts"] = {"features": ["vol_std", "trend_dd", "trend_intra", "osc_chop"], "anchor": True, "out_dim": 1}
    g["anchor_all11"] = {"features": list(V.FEATURES), "anchor": True, "out_dim": 1}
    g["anchor_4blocks"] = {"features": U, "anchor": True, "out_dim": 4}   # 6시간 블록별 alpha
    return g


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def logit(p):
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


def train_gate(x, pa, pc, y, seed, cfg, spec, anchor_alpha, val=None, fixed_epochs=None):
    """x, pa, pc, y: 학습용 텐서. val: (x, pa, pc, y) 또는 None. 반환: model, best_epoch, history."""
    seed_all(seed)
    model = V.GateMLP(x.shape[1], cfg["hidden"], spec["out_dim"],
                      logit(anchor_alpha) if spec["anchor"] else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(seed)

    def loss_of(xx, aa, cc, yy):
        with torch.no_grad():
            return nn.functional.mse_loss(model(xx, aa, cc)[0], yy).item()

    copy = lambda: {k: v.clone() for k, v in model.state_dict().items()}
    history = []
    best, best_epoch, best_state, stale = float("inf"), 0, copy(), 0
    if val is not None:                       # epoch 0 (학습 전)도 후보로 둔다
        best = loss_of(*val)
        history.append({"epoch": 0, "train_MSE_scaled": loss_of(x, pa, pc, y), "val_MSE_scaled": best})
    epochs = cfg["epochs"] if fixed_epochs is None else fixed_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for i in range(0, len(x), cfg["batch_size"]):
            b = order[i:i + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[b], pa[b], pc[b])[0], y[b])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        row = {"epoch": epoch, "train_MSE_scaled": total / len(x)}
        if val is not None:
            v = loss_of(*val)
            row["val_MSE_scaled"] = v
            if v < best:
                best, best_epoch, best_state, stale = v, epoch, copy(), 0
            else:
                stale += 1
        history.append(row)
        if val is not None and stale >= cfg["patience"]:
            break
    if val is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch, history, best


def load_oof(folder, raw, index):
    """OOF 예측 CSV 를 읽고 원본 PJM 가격과 시각·정답이 맞는지 확인."""
    frame = pd.read_csv(folder / "predictions.csv", parse_dates=["Date"])
    return check_frame(frame, raw, index, folder)


def check_frame(frame, raw, index, where=""):
    if len(frame) % 24 or frame["Date"].duplicated().any():
        raise ValueError(f"{where}: rows must be unique complete days")
    pos = index.get_indexer(frame["Date"])
    if (pos < 0).any() or not (np.diff(pos) == 1).all() or frame["Date"].iloc[0].hour != 0:
        raise ValueError(f"{where}: timestamps are not consecutive hours starting at 00:00")
    if not np.allclose(frame["actual"].to_numpy(), raw[pos, 0], rtol=0, atol=1e-9):
        raise ValueError(f"{where}: 'actual' differs from PJM.csv")
    if not np.isfinite(frame[["A", "C"]].to_numpy()).all():
        raise ValueError(f"{where}: missing A/C predictions")
    return {"dates": pd.DatetimeIndex(frame["Date"].iloc[::24]), "starts": pos[::24],
            "truth": frame["actual"].to_numpy().reshape(-1, 24),
            "A": frame["A"].to_numpy().reshape(-1, 24), "C": frame["C"].to_numpy().reshape(-1, 24)}


def stack(blocks, key):
    return np.concatenate([b[key] for b in blocks])


def run_origin(test_name, order, seed, cfg, gates, frame_raw, out, oof_dir, external):
    raw, index = frame_raw.to_numpy(dtype=np.float64), frame_raw.index
    blocks = {n: load_oof(oof_dir / n / f"seed_{seed}", raw, index) for n in order}
    earlier = order[:order.index(test_name)]
    if len(earlier) < 2:
        raise SystemExit(f"{test_name}: gate 학습에 앞선 OOF 블록이 2개 이상 필요합니다 (현재 {earlier})")
    tr_blocks, va_block, te_block = [blocks[n] for n in earlier[:-1]], blocks[earlier[-1]], blocks[test_name]
    meta = json.loads((oof_dir / test_name / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
    mu, sd = meta["scaling"]["mean"][0], meta["scaling"]["std"][0]      # 평가 origin 의 train 구간 가격 통계

    parts = {"gate_train": {k: stack(tr_blocks, k) for k in ("starts", "truth", "A", "C")},
             "gate_val": {k: va_block[k] for k in ("starts", "truth", "A", "C")},
             "test": {k: te_block[k] for k in ("starts", "truth", "A", "C")}}
    if external is not None:
        parts["test_saved_abc"] = {k: external[k] for k in ("starts", "truth", "A", "C")}
    # 시간순 보장: 학습 < 검증 < 테스트. gate 학습 구간의 마지막 목표일이 test 첫 목표일보다 앞선다.
    assert parts["gate_train"]["starts"].max() < parts["gate_val"]["starts"].min()
    assert parts["gate_val"]["starts"].max() + 24 <= parts["test"]["starts"].min()
    feats = {name: V.gate_features(raw, p["starts"]) for name, p in parts.items()}
    fit = {k: np.concatenate([parts["gate_train"][k], parts["gate_val"][k]]) for k in ("starts", "truth", "A", "C")}
    feat_fit = pd.concat([feats["gate_train"], feats["gate_val"]], ignore_index=True)

    sc = lambda v: torch.from_numpy((v - mu) / sd).float()
    alpha_fixed_tr = V.fixed_alpha(parts["gate_train"]["A"], parts["gate_train"]["C"], parts["gate_train"]["truth"])
    alpha_fixed = V.fixed_alpha(fit["A"], fit["C"], fit["truth"])

    rows, frames = [], {}
    for gname, spec in gates.items():
        names = spec["features"]
        x_tr_raw, x_va_raw = V.transform(feats["gate_train"], names), V.transform(feats["gate_val"], names)
        m1, s1 = V.fit_scaler(x_tr_raw)                                  # gate_train 통계만
        tx = lambda a, m, s: torch.from_numpy((a - m) / s).float()
        p_tr, p_va = parts["gate_train"], parts["gate_val"]
        _, best_epoch, history, best_val = train_gate(
            tx(x_tr_raw, m1, s1), sc(p_tr["A"]), sc(p_tr["C"]), sc(p_tr["truth"]), seed, cfg, spec, alpha_fixed_tr,
            val=(tx(x_va_raw, m1, s1), sc(p_va["A"]), sc(p_va["C"]), sc(p_va["truth"])))
        x_fit_raw = V.transform(feat_fit, names)
        m2, s2 = V.fit_scaler(x_fit_raw)                                 # gate_train+gate_val 통계만
        model, _, _, _ = train_gate(tx(x_fit_raw, m2, s2), sc(fit["A"]), sc(fit["C"]), sc(fit["truth"]),
                                    seed, cfg, spec, alpha_fixed, val=None, fixed_epochs=best_epoch)
        gdir = out / "gates" / gname / test_name / f"seed_{seed}"
        gdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), gdir / "gate.pt")
        pd.DataFrame(history).to_csv(gdir / "history.csv", index=False)
        (gdir / "gate.json").write_text(json.dumps({
            "features": names, "anchor": spec["anchor"], "out_dim": spec["out_dim"], "hidden": cfg["hidden"],
            "best_epoch": best_epoch, "best_val_MSE_scaled": best_val, "alpha_fixed": alpha_fixed,
            "feature_mean": m2.tolist(), "feature_std": s2.tolist(), "price_mean": mu, "price_std": sd,
            "gate_train_blocks": earlier[:-1], "gate_val_block": earlier[-1], "test_block": test_name,
        }, indent=2), encoding="utf-8")

        for pname, p in parts.items():
            with torch.no_grad():
                alpha = model.alpha(tx(V.transform(feats[pname], names), m2, s2)).double().numpy()
            assert alpha.min() >= 0.0 and alpha.max() <= 1.0
            alpha = np.broadcast_to(alpha, p["A"].shape)
            pred = V.blend(alpha, p["A"], p["C"])
            day_alpha = alpha.mean(1)
            if pname.startswith("test"):
                source = "saved_abc" if pname == "test_saved_abc" else "retrained"
                rows.append({
                    "gate": gname, "origin": test_name, "seed": seed, "source": source,
                    "n_features": len(names), "features": "+".join(names), "anchor": spec["anchor"],
                    "out_dim": spec["out_dim"], "best_epoch": best_epoch, "gate_val_MSE_scaled": best_val,
                    "alpha_fixed": alpha_fixed,
                    "MAE_A": V.mae(p["truth"], p["A"]), "RMSE_A": V.rmse(p["truth"], p["A"]),
                    "MAE_C": V.mae(p["truth"], p["C"]), "RMSE_C": V.rmse(p["truth"], p["C"]),
                    "MAE_E50": V.mae(p["truth"], V.blend(0.5, p["A"], p["C"])),
                    "RMSE_E50": V.rmse(p["truth"], V.blend(0.5, p["A"], p["C"])),
                    "MAE_Efixed": V.mae(p["truth"], V.blend(alpha_fixed, p["A"], p["C"])),
                    "RMSE_Efixed": V.rmse(p["truth"], V.blend(alpha_fixed, p["A"], p["C"])),
                    "MAE_gate": V.mae(p["truth"], pred), "RMSE_gate": V.rmse(p["truth"], pred),
                    "alpha_mean": float(day_alpha.mean()), "alpha_std": float(day_alpha.std()),
                    "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
                    "alpha_lt_0.05": float((day_alpha < 0.05).mean()), "alpha_gt_0.95": float((day_alpha > 0.95).mean()),
                })
            if gname in MAIN_GATES:
                stamps = index[(p["starts"][:, None] + np.arange(24)).ravel()]
                frames.setdefault(gname, []).append(pd.DataFrame({
                    "Date": stamps, "actual": p["truth"].ravel(), "pred_A": p["A"].ravel(), "pred_C": p["C"].ravel(),
                    "pred_E50": V.blend(0.5, p["A"], p["C"]).ravel(),
                    "pred_Efixed": V.blend(alpha_fixed, p["A"], p["C"]).ravel(),
                    "pred_gate": pred.ravel(), "alpha": alpha.ravel(), "alpha_fixed": alpha_fixed,
                    "volatility_previous_day": np.repeat(feats[pname]["vol_std"].to_numpy(), 24),
                    "seed": seed, "split": "test" if pname.startswith("test") else pname,
                    "origin": test_name, "source": "saved_abc" if pname == "test_saved_abc" else "retrained",
                }))
    return rows, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-tag", default="main")
    p.add_argument("--test-origins", nargs="+", default=["o2018", "o2017"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--saved-abc", type=Path, default=None,
                   help="기존 PJM_error_analysis/results/abc/<run> 폴더. 기본값은 가장 최근 실행. "
                        "o2018 에서 저장된 A·C 예측에도 같은 gate 를 적용한다")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = {"hidden": args.hidden, "epochs": args.epochs, "patience": args.patience,
           "batch_size": args.batch_size, "lr": args.lr}
    oof_dir = ROOT / "results" / "oof_ac" / args.oof_tag
    order = sorted(d.name for d in oof_dir.glob("o*") if d.is_dir())
    out = args.out or ROOT / "results" / "vgate" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=True)
    gates = dict(MAIN_GATES)
    if not args.no_ablation:
        gates.update(ablation_gates())
    saved = args.saved_abc
    if saved is None:
        runs = sorted(d for d in (V.BASE / "results" / "abc").glob("*") if d.is_dir())
        saved = runs[-1] if runs else None
    repo = ROOT.parent

    def shown(path):                      # 설정 파일에는 저장소 기준 상대 경로만 남긴다
        path = Path(path).resolve()
        return str(path.relative_to(repo)) if path.is_relative_to(repo) else path.name
    (out / "config.json").write_text(json.dumps({
        **cfg, "seeds": args.seeds, "test_origins": args.test_origins, "oof_blocks": order,
        "oof_dir": shown(oof_dir), "saved_abc": shown(saved) if saved else None,
        "loss": "24시간 평균 MSE (평가 origin train 구간 가격 mean/std 로 표준화한 단위)",
        "optimizer": "Adam", "final_model": "gate_train+gate_val refit for best_epoch epochs",
        "gate_seed": "base seed 와 동일", "feature_transform": {k: v[1] for k, v in V.FEATURES.items()},
        "gates": gates, "torch": torch.__version__,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    frame_raw = V.read_frame()
    all_rows, all_frames = [], {}
    for test_name in args.test_origins:
        for seed in args.seeds:
            external = None
            if saved is not None and test_name == "o2018":
                ext = pd.read_csv(saved / "PJM" / f"seed_{seed}" / "predictions.csv", parse_dates=["Date"])
                external = check_frame(ext, frame_raw.to_numpy(dtype=np.float64), frame_raw.index, "saved abc")
            rows, frames = run_origin(test_name, order, seed, cfg, gates, frame_raw, out, oof_dir, external)
            all_rows += rows
            for g, fs in frames.items():
                all_frames.setdefault(g, []).extend(fs)
            r = {x["gate"]: x for x in rows if x["source"] == "retrained"}
            print(f"{test_name} seed={seed} alpha_fixed={rows[0]['alpha_fixed']:.3f} | RMSE A={rows[0]['RMSE_A']:.3f} "
                  f"C={rows[0]['RMSE_C']:.3f} E50={rows[0]['RMSE_E50']:.3f} Efixed={rows[0]['RMSE_Efixed']:.3f} | "
                  + " ".join(f"{g}={r[g]['RMSE_gate']:.3f}" for g in MAIN_GATES), flush=True)
    pd.DataFrame(all_rows).to_csv(out / "ablation.csv", index=False)
    for g, fs in all_frames.items():
        pd.concat(fs, ignore_index=True).to_csv(out / f"predictions_{g}.csv", index=False)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
