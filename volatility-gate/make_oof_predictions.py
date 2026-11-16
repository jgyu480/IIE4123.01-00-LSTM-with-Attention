"""Rolling-origin(walk-forward) OOF 예측 생성: 모델 A와 C.

각 origin 은 배포 때와 똑같은 절차를 과거 시점에서 반복한다.

    [ train ............ ][ val 364일 (early stopping, k 선택) ][ block 364일 (예측만) ]

block 의 실제 가격은 그 origin 의 A·C 학습, early stopping, k 선택, 표준화 어디에도 쓰이지 않는다.
따라서 block 예측은 gate 학습에 쓸 수 있는 OOF 예측이다.

    o2018 (trim   0일): 기존 실험과 동일. block = 2017-12-26 ~ 2018-12-24  (최종 평가용)
    o2017 (trim 364일): block = 2016-12-27 ~ 2017-12-25
    o2016 (trim 728일): block = 2015-12-29 ~ 2016-12-26
    o2015 (trim 1092일): block = 2014-12-30 ~ 2015-12-28

A 학습 루프는 train_01_baseline.main 의 seed 순서·하이퍼파라미터를 그대로 따르고,
C 는 train_02_bc 의 함수(search_features, neighbors, mean_prediction, train_variant)를 그대로 호출한다.

실행:  python make_oof_predictions.py            (volatility-gate/ 안에서)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import vgate_lib as V                 # 먼저 import: PJM_error_analysis/ 를 sys.path 에 추가한다
import train_01_baseline as A         # noqa: E402  (PJM_error_analysis/, 수정하지 않음)
import train_02_bc as BC              # noqa: E402  (PJM_error_analysis/, 수정하지 않음)

ROOT = Path(__file__).resolve().parent


def train_a(datasets, seed, cfg, device, folder):
    """train_01_baseline.main 의 seed별 학습 블록과 같은 순서."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets[0], cfg["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(datasets[1], cfg["batch_size"], shuffle=False)
    model = A.Baseline().to(device)
    initial = BC.state(model)
    torch.save(initial, folder / "initial_state.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_epoch, stale, log = float("inf"), 0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for history, forecast, weekday, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history.to(device), forecast.to(device), weekday.to(device))
            loss = nn.functional.mse_loss(prediction, target.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * target.numel()
            count += target.numel()
        val_loss, _ = A.evaluate(model, val_loader, device)
        log.append({"epoch": epoch, "train_MSE_scaled": total / count, "val_MSE_scaled": val_loss})
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(BC.state(model), folder / "best_model.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    pd.DataFrame(log).to_csv(folder / "history.csv", index=False)
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    return model, initial, {"best_epoch": best_epoch, "best_val_MSE_scaled": best}


def run(trim, seed, cfg, device, out):
    name = V.origin_name(trim)
    folder = out / name / f"seed_{seed}"
    if (folder / "predictions.csv").exists():
        print(f"skip (done): {folder}", flush=True)
        return
    started = time.time()
    w = V.load_window(trim, cfg["val_days"], cfg["test_days"])
    datasets, info = w["datasets"], w["info"]
    ntrain, nval, ntest = map(len, datasets)
    scale = lambda x: x * info["std"][0] + info["mean"][0]

    model_a, initial, a_detail = train_a(datasets, seed, cfg, device, folder / "A")
    _, a_scaled = A.evaluate(model_a, DataLoader(datasets[2], cfg["batch_size"]), device)

    # --- train_02_bc.main 의 seed별 블록과 같은 절차 ---
    tensors = tuple(torch.cat([ds.tensors[j] for ds in datasets]) for j in range(4))
    z = BC.search_features(model_a, tensors, device, cfg["batch_size"])
    grid = sorted(set(min(k, ntrain) for k in cfg["k_grid"]))
    all_ids, all_mask, _ = BC.neighbors(z, ntrain, max(grid))
    val_slice, test_slice = slice(ntrain, ntrain + nval), slice(ntrain + nval, None)
    trials = []
    for k in grid:
        pred = BC.mean_prediction(all_ids[val_slice, :k], all_mask[val_slice, :k], tensors[3])
        trials.append({"k": k, "val_MSE_scaled": (pred - tensors[3][val_slice]).square().mean().item()})
    k = min(trials, key=lambda x: (x["val_MSE_scaled"], x["k"]))["k"]
    ids, mask = all_ids[:, :k].clone(), all_mask[:, :k].clone()
    assert int(ids.max()) < ntrain, "case pool must be training days only"
    b_scaled = BC.mean_prediction(ids[test_slice], mask[test_slice], tensors[3]).numpy().astype(np.float64)
    c_scaled, _, c_detail = BC.train_variant(
        initial, "attention", seed, tensors, ids, mask, ntrain, nval, cfg, device, folder / "C")

    preds = {"A": scale(a_scaled), "B": scale(b_scaled), "C": scale(c_scaled)}
    truth = w["truth"]
    stamps = [day + pd.Timedelta(hours=h) for day in w["test_dates"] for h in range(24)]
    table = {"Date": stamps, "actual": truth.ravel(), **{n: p.ravel() for n, p in preds.items()}}
    A.write_json(folder / "metrics.json", {
        "origin": name, "trim_days": trim, "seed": seed, "splits": info["splits"],
        "scaling": {"mean": info["mean"], "std": info["std"]}, "high_threshold": info["high_threshold"],
        "A": a_detail, "C": c_detail, "selected_k": k, "k_trials": trials,
        "block": {n: {"MAE": V.mae(truth, p), "RMSE": V.rmse(truth, p)} for n, p in preds.items()},
        "seconds": time.time() - started,
    })
    pd.DataFrame(table).to_csv(folder / "predictions.csv", index=False)   # 마지막에 저장 = 완료 표시
    print(f"{name} seed={seed} A MAE={V.mae(truth, preds['A']):.4f} RMSE={V.rmse(truth, preds['A']):.4f} | "
          f"C MAE={V.mae(truth, preds['C']):.4f} RMSE={V.rmse(truth, preds['C']):.4f} | k={k} "
          f"({time.time() - started:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trims", type=int, nargs="+", default=[0, 364, 728, 1092],
                   help="끝에서 버릴 일수. 0 = 기존 실험(o2018)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--k-grid", type=int, nargs="+", default=[3, 5, 10, 20])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tag", default="main")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    cfg = {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "lr": args.lr,
           "val_days": V.BLOCK_DAYS, "test_days": V.BLOCK_DAYS, "k_grid": args.k_grid}
    out = ROOT / "results" / "oof_ac" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    A.write_json(out / "config.json", {**cfg, "seeds": args.seeds, "trims": args.trims, "threads": args.threads,
                                       "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
                                       "protocol": "rolling origin: train | val(ES, k) | block(predict only)"})
    for trim in args.trims:
        for seed in args.seeds:
            run(trim, seed, cfg, torch.device("cpu"), out)


if __name__ == "__main__":
    main()
