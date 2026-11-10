"""Four follow-up candidates with reproducible plans and merged results."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import train_01_baseline as A
import train_02_bc as BC
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
NAMES = [
    "D_blend", "E_price_attention", "F_time_mean",
    "F_time_attention", "G_direct_lags"
]
MIX = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+".tmp")
    A.write_json(tmp, obj)
    tmp.replace(p)


def best_mix(base, case, target):
    trials = [
        (float(np.mean(((1-lam)*base+lam*case-target)**2)), lam)
        for lam in MIX
    ]
    return min(trials)


class Model(nn.Module):
    def __init__(self, name, initial):
        super().__init__()
        self.name = name

        if name == "E_price_attention":
            self.norm = nn.LayerNorm(119)
            self.q = nn.Linear(119, 32)
            self.k = nn.Linear(119, 32)
            return

        self.base = A.Baseline()
        self.base.load_state_dict(initial)

        if name == "G_direct_lags":
            old = self.base.mlp[0]
            self.base.mlp[0] = nn.Linear(191, 64)
            with torch.no_grad():
                self.base.mlp[0].weight[:, :119].copy_(old.weight)
                self.base.mlp[0].weight[:, 119:].zero_()
                self.base.mlp[0].bias.copy_(old.bias)
        else:
            self.q = nn.Linear(75, 32)
            self.k = nn.Linear(64, 32)
            self.v = nn.Linear(64, 64)
            self.delta = nn.Sequential(
                nn.Linear(75, 32), nn.ReLU(), nn.Linear(32, 1)
            )
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
            angle = torch.arange(24)*2*math.pi/24
            self.register_buffer(
                "hours", torch.stack((angle.sin(), angle.cos()), -1)
            )

    def forward(self, batch, bundle, device):
        x, f, w, y, z, ids, mask, base_pred = bundle

        if self.name == "E_price_attention":
            valid = mask[batch].to(device)
            candidates = ids[batch]
            query = self.q(self.norm(z[batch].to(device)))
            keys = self.k(self.norm(z[candidates].to(device)))
            logits = (query[:, None]*keys).sum(-1)/math.sqrt(32)
            weights = logits.masked_fill(~valid, -1e9).softmax(1)*valid
            weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-12)
            values = y[candidates].to(device)
            price = (weights[..., None]*values).sum(1)
            price = torch.where(
                valid.any(1)[:, None],
                price,
                base_pred[batch].to(device)
            )
            return price, weights

        xx, ff, ww = (t[batch].to(device) for t in (x, f, w))
        sequence, (h, _) = self.base.lstm(xx)
        current = torch.cat((h[-1], ff.flatten(1), ww), 1)

        if self.name == "G_direct_lags":
            lags = torch.cat((
                xx[:, -24:, 0],
                xx[:, -48:-24, 0],
                xx[:, :24, 0]
            ), 1)
            return self.base.mlp(torch.cat((current, lags), 1)), None

        n = len(batch)
        hour = self.hours[None].expand(n, -1, -1)
        weekday = ww[:, None].expand(-1, 24, -1)

        if self.name == "F_time_attention":
            query_in = torch.cat((
                h[-1][:, None].expand(-1, 24, -1),
                ff, weekday, hour
            ), -1)
            logits = torch.bmm(
                self.q(query_in), self.k(sequence).transpose(1, 2)
            )/math.sqrt(32)
            weights = logits.softmax(-1)
        else:
            weights = sequence.new_full((n, 24, 168), 1/168)

        context = torch.bmm(weights, self.v(sequence))
        correction = self.delta(
            torch.cat((context, ff, weekday, hour), -1)
        ).squeeze(-1)
        return self.base.mlp(current)+correction, weights


@torch.no_grad()
def predict(model, indices, bundle, cfg, device, collect=False):
    model.eval()
    outputs, weights = [], []
    for batch in DataLoader(indices, cfg["batch_size"]):
        pred, weight = model(batch, bundle, device)
        outputs.append(pred.cpu().numpy())
        if collect and weight is not None:
            weights.append(weight.cpu().numpy())
    return (
        np.concatenate(outputs).astype(np.float64),
        np.concatenate(weights) if weights else None
    )


def fit(name, seed, initial, bundle, ntrain, nval, cfg, device, folder):
    BC.seed_all(seed)
    model = Model(name, initial).to(device)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(BC.state(model), folder/"initial_state.pt")

    if name != "E_price_attention":
        baseline = A.Baseline().to(device)
        baseline.load_state_dict(initial)
        with torch.no_grad():
            probe = torch.arange(min(4, ntrain))
            expected = baseline(
                *(t[probe].to(device) for t in bundle[:3])
            )
            actual, _ = model(probe, bundle, device)
            torch.testing.assert_close(
                actual, expected, rtol=1e-5, atol=1e-6
            )
        del baseline

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        torch.arange(ntrain), cfg["batch_size"],
        shuffle=True, generator=generator
    )
    val_ids = torch.arange(ntrain, ntrain+nval)
    target_val = bundle[3][val_ids].numpy().astype(np.float64)
    a_val = bundle[7][val_ids].numpy().astype(np.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best, stale, best_epoch, selected = float("inf"), 0, 0, None
    if name == "E_price_attention":
        best = float(np.mean((a_val-target_val)**2))
        selected = 0.0
        torch.save(BC.state(model), folder/"best_model.pt")

    log = []
    started = time.perf_counter()

    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        total, count = 0.0, 0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch, bundle, device)

            # E: A는 고정. 학습 중 사례 반영 비중은 0.25로 고정.
            if name == "E_price_attention":
                prediction = (
                    0.75*bundle[7][batch].to(device)+0.25*prediction
                )

            target = bundle[3][batch].to(device)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            total += loss.item()*target.numel()
            count += target.numel()

        val_pred, _ = predict(model, val_ids, bundle, cfg, device)

        # E의 최종 결합 비중과 epoch는 검증 MSE로 함께 선택.
        if name == "E_price_attention":
            score, lam = best_mix(a_val, val_pred, target_val)
        else:
            score = float(np.mean((val_pred-target_val)**2))
            lam = None

        if not np.isfinite(score):
            raise RuntimeError("Nonfinite validation loss")

        log.append({
            "epoch": epoch,
            "train_MSE_scaled": total/count,
            "val_MSE_scaled": score,
            "lambda": lam
        })
        pd.DataFrame(log).to_csv(folder/"history.csv", index=False)
        print(
            f"{folder.parent.parent.name} seed={seed} {name} "
            f"epoch={epoch:03d} train={total/count:.6f} val={score:.6f}",
            flush=True
        )

        if score < best:
            best, best_epoch, stale, selected = score, epoch, 0, lam
            torch.save(BC.state(model), folder/"best_model.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break

    model.load_state_dict(torch.load(
        folder/"best_model.pt", map_location="cpu", weights_only=True
    ))
    return model, {
        "best_epoch": best_epoch,
        "best_val_MSE_scaled": best,
        "lambda": selected,
        "train_seconds": time.perf_counter()-started,
        "parameters": sum(p.numel() for p in model.parameters())
    }


def refresh(out, bc_run):
    original = pd.read_csv(bc_run/"metrics_all_seeds.csv")
    plan = read(out/"plan.json")
    original = original[
        original["market"].isin(plan["markets"])
        & original["seed"].isin(plan["seeds"])
    ].copy()
    original["source_run"] = str(bc_run)
    rows = [
        read(p) for p in sorted(out.glob("*/seed_*/*/metrics.json"))
    ]
    frame = pd.concat([original, pd.DataFrame(rows)], ignore_index=True)
    frame.to_csv(out/"metrics_all_seeds.csv", index=False)
    metric_names = [
        "MAE", "RMSE", "high_RMSE", "total_exceedance_MAE",
        "peak_hour_MAE", "high_recall", "false_positive_rate"
    ]
    frame.groupby(["market", "model"])[metric_names].agg(
        ["mean", "std"]
    ).to_csv(out/"summary.csv")


def prepare(args):
    if args.bc_run is None:
        candidates = sorted((ROOT/"results"/"abc").glob("*"))
        if not candidates:
            raise SystemExit(
                "ABC 결과가 없습니다. --bc-run으로 지정하세요."
            )
        bc_run = candidates[-1].resolve()
    else:
        bc_run = args.bc_run.expanduser().resolve()

    bc_config = read(bc_run/"config.json")
    a_run = Path(bc_config["a_run"])
    cfg = read(a_run/"config.json")
    markets = args.markets or cfg["markets"]
    seeds = args.seeds or cfg["seeds"]

    if (
        not set(markets) <= set(cfg["markets"])
        or not set(seeds) <= set(cfg["seeds"])
    ):
        raise SystemExit("선택한 시장/seed가 A 설정에 없습니다.")

    files = [
        ROOT/p for p in [
            "train_01_baseline.py",
            "train_02_bc.py",
            "train_03_candidates.py"
        ]
    ]
    for market in markets:
        files.append(ROOT/"data"/f"{market}.csv")
        for seed in seeds:
            for run, names in [
                (a_run, [
                    "best_model.pt", "initial_state.pt",
                    "preprocessing.json", "metrics.json"
                ]),
                (bc_run, [
                    "candidate_indices.npz", "retrieval.json",
                    "metrics.json", "preprocessing.json"
                ])
            ]:
                files.extend(
                    run/market/f"seed_{seed}"/name for name in names
                )
    files.extend([
        a_run/"config.json", bc_run/"config.json",
        bc_run/"metrics_all_seeds.csv"
    ])

    for p in files:
        if not p.is_file():
            raise SystemExit(
                f"결과가 미완료이거나 파일이 없습니다: {p}"
            )

    plan = {
        "version": 1,
        "a_run": str(a_run),
        "bc_run": str(bc_run),
        "a_config": cfg,
        "markets": markets,
        "seeds": seeds,
        "models": NAMES,
        "lambda_grid": MIX,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version()
        },
        "file_sha256": {
            str(p.resolve()): digest(p) for p in files
        },
        "protocol": (
            "D/E use frozen trained A and B-selected candidates. "
            "F/G start from A initial weights. MSE; "
            "A optimizer/epochs/early stopping. "
            "E training mix=.25; validation selects epoch and mixture "
            "jointly. No test selection."
        ),
        "resume": (
            "Completed model folders are skipped; "
            "interrupted model restarts from its initial state."
        )
    }
    write(ROOT/"candidate_plan.json", plan)
    print(
        "준비 완료: candidate_plan.json. "
        "이 준비 단계에서는 학습을 실행하지 않았습니다.",
        flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--bc-run", type=Path)
    parser.add_argument("--markets", nargs="+", choices=["PJM", "FR"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--plan", type=Path, default=ROOT/"candidate_plan.json"
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.run_all and (
        args.prepare or args.resume
        or args.plan != ROOT/"candidate_plan.json"
    ):
        parser.error(
            "--run-all은 --prepare/--resume/--plan과 함께 쓸 수 없습니다."
        )
    if args.run_all:
        prepare(args)
        args.bc_run, args.markets, args.seeds = None, None, None
    if args.prepare:
        prepare(args)
        return
    if args.bc_run or args.markets or args.seeds:
        parser.error(
            "--bc-run/--markets/--seeds는 --prepare 또는 --run-all과 "
            "함께 사용하세요."
        )

    plan = read(args.plan)
    for path, expected in plan["file_sha256"].items():
        if not Path(path).is_file() or digest(Path(path)) != expected:
            raise SystemExit(
                f"계획 이후 파일이 변경됐습니다. "
                f"재확인 후 --prepare: {path}"
            )

    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version()
    }
    if versions != plan["versions"]:
        raise SystemExit("준비 때와 실행 환경이 다릅니다.")

    cfg = plan["a_config"]
    if torch.__version__ != cfg["torch"]:
        raise SystemExit("A 실험과 PyTorch 버전이 다릅니다.")

    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA가 필요합니다.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    a_run, bc_run = Path(plan["a_run"]), Path(plan["bc_run"])
    if args.resume:
        out = args.resume.resolve()
        if read(out/"plan.json") != plan:
            raise SystemExit(
                "재개할 결과의 계획이 현재 계획과 다릅니다."
            )
    else:
        out = ROOT/"results"/"candidates"/datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S_%fZ")
        out.mkdir(parents=True, exist_ok=False)
        write(out/"plan.json", plan)
        for p in [
            ROOT/"train_01_baseline.py",
            ROOT/"train_02_bc.py",
            Path(__file__)
        ]:
            dest = out/"source"/p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    print(f"출력: {out}", flush=True)

    for market in plan["markets"]:
        datasets, info, dates, truth, _ = A.load_market(
            market, cfg["val_days"], cfg["test_days"]
        )
        ntrain, nval, ntest = map(len, datasets)
        tensors = tuple(
            torch.cat([ds.tensors[j] for ds in datasets])
            for j in range(4)
        )
        val_ids = torch.arange(ntrain, ntrain+nval)
        test_ids = torch.arange(ntrain+nval, ntrain+nval+ntest)

        for seed in plan["seeds"]:
            src = a_run/market/f"seed_{seed}"
            old_bc = bc_run/market/f"seed_{seed}"

            for old in [
                read(src/"preprocessing.json"),
                read(old_bc/"preprocessing.json")
            ]:
                for key in [
                    "data_sha256", "columns", "mean", "std",
                    "splits", "high_threshold"
                ]:
                    if old[key] != info[key]:
                        raise RuntimeError(
                            f"전처리 불일치: {market}, {key}"
                        )

            checkpoint = torch.load(
                src/"best_model.pt",
                map_location="cpu", weights_only=True
            )
            frozen = A.Baseline().to(device)
            frozen.load_state_dict(checkpoint)
            z = BC.search_features(
                frozen, tensors, device, cfg["batch_size"]
            )
            with torch.no_grad():
                base_pred = frozen.mlp(z.to(device)).cpu()

            expected = read(src/"metrics.json")["baseline"]["MAE"]
            base_test = (
                base_pred[test_ids].numpy().astype(np.float64)
                * info["std"][0]+info["mean"][0]
            )
            actual = A.metrics(
                truth, base_test, info["high_threshold"]
            )["MAE"]
            if not np.isclose(
                actual, expected, rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError("A 재평가 불일치")
            del frozen

            with np.load(old_bc/"candidate_indices.npz") as archive:
                ids = torch.from_numpy(
                    archive["ids"].copy()
                ).long()
                mask = torch.from_numpy(
                    archive["mask"].copy()
                ).bool()

            if (
                ids.shape != mask.shape or len(ids) != len(z)
                or ids.min() < 0 or ids.max() >= ntrain
            ):
                raise RuntimeError("사례 인덱스가 올바르지 않습니다.")
            if not (
                (ids < torch.arange(len(z))[:, None]) | ~mask
            ).all():
                raise RuntimeError("현재/미래 사례 참조가 발견됐습니다.")

            bundle = (*tensors, z, ids, mask, base_pred)
            initial = torch.load(
                src/"initial_state.pt",
                map_location="cpu", weights_only=True
            )
            timestamps = [
                pd.Timestamp(day)+pd.Timedelta(hours=h)
                for day in dates for h in range(24)
            ]

            for name in NAMES:
                folder = out/market/f"seed_{seed}"/name
                if (folder/"metrics.json").is_file():
                    print(
                        f"SKIP {market} seed={seed} {name}",
                        flush=True
                    )
                    continue
                folder.mkdir(parents=True, exist_ok=True)

                if name == "D_blend":
                    case_val = BC.mean_prediction(
                        ids[val_ids], mask[val_ids], tensors[3]
                    ).numpy().astype(np.float64)
                    score, lam = best_mix(
                        base_pred[val_ids].numpy(),
                        case_val,
                        tensors[3][val_ids].numpy()
                    )
                    case_test = BC.mean_prediction(
                        ids[test_ids], mask[test_ids], tensors[3]
                    ).numpy().astype(np.float64)
                    pred = (
                        (1-lam)*base_pred[test_ids].numpy()
                        + lam*case_test
                    )
                    detail = {
                        "best_epoch": 0,
                        "best_val_MSE_scaled": score,
                        "lambda": lam,
                        "train_seconds": 0.0,
                        "parameters": 0
                    }
                    write(folder/"mixture.json", {
                        "lambda": lam, "grid": MIX
                    })
                else:
                    model, detail = fit(
                        name, seed, initial, bundle,
                        ntrain, nval, cfg, device, folder
                    )
                    pred, weights = predict(
                        model, test_ids, bundle, cfg, device,
                        collect=True
                    )
                    if name == "E_price_attention":
                        lam = detail["lambda"]
                        pred = (
                            (1-lam)*base_pred[test_ids].numpy()
                            + lam*pred
                        )
                    if weights is not None:
                        np.savez_compressed(
                            folder/"attention.npz",
                            weights=weights
                        )
                    del model

                pred = pred*info["std"][0]+info["mean"][0]
                scores = A.metrics(
                    truth, pred, info["high_threshold"]
                )
                pd.DataFrame({
                    "Date": timestamps,
                    "actual": truth.ravel(),
                    "prediction": pred.ravel()
                }).to_csv(folder/"predictions.csv", index=False)
                write(folder/"preprocessing.json", info)
                write(folder/"metrics.json", {
                    "market": market,
                    "seed": seed,
                    "model": name,
                    "source_run": str(out),
                    **detail, **scores
                })
                refresh(out, bc_run)
                print(
                    f"{market} seed={seed} {name}: "
                    f"MAE={scores['MAE']:.4f} "
                    f"RMSE={scores['RMSE']:.4f}",
                    flush=True
                )

    refresh(out, bc_run)
    print(f"완료: {out}", flush=True)


if __name__ == "__main__":
    main()
