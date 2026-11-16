"""Volatility gate 자동 테스트.   실행:  python -m pytest tests/test_volatility_gate.py -q   (volatility-gate/ 안에서)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vgate_lib as V                    # noqa: E402  (먼저 import: PJM_error_analysis/ 를 sys.path 에 추가)
import train_01_baseline as A            # noqa: E402
import train_volatility_gate as G     # noqa: E402

CFG = {"hidden": 8, "epochs": 3, "patience": 10, "batch_size": 32, "lr": 0.001}
ABC_RUNS = sorted(d for d in (V.BASE / "results" / "abc").glob("*") if d.is_dir())


@pytest.fixture(scope="module")
def frame():
    return V.read_frame()


@pytest.fixture(scope="module")
def raw(frame):
    return frame.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- gate 구조
@pytest.mark.parametrize("anchor,out_dim", [(None, 1), (0.3, 1), (None, 4), (-2.0, 4)])
def test_alpha_in_unit_interval_and_24_outputs(anchor, out_dim):
    torch.manual_seed(0)
    gate = V.GateMLP(5, 8, out_dim, anchor)
    for p in gate.parameters():                       # 극단적인 가중치에서도 범위를 지켜야 한다
        torch.nn.init.normal_(p, std=50.0)
    x = torch.randn(64, 5) * 100
    pa, pc = torch.randn(64, 24), torch.randn(64, 24)
    pred, alpha = gate(x, pa, pc)
    assert pred.shape == (64, 24)
    assert alpha.shape == ((64, 1) if out_dim == 1 else (64, 24))
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0
    assert torch.isfinite(pred).all()


def test_alpha_zero_is_A_and_alpha_one_is_C():
    rng = np.random.default_rng(0)
    a, c = rng.normal(size=(10, 24)), rng.normal(size=(10, 24))
    assert np.array_equal(V.blend(0.0, a, c), a)
    assert np.array_equal(V.blend(1.0, a, c), c)
    for bias, target in ((-1e4, "A"), (1e4, "C")):    # 네트워크 출력이 포화될 때도 동일
        gate = V.GateMLP(3, 8, 1)
        with torch.no_grad():
            gate.net[2].weight.zero_()
            gate.net[2].bias.fill_(bias)
        pred, _ = gate(torch.randn(10, 3), torch.from_numpy(a), torch.from_numpy(c))
        assert torch.equal(pred, torch.from_numpy(a if target == "A" else c))


def test_anchored_gate_starts_at_fixed_weight():
    gate = V.GateMLP(4, 8, 1, anchor_logit=G.logit(0.37))
    alpha = gate.alpha(torch.randn(20, 4) * 10)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.37), atol=1e-6)


def test_fixed_alpha_recovers_known_mixture():
    rng = np.random.default_rng(1)
    a, c = rng.normal(size=(50, 24)), rng.normal(size=(50, 24))
    assert abs(V.fixed_alpha(a, c, 0.3 * a + 0.7 * c) - 0.7) < 1e-9
    assert V.fixed_alpha(a, c, 3 * c - 2 * a) == 1.0          # [0,1] 로 잘림


# ---------------------------------------------------------------- 특징의 인과성
def test_previous_day_volatility_excludes_target_day(raw):
    starts = np.arange(168, 168 + 24 * 40, 24)
    feats = V.gate_features(raw, starts)
    for i, t in enumerate(starts):
        assert np.isclose(feats["vol_std"][i], raw[t - 24:t, 0].std())
        assert not np.isclose(feats["vol_std"][i], raw[t:t + 24, 0].std())
    changed = raw.copy()
    for t in starts:
        changed[t:t + 24, 0] = 0.0                    # 목표일 실제 가격을 지워도 그날의 특징은 그대로
        one = V.gate_features(changed, [t]).iloc[0]
        assert np.allclose(one.to_numpy(), feats.iloc[(t - 168) // 24].to_numpy())
        changed[t:t + 24, 0] = raw[t:t + 24, 0]


def test_future_rows_do_not_change_past_features(raw):
    starts = np.arange(168, len(raw), 24)
    base = V.gate_features(raw, starts)
    rng = np.random.default_rng(0)
    for cut in rng.choice(starts[10:-10], size=5, replace=False):
        future = raw.copy()
        future[cut:] = future[cut:] * 7.0 + 123.0     # cut 이후의 모든 열(가격·예보)을 변경
        past = starts < cut                           # cut 보다 앞선 목표일
        assert np.array_equal(V.gate_features(future, starts[past]).to_numpy(), base[past].to_numpy())
        price_only = raw.copy()
        price_only[cut:, 0] = -999.0                  # 목표일(cut) 이후 '가격'만 바꾸면 cut 날의 특징도 불변
        k = int(np.flatnonzero(starts == cut)[0])
        assert np.array_equal(V.gate_features(price_only, [cut]).to_numpy()[0], base.iloc[k].to_numpy())


def test_dispersion_trend_oscillation_are_distinct():
    """같은 표준편차라도 추세형/진동형 하루가 서로 다른 특징 값을 갖는다."""
    base = np.full(144, 30.0)
    ramp = np.linspace(20, 40, 24)                    # 단조 상승(추세)
    saw = np.where(np.arange(24) % 2 == 0, 1.0, -1.0) * ramp.std() + 30  # 같은 std 의 톱니(진동)
    feats = {}
    for name, day in (("ramp", ramp), ("saw", saw)):
        raw = np.column_stack([np.concatenate([base, day, np.zeros(24)]), np.ones(192), np.ones(192)])
        feats[name] = V.gate_features(raw, [168]).iloc[0]
    assert np.isclose(feats["ramp"]["vol_std"], feats["saw"]["vol_std"])
    assert feats["ramp"]["trend_intra"] > 15 and abs(feats["saw"]["trend_intra"]) < 15
    assert np.isclose(feats["ramp"]["osc_chop"], 1.0, atol=1e-4) and feats["saw"]["osc_chop"] > 20


# ---------------------------------------------------------------- 분할·전처리
def test_window_trim0_equals_original_loader():
    (tr, va, te), info, dates, truth, _ = A.load_market("PJM", 364, 364)
    w = V.load_window(0)
    for ds_old, ds_new in zip((tr, va, te), w["datasets"]):
        assert all(torch.equal(x, y) for x, y in zip(ds_old.tensors, ds_new.tensors))
    assert info["mean"] == w["info"]["mean"] and info["std"] == w["info"]["std"]
    assert info["splits"] == w["info"]["splits"] and dates.equals(w["test_dates"])
    assert np.array_equal(truth, w["truth"])


@pytest.mark.parametrize("trim", [0, 364, 728, 1092])
def test_scaler_uses_training_hours_only(trim, tmp_path, raw, frame):
    w = V.load_window(trim)
    val_start = len(w["raw"]) - 2 * 364 * 24
    assert np.allclose(w["info"]["mean"], raw[:val_start].mean(0))
    assert np.allclose(w["info"]["std"], raw[:val_start].std(0))
    s = w["info"]["splits"]
    assert s["train"]["last_target"] < s["val"]["first_target"] <= s["val"]["last_target"] < s["test"]["first_target"]
    # 검증·테스트 구간 값을 바꿔도 표준화 통계와 train 텐서는 변하지 않는다
    other = frame.copy()
    other.iloc[val_start:] = other.iloc[val_start:].to_numpy() * 5.0 + 50.0
    path = tmp_path / "PJM.csv"
    other.to_csv(path)
    w2 = V.load_window(trim, csv=path)
    assert w2["info"]["mean"] == w["info"]["mean"] and w2["info"]["std"] == w["info"]["std"]
    assert all(torch.equal(x, y) for x, y in zip(w["datasets"][0].tensors, w2["datasets"][0].tensors))


def test_oof_blocks_are_consecutive_and_disjoint():
    blocks = [V.load_window(t)["info"]["splits"] for t in (1092, 728, 364, 0)]
    for earlier, later in zip(blocks, blocks[1:]):
        assert pd.Timestamp(earlier["test"]["last_target"]) + pd.Timedelta(days=1) == pd.Timestamp(later["test"]["first_target"])
        assert earlier["test"] == later["val"]        # 한 origin 의 block = 다음 origin 의 val 구간
    assert [V.origin_name(t) for t in (1092, 728, 364, 0)] == ["o2015", "o2016", "o2017", "o2018"]


# ---------------------------------------------------------------- 저장된 A·C 결과
@pytest.mark.skipif(not ABC_RUNS, reason="results/abc 없음")
def test_saved_predictions_align_and_match_reported_metrics(frame, raw):
    run = ABC_RUNS[-1]
    reported = pd.read_csv(run / "metrics_all_seeds.csv")
    dates = None
    for seed in (42, 43, 44):
        f = pd.read_csv(run / "PJM" / f"seed_{seed}" / "predictions.csv", parse_dates=["Date"])
        b = G.check_frame(f, raw, frame.index, "saved")            # 시각 연속성, actual == PJM.csv, 결측 없음
        assert len(b["dates"]) == 364 and str(b["dates"][0].date()) == "2017-12-26"
        assert dates is None or b["dates"].equals(dates)           # seed 간 날짜 동일
        dates = b["dates"]
        saved = json.loads((run / "PJM" / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
        for name in ("A", "C"):
            rep = reported[(reported.market == "PJM") & (reported.seed == seed) & (reported.model == name)].iloc[0]
            assert np.isclose(V.mae(b["truth"], b[name]), rep.MAE, rtol=0, atol=1e-9)
            assert np.isclose(V.rmse(b["truth"], b[name]), rep.RMSE, rtol=0, atol=1e-9)
            assert np.isclose(V.mae(b["truth"], b[name]), saved[name]["MAE"], rtol=0, atol=1e-9)


def test_check_frame_rejects_misaligned_rows(frame, raw):
    t = 168 + 24 * 100
    good = pd.DataFrame({"Date": frame.index[t:t + 48], "actual": raw[t:t + 48, 0], "A": 1.0, "C": 2.0})
    G.check_frame(good, raw, frame.index)
    shifted = good.assign(actual=np.roll(good["actual"].to_numpy(), 1))
    gap = good.drop(index=5).iloc[:24]
    late = good.assign(Date=good["Date"] + pd.Timedelta(hours=1))
    nan = good.assign(C=np.nan)
    for bad in (shifted, gap, late, nan):
        with pytest.raises(ValueError):
            G.check_frame(bad, raw, frame.index)


# ---------------------------------------------------------------- 학습 과정의 누수
def _fake_oof(folder, frame, seed=42, bump_from=None):
    """원본 가격 + 잡음으로 만든 가짜 A·C OOF. bump_from 이후(=test 블록) 예측만 다르게 만들 수 있다."""
    raw = frame.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(0)
    for trim in (1092, 728, 364, 0):
        w = V.load_window(trim)
        name = V.origin_name(trim)
        pos = (w["starts"][w["masks"][2]][:, None] + np.arange(24)).ravel()
        y = raw[pos, 0]
        a, c = y + rng.normal(0, 4, len(y)), y + rng.normal(0, 3, len(y))
        if bump_from is not None and pos[0] >= bump_from:
            a, c = a * 3 + 100, c * -2 - 50
        d = folder / name / f"seed_{seed}"
        d.mkdir(parents=True)
        pd.DataFrame({"Date": frame.index[pos], "actual": y, "A": a, "C": c}).to_csv(d / "predictions.csv", index=False)
        (d / "metrics.json").write_text(json.dumps({"scaling": {"mean": w["info"]["mean"], "std": w["info"]["std"]}}))
    return ["o2015", "o2016", "o2017", "o2018"]


def test_gate_training_ignores_test_block_and_future_rows(tmp_path, frame, monkeypatch):
    """test 블록의 A·C 예측과 그 이후 원본 행을 바꿔도 학습된 gate(가중치·epoch·표준화·고정가중치)가 같아야 한다."""
    gates = {"gate_struct": G.MAIN_GATES["gate_struct"], "gate_anchor": G.MAIN_GATES["gate_anchor"]}
    test_start = V.load_window(0)["starts"][V.load_window(0)["masks"][2]][0]
    seen = []
    real = V.fit_scaler
    monkeypatch.setattr(V, "fit_scaler", lambda x: (seen.append(len(x)), real(x))[1])

    order = _fake_oof(tmp_path / "oof1", frame)
    G.run_origin("o2018", order, 42, CFG, gates, frame, tmp_path / "out1", tmp_path / "oof1", None)
    assert set(seen) == {364 * 2, 364 * 3}            # 표준화 통계: gate_train(2블록), gate_train+val(3블록)만

    _fake_oof(tmp_path / "oof2", frame, bump_from=test_start)      # test 블록 예측을 전혀 다른 값으로
    future = frame.copy()                                          # test 블록의 수요예보 열도 변경(가격은 정합성 검사 대상)
    future.iloc[test_start:, 1:] = future.iloc[test_start:, 1:].to_numpy() * 2.0
    G.run_origin("o2018", order, 42, CFG, gates, future, tmp_path / "out2", tmp_path / "oof2", None)

    for g in gates:
        p1, p2 = (tmp_path / o / "gates" / g / "o2018" / "seed_42" for o in ("out1", "out2"))
        s1, s2 = torch.load(p1 / "gate.pt", weights_only=True), torch.load(p2 / "gate.pt", weights_only=True)
        assert all(torch.equal(s1[k], s2[k]) for k in s1)
        assert (p1 / "gate.json").read_text() == (p2 / "gate.json").read_text()
        assert (p1 / "history.csv").read_text() == (p2 / "history.csv").read_text()


def test_gate_needs_two_earlier_blocks(tmp_path, frame):
    order = _fake_oof(tmp_path / "oof", frame)
    with pytest.raises(SystemExit):                    # o2016 앞에는 블록이 하나뿐 -> train/val 분리가 불가능
        G.run_origin("o2016", order, 42, CFG, {"gate_vol": G.MAIN_GATES["gate_vol"]}, frame, tmp_path / "o", tmp_path / "oof", None)
