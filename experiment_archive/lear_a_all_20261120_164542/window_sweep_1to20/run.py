"""Compare 1–20-week lookbacks for weekly MAE-optimal A+LEAR weights."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = tuple(range(1, 21))


def find_source(here):
    suffix = Path("results/step1/split2_2018/export")
    for parent in (here, *here.parents):
        for candidate in (
            parent / suffix,
            parent / "project/adaptive-weighted-LSTM-Attention" / suffix,
        ):
            if all(
                (candidate / f"predictions_{s}.csv").is_file()
                for s in ("val", "test")
            ):
                return candidate
    raise FileNotFoundError(
        "入力を見つけられません。--source に export フォルダを指定してください。"
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    frame = pd.read_csv(path)
    required = ["Date", "actual", "A", "LEAR"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: 없는 열 {sorted(missing)}")

    frame = frame[required].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.sort_values("Date").reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"빈 입력: {path}")
    if frame["Date"].duplicated().any():
        raise ValueError(f"중복 시간: {path}")
    if not np.isfinite(frame[["actual", "A", "LEAR"]].to_numpy(float)).all():
        raise ValueError(f"결측값 또는 무한값: {path}")
    if not frame["Date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"누락되거나 연속되지 않은 시간: {path}")
    return frame


def complete(frame, start, hours):
    expected = pd.date_range(start, periods=hours, freq="h")
    return np.array_equal(
        frame["Date"].to_numpy(), expected.to_numpy()
    )


def optimal_mae_weight(frame):
    """Exact constrained absolute-error minimizer via weighted median."""
    y, a, l = (
        frame[c].to_numpy(float) for c in ("actual", "A", "LEAR")
    )
    delta = a-l
    active = np.abs(delta) > 1e-12

    if not active.any():
        return 0.5

    crossing = (y[active]-l[active])/delta[active]
    mass = np.abs(delta[active])
    order = np.argsort(crossing)
    index = np.searchsorted(
        np.cumsum(mass[order]), mass.sum()/2, side="left"
    )
    return float(np.clip(crossing[order][index], 0, 1))


def predict(frame, weight):
    return (
        weight*frame["A"].to_numpy(float)
        + (1-weight)*frame["LEAR"].to_numpy(float)
    )


def metrics(actual, prediction):
    error = np.asarray(prediction)-np.asarray(actual)
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, default=here)
    args = parser.parse_args()

    source = args.source.resolve() if args.source else find_source(here)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    guard = (out/"run.lock").open("w")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("같은 폴더에서 이미 분석 중입니다.")

    paths = {
        split: source/f"predictions_{split}.csv"
        for split in ("val", "test")
    }
    input_hashes = {s: sha256(p) for s, p in paths.items()}
    val, test = (load(paths[s]) for s in ("val", "test"))

    if val["Date"].max() >= test["Date"].min():
        raise ValueError("검증·테스트 기간이 겹칩니다.")

    history = pd.concat([val, test], ignore_index=True)
    history = history.sort_values("Date").reset_index(drop=True)

    if not history["Date"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("검증과 테스트 사이에 시간 누락이 있습니다.")

    test["week_start"] = (
        test["Date"].dt.to_period("W-SUN").dt.start_time
    )

    weekly_rows = []
    prediction_blocks = []
    excluded = []

    for start, target in test.groupby("week_start", sort=True):
        if not complete(target, start, 168):
            excluded.append({
                "week_start": str(start.date()),
                "reason": "incomplete_test_week",
            })
            continue

        # 모든 기간을 정확히 같은 주에서 비교한다.
        longest_begin = start-pd.Timedelta(weeks=max(WINDOWS))
        longest = history[
            (history["Date"] >= longest_begin)
            & (history["Date"] < start)
        ]
        if not complete(longest, longest_begin, max(WINDOWS)*168):
            excluded.append({
                "week_start": str(start.date()),
                "reason": "insufficient_20_week_history",
            })
            continue

        block = target[["Date", "actual", "A", "LEAR"]].copy()
        block["week_start"] = start

        for weeks in WINDOWS:
            begin = start-pd.Timedelta(weeks=weeks)
            past = history[
                (history["Date"] >= begin)
                & (history["Date"] < start)
            ]
            if not complete(past, begin, weeks*168):
                raise ValueError(f"{start}: {weeks}주 과거 자료 누락")
            if past["Date"].max() >= target["Date"].min():
                raise ValueError("미래 정답이 비율 계산에 포함되었습니다.")

            weight = optimal_mae_weight(past)
            pred = predict(target, weight)
            block[f"pred_{weeks:02d}w"] = pred

            weekly_rows.append({
                "week_start": str(start.date()),
                "lookback_weeks": weeks,
                "past_start": str(past["Date"].min()),
                "past_end": str(past["Date"].max()),
                "past_hours": len(past),
                "A_weight": weight,
                "LEAR_weight": 1-weight,
                **metrics(target["actual"], pred),
            })

        prediction_blocks.append(block)

    if not prediction_blocks:
        raise ValueError("모든 기간을 함께 평가할 완전한 주가 없습니다.")

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    weekly = pd.DataFrame(weekly_rows)
    n_weeks = weekly["week_start"].nunique()

    if len(predictions) != n_weeks*168:
        raise AssertionError("평가 시간 수 불일치")
    if not weekly.groupby("lookback_weeks").size().eq(n_weeks).all():
        raise AssertionError("방법별 평가 주 수 불일치")

    summary_rows = []
    for weeks in WINDOWS:
        by_week = weekly[weekly["lookback_weeks"] == weeks]
        summary_rows.append({
            "lookback_weeks": weeks,
            **metrics(
                predictions["actual"], predictions[f"pred_{weeks:02d}w"]
            ),
            "mean_A_weight": float(by_week["A_weight"].mean()),
            "std_A_weight": float(by_week["A_weight"].std(ddof=0)),
            "evaluation_weeks": n_weeks,
            "evaluation_hours": len(predictions),
        })

    scores = pd.DataFrame(summary_rows)
    reference = scores.set_index("lookback_weeks").loc[20]
    for name in ("MAE", "RMSE"):
        scores[f"{name}_gain_vs_20w_pct"] = (
            100*(reference[name]-scores[name])/reference[name]
        )
        scores[f"{name}_rank"] = scores[name].rank(
            method="min"
        ).astype(int)

    # 결과 확인용 순위이며, 평가 후 선택한 최적 기간이다.
    ranking = scores.sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    best_mae = scores.loc[scores["MAE"].idxmin()]
    best_rmse = scores.loc[scores["RMSE"].idxmin()]

    for split, path in paths.items():
        if sha256(path) != input_hashes[split]:
            raise RuntimeError(f"실행 중 입력 파일 변경: {path}")

    tables = {
        "scores": scores,
        "ranking_by_MAE": ranking,
        "weekly_results": weekly,
        "predictions": predictions,
        "weekly_A_weights": weekly.pivot(
            index="week_start", columns="lookback_weeks", values="A_weight"
        ).reset_index(),
        "excluded_weeks": pd.DataFrame(
            excluded, columns=["week_start", "reason"]
        ),
    }
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)

    settings = {
        "lookback_weeks": list(WINDOWS),
        "weight_objective": "MAE",
        "forecast": "w*A+(1-w)*LEAR; 0<=w<=1",
        "weight_scope": "One scalar per method per week; all 24 hours",
        "update": "Monday; fixed for following week",
        "information": "Actuals strictly before evaluated week only",
        "same_evaluation_dates_for_all_methods": True,
        "evaluation_weeks": int(n_weeks),
        "evaluation_hours": len(predictions),
        "start": str(predictions["Date"].min()),
        "end": str(predictions["Date"].max()),
        "input_sha256": input_hashes,
        "best_MAE_lookback_exploratory": int(best_mae["lookback_weeks"]),
        "best_RMSE_lookback_exploratory": int(best_rmse["lookback_weeks"]),
        "warning": (
            "Exploratory test sweep after earlier comparisons; "
            "not independent validation of the best lookback or BOA."
        ),
    }
    (out/"settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 과거 참조 기간 1~20주에 따른 A+LEAR 결합 성능",
        "",
        "## 분석 목적",
        "",
        "결합 비율을 산출할 때 과거 참조 기간이 예측 성능에 미치는 영향을 탐색한다.",
        "기존 모델을 재학습하지 않고 저장된 A·LEAR 예측을 이용한다.",
        "",
        "## 통제한 조건",
        "",
        "- 참조 기간: 직전 1~20주. 이 조건만 변경한다.",
        "- 매주 월요일 과거 MAE가 최소인 비율을 구해 다음 1주에 적용한다.",
        "- 결합식: w*A+(1-w)*LEAR, 0<=w<=1.",
        "- 시간대 구분 없이 한 주에 하나의 비율을 사용한다.",
        "- 평가 주의 정답은 해당 주 비율 산출에 사용하지 않는다.",
        "- 20주 과거가 완전한 주만 포함하여 모든 방법을 동일 날짜에서 비교한다.",
        "- RMSE는 MAE로 선택한 비율의 보조 평가 지표이다.",
        "",
        "## 평가 범위",
        "",
        f"- {settings['start']} ~ {settings['end']}",
        f"- {n_weeks}주 / {len(predictions):,}시간",
        "",
        "## 전체 결과",
        "",
        "| 참조 기간 | MAE ↓ | RMSE ↓ | 평균 A 비율 |",
        "|---|---:|---:|---:|",
    ]
    for row in scores.to_dict("records"):
        lines.append(
            f"| {int(row['lookback_weeks'])}주 | "
            f"{row['MAE']:.6f} | {row['RMSE']:.6f} | "
            f"{100*row['mean_A_weight']:.2f}% |"
        )
    lines += [
        "",
        f"- 평가 후 확인한 최소 MAE 기간: {int(best_mae['lookback_weeks'])}주.",
        f"- 평가 후 확인한 최소 RMSE 기간: {int(best_rmse['lookback_weeks'])}주.",
        "",
        "## 해석의 범위",
        "",
        "짧은 기간은 최근 변화에 빠르게 반응하지만 추정이 불안정할 수 있고,",
        "긴 기간은 더 많은 자료를 쓰지만 오래된 성능을 반영할 수 있다.",
        "구체적인 경향은 전체 결과표와 주별 결과를 함께 보고 판단한다.",
        "",
        "이 분석은 이미 확인한 테스트에서 수행한 20개 기간의 추가 탐색이다.",
        "여기서 가장 좋은 기간을 골랐다고 일반화 성능이 검증된 것은 아니다.",
        "통계적 유의성이나 BOA 자체의 우수성도 이 분석으로 입증하지 않는다.",
        "과거 전체를 누적하는 BOA와 짧은 기간만 사용하는 방식도 구분해야 한다.",
        "",
        "## 파일",
        "",
        "- scores.csv: 기간 순 전체 성능 및 20주 대비 개선율",
        "- ranking_by_MAE.csv: MAE 순 결과",
        "- weekly_results.csv: 기간별·주별 비율과 성능",
        "- weekly_A_weights.csv: 주별 A 비율 비교",
        "- predictions.csv: 동일 시간에서 모든 방법의 예측",
        "- excluded_weeks.csv: 평가에서 제외한 주와 사유",
        "- settings.json: 실험 조건·날짜·입력 SHA-256",
        "- run.py / run.log: 실행 코드와 로그",
        "",
        "## 재실행",
        "",
        "필요 패키지: numpy, pandas.",
        "저장소 루트에서 다음 명령으로 실행할 수 있다.",
        "",
        "```bash",
        "python3 experiment_archive/lear_a_all_20261120_164542/"
        "window_sweep_1to20/run.py",
        "```",
        "",
        "원래 프로젝트 또는 아카이브 project/의 예측 CSV를 자동으로 찾는다.",
        "직접 지정하려면 --source /path/to/export --out /path/to/output을 사용한다.",
        "",
        "이 추가 분석은 상위 원본 보관 manifest 집계와 별도로 관리한다.",
        "",
    ]
    (out/"README.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n[실험 설정]")
    print("직전 1~20주 MAE로 비율 선택 → 다음 1주 적용")
    print("모든 방법이 동일한 날짜에서 평가됨")
    print(f"{settings['start']} ~ {settings['end']}")
    print(f"{n_weeks}주 / {len(predictions):,}시간")

    print("\n[참조 기간 순 전체 결과]")
    print(scores.to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))

    print("\n[탐색적 최소 오차]")
    print(
        f"MAE: {int(best_mae['lookback_weeks'])}주, "
        f"{best_mae['MAE']:.6f}"
    )
    print(
        f"RMSE: {int(best_rmse['lookback_weeks'])}주, "
        f"{best_rmse['RMSE']:.6f}"
    )
    print("최적 기간의 일반화·통계적 유의성·BOA 효과를 입증한 것은 아닙니다.")
    print(f"\n완료. 결과 위치: {out}")


if __name__ == "__main__":
    main()
