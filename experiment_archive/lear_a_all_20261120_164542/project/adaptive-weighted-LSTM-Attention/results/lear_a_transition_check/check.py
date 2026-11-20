from pathlib import Path

# Reuse only the verified data-loading section of the previous diagnosis.
loader_path = Path("results/lear_a_diagnosis/analyze.py").resolve()
if not loader_path.is_file():
    raise SystemExit(f"기존 진단 코드가 없습니다: {loader_path}")
source = loader_path.read_text()
marker = "# 1. Error concentration"
if marker not in source:
    raise SystemExit("기존 진단 코드의 데이터 읽기 구간을 찾지 못했습니다.")

namespace = {"__file__": str(loader_path), "__name__": "__data_loader__"}
exec(compile(source.split(marker, 1)[0], str(loader_path), "exec"), namespace)

import json
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("error", category=ConvergenceWarning)

Y, A, L, B, H, F = [namespace[k] for k in ["Y", "A", "L", "B", "H", "F"]]
dates, parts, frame = [namespace[k] for k in ["dates", "parts", "frame"]]
daily = namespace["daily"]
OUT = Path("results/lear_a_transition_check").resolve()
OUT.mkdir(parents=True, exist_ok=True)
notes = []

def say(message):
    print(message, flush=True)
    notes.append(str(message))

def save(name, rows):
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(OUT / name, index=False)
    return table

val = np.flatnonzero(parts == "val")
test = np.flatnonzero(parts == "test")

# Seven days of previous model errors must be available.
val = val[val >= 7]
nfit = int(len(val) * 0.60)
ntune = int(len(val) * 0.20)
fit = val[:nfit]
tune = val[nfit:nfit+ntune]
hold = val[nfit+ntune:]

previous = H[:, -24:, 0]
prior_mean = previous.mean(axis=1)
actual_mean = Y.mean(axis=1)
change = actual_mean-prior_mean

# Labels describe outcomes, not forecast-time inputs.
# Thresholds are estimated on fit days only and frozen.
low_change = min(float(np.quantile(change[fit], 0.10)), -1e-6)
high_change = max(float(np.quantile(change[fit], 0.90)), 1e-6)
high_price = float(np.quantile(prior_mean[fit], 0.90))

transition = np.where(
    change < low_change, "drop",
    np.where(change > high_change, "rise", "middle")
)
high_previous = prior_mean > high_price

mse_a = np.mean((A-Y)**2, axis=1)
mse_l = np.mean((L-Y)**2, axis=1)
winner = np.where(mse_a < mse_l, "A", "LEAR")

say("\n[설정]")
say(
    f"학습 {len(fit)}일 / 설정선택 {len(tune)}일 / "
    f"검증 홀드아웃 {len(hold)}일 / 테스트 {len(test)}일"
)
say(
    f"하락 기준: 하루 평균 변화 < {low_change:.3f}, "
    f"상승 기준: > {high_change:.3f}, "
    f"전날 고가격 기준: > {high_price:.3f}"
)
say("위 기준은 모두 학습 구간에서 정했으며 테스트에서 변경하지 않습니다.")

# 1. What actually happens in high-price periods?
event_rows = []
for split_name, ids in [("validation_holdout", hold), ("test", test)]:
    for state in ["all", "high_previous"]:
        active = ids if state == "all" else ids[high_previous[ids]]
        for label in ["drop", "middle", "rise"]:
            selected = active[transition[active] == label]
            if not len(selected):
                continue
            row = {
                "split": split_name, "state": state,
                "outcome": label, "days": len(selected),
                "A_MAE": np.abs(A[selected]-Y[selected]).mean(),
                "LEAR_MAE": np.abs(L[selected]-Y[selected]).mean(),
                "A_better_day_pct": 100*np.mean(winner[selected] == "A"),
            }
            if np.isfinite(B[selected]).all():
                row["Recent_MAE"] = np.abs(B[selected]-Y[selected]).mean()
                row["Recent_bias"] = np.mean(B[selected]-Y[selected])
            event_rows.append(row)
event_table = save("01_transition_groups.csv", event_rows)
say("\n[실제 전환 유형별 성능]")
say(event_table.to_string(index=False))

# 2. Forecast-time feature groups.
price_features = {
    "prev_mean": prior_mean,
    "prev_std": previous.std(axis=1),
    "prev_day_change": prior_mean-H[:, -48:-24, 0].mean(axis=1),
    "prev_vs_week": prior_mean-H[:, :, 0].mean(axis=1),
    "last6_vs_prev_mean": H[:, -6:, 0].mean(axis=1)-prior_mean,
    "last6_vs_first6": H[:, -6:, 0].mean(axis=1)-H[:, -24:-18, 0].mean(axis=1),
    "weekday_sin": np.sin(2*np.pi*dates.dayofweek.to_numpy()/7),
    "weekday_cos": np.cos(2*np.pi*dates.dayofweek.to_numpy()/7),
}

load_features = {}
for channel, name in [(0, "system"), (1, "zonal")]:
    curve = F[:, :, channel]
    old = H[:, -24:, channel+1]
    load_features.update({
        f"{name}_mean": curve.mean(axis=1),
        f"{name}_day_change": (curve-old).mean(axis=1),
        f"{name}_evening_change": (curve[:, 16:21]-old[:, 16:21]).mean(axis=1),
        f"{name}_max_ramp": np.diff(curve, axis=1).max(axis=1),
        f"{name}_min_ramp": np.diff(curve, axis=1).min(axis=1),
        f"{name}_evening_vs_daily": curve[:, 16:21].mean(axis=1)-curve.mean(axis=1),
    })

model_features = {
    "A_mean_minus_previous": A.mean(axis=1)-prior_mean,
    "LEAR_mean_minus_previous": L.mean(axis=1)-prior_mean,
    "A_minus_LEAR_mean": (A-L).mean(axis=1),
    "A_LEAR_abs_gap": np.abs(A-L).mean(axis=1),
    "A_LEAR_evening_gap": (A[:, 16:21]-L[:, 16:21]).mean(axis=1),
    "A_evening_vs_daily": A[:, 16:21].mean(axis=1)-A.mean(axis=1),
    "LEAR_evening_vs_daily": L[:, 16:21].mean(axis=1)-L.mean(axis=1),
}
window_names = ["LEAR_w56", "LEAR_w84", "LEAR_w1092", "LEAR_w1456"]
if all(name in frame for name in window_names):
    windows = [daily(name) for name in window_names]
    short = (windows[0]+windows[1])/2
    long = (windows[2]+windows[3])/2
    model_features.update({
        "short_minus_long_mean": (short-long).mean(axis=1),
        "short_long_abs_gap": np.abs(short-long).mean(axis=1),
        "short_minus_previous": short.mean(axis=1)-prior_mean,
        "long_minus_previous": long.mean(axis=1)-prior_mean,
    })

error_features = {}
for name, prediction in [("A", A), ("LEAR", L)]:
    bias = (prediction-Y).mean(axis=1)
    daily_mse = np.mean((prediction-Y)**2, axis=1)
    # Shift first: target-day actuals never enter their own inputs.
    for days in [3, 7]:
        error_features[f"{name}_bias_last{days}"] = (
            pd.Series(bias).shift(1).rolling(days, min_periods=days).mean().to_numpy()
        )
        error_features[f"{name}_mse_last{days}"] = (
            pd.Series(daily_mse).shift(1).rolling(days, min_periods=days).mean().to_numpy()
        )

feature_groups = {
    "price_history": price_features,
    "price_plus_load": {**price_features, **load_features},
    "price_load_models": {
        **price_features, **load_features, **model_features,
    },
    "all_plus_recent_errors": {
        **price_features, **load_features, **model_features, **error_features,
    },
}
all_features = pd.DataFrame(feature_groups["all_plus_recent_errors"])
if not np.isfinite(all_features.iloc[np.r_[fit, tune, hold, test]].to_numpy()).all():
    raise ValueError("입력 특징에 NaN/Inf가 있습니다.")

# 3. Learnability checks.
# Task A: predict next-day drop/middle/rise.
# Task B: predict whether A or LEAR has smaller daily MSE.
tasks = {"transition": transition, "model_winner": winner}
scores = []
confusions = []
settings = {}
test_predictions = {}

def classification_scores(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_F1": f1_score(actual, predicted, zero_division=0, average="macro"),
    }

for task, target in tasks.items():
    labels = np.unique(target[fit])
    if len(labels) < 2:
        say(f"{task}: 학습 구간에 클래스가 하나뿐이라 건너뜁니다.")
        continue

    settings[task] = {}
    for feature_name, columns in feature_groups.items():
        names = list(columns)
        x = pd.DataFrame(columns).to_numpy()
        best = None
        for strength in [0.001, 0.01, 0.1, 1.0]:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=strength, class_weight="balanced",
                    max_iter=5000, random_state=42,
                ),
            )
            model.fit(x[fit], target[fit])
            predicted = model.predict(x[tune])
            score = f1_score(
                target[tune], predicted, average="macro", zero_division=0
            )
            rank = (-score, strength)
            if best is None or rank < best[0]:
                best = (rank, strength)

        strength = best[1]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=strength, class_weight="balanced",
                max_iter=5000, random_state=42,
            ),
        )
        # Freeze before the independent validation holdout and test.
        refit = np.r_[fit, tune]
        model.fit(x[refit], target[refit])
        settings[task][feature_name] = {"C": strength, "features": names}

        for split_name, ids in [("validation_holdout", hold), ("test", test)]:
            prediction = model.predict(x[ids])
            row = {
                "task": task, "features": feature_name, "split": split_name,
                **classification_scores(target[ids], prediction),
            }
            if task == "model_winner":
                class_index = list(model.classes_).index("A")
                probability = model.predict_proba(x[ids])[:, class_index]
                binary = (target[ids] == "A").astype(int)
                row["AUC_A_wins"] = (
                    roc_auc_score(binary, probability)
                    if len(np.unique(binary)) == 2 else np.nan
                )
            scores.append(row)
            matrix = confusion_matrix(target[ids], prediction, labels=labels)
            for i, truth_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    confusions.append({
                        "task": task, "features": feature_name,
                        "split": split_name, "actual": truth_label,
                        "predicted": pred_label, "count": matrix[i, j],
                    })
            if split_name == "test":
                test_predictions[f"{task}__{feature_name}"] = prediction

        coefficients = model.named_steps["logisticregression"].coef_
        save(
            f"coefficients_{task}_{feature_name}.csv",
            [
                {"row": i, "feature": name, "coefficient": coefficients[i, j]}
                for i in range(coefficients.shape[0])
                for j, name in enumerate(names)
            ],
        )

    # Two reference classifiers.
    x_dummy = np.zeros((len(Y), 1))
    for strategy in ["most_frequent", "stratified"]:
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        refit = np.r_[fit, tune]
        dummy.fit(x_dummy[refit], target[refit])
        for split_name, ids in [("validation_holdout", hold), ("test", test)]:
            scores.append({
                "task": task, "features": f"dummy_{strategy}",
                "split": split_name,
                **classification_scores(target[ids], dummy.predict(x_dummy[ids])),
            })

score_table = save("02_predictability.csv", scores)
save("03_confusion_matrices.csv", confusions)
say("\n[입력으로 전환 / 모델 우열을 구분할 수 있는가]")
say(score_table.to_string(index=False))

# 4. High-price drop vs other outcomes: descriptive feature differences.
# Useful only when the pattern appears beyond one exceptional episode.
difference_rows = []
feature_sd = all_features.iloc[fit].std(ddof=0).clip(lower=1e-6)
for split_name, ids in [("validation_holdout", hold), ("test", test)]:
    high = ids[high_previous[ids]]
    falling = high[transition[high] == "drop"]
    other = high[transition[high] != "drop"]
    say(
        f"\n{split_name}: 전날 고가격 {len(high)}일, "
        f"이후 급락 {len(falling)}일, 그 외 {len(other)}일"
    )
    if len(falling) >= 2 and len(other) >= 2:
        delta = (
            all_features.iloc[falling].mean()
            - all_features.iloc[other].mean()
        )/feature_sd
        for feature in delta.index:
            difference_rows.append({
                "split": split_name, "feature": feature,
                "standardized_drop_minus_other": delta[feature],
                "drop_days": len(falling), "other_days": len(other),
            })
differences = save("04_high_price_drop_features.csv", difference_rows)
if not differences.empty:
    say("\n[고가격 이후 급락일의 특징: 절댓값 상위 8개]")
    for split_name in ["validation_holdout", "test"]:
        selected = differences[differences["split"] == split_name].copy()
        selected["absolute"] = selected["standardized_drop_minus_other"].abs()
        say(
            selected.sort_values("absolute", ascending=False)
            .head(8).drop(columns="absolute").to_string(index=False)
        )

# 5. Attach outcomes, known signals, predictions to the worst dates.
daily_mse = np.mean((B[test]-Y[test])**2, axis=1)
order = np.argsort(daily_mse)[::-1][:12]
case_rows = []
for local in order:
    t = test[local]
    row = {
        "date": str(dates[t].date()),
        "Recent_RMSE": float(np.sqrt(daily_mse[local])),
        "Recent_bias": float(np.mean(B[t]-Y[t])),
        "outcome": transition[t],
        "previous_high": bool(high_previous[t]),
        "actual_day_change": float(change[t]),
        "A_or_LEAR_winner": winner[t],
    }
    for feature in [
        "prev_day_change", "last6_vs_prev_mean",
        "system_day_change", "A_mean_minus_previous",
        "LEAR_mean_minus_previous", "short_minus_long_mean",
    ]:
        if feature in all_features:
            row[feature] = float(all_features.iloc[t][feature])
    for key, prediction in test_predictions.items():
        if key.endswith("all_plus_recent_errors"):
            row[key] = prediction[local]
    case_rows.append(row)
case_table = save("05_worst_day_checks.csv", case_rows)
say("\n[최악의 날짜에서 실제 전환과 사전 신호]")
say(case_table.to_string(index=False))

# Label distributions expose small-sample and class-shift limitations.
distributions = []
for task, target in tasks.items():
    for split_name, ids in [
        ("fit", fit), ("tune", tune),
        ("validation_holdout", hold), ("test", test),
    ]:
        values, counts = np.unique(target[ids], return_counts=True)
        for label, count in zip(values, counts):
            distributions.append({
                "task": task, "split": split_name,
                "label": label, "days": int(count),
                "share": float(count/len(ids)),
            })
save("06_label_counts.csv", distributions)
say("\n[클래스별 날짜 수]")
say(pd.DataFrame(distributions).to_string(index=False))

settings["thresholds"] = {
    "drop": low_change, "rise": high_change, "previous_high": high_price,
}
settings["interpretation"] = (
    "Exploratory diagnostics. Outcome labels use actuals; input features do not. "
    "Predicting the better individual model is different from predicting optimal blend weights."
)
(OUT/"settings.json").write_text(json.dumps(settings, indent=2))
(OUT/"summary.txt").write_text("\n".join(notes), encoding="utf-8")
say(f"\n완료. 결과 위치: {OUT}")
