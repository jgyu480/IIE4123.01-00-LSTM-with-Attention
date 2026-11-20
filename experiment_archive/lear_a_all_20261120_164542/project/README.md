# PJM 하루 전 전기요금 예측

PJM 시장의 다음 날 24시간 전기요금을 예측하는 팀 프로젝트다.
최근 7일의 시계열과 목표일의 예보·요일 정보를 사용해 세 모델을 비교하고,
고가격 시간대와 급등·급락 구간의 오차를 분석했다.

## 비교 모델

| 모델 | 예측 방식 |
| --- | --- |
| A · LSTM | 최근 7일의 시계열을 처리한 뒤 목표일 정보와 결합해 가격 24개를 예측한다. |
| B · 유사일 평균 | 목표일과 비슷한 과거 날짜의 실제 가격곡선을 평균한다. |
| C · 사례 어텐션 | 목표일에 따라 과거 사례별 가중치를 계산해 예측에 반영한다. |

## 프로젝트 구성

| 구분 | 위치 |
| --- | --- |
| 학습 코드 | [기준 모델](PJM_error_analysis/train_01_baseline.py) · [모델 B·C](PJM_error_analysis/train_02_bc.py) · [추가 실험](PJM_error_analysis/train_03_candidates.py) |
| 입력 데이터 | [PJM 시간별 데이터](PJM_error_analysis/data/PJM.csv) |
| 실험 결과 | [저장된 예측과 지표](PJM_error_analysis/results/) |
| 결과 해석 | [모델별 성능 비교와 오차 분석](docs/PJM_error_analysis/README.md) |
| 실행 환경 | [필요 패키지](requirements_baseline.txt) |

## 기준 모델 실행

저장소 루트에서 실행한다. `--check-only`로 데이터와 분할을 먼저 확인한다.

    python3 -m pip install -r requirements_baseline.txt
    python3 PJM_error_analysis/train_01_baseline.py --markets PJM --check-only
    python3 PJM_error_analysis/train_01_baseline.py --markets PJM

학습 결과는 `PJM_error_analysis/results/baseline/`에 저장된다.
