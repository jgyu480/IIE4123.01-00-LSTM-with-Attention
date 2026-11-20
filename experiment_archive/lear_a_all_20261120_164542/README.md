# LEAR + A 전체 실험 기록

35개 앙상블 실험과 후속 실험·진단을 함께 보관한 기록이다.
원래 로컬 파일을 복사했으며 실험을 다시 실행하지 않았다.

## 폴더 안내

| 실험 | 내용 | 파일 수 | 경로 |
|---|---|---:|---|
| ensemble35_pjm2018 | 7개 모델 조합 × 5개 결합 방식 = 35개 앙상블 실험 | 8 | [ensemble35_pjm2018](project/adaptive-weighted-LSTM-Attention/results/ensemble35_pjm2018/) |
| step1 | A·B·C·LEAR 기본 실험, seed별 체크포인트와 검증·테스트 예측 | 102 | [step1](project/adaptive-weighted-LSTM-Attention/results/step1/) |
| lear_a_context_attention | Hourly·Recent·Attention 비교와 OriginalRecent 결과 | 6 | [lear_a_context_attention](project/adaptive-weighted-LSTM-Attention/results/lear_a_context_attention/) |
| lear_a_case_attention_v2 | KernelCase·LearnedCase·CDelta 사례 기반 결합 | 8 | [lear_a_case_attention_v2](project/adaptive-weighted-LSTM-Attention/results/lear_a_case_attention_v2/) |
| lear_a_neural_attention | MLP·HorizonAttention·WindowAttention·HistoryAttention | 34 | [lear_a_neural_attention](project/adaptive-weighted-LSTM-Attention/results/lear_a_neural_attention/) |
| lear_a_diagnosis | 오차 분해·극단 날짜·oracle·기간별 LEAR·입력 가치 진단 | 18 | [lear_a_diagnosis](project/adaptive-weighted-LSTM-Attention/results/lear_a_diagnosis/) |
| lear_a_transition_check | 가격 전환 및 모델 우위 예측 가능성 점검 | 18 | [lear_a_transition_check](project/adaptive-weighted-LSTM-Attention/results/lear_a_transition_check/) |
| lear_a_transition_attention | 전환·과거 오차 기반 어텐션과 MeanErrorMemory 비교 | 40 | [lear_a_transition_attention](project/adaptive-weighted-LSTM-Attention/results/lear_a_transition_attention/) |
| lear_a_simple_improvements | LevelShape·Reconcile·BiasMix·BOA 온라인 결합 | 19 | [lear_a_simple_improvements](project/adaptive-weighted-LSTM-Attention/results/lear_a_simple_improvements/) |

파일 수가 0인 항목은 업로드 당시 로컬에서 찾지 못한 항목이다.
위 목록 외의 results 하위 파일도 함께 포함했다.

## 35개 앙상블 실험

모델 조합: LEAR+A, LEAR+B, LEAR+C, LEAR+A+B,
LEAR+A+C, LEAR+B+C, LEAR+A+B+C.

각 조합의 결합 방식: equal, static, hourly, volatility, attention.
총 7 × 5 = 35개 설정이며 seed별 학습 횟수와는 구분한다.

## 원래 경로

project/ 아래에 원래 저장소의 상대 경로를 그대로 유지했다.
프로젝트 코드: project/adaptive-weighted-LSTM-Attention/
전체 실험 결과: project/adaptive-weighted-LSTM-Attention/results/
전체 파일 목록·크기·SHA-256: manifest.csv

## 큰 파일 복원

이 README와 같은 폴더에서 다음 명령을 실행한다.

    python3 restore_and_verify.py

90 MiB를 초과하는 파일을 복원하고 모든 파일의 SHA-256을 검사한다.
체크포인트를 사용하기 전 이 명령을 실행하면 된다.

## 보관 범위

- 파일 수: 349
- 저장된 체크포인트 파일 수: 77
- 분할 전 전체 크기: 0.044 GiB
- 로컬 코드 기준 커밋: af2e397054c7553bf7aeedd544e72c22d93a004f
- Python 가상환경과 임시 파일은 제외했다.
- LEAR가 예측 파일만 저장한 경우 별도 체크포인트는 생성하지 않았다.
- OriginalRecent와 후속 TrainingRecent는 서로 다른 기준 예측이다.
- 원래 실행 코드가 로컬에 남아 있지 않은 실험은 결과만 보관된다.


## 추가 분석: 과거 참조 기간 1~20주

[실험 방법·전체 결과·실행 코드 보기](window_sweep_1to20/README.md)

모든 방법을 동일한 51주·8,568시간에서 비교했다. 테스트 기간에서 수행한 추가 탐색이며 BOA의 우수성을 입증하는 실험과는 구분한다.
