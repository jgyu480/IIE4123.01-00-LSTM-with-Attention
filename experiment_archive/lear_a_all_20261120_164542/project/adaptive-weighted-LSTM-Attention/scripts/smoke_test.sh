#!/usr/bin/env bash
# Quick end-to-end check before pushing or on a new machine: 1 seed, 2 epochs, both splits.
# Usage: bash scripts/smoke_test.sh            (CPU)
#        DEVICE=cuda bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-../.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
rm -rf results/smoke
"$PY" scripts/01_make_oof.py --tag smoke --seeds 42 --epochs 2 --device "$DEVICE"
"$PY" scripts/02_res_knn.py --tag smoke --seeds 42
"$PY" scripts/03_summarize.py --tag smoke --n-boot 200
"$PY" scripts/05_train_case_attn.py --tag smoke --seeds 42 --epochs 2 --device "$DEVICE"
"$PY" scripts/06_gate.py --tag smoke --seeds 42
"$PY" scripts/03_summarize.py --tag smoke --stage gate --references LSTM CaseAttn --n-boot 200
"$PY" scripts/07_learned_gate.py --tag smoke --seeds 42
"$PY" scripts/03_summarize.py --tag smoke --stage gate_learn --references LSTM CaseAttn Gate-rule --n-boot 200
rm -rf results/smoke
echo "SMOKE OK ($DEVICE)"
