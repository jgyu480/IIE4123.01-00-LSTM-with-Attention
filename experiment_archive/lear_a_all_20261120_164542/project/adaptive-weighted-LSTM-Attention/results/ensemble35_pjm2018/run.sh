#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ../.venv/bin/python ]]; then
    PYTHON=../.venv/bin/python
  elif python3 -c 'import torch, numpy, pandas, sklearn, scipy, joblib' 2>/dev/null; then
    PYTHON=python3
  else
    python3 -m venv ../.venv
    PYTHON=../.venv/bin/python
  fi
fi
if ! "$PYTHON" -c 'import torch, numpy, pandas, sklearn, scipy, joblib'; then
  "$PYTHON" -m pip install -r requirements.txt
fi

device="$("$PYTHON" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
split=split2_2018
tag=step1
base="results/$tag/$split"
out=results/ensemble35_pjm2018
common=(--tag "$tag" --splits "$split")

echo "repo=$(git rev-parse --short HEAD) device=$device split=$split"

if [[ ! -s "$base/export/predictions_val.csv" || ! -s "$base/export/predictions_test.csv" ]]; then
  "$PYTHON" -u scripts/01_make_oof.py "${common[@]}" --device "$device"
  "$PYTHON" -u scripts/05_train_case_attn.py "${common[@]}" --device "$device" --skip-folds
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -u scripts/08_lear.py "${common[@]}" --jobs "${LEAR_JOBS:-4}"
  "$PYTHON" -u scripts/10_export_predictions.py "${common[@]}"
fi

"$PYTHON" -u "$out/combine.py" "$base/export" "$out"
