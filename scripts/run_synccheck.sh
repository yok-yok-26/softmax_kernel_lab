#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mode=${1:-baseline}
mkdir -p reports/synccheck
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  echo "framework_torch uses PyTorch runtime; compute-sanitizer synccheck is not used for this framework baseline." | tee reports/synccheck/latest_framework_torch.log
else
  compute-sanitizer --tool synccheck --error-exitcode 1 --log-file "reports/synccheck/latest_${mode}.log" ./build/debug/softmax_test "$mode"
  echo "synccheck log: reports/synccheck/latest_${mode}.log"
fi
