#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mode=${1:-baseline}
mkdir -p reports/racecheck
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  echo "framework_torch uses PyTorch runtime; compute-sanitizer racecheck is not used for this framework baseline." | tee reports/racecheck/latest_framework_torch.log
else
  compute-sanitizer --tool racecheck --error-exitcode 1 --log-file "reports/racecheck/latest_${mode}.log" ./build/debug/softmax_test "$mode"
  echo "racecheck log: reports/racecheck/latest_${mode}.log"
fi
