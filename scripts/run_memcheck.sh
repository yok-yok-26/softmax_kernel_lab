#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mode=${1:-baseline}
mkdir -p reports/memcheck
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  echo "framework_torch uses PyTorch runtime; compute-sanitizer memcheck is not used for this framework baseline." | tee reports/memcheck/latest_framework_torch.log
else
  compute-sanitizer --tool memcheck --error-exitcode 1 --log-file "reports/memcheck/latest_${mode}.log" ./build/debug/softmax_test "$mode"
  echo "memcheck log: reports/memcheck/latest_${mode}.log"
fi
