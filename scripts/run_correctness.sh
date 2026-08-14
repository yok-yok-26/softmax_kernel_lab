#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-baseline}
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  /home/silenceduke/miniconda3/bin/conda run -n YOLO26 python scripts/framework_torch_softmax.py correctness
else
  ./build/debug/softmax_test "$mode"
fi
