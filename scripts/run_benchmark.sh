#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-baseline}
rows=${2:-4096}
cols=${3:-1024}
iters=${4:-100}
warmup=${5:-10}
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  /home/silenceduke/miniconda3/bin/conda run -n YOLO26 python scripts/framework_torch_softmax.py benchmark "$rows" "$cols" "$iters" "$warmup"
else
  ./build/release/softmax_bench "$mode" "$rows" "$cols" "$iters" "$warmup"
fi
