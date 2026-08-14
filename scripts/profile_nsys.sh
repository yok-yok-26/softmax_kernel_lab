#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mode=${1:-baseline}
rows=${2:-4096}
cols=${3:-1024}
mkdir -p reports/nsys
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output "reports/nsys/latest_framework_torch" /home/silenceduke/miniconda3/bin/conda run -n YOLO26 python scripts/framework_torch_softmax.py benchmark "$rows" "$cols" 1 0 --single-launch
  echo "nsys report: reports/nsys/latest_framework_torch.nsys-rep"
else
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output "reports/nsys/latest_${mode}" ./build/release/softmax_bench "$mode" "$rows" "$cols" 1 0 --single-launch
  echo "nsys report: reports/nsys/latest_${mode}.nsys-rep"
fi
