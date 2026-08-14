#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mode=${1:-baseline}
rows=${2:-4096}
cols=${3:-1024}
mkdir -p reports/ncu
if [[ "$mode" == "framework_torch" || "$mode" == "torch" ]]; then
  ncu --force-overwrite --profile-from-start off --set full --target-processes all --export "reports/ncu/latest_framework_torch" /home/silenceduke/miniconda3/bin/conda run -n YOLO26 python scripts/framework_torch_softmax.py benchmark "$rows" "$cols" 1 0 --single-launch | tee "reports/ncu/latest_framework_torch_raw.txt"
  echo "ncu report: reports/ncu/latest_framework_torch.ncu-rep"
  echo "ncu raw text: reports/ncu/latest_framework_torch_raw.txt"
else
  ncu --force-overwrite --set full --target-processes all --export "reports/ncu/latest_${mode}" ./build/release/softmax_bench "$mode" "$rows" "$cols" 1 0 --single-launch | tee "reports/ncu/latest_${mode}_raw.txt"
  echo "ncu report: reports/ncu/latest_${mode}.ncu-rep"
  echo "ncu raw text: reports/ncu/latest_${mode}_raw.txt"
fi
