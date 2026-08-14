#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-O3 -lineinfo"
cmake --build build/release -j"$(nproc)"
