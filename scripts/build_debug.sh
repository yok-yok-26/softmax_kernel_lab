#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DDEBUG_CUDA_SYNC=ON -DCMAKE_CUDA_FLAGS="-G -g -lineinfo"
cmake --build build/debug -j"$(nproc)"
