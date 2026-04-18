#!/usr/bin/env bash
# Build llama.cpp SYCL backend with B70 cherry-picks applied.
# Prereqs: Intel oneAPI Base Toolkit 2024.2+ installed at /opt/intel/oneapi
#          git, cmake, ninja-build, build-essential
set -euo pipefail

SRC="${SRC:-/opt/llama.cpp/llama-sycl-src}"
BUILD="${BUILD:-$SRC/build-f16}"
BASE_SHA="${BASE_SHA:-073bb2c20b5b2c919469653214aaa1a9895816a2}"  # llama.cpp master pinned 2026-04
PATCHES="$(cd "$(dirname "$0")/.." && pwd)/patches"

if [ ! -d "$SRC" ]; then
  sudo mkdir -p "$(dirname "$SRC")"
  sudo chown "$USER:$USER" "$(dirname "$SRC")"
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi

cd "$SRC"
git fetch origin
git checkout "$BASE_SHA"
git branch -D b70-kit 2>/dev/null || true
git checkout -b b70-kit
git am --3way "$PATCHES"/*.patch || { echo "patch apply failed"; exit 1; }

source /opt/intel/oneapi/setvars.sh --force

rm -rf "$BUILD"
cmake -B "$BUILD" -S "$SRC" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

cmake --build "$BUILD" -j"$(nproc)"

echo
echo "Built: $BUILD/bin/llama-server"
"$BUILD/bin/llama-server" --version 2>&1 | head -3
