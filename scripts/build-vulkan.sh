#!/usr/bin/env bash
# Build llama.cpp Vulkan backend with B70 cherry-picks applied.
# Prereqs: Mesa 26.0.5+ (see install-mesa.sh), glslc (from shaderc or glslang),
#          git, cmake, ninja-build, libvulkan-dev
set -euo pipefail

SRC="${SRC:-/opt/llama.cpp/llama-vulkan-src}"
BUILD="${BUILD:-/opt/llama.cpp/llama-vulkan-build}"
BASE_SHA="${BASE_SHA:-073bb2c20b5b2c919469653214aaa1a9895816a2}"
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

rm -rf "$BUILD"
cmake -B "$BUILD" -S "$SRC" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT=ON

cmake --build "$BUILD" -j"$(nproc)"

echo
echo "Built: $BUILD/bin/llama-server"
"$BUILD/bin/llama-server" --version 2>&1 | head -3
