#!/usr/bin/env bash
# Qwen3-Coder-30B-A3B MoE on SYCL3 (port 8001).
# Migrated 2026-04-18: GGML_SYCL_DISABLE_OPT=1 fixes MoE+server SEGV (slot init reorder-MMVQ bug).
# Bench: tg32 23→61 tok/s (2.65x vs Vulkan). PP slightly slower (276→203).
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
export GGML_SYCL_DISABLE_OPT=1
exec /opt/llama.cpp/llama-sycl-src/build-f16/bin/llama-server     --model /mnt/models/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf     --device SYCL3     -ngl 999 -c 16384     --parallel 1     --batch-size 2048 --ubatch-size 512     --defrag-thold 0.1     --host 0.0.0.0 --port 8001     --alias Qwen3-Coder-30B-A3B     -t 1 --jinja --reasoning off     --no-warmup --log-file /tmp/llama-coder.log
