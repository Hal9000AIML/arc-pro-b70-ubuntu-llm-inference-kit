#!/usr/bin/env bash
# Qwen3-Next-80B-A3B-Thinking IQ3_XXS on SYCL2 (port 8004). Migrated 2026-04-18.
# 80B MoE 3B active, 28.7GB model. ctx 16384 for f16 KV fit on 32GB.
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export GGML_SYCL_DISABLE_OPT=1
exec /opt/llama.cpp/llama-sycl-src/build-f16/bin/llama-server     --model /mnt/models/Qwen3-Next-80B-A3B-Thinking/Qwen3-Next-80B-A3B-Thinking.i1-IQ3_XXS.gguf     --device SYCL2     -ngl 999 -c 16384     --parallel 1     --batch-size 2048 --ubatch-size 512     -fa 0     --host 0.0.0.0 --port 8004     --alias Qwen3-Next-80B-Thinking     -t 1 --jinja     --no-warmup --log-file /tmp/llama-reasoning.log
