#!/usr/bin/env bash
# Gemma 4 26B-A4B Q8_0 on SYCL1 (port 8000). Migrated 2026-04-18 with DISABLE_OPT=1 fix.
# Reduced ctx 65536→32768 and parallel 2→1 for SYCL fitting headroom on 32GB B70.
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export GGML_SYCL_DISABLE_OPT=1
exec /opt/llama.cpp/llama-sycl-src/build-f16/bin/llama-server     --model /mnt/models/gemma-4-26B-A4B-it-Q8_0.gguf     --device SYCL1     -ngl 999 -c 32768     --parallel 1     --batch-size 2048 --ubatch-size 512     --defrag-thold 0.1     -fa 0     --host 0.0.0.0 --port 8000     --alias gemma-4-26B-A4B     -t 1     --chat-template-file /mnt/models/gemma-4-26B-A4B-it/chat_template.jinja     --jinja --reasoning off     --no-warmup --log-file /tmp/llama-gemma.log
