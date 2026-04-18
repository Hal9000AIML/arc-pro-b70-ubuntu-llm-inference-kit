#!/usr/bin/env bash
exec /opt/llama.cpp/llama-vulkan-build/bin/llama-server     --model /mnt/models/Qwen_Qwen3-4B-Instruct-2507-Q6_K.gguf     --device Vulkan3     -ngl 999 -c 16384     --parallel 2     --batch-size 1024 --ubatch-size 512     --host 0.0.0.0 --port 8002     --alias Qwen3-4B-Instruct     -t 1 --jinja --reasoning off     --no-warmup --log-file /tmp/llama-fast.log
