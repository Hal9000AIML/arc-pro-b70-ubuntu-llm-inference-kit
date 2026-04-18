# Benchmarks

All numbers are single-stream generation of the Fibonacci prompt:

> Write an iterative Python function that computes the nth Fibonacci number.

with `max_tokens=300`, `temperature=0.1`, `/no_think` for Qwen3. Values are `timings.predicted_per_second` from llama.cpp's `/v1/chat/completions` response.

## 2026-04-18 reference run (4× B70 box, 5 concurrent servers)

| Port | Model | GPU | Backend | pp tok/s | tg tok/s |
|---|---|---|---|---|---|
| 8000 | gemma-4-26B-A4B Q8_0 | 1 | SYCL | 68 | 26.4 |
| 8001 | Qwen3-Coder-30B-A3B Q5_K_M | 3 | SYCL (DISABLE_OPT) | 50 | 57.7 |
| 8002 | Qwen3-4B-Instruct Q6_K | 3 | Vulkan | 90 | 33.0 |
| 8003 | Qwen3.6-35B-A3B Q6_K_XL + 0.6B draft | 0 | Vulkan (spec) | 81 | 25.0 |
| 8004 | Qwen3-Next-80B-A3B IQ3_XXS | 2 | SYCL (DISABLE_OPT) | 36 | 21.2 |

GPU3 is shared between :8001 (code) and :8002 (fast). Both are benchmarked with the other server idle (warm but not generating). Concurrent generation on both degrades each by ~30%.

## Regression guardrails

If your numbers are dramatically below these, check in order:

1. **Mesa version** — Vulkan perf regresses hard without the `kisak/kisak-mesa` PPA. `apt-cache policy mesa-vulkan-drivers` should show 26.0.5+.
2. **SYCL cache poisoning** — if a SYCL tier is 5–10× slower than expected right after a restart, `rm -rf ~/.cache/libsycl_cache ~/.cache/neo_compiler_cache` and restart. Most common after kernel or driver update.
3. **Env vars** — confirm `GGML_SYCL_DISABLE_OPT=1` set (otherwise MoE SEGVs or dense runs reorder-buggy kernels).
4. **Two servers on one GPU, both SYCL** — 10× penalty. One of them must be Vulkan. See backend-selection.md Rule 4.
5. **Flash attention on SYCL MoE** — try `-fa 0`. FA + SYCL + MoE is a known crash path on B70.

## Solo-GPU numbers (no co-tenant)

For reference, these are the same tiers when they own their GPU:

| Model | Backend | Solo tg | Co-tenant tg | Penalty |
|---|---|---|---|---|
| Qwen3-Coder-30B Q5_K_M | SYCL | 60 | 57.7 | 4% (if pair is Vulkan 4B) |
| Qwen3-Coder-30B Q5_K_M | SYCL | 60 | 7 | 88% (if pair is SYCL anything) |
| gemma-4-26B Q8_0 | SYCL | 27 | 26.4 | 2% |
| Qwen3-Next-80B IQ3_XXS | SYCL | 22 | 21.2 | 4% |

The lesson: SYCL + SYCL on one card is catastrophic. SYCL + Vulkan on one card is fine if the Vulkan one is small.
