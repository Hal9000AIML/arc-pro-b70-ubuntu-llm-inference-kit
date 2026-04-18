# Backend selection rules for Arc Pro B70

Pick SYCL vs Vulkan per model. These rules come from measured regressions on 4× B70 across Qwen3, Gemma 4, and Llama family models.

## Rule 1 — Dense models: SYCL

Any dense (non-MoE) model benefits from SYCL's oneMKL/oneDNN paths. Vulkan on B70 is 2–3× slower for TG on dense transformers.

Examples: Gemma 4 26B, Qwen3-14B, Qwen3-32B, Llama 3.3 70B, Qwen2.5-32B.

Runtime env:
```
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export GGML_SYCL_DISABLE_OPT=1   # still recommended; costs ~5% perf but fixes rare reorder crashes
source /opt/intel/oneapi/setvars.sh --force
```

## Rule 2 — MoE models: Vulkan, or SYCL with DISABLE_OPT=1

MoE servers hit a SYCL slot-init SEGV at model load. Workaround: set `GGML_SYCL_DISABLE_OPT=1` before launching — this disables the fused-reorder-MMVQ path that crashes. If that still fails, fall back to Vulkan.

Examples: Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B, Qwen3-Next-80B-A3B, Mixtral.

Preferred: SYCL with `DISABLE_OPT=1` (tg wins). Fallback: Vulkan.

## Rule 3 — Speculative decoding: Vulkan

SYCL speculative decoding (draft + target on same GPU) is unstable on B70 — kernel cache contention between the two models causes intermittent server death. Vulkan is stable for draft models.

If you want speculative on SYCL, run draft on a different physical GPU than the target model.

## Rule 4 — Co-tenant GPUs: lighter model gets Vulkan

When two llama-server processes must share one B70 (VRAM permits but compute contends), put the smaller/lighter model on Vulkan. Vulkan yields the card more cooperatively under pressure than SYCL does. Measured on GPU3 with Qwen3-Coder-30B (SYCL) + Qwen3-4B (Vulkan): code tier held 57 tok/s, fast tier 33 tok/s. Flipping fast tier to SYCL drops code tier to 7 tok/s (SYCL+SYCL on one card is a 10× penalty).

## Rule 5 — Flash attention

`--flash-attn on` works on Vulkan for most models on B70 (Xe2 has decent coopmat support with Mesa 26+). On SYCL, FA is more fragile: safe for dense, often crashes on MoE. Default: `-fa 0` for SYCL MoE, `--flash-attn on` for Vulkan.

## Quick decision table

| Architecture | Shared GPU? | Spec decoding? | Backend |
|---|---|---|---|
| dense | no | no | SYCL |
| dense | yes (smaller) | no | Vulkan |
| MoE | no | no | SYCL + DISABLE_OPT=1 |
| MoE | yes | no | Vulkan |
| any | — | yes | Vulkan |
