# b70-llamacpp-kit

Intel Arc Pro B70 (BMG G31, Xe2, 32GB) tuning kit for `llama.cpp`. Reproduces the build + runtime configuration that drives a 4× B70 inference box running 5 concurrent llama-server tiers.

## What this kit is for

If you have one or more Intel Arc Pro B70 cards and you want them to be useful for local LLM inference, out-of-the-box `llama.cpp` leaves a lot on the table. This kit captures:

- 11 cherry-picked commits on top of a known-good `llama.cpp` base (BF16 GET_ROWS, MoE MMVQ fused TG, K-quant native subgroup DMMV, Xe2 Vulkan warptile, oneMKL small-matmul path, Q8_0 reorder fix, etc.)
- Correct SYCL build flags (`GGML_SYCL_F16=ON`, HOST_MEM_FALLBACK, DNN/graph enabled)
- Runtime env vars that matter on Xe2 (`GGML_SYCL_DISABLE_OPT=1`, `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`)
- Mesa 26.0.5 from the `kisak/kisak-mesa` PPA (enables `VK_KHR_shader_bfloat16` + `VK_KHR_shader_integer_dot_product` on BMG)
- Per-model start scripts (dense Q8 on SYCL; MoE on Vulkan to avoid SYCL MoE slot-init SEGV)
- Systemd unit template so tiers survive reboots
- Known-good benchmark numbers so you know when something regressed

## Hardware tested

| | |
|---|---|
| GPU | 4× Intel Arc Pro B70 (BMG G31, Xe2, 32GB GDDR6 each) |
| Host | AMD Threadripper 1900X, 128GB DDR4 |
| OS | Ubuntu 24.04 (kernel 6.8+) |
| Backends | llama.cpp SYCL (oneAPI 2024.2+) and llama.cpp Vulkan (Mesa 26.0.5) |

## Headline numbers (single-stream, identical 300-tok prompt)

| Tier | Model | Backend | GPU | tg tok/s | Notes |
|---|---|---|---|---|---|
| chat | gemma-4-26B-A4B Q8_0 | SYCL | 1 | 26.4 | dense, SYCL wins over Vulkan |
| code | Qwen3-Coder-30B-A3B Q5_K_M | SYCL | 3 | 57.7 | MoE; DISABLE_OPT=1 required |
| fast | Qwen3-4B-Instruct Q6_K | Vulkan | 3 | 33.0 | co-tenant with code tier |
| agentic | Qwen3.6-35B-A3B Q6_K_XL + 0.6B draft | Vulkan | 0 | 25.0 | speculative decoding |
| reasoning | Qwen3-Next-80B-A3B IQ3_XXS | SYCL | 2 | 21.2 | 80B MoE, 3B active |

See `docs/benchmarks.md` for methodology and regression guardrails.

## Quick start

```bash
# 1. Install Mesa 26.0.5 (Vulkan backend)
sudo bash scripts/install-mesa.sh

# 2. Install oneAPI (SYCL backend) — Intel's repo, not packaged here
# Follow https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# 3. Build both backends with the cherry-picks applied
bash scripts/build-sycl.sh    # ~25 min
bash scripts/build-vulkan.sh  # ~5 min

# 4. Install systemd template
sudo cp systemd/llamacpp@.service /etc/systemd/system/
sudo systemctl daemon-reload

# 5. Drop your model files in /mnt/models, edit scripts/start_*.sh paths,
#    put them in ~/, then:
systemctl --user start llamacpp@8000  # etc.
```

## Why two backends?

On B70 the two llama.cpp backends have different strengths:

- **SYCL** wins on dense models (Gemma 4, Qwen3-14B/32B) by 2–3× over Vulkan for token generation, thanks to oneMKL/oneDNN paths and the cherry-picked MMVQ kernels.
- **Vulkan** is required for MoE models with speculative decoding draft models; SYCL has a slot-init SEGV on MoE servers (mitigated but not fully fixed by `GGML_SYCL_DISABLE_OPT=1`).

`docs/backend-selection.md` has the rules.

## What's NOT in this kit

- Windows support. These patches and flags are Linux-only. Intel's Windows Arc stack is a different beast.
- Model files. Bring your own GGUFs. The start scripts reference `/mnt/models/...` paths; edit to yours.
- An installer. This is artifacts + scripts; you run them.

## Repository layout

```
patches/        11 cherry-pick .patch files (apply to llama.cpp master@073bb2c20)
scripts/        build-sycl.sh, build-vulkan.sh, install-mesa.sh, start_*.sh
systemd/        llamacpp@.service template
docs/           build.md, tuning.md, backend-selection.md, benchmarks.md
```

## License

MIT. The patches are derivative work of llama.cpp (MIT). Upstream authors credited in each `.patch` file via `git format-patch` headers.
