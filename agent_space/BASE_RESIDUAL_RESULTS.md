# Base residual: L2QER low-rank correction on ALL frozen FP4 base linears — results

Generalization of the Phase-2 lm_head residual (`RESIDUAL_PHASE2_RESULTS.md`) to
every frozen FP4 base linear under LoRA/QLoRA. PTQ A/B harness
(`agent_space/base_residual_probe.py`, no training) on Qwen3-0.6B: all 196
eligible body linears (q/k/v/o, gate/up/down; lm_head excluded — it has its own
block) swapped to the shipped frozen FP4 base (`NVFP4FastComputeBaseLinear`,
the LoRA compute-mode default on this machine), scored as logit
KL(bf16 ‖ fp4-body) against the unmodified bf16 model on a held-out 512-token
eval slice (512 separate calibration tokens).

Construction is the SHIPPED code path: `attach_base_residual` →
`_build_residual` (the same whitened SVD the lm_head uses), with
`g = sqrt(mean_t x_t²)` per INPUT channel captured by forward pre-hooks on the
still-bf16 model BEFORE the quantize swap (the QLoRA storage mode discards the
bf16 master right after packing, so the error `E = W − dequant(Q(W))` must be
factored at swap time).

## Headline: logit KL vs the bf16 model (Qwen3-0.6B, full FP4 body)

| method        | KL(bf16‖cand) | vs fp4 base | top-1 agree |
|---------------|--------------:|------------:|------------:|
| fp4_baseline  | 0.07391       |      —      | 0.9512      |
| plain_k16     | 0.06978       |   −5.6%     | 0.9492      |
| l2qer_k8      | 0.05321       |  −28.0%     | 0.9551      |
| **l2qer_k16** | **0.05404**   | **−26.9%**  | **0.9570**  |
| l2qer_k32     | 0.05291       |  −28.4%     | 0.9473      |

- **Rank-16 L2QER residuals cut the FP4-body logit KL by ~27%** and raise top-1
  agreement 0.951 → 0.957. The gain saturates by rank 8–16 (rank 32 buys ~nothing,
  consistent with the lm_head finding), so the default rank is 16.
- **Plain SVD is again a dead end** (−5.6% ≈ noise): the e2m1 error is near
  full-rank; only the activation weighting concentrates the output-relevant
  error into a low-rank subspace. This is why an unavailable calibration
  DISABLES the residual (loud warning) instead of degrading to plain SVD.

## Load-time overhead (one-time, at model load)

| phase                                | wall time |
|--------------------------------------|----------:|
| calibration forward (512 tokens, hooks on 196 linears) | 0.40 s |
| 196 × SVD + attach (rank 16, GPU)    | 11.7 s    |
| (reference: the FP4 swap itself)     | 0.46 s    |
| A/B buffer memory (rank 16, whole model) | 0.62 MiB |

The SVD total is per-call-overhead dominated (196 small fp32 GPU SVDs); fine as
a load-time cost, scales linearly with layer count and ~quadratically with
hidden size (a 9B-scale body is minutes, still load-time-acceptable).

## Step-time overhead (PERMANENT: train + inference) — the honest number

| measurement (fwd+bwd, median)                         | base    | +residuals | overhead |
|-------------------------------------------------------|--------:|-----------:|---------:|
| whole 0.6B model, 1081 tokens, eager                  | 160.9 ms| 186.1 ms   | **+15.7%** |
| whole 0.6B model, 1081 tokens, torch.compile          | 98.7 ms | 120.8 ms   | **+22.4%** |
| one linear 1024→3072, M=4096, eager                   | 0.65 ms | 0.98 ms    | +51%     |
| one linear 4096→12288 (9B-ish), M=8192, eager         | 2.66 ms | 3.97 ms    | +49%     |

**The design brief's ~+1% step-cost assumption did NOT hold.** The residual's
FLOPs really are <1% of the base GEMM, but the shipped bolt-on implementation
materializes and adds the `[tokens, out]` correction (and reads `[tokens, out]`
again in dgrad) — pure memory bandwidth, the exact effect Phase 2 measured as
the lm_head "bolt-on +36%". It does not amortize with wider layers (the FP4
base GEMM speeds up too) and default torch.compile does not fuse the add into
the cuBLAS GEMM epilogue (`torch.addmm` measured no better). A useful framing:
the residual costs about as much as a SECOND rank-16 LoRA adapter pair per
layer — the trainable adapters already run the same unfused skinny-matmul
shapes every step.

Per the brief, the feature ships DEFAULT ON (explicit product decision), with
the measured cost documented in the schema field descriptions and
`enabled: false` as the pure-throughput opt-out.

### Paths to cheap (not done here)
1. Fuse the correction add into the FP4 GEMM epilogue (custom kernel or
   max-autotune Triton matmul templates) — would leave only the thin `[M,k]`
   matmuls (~1–2%).
2. Piggyback on the LoRA adapter matmuls (concat `[B; lora_A]`, `[A, lora_B]`
   shapes) — halves the launch/traffic count but entangles frozen and trainable
   parameters.

## Correctness / wiring (unit-tested)

- Forward: `y = Q(W)x + (x@Bᵀ)@Aᵀ` in all four frozen base classes
  (`NVFP4FrozenBaseLinear`, `NVFP4ComputeBaseLinear`, and both MSLK-fast
  variants) AND in the fused-LoRA kernel entry points (`nvfp4_base_fprop`,
  `nvfp4_base_fprop_many`, `nvfp4_base_dgrad`), so fused-kernel runs stay
  consistent with module forwards.
- Backward: the module-forward correction is a plain differentiable bolt-on, so
  autograd supplies the EXACT dgrad `dX += (dY@A)@B` and no wgrad (A/B are
  buffers). Verified: corrected-minus-base grad equals the analytic term to
  <1e-2 of total grad norm (bf16 accumulation rounding).
- torch.compile: 0 graph breaks with residuals attached (the None-check is a
  module attribute dynamo specializes once).
- FSDP: A/B are tiny plain bf16 buffers (replicated, not sharded); existing
  FSDP all-gather/pickling tests stay green. Buffers are non-persistent
  (reconstructed deterministically at load), so checkpoints are unchanged.
- Default-on: a bare `nvfp4_training: {enabled: true}` LoRA config calibrates
  and attaches residuals to every swapped base; `base_residual.enabled: false`
  disables; `base_mode: hp` and full-FT skip with a debug log; unavailable
  calibration disables loudly (never plain-SVD).

## Files
- `agent_space/base_residual_probe.py` — the PTQ A/B harness (KL, load
  overhead, step-time A/B; `--compile-step` for the compiled number).
- `agent_space/base_residual_qwen3_0.6b.json` — raw probe output.
- shipped: `src/axolotl/kernels/nvfp4_residual.py` (`attach_base_residual`,
  reusing `_build_residual`), `src/axolotl/utils/nvfp4_training.py`
  (`_apply_base_residual[_dgrad]`, `BaseResidualPlan`, swap-time attach in
  `convert_lora_base_to_nvfp4`), `src/axolotl/loaders/patch_manager.py`
  (`_nvfp4_prepare_base_residual` pre-swap calibration), schema
  `nvfp4_training.base_residual` (default ON), tests in
  `tests/e2e/test_nvfp4_training.py::TestNVFP4BaseResidual` and
  `tests/e2e/test_nvfp4_integration.py`.
