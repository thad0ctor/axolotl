# SparseLoRA throughput optimization

Branch `perf/sparselora-throughput`. Goal: raise the *whole-model* training
speedup of the SparseLoRA plugin, which is far smaller than the per-projection
microbench (1.9-6.7x) because most of a step is dense work that context-only
sparsity cannot touch.

Setup for all synthetic numbers: RTX 3090 Ti (GPU 7), bf16, gradient
checkpointing **on**, Llama-8B dims (24 layers, hidden 4096, inter 14336, 32 q /
8 kv heads), mbs 1, forced uniform sparsity, context fraction 0.75 (the 25%
output tokens stay dense by design). Speedups are vs the dense LoRA baseline
(plugin off). Timings are order-robust: orig/fast modes interleaved per round,
min over 5 rounds (medians agree).

## 1. Profiling - where a sparse step goes

Per-sub-op CUDA-event breakdown of a fwd+bwd step (instrumented, so absolute ms
is inflated; the *attribution* is the point). Gradient checkpointing recomputes
the forward in backward, so these timers capture forward **and** recompute.

seq 2048, s=0.7, **before** (boolean token path):

| op | ms/step | % step |
|----|--------:|-------:|
| sparse_linear (all sparse projections) | 417.9 | 35.4% |
| token_join (scatter rejoin)            |  43.3 |  3.7% |
| predict (2 bmm + silu_mul + topk)      |  39.8 |  3.4% |
| token_splits (boolean gather)          |  31.8 |  2.7% |
| **untracked** (attn core / RMSNorm / lm_head over 128k vocab / embeddings / LoRA / dense down_proj / loss) | **649.1** | **54.9%** |

Reading: **~55% of every step is dense compute the plugin structurally cannot
sparsify** - context-only sparsity leaves attention-core, norms, the 128k-vocab
lm_head, embeddings, and all output-token projections full size, and gradient
checkpointing recomputes all of it in backward. Sparse projections are ~35%. The
plugin's own overhead (predict + split + join) was ~9.8%; split+join (boolean
gather/scatter) was ~6.4% - the addressable target.

## 2. Optimization that worked - contiguous-block token split/join

`_compute_output_token_mask` builds the mask with a single
`masks[..., left:right] = True` using one `left`/`right` for the whole batch, so
the mask is always a **contiguous block**. The boolean ops are then identical to
slicing:

- `x[mask]`  (dense)  == `x[:, left:right]`
- `x[~mask]` (sparse) == `cat([x[:, :left], x[:, right:]], dim=1)`
- the `scatter_add` rejoin == `cat([sparse[:, :left], dense, sparse[:, left:]], dim=1)`

This avoids materializing the bool mask every step, the two gather kernels per
split, and the scatter on rejoin. `left`/`right` are already on the host (the
vendored builder already `.item()`s them), so no extra sync.

Implemented **plugin-side** in `fast_tokens.py` (no `_vendor` edits):
`_compute_output_token_mask` is rebound to return a tiny
`MaskContext(left, right, total)`; `SparseModule.token_splits`/`token_join` are
rebound to slice/cat; installed from `plugin.py` just before the
`torch.compiler.disable` boundaries. Degenerate cases (all-context -> all-sparse,
all-output -> all-dense, empty block) are handled to reproduce the boolean mask
exactly.

**Correctness: bit-identical** (not "within tolerance"). `test_fast_tokens.py`
asserts `torch.equal` for split, transformed-rejoin and round-trip across 7
boundary cases, plus a GPU test asserting `torch.equal` logits between boolean
and fast paths on a real applied tiny Llama. Full suite (36 tests) green.

**Profile after** (seq 2048, s=0.7): token_join 43.3 -> **10.9** ms, token_splits
31.8 -> **8.0** ms (~75% off each, ~56 ms/step saved); predict and the dense
remainder unchanged, as expected.

### Synthetic whole-model speedup (Llama-8B dims)

| seq  | s    | dense ms | orig ms (x)    | fast ms (x)    | gain  |
|------|------|---------:|---------------:|---------------:|------:|
| 2048 | 0.70 | 1472.9   | 1195.8 (1.23x) | 1076.8 (1.37x) | 1.11x |
| 2048 | 0.90 | 1472.9   | 1064.9 (1.38x) |  931.4 (1.58x) | 1.14x |
| 4096 | 0.70 | 3001.9   | 2170.7 (1.38x) | 2018.2 (1.49x) | 1.08x |
| 4096 | 0.90 | 3001.9   | 1882.0 (1.60x) | 1722.9 (1.74x) | 1.09x |

The fast path lifts the whole-model speedup at ctx=0.75 from the **1.32x/1.44x
baseline** to **1.58x (s=0.9, seq2048)** and **1.74x (s=0.9, seq4096)**. The
relative gain is largest at seq2048 / high sparsity (split/join is a bigger
share when matmuls are small) and shrinks slightly at seq4096 as matmuls
dominate, though the absolute speedup is higher there.

> Methodology note: a first pass timed orig then fast back-to-back on one model
> and reported a spurious 0.67x at s=0.9 (fast slower than *dense*) - an
> allocator/thermal artifact of the second timing loop. Interleaving modes per
> round and taking the min removed it. Don't trust back-to-back A/B timing.

## 3. Real-model sanity check - Llama-3.2-1B (real weights)

Local checkpoint, attention-only LoRA, real-weight predictor factors, seq 2048,
ctx 0.75, gc on, GPU 7. tokens/sec, plugin off vs on, boolean vs fast:

| config | ms/step | x vs dense | tok/s |
|--------|--------:|-----------:|------:|
| dense (plugin off)        | 386.3 | 1.00x | 5301 |
| s=0.70 orig (boolean)     | 446.8 | **0.86x** | 4583 |
| s=0.70 fast (slice/cat)   | 388.0 | 1.00x | 5279 |
| s=0.90 orig (boolean)     | 411.2 | **0.94x** | 4980 |
| s=0.90 fast (slice/cat)   | 354.8 | **1.09x** | 5773 |

split/join gain: **1.15x (s=0.7), 1.16x (s=0.9)**.

The headline real-model finding: on this small model the **original boolean path
is slower than dense** (0.86x / 0.94x) - the tiny transformer body's matmul
savings don't cover the plugin overhead while the 128k-vocab lm_head stays
dense. The fast path's 1.15-1.16x is exactly what makes SparseLoRA pay off here:
it flips s=0.9 from a net loss (0.94x) to a 1.09x win and brings s=0.7 to
break-even. The relative gain is even larger than on 8B dims because split/join
is a bigger fraction of the small model's total. The optimization survives the
full real-weight pipeline.

## 4. Considered / not pursued (honest negatives)

- **Predictor fusion** (two bmm + silu_mul): ~3.4% of a step and already
  `torch.compile`d; not worth vendored churn.
- **gate/up sparse-proj fusion** (stack weights, one GEMM): they share input and
  indices, but they live inside the ~35% useful-matmul bucket, and the per-step
  weight-slice + cat to fuse eats the saving. Not landed.
- **Caching predicted indices across the GC recompute**: predict is only 3.4%
  and a safe cache keyed on recomputed activations is fragile for that ceiling.
- **The ~55% dense remainder is the real ceiling.** Context-only sparsity leaves
  attention-core, norms, embeddings, the 128k-vocab lm_head and output-token
  projections dense, and gc recomputes them in backward. No plugin-local change
  reaches it; it needs a different sparsity scheme (out of scope, changes
  semantics).

## 5. Files changed

- `src/axolotl/integrations/sparselora/fast_tokens.py` - new; fast path + installer.
- `src/axolotl/integrations/sparselora/plugin.py` - install before compile boundaries.
- `tests/integrations/sparselora/test_fast_tokens.py` - new; bit-identity + GPU logit parity + idempotence.
- `tests/integrations/sparselora/conftest.py` - autouse fixture restores the rebound globals (no cross-test leak).

No `_vendor/` files were edited.
