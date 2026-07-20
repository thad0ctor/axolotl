# MegaTrain vendored source provenance

## Upstream source

- Project: [MegaTrain](https://github.com/DLYuanGod/MegaTrain)
- Upstream package: `infinity`
- Commit: `7f5c9597e5b20bb618932c77c922e8eac4a11c4d`
- Commit branch at retrieval: `main`
- Upstream version: `0.3.0`, from `infinity/__init__.py`
- Conflicting distribution version: `0.2.0`, from `setup.py`
- Retrieval date: 2026-07-19
- License: Apache-2.0

The upstream `LICENSE` is copied to the integration's `LICENSE` file
(`../LICENSE` relative to this file).

## Why the source is vendored

MegaTrain is not published on PyPI. Its upstream packaging also has conflicting
dependency and version metadata, conditionally builds a CUDA extension, includes
stale package metadata and compiled artifacts, and installs the generic top-level
package name `infinity`. Vendoring the small, pure-Python runtime gives Axolotl a
fixed, auditable source revision without introducing a top-level package collision
or an extension build at installation time.

## Retrieval and pruning

The snapshot was created from Git objects rather than the upstream working tree:

```bash
git archive 7f5c9597e5b20bb618932c77c922e8eac4a11c4d infinity LICENSE
```

The following tracked paths were removed from the archived `infinity` tree:

- `infinity/cuda_pipeline/` in its entirety, including its C++/CUDA sources,
  build scripts, tests, `build/`, `.so` and `.o` files, and egg-info metadata.
  Nothing upstream imports its `infinity_memory_ops` extension, so removing it
  costs no functionality.
- Every tracked `infinity/**/__pycache__/` directory and its `.pyc` files.
- Every module the integration never imports: `infinity/adapters/`,
  `infinity/csrc/`, `infinity/data/`, `infinity/memory/`, `infinity/ops/`,
  `infinity/runtime/`, `infinity/scheduler/`, `infinity/config/yaml_loader.py`,
  `infinity/model/transformer.py`, `infinity/optimizer.py`,
  `infinity/profiler.py`, `infinity/simple_profiler.py`, and
  `infinity/true_cpu_offloading.py`. These are unused upstream as well —
  `infinity/ops/attention.py` carries its own deprecation warning, and
  `scheduler/`, `runtime/`, `adapters/`, `profiler.py`, `transformer.py`, and
  `true_cpu_offloading.py` have no callers anywhere in the upstream repository.
  Several contain defects (an interleaved `rotate_half` against a half-split
  cos/sin cache in `ops/layers.py`; a scheduler whose backward pass cannot
  complete). The re-exports that pulled `data/datasets.py`,
  `config/yaml_loader.py`, and `model/transformer.py` into every import were
  removed from the corresponding `__init__.py` files.

The repository-level `build/`, `csrc/`, `verl/`, and egg-info trees were never
selected for the archive and are not part of this vendored copy.

## Local changes

The retained package was relocated to
`axolotl.integrations.megatrain._vendor.infinity`. Absolute imports were
mechanically rewritten as follows:

- `from infinity...` became
  `from axolotl.integrations.megatrain._vendor.infinity...`.
- `import infinity...` became an import through
  `axolotl.integrations.megatrain._vendor.infinity...`.

The optional `import infinity_memory_ops` in `infinity/csrc/__init__.py` was left
unchanged. String literals and documentation references that do not import the
package were not rewritten. An Axolotl-authored `_vendor/__init__.py` was added
outside the upstream snapshot to establish the private namespace.

The following compatibility and correctness edits were made after the mechanical
rewrite:

- `infinity/model/cpu_master.py` imports
  `transformers.masking_utils.create_masks_for_generate`, adds
  `_prepare_attention_mask()`, and uses it in both single-GPU streamed forward
  paths. This converts Axolotl's 2D padding mask into the backend-specific causal
  mask expected by Transformers 5.14.1. Passing the upstream integer mask
  directly caused an SDPA dtype failure.
- `infinity/model/cpu_master.py` adds an optional `global_valid_tokens` argument
  to `forward_and_backward()` and forwards it to the existing single-GPU loss
  implementation. Axolotl uses the accumulation window's token count so uneven
  microbatches produce the same token-mean gradient as an unaccumulated batch.
- `infinity/model/cpu_master.py` uses Transformers' mask mapping for homogeneous
  full or sliding-window attention. Mixed schedules and stateful recurrent,
  chunked, or sparse attention layer types are rejected by the streamed runtime.
- `infinity/model/cpu_master.py` records each layer's CUDA RNG state during the
  original forward and replays it without advancing the global RNG stream during
  activation recomputation. This keeps dropout masks identical between the
  original and recomputed forwards.
- `infinity/model/cpu_master.py` reserves and zero-fills every parameter's
  gradient-slab region and carries a gradient-presence bitmap to the CPU worker.
  Unused parameters retain `grad=None`, while later gradients are decoded at the
  correct offsets in the layer, head/norm, and embedding collectors.
- `infinity/model/cpu_master.py` computes fallback cross entropy in FP32 and
  backpropagates each normalized logits chunk immediately. This matches
  Transformers' causal-LM loss precision while bounding retained logits memory.
- `infinity/model/cpu_master.py` does not reserve a separate head-gradient slab
  region when the language-model head is tied to the embedding, because that
  gradient is collected through the shared embedding tensor.
- `infinity/model/cpu_master.py` creates GPU working modules by casting only
  their parameters to the configured compute dtype. CPU master parameters and
  gradients remain FP32, while FP32 rotary and other floating buffers retain
  their precision on GPU.
- `infinity/model/cpu_master.py` and `infinity/model/mp_worker.py` promote BF16
  gradient slabs into gradients allocated with the CPU master parameter dtype.
  Both the single-process and vendored multiprocessing paths use the same
  parameter-only working-module conversion.
- `infinity/model/cpu_master.py` removes a duplicate, unused final-normalization
  forward that retained an unnecessary autograd graph before the actual loss
  forward rebuilt the same activation.

Each modified file additionally carries a three-line header naming the upstream
project, the pinned revision, and the fact that Axolotl changed it, as required
by Apache-2.0 section 4(b). Every retained vendored file is modified, so all of them carry the header.

### Correctness fixes to the single-GPU path

- `infinity/model/cpu_master.py` no longer lets a failure in the gradient
  accumulation worker thread hang training. The thread previously died without
  calling `task_done()`, so `grad_task_queue.join()` and the slab free-list
  blocked forever with no traceback. The worker now records the exception, keeps
  draining the queue, and the error is re-raised on the training thread.
- `infinity/model/cpu_master.py` raises a clear error when
  `_sync_params_to_gpu()` runs after `release_gpu_buffers()` instead of failing
  with `'NoneType' object has no attribute 'parameters'`.
- `infinity/model/cpu_master.py` keys the streamed-layer grouping on the layer's
  parameter structure rather than `hash(structure)`; a collision would have
  silently merged two differently-shaped layers.
- `infinity/model/cpu_master.py` sums the chunked cross-entropy contributions to
  the language-model head and final norm in FP32. They are BF16 leaves, so
  accumulating one gradient per chunk in `.grad` drifted by roughly
  `sqrt(num_chunks) * eps` on the largest matrix in the model. The FP32
  accumulator is allocated only when the sequence spans more than one chunk.

### Multi-GPU (data-parallel) fixes

The vendored multiprocessing path was unreachable from Axolotl and had not been
exercised. It is now reachable through `megatrain_devices`, and the following
defects were fixed:

- `infinity/model/mp_worker.py` reserved gradient-slab space only for parameters
  that had a gradient, while the reader assumed a dense layout over every
  parameter. Any unused parameter shifted every subsequent parameter's gradient
  by one slot and fed uninitialized slab bytes into the gap. All three
  collectors (layer, head, embedding) now use the same
  `_copy_parameter_grads_to_slab` / `_accumulate_parameter_grads_from_slab`
  helpers and gradient-presence bitmap as the single-GPU path.
- `infinity/model/cpu_master.py` forwards `global_valid_tokens` to the
  multiprocessing path, which previously dropped it and renormalized by its own
  microbatch-local token count.
- `infinity/model/cpu_master.py` no longer zeroes the shared gradients on every
  microbatch, which discarded all but the last microbatch of a gradient
  accumulation window.
- `infinity/model/cpu_master.py` shards each batch with `_shard_bounds` so every
  sample is covered exactly once; the previous `B // world_size` slice silently
  dropped the remainder. Batches smaller than the worker count are rejected.
- `infinity/model/mp_state.py` gains `reattach_and_zero_detached_grads()`, and
  `CPUMasterModel.ensure_grads_attached()` calls it before each streamed step.
  Hugging Face's `Module.zero_grad()` defaults to `set_to_none=True`, which
  detached the CPU masters from the shared-memory buffers the workers accumulate
  into and would otherwise have made every optimizer step a no-op.
- `infinity/model/mp_worker.py` builds its attention mask with the shared
  `_prepare_attention_mask` helper instead of a hand-rolled causal-or-padding
  mask. For `sdpa` and `eager` the sliding window lives entirely in the mask, so
  the previous mask silently gave sliding-window models full-context attention.
  It also inherits the mixed-schedule and chunked-attention rejections, and
  resolves the per-layer-type mask mapping.
- `infinity/model/mp_worker.py` replays each layer's CUDA RNG state during
  activation recomputation, matching the single-GPU path, so dropout masks agree
  between the original and recomputed forwards.
- `infinity/model/mp_worker.py` reports failures instead of dying. Every command
  is answered exactly once, with the traceback carried back on a new
  `WorkerResult.error` field; the parent surfaces it through
  `CPUMasterModel._await_worker_result`, which also polls worker liveness and
  applies a timeout rather than blocking forever.
- `infinity/model/mp_worker.py` backpropagates each normalized loss chunk
  immediately rather than retaining the whole chunked graph, computes fallback
  cross entropy in FP32, and uses the same FP32 head-gradient accumulator as the
  single-GPU path.
- `infinity/model/mp_worker.py` runs the initial embedding forward under
  `torch.no_grad()` and drops a duplicate final-normalization forward whose
  result was discarded; both previously retained an unused autograd graph for
  the whole step.
- `infinity/model/mp_worker.py` raises `NotImplementedError` for vision inputs
  instead of silently ignoring `pixel_values` and `vision_kwargs`.
- `infinity/model/mp_state.py` strips Transformers' `_hidden_kernels` caches from
  the shared modules before spawning workers. Transformers 5.x attaches
  `{name: Func()}` to every attention module, where `Func` is defined inside a
  function and therefore unpicklable, which made worker spawn fail outright with
  `Can't pickle local object '_create_func_module.<locals>.Func'`. The dict is
  only a discovery cache for `kernelize()`, which this integration never calls.
- `infinity/model/cpu_master.py` and `infinity/model/mp_worker.py` release each
  FP32 head-gradient accumulator as it is consumed, so the largest one is not
  pinned on the GPU for the duration of the layer backward.
- `infinity/config/training.py` gains `fp32_head_grad` (default on). The FP32
  head-gradient accumulator costs 4 bytes per `lm_head` element on each GPU
  (~2 GB for a 128k x 4096 head), which matters on the memory-constrained setups
  layer streaming exists to serve, so it can be turned off.
- `infinity/model/cpu_master.py` idles surplus workers instead of raising when a
  batch has fewer rows than there are GPUs. `dataloader_drop_last` defaults to
  false, so the trailing batch of an epoch is short whenever the dataset does not
  divide evenly, and failing there would kill the run after a full epoch.
- `infinity/model/cpu_master.py` and `infinity/model/mp_worker.py` leave
  `grad=None` for head/norm parameters that never received a gradient rather than
  writing a zero, so the slab presence bitmap keeps skipping them and the
  optimizer does not apply decay to untouched parameters.
- `infinity/model/cpu_master.py` shards `_forward_logits_multiprocess` with
  `_shard_bounds` too; it previously dropped the remainder onto the last rank and
  handed empty slices to the others when the batch was smaller than the worker
  count.
- `infinity/model/mp_worker.py` drains its gradient queue in a `finally` and
  clears the recorded worker error after every reply, so a failed step cannot
  leave in-flight writes landing in shared gradients or report a stale traceback
  against a later command.
- `infinity/config/training.py` requires the batch to hold at least one sample
  per worker instead of being exactly divisible by the worker count, matching
  the remainder-distributing `_shard_bounds` split.
- `infinity/model/mp_state.py` tries versioned `libcudart` sonames (a pip
  `torch` ships `libcudart.so.12`, not the unversioned name) and declares
  `argtypes`/`restype` before calling `cudaHostRegister`. It also verifies that
  the shared-gradient list length matches the parameters it re-attached.

### Removal of the VERL-only API surface

`verl/` — the upstream RL stack that is the sole caller of these methods — was
never vendored, so the following were removed from `infinity/model/`:

- `forward_and_backward_custom_loss()`, used only by VERL's GRPO loss.
- `forward_logits()`, `_forward_hidden()`, `_forward_logits_multiprocess()`, and
  the worker's `_run_forward_logits()`, used only for VERL rollout log-probs.
- `rebuild_gpu_buffers()`, `_rebuild_gpu_buffers_multiprocess()`, and the
  worker's `_worker_rebuild_gpu()`, used only by VERL to free VRAM for a
  colocated rollout engine and rebuild afterwards. `release_gpu_buffers()` is
  retained because the trainer calls it on shutdown.
- The now-unreachable `FORWARD_LOGITS` and `REBUILD_GPU` worker commands and the
  `WorkerResult.logits` field.
- `_prepare_4d_causal_mask()` (superseded by `_prepare_attention_mask`) and
  `zero_grad()` (Hugging Face drives gradient zeroing).

Re-adding them alongside a future VERL integration is mechanical: they are a
matched set and this file pins the upstream commit.

- `infinity/model/cpu_master.py` writes merged vision embeddings into the
  destination tensor by index. `merged[img_mask][:n] = ...` assigns into the
  temporary that boolean indexing returns, so image embeddings were silently
  discarded and the model saw text embeddings at every image position.
- `infinity/model/cpu_master.py` and `infinity/model/mp_worker.py` pair the
  CPU-to-GPU parameter-sync iterators with `zip(..., strict=True)`, so an
  unexpected architecture fails loudly instead of syncing a truncated set of
  weights.

### Known remaining gaps

- `CPUMasterModel._prepare_4d_causal_mask` is retained from upstream but is no
  longer referenced. It builds a plain causal-or-padding mask and would drop
  sliding-window attention for `sdpa` and `eager`; use `_prepare_attention_mask`.
- The vision/VLM path in `infinity/model/cpu_master.py` is not used by Axolotl
  (the plugin rejects multimodal models). The embedding merge is fixed, but the
  encoder and projector still run under `no_grad`, so the path is incomplete.
Any future compatibility or behavioral change must be listed here with the
affected files and rationale.

## Updating from upstream

1. Choose and record a new immutable upstream commit.
2. Re-run `git archive <commit> infinity LICENSE` into a clean temporary
   directory.
3. Remove `infinity/cuda_pipeline/`, all caches, compiled artifacts, build
   directories, and package metadata.
4. Rewrite absolute `infinity` imports to the private Axolotl namespace and
   verify that no top-level `infinity` import remains.
5. Diff the result against this snapshot, review every upstream change, and
   reapply any hand edits documented above.
6. Replace the vendored tree and license, then update the commit, versions,
   retrieval date, prune list, and local-edit record in this file.
7. Run the vendored import checks, integration tests, package build inspection,
   and Axolotl lint suite.
