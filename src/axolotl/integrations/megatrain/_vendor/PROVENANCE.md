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
- Every tracked `infinity/**/__pycache__/` directory and its `.pyc` files.

Nothing else was removed from the archived `infinity` tree. In particular,
`infinity/csrc/__init__.py` remains because it safely treats the optional
top-level `infinity_memory_ops` extension as unavailable. The repository-level
`build/`, `csrc/`, `verl/`, and egg-info trees were never selected for the
archive and are not part of this vendored copy.

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
