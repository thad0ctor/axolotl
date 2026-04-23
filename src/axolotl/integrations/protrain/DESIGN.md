## Purpose

This package is a from-scratch Python implementation of the ProTrain memory manager (MLSys 2026, arXiv 2406.08334), shipped as an **Axolotl plugin** (`BasePlugin` subclass). It owns per-rank memory policy on top of ZeRO-3: hierarchical chunk management for model states (params / grads / optim states), interleaved block management for activations, a memory-aware profiler, a 4-knob cost model, and an automatic searcher. It does NOT own data parallelism collectives (delegates to `torch.distributed`), training-loop control flow, trainer orchestration, TP/PP, FP8, or any changes to Axolotl core files. Activation is opt-in via `plugins: [axolotl.integrations.protrain]` in the user YAML; mutual exclusion with `deepspeed:` and `fsdp:` is enforced by a pydantic validator in `args.py`.

## Directory Layout

```
src/axolotl/integrations/protrain/
├── __init__.py                  # re-exports ProTrainArgs + ProTrainPlugin
├── DESIGN.md                    # this file
├── plugin.py                    # BasePlugin subclass: get_input_args / post_model_load / create_optimizer
├── args.py                      # ProTrainArgs pydantic model + DS/FSDP mutex validator
├── types.py                     # shared dataclasses (ProfilerTrace, ChunkLayout, ...)
├── profiler/
│   ├── __init__.py
│   ├── trace.py                 # single-iter forward/backward hook driver
│   ├── memory_deltas.py         # intra-op + inter-op Δ capture via cuda.memory_stats
│   ├── on_demand.py             # allocate-before-use / free-after tensor mode
│   ├── hw_bench.py              # H2D/D2H + NCCL gather/reduce microbenchmarks
│   └── cache.py                 # on-disk cache keyed by (arch_hash, bs, seq, sku, world)
├── chunk/
│   ├── __init__.py
│   ├── layout.py                # param→chunk assignment, exec-order intra-chunk reorder
│   ├── sizing.py                # S_chunk grid search over {32,64,128,256} MB
│   ├── manager.py               # persistent/non-persistent split, gather/offload drivers
│   ├── buffer_pool.py           # pre-allocated chunk buffer pool, forward→backward reuse
│   ├── pinned_alloc.py          # ctypes → cudaHostAlloc, precise-size (App B.2)
│   └── optim.py                 # DeepSpeedCPUAdam adapter (non-persist) + GPU FusedAdam (persist)
├── block/
│   ├── __init__.py
│   ├── strategy.py              # BlockMode enum {NONE, CKPT, SWAP}
│   ├── dispatcher.py            # per-block forward wrapper honoring selected mode
│   ├── checkpoint.py            # CKPT path (torch.utils.checkpoint adapter)
│   ├── swap.py                  # SWAP no-op stub gated by PROTRAIN_ENABLE_SWAP env flag
│   └── layout_rules.py          # placement rules: swap-early / unopt-late / interleave
├── cost/
│   ├── __init__.py
│   ├── runtime.py               # Eqs. 2–7, per-chunk max(compute, comm) roofline
│   ├── memory.py                # Eqs. 8–11, op-walk peak + α=1.10 fragmentation
│   └── bandwidth.py             # contention model when n_swap>0 competes with prefetch
├── search/
│   ├── __init__.py
│   ├── knobs.py                 # CostConfig + bound derivation (N_chunk, N_block, N_interval)
│   └── exhaustive.py            # 4-knob enumeration with memory-ascending pruning
├── runtime/
│   ├── __init__.py
│   ├── streams.py               # single-stream alloc scheme (App B.2)
│   ├── scheduler.py             # prefetch / reduce-offload / CPU-step / swap orchestration
│   └── hooks.py                 # install/uninstall fwd/bwd hooks on the user model
└── api/
    ├── __init__.py
    ├── model_wrapper.py         # protrain_model_wrapper() — called from plugin.post_model_load
    └── optim_wrapper.py         # protrain_optimizer_wrapper() — called from plugin.create_optimizer
```

## Module Specs

Every entry: Inputs · Outputs · Paper ref · Milestone.

### plugin.py (M5)

- `class ProTrainPlugin(BasePlugin)` — thin shim.
  - `get_input_args() -> "axolotl.integrations.protrain.args.ProTrainArgs"`.
  - `post_model_load(cfg, model)` — constructs `HardwareProfile`, runs profiler (cached), calls `protrain_model_wrapper(model, ...)`, stashes `WrappedModel` on `cfg` for `create_optimizer` to pick up.
  - `create_optimizer(cfg, trainer) -> Optimizer` — returns `protrain_optimizer_wrapper(wrapped_model)`; returns `None` when plugin is inactive.
  - `post_trainer_create(cfg, trainer)` — installs any trainer-level callbacks if needed for metric reporting.

### args.py (M5)

- `class ProTrainArgs(BaseModel)` — fields: `protrain_auto_memory: bool = True`, optional manual knob overrides `protrain_n_persist / n_buffer / n_swap / n_checkpoint` for debugging, `protrain_cache_dir: Path | None`.
- `model_validator` — rejects `plugins: [...protrain...]` + (`deepspeed` set) or (`fsdp` / `fsdp_config` set). Pattern cloned from `integrations/spectrum/args.py:32-47`.

### profiler/ (M1)

- `trace.py` — `run_trace(model: nn.Module, batch: dict, cfg: ProfilerConfig) -> ProfilerTrace`. Installs pre/post fwd + bwd hooks, records op order, delegates Δ capture. §3.2.
- `memory_deltas.py` — `intra_op_delta(op) -> int`, `inter_op_delta(prev, curr) -> int` from `torch.cuda.memory_stats()`. Catches the ~17% invisible peak. §3.2, App A.2.
- `on_demand.py` — `class OnDemandTensorMgr` context; `allocate_inputs(op)` / `free_after(op)`. Enables profiling models larger than single-GPU. §3.2.
- `hw_bench.py` — `measure_pcie() -> BW`, `measure_nccl(world_size) -> NcclTable`. §3.2.
- `cache.py` — `load(key) -> ProfilerTrace | None`, `save(key, trace)`. Key = `(arch_hash, bs, seq, sku, world)`. §7.

### chunk/ (M2)

- `layout.py` — `build_layout(model, exec_order: list[ParamId], S_chunk: int) -> ChunkLayout`. Groups params per transformer block, reorders intra-chunk by first use, shared params at first occurrence. §3.1.1.
- `sizing.py` — `pick_S_chunk(model_state_sizes: list[int], candidates=(32<<20, 64<<20, 128<<20, 256<<20)) -> int`. Simulates fragmentation waste; returns argmin. App B.1.
- `manager.py` — `class ChunkManager`; `gather(chunk_id)`, `offload(chunk_id)`, `mark_persistent(first_n)`. §3.1.1.
- `buffer_pool.py` — `class BufferPool(n_buffer: int, S_chunk: int)`; `acquire() / release()`; carries forward-resident buffers into backward. §3.1.1, §5.
- `pinned_alloc.py` — `pinned_alloc(n_buffer, S_chunk) -> HostMemory`. `ctypes` → `cudaHostAlloc` with exact byte count. App B.2.
- `optim.py` — wraps `deepspeed.ops.adam.DeepSpeedCPUAdam` for non-persistent chunks, `apex.optimizers.FusedAdam` (or torch `FusedAdam`) for persistent. `step_async(chunk_id)` for CPU path to overlap GPU bwd. §5.

### block/ (M3)

- `strategy.py` — `class BlockMode(Enum){NONE, CKPT, SWAP}`; `BlockStrategyMap = dict[int, BlockMode]`. §3.1.2.
- `dispatcher.py` — `wrap_block(block: nn.Module, mode: BlockMode) -> nn.Module`. §3.1.2.
- `checkpoint.py` — thin wrapper over `torch.utils.checkpoint.checkpoint` (use_reentrant=False). §3.1.2.
- `swap.py` — no-op stub; raises if `PROTRAIN_ENABLE_SWAP` unset and `BlockMode.SWAP` requested. §3.1.2.
- `layout_rules.py` — `assign_modes(n_swap, n_checkpoint, N_block) -> BlockStrategyMap`. Swap-early / unopt-late / interleave. §3.1.2.

### cost/ (M4)

- `runtime.py` — `estimate_runtime(cfg, trace, layout) -> float`. Implements **Eqs. 2–7**: `T_iter = T_fwd + max(T_bwd + T_gpu_optim, T_cpu_optim)`, per-chunk `max(compute, comm)` roofline. §3.3, App A.1.
- `memory.py` — `estimate_peak(cfg, trace, layout, block_map) -> int`. Implements **Eqs. 8–10** (op-walk) and **Eq. 11** (α = 1.10 fragmentation). Bumps at first op of each CKPT block. §3.3, App A.2.
- `bandwidth.py` — `effective_bw(cfg, hw) -> float`. Derates prefetch BW when `n_swap > 0`. §3.3.

### search/ (M4)

- `knobs.py` — `CostConfig` dataclass + `derive_bounds(trace, layout) -> Bounds(N_chunk, N_block, N_interval)`. §3.3.
- `exhaustive.py` — `search(trace, layout, capacity_bytes) -> SearchResult`. Enumerates 4-tuple in memory-ascending order, prunes OOM, returns argmin(T_iter). §3.3.

### runtime/ (M2+M3 integration)

- `streams.py` — single-default-stream allocator, manual dealloc sync. App B.2.
- `scheduler.py` — orchestrates (a) param prefetch, (b) grad reduce+offload, (c) CPU optimizer step, (d) activation swap. Respects `cost/bandwidth.py` budgets. §5, §6.
- `hooks.py` — `install(model)` / `uninstall()`; wires chunk & block managers into fwd/bwd. §1.

### api/ (M4)

- `model_wrapper.py` — `protrain_model_wrapper(model, model_config, hardware_profile) -> WrappedModel`. §1.
- `optim_wrapper.py` — `protrain_optimizer_wrapper(wrapped_model) -> Optimizer`. §1.

## Key Data Structures

All live in `types.py`. Fields expand during M1–M4:

```python
@dataclass(frozen=True)
class ProfilerTrace:
    op_order: list[OpRecord]                  # per-op: id, module_path, shape_sig
    intra_op_delta: dict[OpId, int]           # bytes
    inter_op_delta: dict[OpId, int]           # bytes
    activation_sizes: dict[BlockId, int]
    model_state_bytes: int
    pcie_h2d_bps: float
    pcie_d2h_bps: float
    nccl_gather_s: dict[int, float]
    nccl_reduce_s: dict[int, float]
    arch_hash: str; bs: int; seq: int; sku: str; world: int

@dataclass(frozen=True)
class ChunkLayout:
    S_chunk: int
    N_chunk: int
    chunks: list[list[ParamId]]
    param_to_chunk: dict[ParamId, int]
    block_to_chunks: dict[BlockId, list[int]]

BlockStrategyMap = dict[int, BlockMode]

@dataclass(frozen=True)
class CostConfig:
    n_persist: int
    n_buffer: int
    n_swap: int
    n_checkpoint: int

@dataclass(frozen=True)
class SearchResult:
    cfg: CostConfig
    block_map: BlockStrategyMap
    predicted_peak_bytes: int
    predicted_iter_s: float
```

## Plugin Integration (M5)

Zero diffs to Axolotl core files. The entire Axolotl surface consumed:

- `BasePlugin` subclass at `src/axolotl/integrations/protrain/plugin.py`
- `get_input_args` returns `ProTrainArgs` → pydantic merge handled by `axolotl/utils/schemas/config.py:1275` (`plugins:` field)
- `post_model_load(cfg, model)` hook — wraps post-LoRA so frozen LoRA base params contribute to persistent-chunk memory only
- `create_optimizer(cfg, trainer)` hook — returns ProTrain optimizer; `None` if disabled
- Example YAML: `examples/protrain/3090-7b-lora.yml` — opts in via `plugins: [axolotl.integrations.protrain]`

## Cross-Module Dependency Graph

- `types.py` — depended on by everyone; depends on nothing.
- `profiler/*` — independent (M1). Depends only on `types.py` and `torch`.
- `chunk/*` — independent of profiler and block (M2). Uses `runtime/streams.py` and `runtime/hooks.py`.
- `block/*` — independent of profiler and chunk (M3). Uses `runtime/hooks.py`.
- `cost/*` — reads `ProfilerTrace` + `ChunkLayout` + `BlockStrategyMap` as **data**; no code-level dep on chunk/block internals (M4).
- `search/*` — depends on `cost/*` and `types.py` only (M4).
- `api/*` — depends on everything; built last.
- `plugin.py` — consumes `api/*` only; M5. Supports M1→M4 parallel fan-out: profiler, chunk, block run concurrently; cost+search starts once `ProfilerTrace` schema is frozen at end of M1.

## Out of Scope

Mirrors `plan.md`:
- A100/H100, NVLink, InfiniBand, multi-node
- TP, PP, any non-ZeRO-3 parallelism
- FP8/FP4, quantization, FlashAttention variants
- Windows / macOS
- Edits to Axolotl core files outside this plugin package — ProTrain is additive, DeepSpeed/FSDP/Unsloth paths unchanged

## Design Decisions (previously open questions, now resolved)

1. **α fragmentation factor = 1.10** — matches paper's "up to 10% overestimate" (§3.3). M1 records ground truth; M4 can recalibrate if observed 3090 fragmentation diverges.
2. **Pinned-memory allocator:** `ctypes` → `cudaHostAlloc` directly. ~50 LOC, zero new deps, matches App B.2 precisely (avoids `CUDAHostAllocator` pow-2 rounding). DeepSpeed's `PinnedMemoryAllocator` rejected: may inherit same wart, adds import-graph weight.
3. **CPU FusedAdam source:** `deepspeed.ops.adam.DeepSpeedCPUAdam`. Paper builds directly on ZeRO-Offload's CPU Adam. Pure-Python reimpl is >10× slower and would collapse the T_bwd / T_cpu_optim overlap window the cost model assumes. DeepSpeed is already in Axolotl's env.
4. **S_chunk grid:** `{32, 64, 128, 256} MB`. 7B Llama blocks are ~200 MB fp16 → chunks want to be block-scale. 16 MB is too fine-grained; per-chunk sync overhead dominates. M2 agent extends the grid if optimum lands at an endpoint.
5. **SWAP path:** no-op stub gated by `PROTRAIN_ENABLE_SWAP` env flag. Searcher test asserts `n_swap=0` is selected on 3090. ~30 LOC; exercises M4 bound logic end-to-end. Deletable if M6 confirms we never need it.
