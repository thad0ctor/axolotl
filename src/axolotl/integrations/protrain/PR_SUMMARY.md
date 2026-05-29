<!--- Provide a general summary of your changes in the Title above -->

# Description

This PR adds an opt-in ProTrain integration for Axolotl under
`src/axolotl/integrations/protrain/`.

ProTrain provides automatic chunked parameter residency, CPU offload, Mode A/B/C
execution, block-level activation checkpoint/offload policy, and
checkpoint/resume support for Axolotl LoRA/QLoRA and bounded full-FT workloads,
meaning full fine-tune shapes validated within the documented hardware and
sequence-length limits.

It is disabled by default. Runtime behavior requires both:

```yaml
plugins:
  - axolotl.integrations.protrain.ProTrainPlugin
protrain_auto_memory: true
```

Listing the plugin alone only registers schema. It does not alter model loading,
optimizer creation, embedding dtype conversion, checkpointing, or runtime hooks.
Existing Axolotl configs that do not enable this plugin should behave unchanged.

The integration deliberately rejects conflicting memory backends and unsafe
settings rather than silently composing them.

Detailed implementation notes, feature scope, and benchmark evidence are in
[`PR_PROPOSAL.md`](PR_PROPOSAL.md).

## Motivation and Context

Axolotl users can hit 24 GiB GPU limits when fine-tuning larger LoRA/QLoRA or
bounded full-FT workloads. ProTrain adds an Axolotl-native memory manager that
keeps the normal config-driven training flow while choosing between:

| Mode | Purpose |
|---|---|
| Mode A | GPU-resident execution when the selected model/training shape fits. |
| Mode B | Replicated CPU offload when GPU memory is binding and each rank has enough host RAM for the non-persistent chunks. |
| Mode C | ZeRO-3-style sharded CPU offload when replicated CPU offload would exceed per-rank host RAM or a sharded layout is explicitly requested. |

The main resume contract is intentionally conservative:

| Resume path | Status |
|---|---|
| Normal save/resume | Supported. |
| Same-world optimizer resume | Supported when `protrain_save_optimizer_state: true`. |
| Cross-world optimizer resume | Advanced opt-in via `protrain_allow_online_reshard: true`; fails closed without required sidecar metadata. |

### Fail-closed behavior

ProTrain prefers explicit failure over silent fallback. These cases raise at
config, startup, or resume time:

- ProTrain auto-memory enabled without plugin registration.
- DeepSpeed, FSDP, or Axolotl-level `gradient_checkpointing` combined with ProTrain.
- Unsupported optimizer family.
- Multiple forced execution modes enabled.
- Cross-world optimizer resume requested without sidecar metadata.
- PEFT or Transformers API surface outside validated guardrails.
- Runtime calibrated memory prediction exceeds configured device capacity.

## How has this been tested?

The committed validation runner is the reviewer-scale acceptance path:

```bash
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite maintainer
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite full --gpu-devices 1,2,4,5,7 --keep-cache
```

Latest recorded validation:

GPU lanes are opt-in/self-hosted; standard CI remains CPU-only.

| Lane | Result | Coverage |
|---|---|---|
| Default ProTrain pytest | `632 passed, 17 skipped, 180 deselected` | CPU-safe unit and integration coverage. |
| `cpu-core` | PASS | Validators, cost/search math, calibrated memory gate, metadata, layout determinism, non-finite guard. |
| `cpu-surface` | PASS | Mode selection, force-mode safety, save/resume hooks, Path B LoRA ownership, debug/watchdog hooks. |
| `merge-surface` | PASS | `merge-lora` CLI and LoRA/QLoRA/rsLoRA/DoRA/MoE merge math. |
| `single-gpu-edge` | PASS | DoRA, multi-adapter switching, vision-LM ownership, LoRA offload runtime, A<->C resume. |
| `single-gpu` | PASS | 8B QLoRA 50-step train, checkpoint, 10-step resume, finite losses, bounded loss continuity. |
| `two-gpu` | PASS | Forced Mode C finite train, same-world optimizer save/resume, non-finite boundary checks. |

Critical regression coverage:

| Area | Covered by |
|---|---|
| Plugin inertness and schema-only registration | CPU/default tests. |
| DeepSpeed/FSDP/gradient-checkpointing validator rejection | CPU/default tests. |
| Calibrated memory gate | `cpu-core`. |
| Trainer resume hook restore/re-offload ordering | `cpu-surface`. |
| PEFT LoRA container hook shape preservation | CPU/default and `single-gpu-edge`. |
| Mode C DDP bypass | `cpu-surface` and `two-gpu`. |
| Same-world optimizer resume | `two-gpu`. |
| Cross-world fail-closed behavior without sidecars | CPU/default focused tests. |

Representative hardware validation summarized from `PR_PROPOSAL.md`:

| Shape | Result |
|---|---|
| 8B BF16 LoRA on 24 GiB | Resident memory drops from 15.83 GiB to 3.08 GiB. |
| Llama-13B 4-bit LoRA on one 3090 | Fits through seq=2048. |
| Qwen3.5-27B 4-bit LoRA on one 3090 | Fits at seq=128; Mode B reaches seq=256. |
| Qwen3.5-4B full-FT forced Mode C | Local 5x 3090-class train/save/resume coverage. |
| Qwen3.5-9B full-FT forced Mode C | High-memory 2-rank train/save/resume coverage. |

## AI Usage Disclaimer

Yes. ChatGPT/Codex and Claude were used to assist with implementation,
debugging, test orchestration, documentation drafting, and review triage. Human
review, local execution, and commit decisions were performed by the PR author.

## Screenshots (if appropriate)

Not applicable.

## Types of changes

- [x] New feature
- [x] Performance improvement
- [x] Documentation update
- [x] Tests / validation coverage
- [ ] Bug fix

## Social Handles (Optional)
