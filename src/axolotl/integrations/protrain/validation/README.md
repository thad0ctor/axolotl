# ProTrain Validation

Run from the repo root with this worktree on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite maintainer
```

This is not a request for mandatory GPU CI in upstream GitHub Actions. The CPU
lanes are suitable for ordinary CI; the GPU lanes are an opt-in validation
command for reviewers or developers with qualifying CUDA hardware. Missing GPU
hardware is reported as `SKIP`, not hidden and not treated as a failure.

Suites:

| Suite | Hardware | What it proves |
|---|---:|---|
| `cpu-core` | CPU | Validators, cost/search math, calibrated memory gates, checkpoint metadata, chunk layout determinism, optimizer metadata, model-family ownership, and non-finite optimizer-boundary guards. |
| `cpu-surface` | CPU | Mode A/B/C selection, force-mode guards, Mode C DDP bypass, full-FT save hooks, resume lifecycle, Path B LoRA ownership/parity, alpha logs, and debug/watchdog hooks. |
| `merge-surface` | CPU | `merge-lora` CLI dispatch plus LoRA, QLoRA, ProTrain block-key, rsLoRA, DoRA, and MoE merge math. |
| `maintainer` | CPU | `cpu-core` + `cpu-surface` + `merge-surface`; the best default pre-review lane. |
| `cpu-full` | CPU | The full default ProTrain pytest suite. |
| `single-gpu` | 1x 24 GiB | 8B QLoRA train for 50 steps, checkpoint, resume, finite logged losses, finite logged grad norms when present, and resume loss continuity. |
| `single-gpu-edge` | 1x 24 GiB | DoRA, multi-adapter switching, vision-LM hybrid ownership, LoRA offload runtime, and A<->C cross-mode resume on tiny models. |
| `two-gpu` | 2 GPUs | Forced Mode C finite training, same-world optimizer save/resume, and numerical-stability checks using the maintained pytest entry points. |
| `full` | CPU + GPUs | Runs all lanes in order. Missing hardware is reported as `SKIP`, not hidden. |

Useful flags:

```bash
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite full --dry-run
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite maintainer --work-dir /tmp/protrain-validation
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite single-gpu --gpu-devices 0 --work-dir /tmp/protrain-validation
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite two-gpu --gpu-devices 0,1 --work-dir /tmp/protrain-validation
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite full --json
```

For a private/self-hosted CI loop, run `--suite full --json` on any qualifying
machine and archive the log directory. The process exits non-zero only on a
failed lane; unavailable hardware remains an explicit `SKIP`.

Acceptance gates are intentionally mechanical: process exit status, expected
checkpoint artifacts, finite logged losses, finite logged grad norms when
present, no `nan`/`inf` markers in train logs, and bounded resume loss
continuity. The runner prints any untested area as a gap so reviewers can
distinguish failed validation from unavailable hardware.

Coverage map:

| Historical/local validation theme | Durable lane |
|---|---|
| P0/P1 blockers from `FinalReview.md` | `maintainer` |
| Mode A/B/C selector and force-mode safety | `cpu-surface` |
| Save/resume, optimizer metadata, safetensors save hooks | `cpu-core`, `cpu-surface`, `single-gpu` |
| Path B LoRA sync and ownership | `cpu-surface` |
| Merge-lora command/reload surface | `merge-surface` |
| Numerical stability and NaN/Inf fail-closed behavior | `cpu-core`, `single-gpu`, `two-gpu` |
| PEFT edge cases from local coverage audits | `single-gpu-edge` |
| 8B QLoRA 24 GiB maintainer acceptance | `single-gpu` |
| Forced Mode C multi-rank maintainer acceptance | `two-gpu` |

Large historical benchmarks such as 30B/35B/9B long runs remain proposal
evidence, not maintainer acceptance gates.
