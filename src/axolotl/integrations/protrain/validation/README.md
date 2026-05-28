# ProTrain Validation

Run from the repo root with this worktree on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite cpu-core
```

Suites:

| Suite | Hardware | What it proves |
|---|---:|---|
| `cpu-core` | CPU | Validators, cost/search math, checkpoint metadata, chunk layout determinism, optimizer metadata, model-family ownership, and non-finite optimizer-boundary guards. |
| `cpu-full` | CPU | The full default ProTrain pytest suite. |
| `single-gpu` | 1x 24 GiB | 8B QLoRA train for 50 steps, checkpoint, resume, finite loss/grad-norm checks, and resume loss continuity. |
| `two-gpu` | 2 GPUs | Forced Mode C finite training plus resume/numerical-stability checks using the maintained pytest entry points. |
| `full` | CPU + GPUs | Runs all lanes in order. Missing hardware is reported as `SKIP`, not hidden. |

Useful flags:

```bash
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite full --dry-run
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite single-gpu --gpu-devices 0 --work-dir /tmp/protrain-validation
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite full --json
```

Acceptance gates are intentionally mechanical: process exit status, expected
checkpoint artifacts, finite logged losses, finite logged grad norms, no
`nan`/`inf` markers in train logs, and bounded resume loss continuity. The
runner prints any untested area as a gap so reviewers can distinguish failed
validation from unavailable hardware.
