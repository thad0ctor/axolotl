# Agent Skills

Axolotl ships **agent skills** — self-contained workflow guides (plus optional scripts) that AI coding assistants can run for repetitive, repo-specific tasks. Each skill is one directory under `.agents/skills/` with a `SKILL.md`. See [`AGENTS.md`](../../AGENTS.md) for the current list.

There is **nothing to install** — skills are committed to the repo and available on clone.

## Using a skill

- **Claude Code** auto-discovers them (via the `.claude/skills` → `.agents/skills` symlink). It loads a skill automatically when your request matches the skill's `description`, or you can invoke one explicitly by name, e.g. `/model-verification-harness`.
- **Codex / Gemini / Antigravity** read `.agents/skills/` natively — same files, no symlink needed.

Skills are **on by default**; run `/skills` to toggle them off/on (Codex, Antigravity, and Claude Code).

## Skills

Each skill's full reference (stages, output interpretation, all flags) lives in its `SKILL.md`. Quick-start per skill:

### `model-verification-harness`

A reproducible smoke gate for model-support PRs. Detects `model_config_type` + architecture features (MoE / multimodal / SSM-hybrid), then runs a tiered gate ladder — config-resolution compat matrix (liger/CCE/kernel flags via the real `load_cfg`), an adaptive wired-everywhere integration checklist, preprocess + chat-template masking + sample-packing checks, and (on GPU) loss-sanity + cross-variant numerical consistency that catches silent kernel-shadowing. Emits a standardized report + manifest with a `0`/`1`/`2` exit code.

It is a heuristic starting point, not an oracle: every finding (and every PASS) should be verified independently against the model's own code and the PR diff before acting.

```bash
# CPU gates (config / integration / preprocess / masking / packing / coverage)
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <hf-id-or-path> --gates G1,G2,G3,G4,G5,G8 --report report.md

# auto-discover what a model-support PR touches, then verify it
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --from-pr <url-or-number> --gates all

# add GPU gates (loss sanity + numerical consistency) with a pinned device
python .agents/skills/model-verification-harness/scripts/verify_model.py \
  --base-model <hf-id-or-path> --gates all --profile full --gpus 1
```

Exit code is `0` clean / `1` findings / `2` could-not-run (no GPU / kernel won't compile / model unloadable), so it can gate CI. Invoke the script directly (it has the `__main__` guard the dataset multiprocess save needs), not via `python -c`.

Or invoke it explicitly by name: **`/model-verification-harness`**.
