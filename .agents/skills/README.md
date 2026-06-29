# Agent Skills

Axolotl ships **agent skills** — self-contained workflow guides (plus optional
scripts) that AI coding assistants can run for repetitive, repo-specific tasks.
Each skill is one directory under `.agents/skills/` with a `SKILL.md`. See
[`AGENTS.md`](../../AGENTS.md) for the current list.

There is **nothing to install** — skills are committed to the repo and available
on clone.

## Using a skill

- **Claude Code** auto-discovers them (via the `.claude/skills` → `.agents/skills`
  symlink). It loads a skill automatically when your request matches the skill's
  `description`, or you can invoke one explicitly by name, e.g.
  `/liger-upstream-sync` or `/model-verification-harness`.
- **Codex / Gemini / Antigravity** read `.agents/skills/` natively — same files,
  no symlink needed.

Skills are **on by default**; run `/skills` to toggle them off/on (Codex,
Antigravity, and Claude Code).

## Skills

Each skill's full reference (stages, output interpretation, all flags) lives in its
`SKILL.md`. Quick-start usage per skill:

### `liger-upstream-sync`

Audits axolotl's Liger integration (`src/axolotl/integrations/liger/plugin.py`)
against `liger-kernel` to catch silent dispatch drift — hand-patches that upstream
now shadows with native dispatch, stale `axolotl_override_liger_fn` entries, and
signature drift. Full reference: [`liger-upstream-sync/SKILL.md`](liger-upstream-sync/SKILL.md).

**Trigger it** — in an assistant that auto-discovers skills, just describe the task;
the skill matches on phrasings like:

- "launch liger skill" / "launch the liger skill" / "run the liger sync skill"
- "update the liger version" / "bump liger-kernel to X.X.X" (or no version) / "upgrade the liger dependency"
- "help me bump liger — what breaks / what will it shadow?"
- "audit the liger integration" / "check liger dispatch drift"
- "this liger-patched model trains but the loss/throughput looks off"
- "review the changes to `integrations/liger/`"
- "did upstream liger add native support for a model we still hand-patch?"

Or invoke it explicitly by name: **`/liger-upstream-sync`**.

**Audit the installed liger** (run in the training env — it introspects the
*installed* liger-kernel, so it reports against the exact version you train with):

```bash
python .agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py
```

Exit code is `0` clean / `1` findings / `2` could-not-audit, so it can gate CI.

**Analyze a new liger version when bumping the pin.** The simplest, most complete
check is to just install the target version and re-run the plain audit so it runs
the **full** check including `[4]` signature drift:

```bash
pip install -U "liger-kernel==0.8.0"   # the version you're bumping the pin to
python .agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py
```

If you don't want to change your environment, preview the version's source
without installing via `--liger-source`. This is keys-only, so signature-
drift check `[4]` is skipped but `[1]`–`[3]` run normally:

```bash
# Option A — download the wheel without installing, then unzip it
pip download liger-kernel==0.8.0 --no-deps -d /tmp/liger-dl
unzip -o /tmp/liger-dl/liger_kernel-*.whl -d /tmp/liger-new
python .agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py \
  --liger-source /tmp/liger-new

# Option B — clone the upstream repo at a release tag
git clone --depth 1 --branch v0.8.0 https://github.com/linkedin/Liger-Kernel /tmp/Liger-Kernel
python .agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py \
  --liger-source /tmp/Liger-Kernel
```

`--liger-source` accepts the `monkey_patch.py` file, a package dir, an unzipped
wheel, or a repo checkout — it locates `liger_kernel/transformers/monkey_patch.py`
within whatever you give it. Upstream repo:
[`linkedin/Liger-Kernel`](https://github.com/linkedin/Liger-Kernel).

### `model-verification-harness`

A reproducible smoke gate for model-support PRs. Detects `model_config_type` +
architecture features (MoE / multimodal / SSM-hybrid), then runs a tiered gate
ladder — config-resolution compat matrix (liger/CCE/kernel flags via the real
`load_cfg`), an adaptive wired-everywhere integration checklist, preprocess +
chat-template masking + sample-packing checks, and (on GPU) loss-sanity +
cross-variant numerical consistency that catches silent kernel-shadowing. Emits
a standardized report + manifest with a `0`/`1`/`2` exit code.

It is a heuristic starting point, not an oracle: every finding (and every PASS)
should be verified independently against the model's own code and the PR diff
before acting.

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

Exit code is `0` clean / `1` findings / `2` could-not-run (no GPU / kernel won't
compile / model unloadable), so it can gate CI. Invoke the script directly (it has
the `__main__` guard the dataset multiprocess save needs), not via `python -c`.

Or invoke it explicitly by name: **`/model-verification-harness`**.

## Adding a skill

Create `.agents/skills/<your-skill>/SKILL.md` with `name` (matching the directory,
lowercase-hyphen) and `description` frontmatter, put any helpers under that
directory, and add a row to the table in [`AGENTS.md`](../../AGENTS.md). The
canonical copy always lives under `.agents/skills/`; vendor paths like
`.claude/skills` are symlinks to it — never duplicate skill content.
`tests/test_agent_skill_layout.py` enforces this layout.

Skills must adhere to [Anthropic's Skill authoring best practices](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices)
— a concise `SKILL.md` (under ~500 lines), a third-person `description` that states
both what the skill does and when to use it, progressive disclosure (split detail
into sibling files referenced one level deep), and forward-slash paths. Frontmatter
must stay within the portable [agentskills.io](https://agentskills.io/specification)
core (`name` ≤64 chars, `description` ≤1024) so the skill works across Claude Code,
Codex, and other agents.

> **Windows note:** the `.claude/skills` symlink materializes correctly only if git
> symlinks are enabled (`git config core.symlinks true`, plus Developer Mode or
> admin). Otherwise use the native `.agents/skills/` path or run the scripts directly.
