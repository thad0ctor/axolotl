# Agent Skills

Axolotl ships **agent skills** — self-contained workflow guides (plus optional
scripts) that AI coding assistants can run for repetitive, repo-specific tasks.
Each skill is one directory under `.agents/skills/` with a `SKILL.md`.

There is **nothing to install** — skills are committed to the repo and available
on clone.

## Using a skill

- **Claude Code** auto-discovers them (via the `.claude/skills` → `.agents/skills`
  symlink). It loads a skill automatically when your request matches the skill's
  `description`, or you can invoke one explicitly by name, e.g.
  `/sparselora-add-model`.
- **Codex / Gemini / Antigravity** read `.agents/skills/` natively — same files,
  no symlink needed.

Skills are **on by default**; run `/skills` to toggle them off/on.

## Skills

Each skill's full reference (stages, output interpretation, all flags) lives in
its `SKILL.md`. Quick-start usage per skill:

### `sparselora-add-model`

Adds or verifies a model architecture's SparseLoRA sparse wiring
(`src/axolotl/integrations/sparselora/arch_wiring.py`). SparseLoRA is
model-agnostic for SwiGLU-MLP + standard-attention transformers (Llama, Qwen2,
Qwen3, Mistral, ...), which auto-register at apply time; this skill confirms a
new architecture works and guides the rare case that needs custom attention
wiring. Full reference:
[`sparselora-add-model/SKILL.md`](sparselora-add-model/SKILL.md).

**Trigger it** — in an assistant that auto-discovers skills, just describe the
task; the skill matches on phrasings like:

- "launch the sparselora add-model skill" / "run the sparselora model skill"
- "make SparseLoRA work on Qwen3 / Mistral / Phi / `<arch>`"
- "does SparseLoRA support `<model>`?"
- "SparseLoRA says 'wiring not available' / 'cannot sparsify this model'"
- "this model trains under SparseLoRA but the loss looks wrong"
- "review the changes to `integrations/sparselora/arch_wiring.py`"

Or invoke it explicitly by name: **`/sparselora-add-model`**.

**Audit an architecture** (run on a GPU for the sparse forward/backward smoke
test; CPU does structural checks only):

```bash
# synthetic tiny model, no download (llama|qwen2|qwen3|mistral|gemma2)
python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py --arch qwen3

# any HuggingFace id — built from its config, no checkpoint, capped layers
python .agents/skills/sparselora-add-model/scripts/audit_sparselora_model.py \
  --base-model Qwen/Qwen2.5-0.5B
```

Exit code is `0` supported / `1` not supported / `2` could not audit, so it can
gate CI.

## Adding a skill

Create `.agents/skills/<your-skill>/SKILL.md` with `name` (matching the
directory, lowercase-hyphen) and `description` frontmatter, put any helpers under
that directory, and add a row to the table in
[`AGENTS.md`](../../AGENTS.md). The canonical copy always lives under
`.agents/skills/`; vendor paths like `.claude/skills` are symlinks to it — never
duplicate skill content. `tests/test_agent_skill_layout.py` enforces this layout.

Skills must adhere to [Anthropic's Skill authoring best practices](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices)
— a concise `SKILL.md` (under ~500 lines), a third-person `description` that
states both what the skill does and when to use it, progressive disclosure, and
forward-slash paths. Frontmatter must stay within the portable
[agentskills.io](https://agentskills.io/specification) core (`name` ≤64 chars,
`description` ≤1024) so the skill works across Claude Code, Codex, and other
agents.

> **Windows note:** the `.claude/skills` symlink materializes correctly only if
> git symlinks are enabled (`git config core.symlinks true`, plus Developer Mode
> or admin). Otherwise use the native `.agents/skills/` path or run the scripts
> directly.
