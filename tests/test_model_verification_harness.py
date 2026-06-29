"""CPU-only, offline tests for the static gates (G1/G2/G8), detection, the exit-code contract, and the report/manifest renderer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / ".agents" / "skills" / "model-verification-harness" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from harness import (  # noqa: E402
    GateContext,
    GateResult,
    GateStatus,
    ModelFeatures,
    exit_code,
)


def _write_config(tmp_path: Path, name: str, cfg: dict) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _features(model_config_type: str, base_model: str, **kw) -> ModelFeatures:
    return ModelFeatures(
        model_config_type=model_config_type, base_model=base_model, **kw
    )


def _ctx(features: ModelFeatures, tmp_path: Path, **options) -> GateContext:
    return GateContext(
        features=features,
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "out",
        selected_gates={"G1", "G2", "G8"},
        options=options,
    )


# --- exit-code contract --------------------------------------------------------


def test_exit_code_aggregation():
    def r(status):
        return GateResult("Gx", "x", status)

    assert exit_code([r(GateStatus.PASS), r(GateStatus.SKIPPED)]) == 0
    assert exit_code([r(GateStatus.PASS), r(GateStatus.FINDINGS)]) == 1
    # could-not-run dominates findings
    assert exit_code([r(GateStatus.FINDINGS), r(GateStatus.COULD_NOT_RUN)]) == 2


def test_exit_code_on_unavailable_skip():
    def r(status):
        return GateResult("Gx", "x", status)

    # skip policy: could-not-run is non-blocking (reflects only real findings)
    assert exit_code([r(GateStatus.COULD_NOT_RUN)], on_unavailable="skip") == 0
    assert exit_code([r(GateStatus.FINDINGS), r(GateStatus.COULD_NOT_RUN)], "skip") == 1


# --- verify_model.main(argv) exit-code contract (no GPU; static gates) ----------


def test_main_g1_exit_clean(tmp_path):
    import verify_model

    d = _write_config(
        tmp_path,
        "llama",
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    code = verify_model.main(
        [
            "--base-model",
            str(d),
            "--gates",
            "G1",
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ]
    )
    # G1 resolves on a healthy llama -> clean (0); if load_cfg is unavailable it self-reports could-not-run (2). Never crash.
    assert code in (0, 2)


def test_main_g2_findings_exit_one(tmp_path):
    import verify_model

    # unknown type with architectures -> G2 model-loads/liger/CCE generic-fallback -> finding -> exit 1 (static-gate contract via main())
    d = _write_config(
        tmp_path,
        "novel",
        {"model_type": "znovelarch_test", "architectures": ["ZNovelArchForCausalLM"]},
    )
    code = verify_model.main(
        [
            "--base-model",
            str(d),
            "--gates",
            "G2",
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ]
    )
    assert code == 1


# --- detection -----------------------------------------------------------------


def test_detect_llama(tmp_path):
    from harness.detect import detect_model

    d = _write_config(
        tmp_path,
        "llama",
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    f = detect_model(str(d), REPO_ROOT)
    assert f.model_config_type == "llama"
    assert not f.is_moe and not f.is_multimodal and not f.is_ssm_hybrid
    assert f.needs_patch  # llama has a flash-attn branch in patch_manager


def test_detect_moe_and_multimodal(tmp_path):
    from harness.detect import detect_model

    moe = _write_config(
        tmp_path,
        "moe",
        {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_experts": 8,
        },
    )
    fm = detect_model(str(moe), REPO_ROOT)
    assert fm.is_moe and fm.extra.get("num_experts") == 8

    mm = _write_config(
        tmp_path,
        "mm",
        {
            "model_type": "qwen2_vl",
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "vision_config": {"depth": 2},
        },
    )
    fmm = detect_model(str(mm), REPO_ROOT)
    assert fmm.is_multimodal


# --- G2 integration ------------------------------------------------------------


def test_g2_llama_passes(tmp_path):
    from harness.gates import g2_integration

    res = g2_integration.run(_ctx(_features("llama", "x"), tmp_path))
    assert res.gate_id == "G2"
    assert res.status in (GateStatus.PASS, GateStatus.FINDINGS)
    # a well-supported dense model should have no missing/generic expected hooks
    assert res.data["counts"]["missing"] == 0
    assert res.data["counts"]["generic_fallback"] == 0


def test_g2_bogus_type_findings(tmp_path):
    from harness.gates import g2_integration

    feats = _features(
        "zzz_nonexistent_arch",
        "x",
        needs_patch=True,
        is_moe=True,
        is_multimodal=True,
        custom_embed_names=False,
    )
    res = g2_integration.run(_ctx(feats, tmp_path))
    assert res.status == GateStatus.FINDINGS
    assert res.data["counts"]["missing"] >= 1


# --- G1 config matrix (resolution-only; needs a local config + axolotl) ---------


def test_g1_resolves_llama(tmp_path):
    from harness.gates import g1_config

    d = _write_config(
        tmp_path,
        "llama",
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    res = g1_config.run(_ctx(_features("llama", str(d)), tmp_path))
    assert res.gate_id == "G1"
    if res.status == GateStatus.COULD_NOT_RUN:
        pytest.skip(f"load_cfg unavailable: {res.summary}")
    matrix = res.data.get("matrix", [])
    assert matrix, "G1 produced no matrix"
    ids = {c["composite_id"] for c in matrix}
    assert "p_dual_ce" in ids  # the reject oracle probe is always present
    dual = next(c for c in matrix if c["composite_id"] == "p_dual_ce")
    assert dual["verdict"] == "rejected"  # mutual-exclusivity oracle fired


# --- report + manifest ---------------------------------------------------------


def test_report_and_manifest(tmp_path):
    from harness import report

    g1 = GateResult(
        "G1",
        "config",
        GateStatus.PASS,
        summary="5/5 resolve",
        data={
            "matrix": [
                {
                    "composite_id": "c0",
                    "flags": {"sample_packing": True},
                    "expect": "resolve",
                    "verdict": "supported",
                    "note": "ok",
                    "warnings": [],
                    "bisect": [],
                },
            ]
        },
    )
    g2 = GateResult(
        "G2",
        "integration",
        GateStatus.PASS,
        summary="all explicit",
        data={
            "checklist": [
                {
                    "hook": "multipack",
                    "file": "f",
                    "gated_on": "packing",
                    "status": "present_explicit",
                    "note": "in registry",
                },
            ],
            "counts": {
                "expected": 1,
                "present_explicit": 1,
                "generic_fallback": 0,
                "missing": 0,
                "not_expected": 0,
            },
        },
    )
    feats = _features("llama", "axolotl-ai-co/tiny-llama-50m")
    ctx = GateContext(features=feats, repo_root=REPO_ROOT, output_dir=tmp_path)
    md = report.render_report(ctx, feats, [g1, g2])
    assert "Model Verification Report" in md and "llama" in md
    assert "Compat matrix" in md

    class _Args:
        seed = 42
        profile = "smoke"

    manifest = report.build_manifest(_Args(), ctx, feats, [g1, g2])
    json.dumps(manifest)  # must be serializable
    assert manifest["model"]["model_config_type"] == "llama"


# --- G8 static coverage --------------------------------------------------------


def test_g8_static_coverage_llama(tmp_path):
    from harness.gates import g8_coverage

    res = g8_coverage.run(_ctx(_features("llama", "x"), tmp_path))
    assert res.gate_id == "G8"
    # llama is referenced across the test tree -> not undefended
    assert res.status in (GateStatus.PASS, GateStatus.FINDINGS)


def test_g8_undefended_type_findings(tmp_path):
    from harness.gates import g8_coverage

    # build the token by concatenation so its contiguous literal never appears in this file (else G8 matches this very test, not "undefended")
    bogus = "qzx" + "nonexistent" + "arch7"
    res = g8_coverage.run(_ctx(_features(bogus, f"acme/{bogus}"), tmp_path))
    assert res.status == GateStatus.FINDINGS  # zero references -> undefended


# --- PR discovery (discover_from_diff) -----------------------------------------


def _newfile(path: str, body: str) -> str:
    lines = [f"+{ln}" for ln in body.splitlines()]
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        f"--- /dev/null\n+++ b/{path}\n"
        "@@ -0,0 +1,1 @@\n" + "\n".join(lines) + "\n"
    )


def _hunk(path: str, added: list[str], context: tuple[str, ...] = ()) -> str:
    body = "".join(f" {c}\n" for c in context) + "".join(f"+{a}\n" for a in added)
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        "@@ -1,1 +1,9 @@\n" + body
    )


def test_discover_new_model_diff():
    from harness.discover import discover_from_diff

    diff = (
        _newfile(
            "examples/glmnova/cfg.yaml",
            "base_model: acme/GLM-Nova-9B\nchat_template: glmnova\n",
        )
        + _newfile(
            "src/axolotl/monkeypatch/models/glmnova/__init__.py",
            "# bespoke kernels\n",
        )
        + _hunk(
            "src/axolotl/monkeypatch/multipack.py",
            ['    "glmnova",'],
            context=('    "glm4",',),
        )
    )
    res = discover_from_diff(diff)
    cands = res["candidates"]
    assert cands, "expected at least one candidate"
    top = cands[0]
    assert top["model_config_type"] == "glmnova"
    assert top["base_model"] == "acme/GLM-Nova-9B"
    assert top["is_new"] is True
    assert {"model_dir", "example", "multipack"} <= set(top["signals"])
    assert (
        top["score"] >= 8
    )  # model dir + example + base_model + multipack + multi-class


def test_discover_family_diff():
    from harness.discover import discover_from_diff

    diff = (
        _newfile("examples/qwen9/cfg.yaml", "base_model: q/Qwen9\n")
        + _hunk(
            "src/axolotl/monkeypatch/multipack.py",
            ['    "qwen9",', '    "qwen9_moe",'],
        )
        + _hunk(
            "src/axolotl/common/architectures.py",
            ['    "qwen9_moe": "Qwen9SparseMoeBlock",'],
        )
    )
    res = discover_from_diff(diff)
    types = {c["model_config_type"] for c in res["candidates"]}
    assert {"qwen9", "qwen9_moe"} <= types
    roots = {c["family_root"] for c in res["candidates"]}
    assert "qwen9" in roots  # qwen9_moe strips to the qwen9 family


def test_discover_extends_only():
    from harness.discover import discover_from_diff

    # "gemma" added to multipack but it already exists in fused-attn (pre-image context)
    diff = _hunk(
        "src/axolotl/loaders/patch_manager.py",
        ['    "gemma",'],
        context=('    "gemma",', '    "llama",'),
    )
    res = discover_from_diff(diff)
    gemma = next(
        (c for c in res["candidates"] if c["model_config_type"] == "gemma"), None
    )
    assert gemma is not None
    assert gemma["is_new"] is False  # pre-existed in the container -> extends


def test_discover_dynamic_only_warns():
    from harness.discover import discover_from_diff

    diff = _hunk(
        "pyproject.toml",
        ['    "transformers==5.12.1",'],
        context=('    "transformers==5.10.2",',),
    )
    res = discover_from_diff(diff)
    # a pin bump alone must not produce a confident new-type candidate
    assert not any(c["is_new"] for c in res["candidates"])
    assert any("transformers" in w for w in res["warnings"])


# --- G4 terminator masking (#3754), pure-function offline ----------------------

# tiny token map: 1=<eos> (special, never in chat), 107=<end_of_turn> (real terminator),
# 9/10/11 = content, -100 = IGNORE
_SPECIAL = {1, 107}


def test_g4_terminator_masked_is_detected():
    from harness.gates import g4_masking

    # assistant content trained, terminator <end_of_turn> left at IGNORE (the bug)
    input_ids = [106, 9, 10, 107, 9, 11, 107]
    labels = [-100, 9, 10, -100, 9, 11, -100]
    status = g4_masking._terminator_status(input_ids, labels, _SPECIAL)
    assert status["terminators_masked"] == 2
    assert status["terminators_trained"] == 0
    assert status["terminator_id"] == 107


def test_g4_terminator_trained_passes():
    from harness.gates import g4_masking

    # terminator is the last trained token of each assistant turn (correct); 50 is a
    # masked user token separating the two assistant turns into distinct trained runs
    input_ids = [106, 9, 10, 107, 50, 9, 11, 107]
    labels = [-100, 9, 10, 107, -100, 9, 11, 107]
    status = g4_masking._terminator_status(input_ids, labels, _SPECIAL)
    assert status["terminators_masked"] == 0
    assert status["terminators_trained"] == 2
    assert status["terminator_id"] == 107


def test_g4_eos_not_found_warning_regex():
    from harness.gates import g4_masking

    msg = (
        "EOS token '<eos>' not found in chat_template. "
        "Please check if your template/EOS token is correct."
    )
    assert g4_masking._EOS_NOT_FOUND_RE.search(msg)
    assert g4_masking._EOS_NOT_FOUND_RE.search(
        "EOT token '<x>' not found in chat_template."
    )
    assert not g4_masking._EOS_NOT_FOUND_RE.search("some unrelated warning")


def test_g4_native_template_first():
    from harness.gates import g4_masking

    # native template must be preferred so family-specific terminator bugs surface
    assert g4_masking._CHAT_TEMPLATE_CANDIDATES[0] == "tokenizer_default"


# --- multimodal detection: nested vision/audio + Phi3V img_processor ------------


def test_detect_phi3v_img_processor_is_multimodal(tmp_path):
    from harness.detect import detect_model

    # Phi-3.5-vision has no vision_config; the signal is img_processor + Phi3V arch
    d = _write_config(
        tmp_path,
        "phi3v",
        {
            "model_type": "phi3_v",
            "architectures": ["Phi3VForCausalLM"],
            "img_processor": {"name": "clip_vision_model"},
        },
    )
    f = detect_model(str(d), REPO_ROOT)
    assert f.is_multimodal  # must route to the MM gate, not the text path


def test_detect_qwen_omni_nested_modality_is_multimodal(tmp_path):
    from harness.detect import detect_model

    # Qwen2.5-Omni nests vision/audio under thinker_config; nothing at top level
    d = _write_config(
        tmp_path,
        "omni",
        {
            "model_type": "qwen2_5_omni",
            "architectures": ["Qwen2_5OmniForConditionalGeneration"],
            "thinker_config": {
                "audio_config": {"d_model": 1},
                "image_token_index": 1,
                "audio_token_index": 2,
            },
        },
    )
    f = detect_model(str(d), REPO_ROOT)
    # without this, the omni model silently validates as text-only (false PASS)
    assert f.is_multimodal


def test_detect_plain_text_not_multimodal(tmp_path):
    from harness.detect import detect_model

    # a nested dict that is not a modality container must not trip the heuristic
    d = _write_config(
        tmp_path,
        "plain",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "rope_scaling": {"type": "linear", "factor": 2.0},
        },
    )
    f = detect_model(str(d), REPO_ROOT)
    assert not f.is_multimodal


# --- G2 lora_kernels attn-class + experts-only MoE registration ----------------


def _g2_row(res, hook):
    for r in res.data["checklist"]:
        if r["hook"] == hook:
            return r["status"]
    return None


def test_g2_lora_kernel_explicit_branch(tmp_path):
    from harness.gates import g2_integration

    # qwen3_vl has a dedicated branch in get_attention_cls_from_config
    res = g2_integration.run(
        _ctx(_features("qwen3_vl", "x"), tmp_path, lora_qkv_kernel=True)
    )
    assert _g2_row(res, "LoRA-kernel attention class") == "present_explicit"


def test_g2_lora_kernel_generic_when_no_branch(tmp_path):
    from harness.gates import g2_integration

    # a novel type that requests lora attn kernels but has no branch -> silently inert
    res = g2_integration.run(
        _ctx(_features("znovel_arch_test", "x"), tmp_path, lora_o_kernel=True)
    )
    assert _g2_row(res, "LoRA-kernel attention class") == "generic_fallback"


def test_g2_lora_kernel_not_requested(tmp_path):
    from harness.gates import g2_integration

    res = g2_integration.run(_ctx(_features("llama", "x"), tmp_path))
    assert _g2_row(res, "LoRA-kernel attention class") == "not_expected"


def test_g2_experts_only_block_present(tmp_path):
    from harness.gates import g2_integration

    # gemma4_text registers experts-in-decoder-layer in EXPERTS_ONLY_BLOCK
    res = g2_integration.run(_ctx(_features("gemma4_text", "x", is_moe=True), tmp_path))
    assert _g2_row(res, "experts-only MoE block") == "present_explicit"


def test_g2_experts_only_block_not_expected_for_standard_moe(tmp_path):
    from harness.gates import g2_integration

    res = g2_integration.run(_ctx(_features("qwen3_moe", "x", is_moe=True), tmp_path))
    assert _g2_row(res, "experts-only MoE block") == "not_expected"


def test_g4_bundled_template_exact_match_only():
    from harness.gates import g4_masking

    # exact enum names resolve; a non-enum family name must NOT (avoid running wrong template)
    assert g4_masking._bundled_template_name("qwen3") == "qwen3"
    assert g4_masking._bundled_template_name("gemma2") is None
    assert g4_masking._bundled_template_name("znovel_arch_test") is None


# --- CodeRabbit-round regressions ----------------------------------------------


def test_g1_oserror_subclass_is_environment_error():
    from harness.gates import g1_config

    # FileNotFoundError/PermissionError/TimeoutError are OSError subclasses -> could-not-run
    assert g1_config._is_environment_error(FileNotFoundError("x"))
    assert g1_config._is_environment_error(PermissionError("x"))
    assert not g1_config._is_environment_error(ValueError("logic bug"))


def test_main_rejects_unknown_gate_token(tmp_path):
    import verify_model

    d = _write_config(
        tmp_path,
        "llama",
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    # a typo like G1,G9 must fail fast (exit 2), not silently run only G1 and exit clean
    code = verify_model.main(
        [
            "--base-model",
            str(d),
            "--gates",
            "G1,G9",
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ]
    )
    assert code == 2


def test_parse_features_cannot_override_control_keys():
    import verify_model

    # feature flags spread first; explicit control keys must win
    feats = verify_model._parse_features("on_unavailable")
    merged = {**feats, "on_unavailable": "fail"}
    assert merged["on_unavailable"] == "fail"
