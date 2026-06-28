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

    # build the token by concatenation so its contiguous literal never appears in
    # this source file (else G8 would match this very test and not be "undefended").
    bogus = "qzx" + "nonexistent" + "arch7"
    res = g8_coverage.run(_ctx(_features(bogus, f"acme/{bogus}"), tmp_path))
    assert res.status == GateStatus.FINDINGS  # zero references -> undefended
