"""Distribution metadata and example contracts for MegaTrain."""

from pathlib import Path

import yaml

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault

PLUGIN = "axolotl.integrations.megatrain.MegaTrainPlugin"
UPSTREAM_SHA = "7f5c9597e5b20bb618932c77c922e8eac4a11c4d"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_DIR = REPO_ROOT / "src/axolotl/integrations/megatrain"


def test_distribution_metadata_includes_megatrain_artifacts():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    package_artifacts = [
        "LICENSE",
        "README.md",
        "ACKNOWLEDGEMENTS.md",
        "_vendor/PROVENANCE.md",
    ]
    for artifact in package_artifacts:
        assert f'"integrations/megatrain/{artifact}"' in pyproject
        assert f"include src/axolotl/integrations/megatrain/{artifact}" in manifest

    assert "include examples/smollm2/megatrain.yaml" in manifest


def test_vendored_snapshot_has_expected_shape_and_provenance():
    vendor_dir = INTEGRATION_DIR / "_vendor/infinity"
    vendor_files = [path for path in vendor_dir.rglob("*") if path.is_file()]

    assert len([path for path in vendor_files if path.suffix == ".py"]) == 34
    assert not [
        path
        for path in vendor_files
        if path.suffix in {".o", ".so"}
        or any(
            part in {"build", "cuda_pipeline", "verl"} or part.endswith(".egg-info")
            for part in path.parts
        )
    ]

    provenance = (INTEGRATION_DIR / "_vendor/PROVENANCE.md").read_text(encoding="utf-8")
    assert "https://github.com/DLYuanGod/MegaTrain" in provenance
    assert UPSTREAM_SHA in provenance
    assert "Upstream version: `0.3.0`" in provenance
    assert "Retrieval date: 2026-07-19" in provenance
    assert "## Retrieval and pruning" in provenance
    assert "## Local changes" in provenance
    assert "## Updating from upstream" in provenance

    license_text = (INTEGRATION_DIR / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text


def test_example_config_is_loadable(monkeypatch, tmp_path):
    monkeypatch.setattr("axolotl.cli.config.prepare_debug_log", lambda _cfg: None)
    monkeypatch.setattr(
        "axolotl.cli.config.gpu_capabilities",
        lambda: (
            {
                "bf16": True,
                "compute_capability": "sm_89",
                "fp8": False,
                "n_gpu": 1,
                "n_node": 1,
                "tf32": True,
            },
            {"torch_version": "2.11.0"},
        ),
    )
    raw_config = yaml.safe_load(
        (REPO_ROOT / "examples/smollm2/megatrain.yaml").read_text(encoding="utf-8")
    )
    raw_config["dataset_prepared_path"] = str(tmp_path / "prepared")
    raw_config["output_dir"] = str(tmp_path / "output")

    config = load_cfg(DictDefault(raw_config))

    assert config.plugins == [PLUGIN]
    assert config.adapter is None
    assert config.bf16 is True
    assert config.sample_packing is False
    assert config.val_set_size == 0
