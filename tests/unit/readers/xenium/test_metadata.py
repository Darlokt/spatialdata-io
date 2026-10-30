"""Experiment, metrics and panel metadata become typed JSON-serializable records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium._metadata import (
    read_experiment,
    read_gene_panel,
    read_metrics,
    read_protein_panel,
)

if TYPE_CHECKING:
    from pathlib import Path

EXPERIMENT = {
    "major_version": 5,
    "minor_version": 2,
    "patch_version": 0,
    "run_name": "run",
    "pixel_size": 0.2125,
    "z_step_size": 3.0,
    "analysis_sw_version": "xenium-3.0.0.15",
    "images": {"morphology_focus_filepath": "morphology_focus/morphology_focus_0000.ome.tif"},
    "xenium_explorer_files": {"cells_zarr_filepath": "cells.zarr.zip"},
    "xenium_ranger": {
        "run_id": "XRrun",
        "version": "xenium-3.1.0.5",
        "command_line": "xeniumranger relabel --xenium-bundle=/home/someone/outs",
        "original_analysis_uuid": "fd0b49ea",
    },
}


def _json(path: Path, payload: object) -> ArtifactFile:
    path.write_text(json.dumps(payload))
    return ArtifactFile(path)


class TestReadExperiment:
    """experiment.xenium yields the version, family, calibration and kept attributes."""

    def test_version_family_calibration_and_attrs(self, tmp_path: Path) -> None:
        experiment = read_experiment(_json(tmp_path / "experiment.xenium", EXPERIMENT))

        assert str(experiment.version) == "5.2.0"
        assert experiment.family == "polygon_sets"
        assert experiment.software == "xenium-3.0.0.15"
        assert experiment.pixel_size == 0.2125
        assert experiment.images == {"morphology_focus_filepath": "morphology_focus/morphology_focus_0000.ome.tif"}
        # Path blocks and the Ranger command line (local paths) are excluded.
        assert "images" not in experiment.attrs
        assert "xenium_explorer_files" not in experiment.attrs
        assert experiment.attrs["xenium_ranger"] == {
            "version": "xenium-3.1.0.5",
            "run_id": "XRrun",
            "original_analysis_uuid": "fd0b49ea",
        }
        assert experiment.attrs["run_name"] == "run"
        json.dumps(dict(experiment.attrs))

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("pixel_size", 0, "finite positive pixel_size"),
            ("pixel_size", -0.2, "finite positive pixel_size"),
            ("pixel_size", True, "finite positive pixel_size"),
            ("pixel_size", "0.2125", "finite positive pixel_size"),
            ("pixel_size", None, "finite positive pixel_size"),
            ("z_step_size", 0, "finite positive z_step_size"),
            ("analysis_sw_version", 1, "software version string"),
            ("images", [], "images as a JSON object"),
            ("images", {"morphology_focus_filepath": ""}, "nonempty relative image paths"),
            ("xenium_ranger", "XRrun", "xenium_ranger as a JSON object"),
        ],
    )
    def test_rejects_malformed_keys(self, tmp_path: Path, key: str, value: object, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_experiment(_json(tmp_path / "experiment.xenium", EXPERIMENT | {key: value}))

    def test_missing_pixel_size(self, tmp_path: Path) -> None:
        experiment = {key: value for key, value in EXPERIMENT.items() if key != "pixel_size"}

        with pytest.raises(ReaderFormatError, match="pixel_size = None"):
            read_experiment(_json(tmp_path / "experiment.xenium", experiment))

    def test_optional_blocks_may_be_absent(self, tmp_path: Path) -> None:
        minimal = {"major_version": 4, "minor_version": 0, "pixel_size": 0.2125, "analysis_sw_version": "R&D"}

        experiment = read_experiment(_json(tmp_path / "experiment.xenium", minimal))

        assert experiment.images == {}
        assert dict(experiment.attrs) == minimal


class TestReadMetrics:
    """metrics_summary.csv is one strict record of exact values."""

    def test_exact_values_and_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text('run_name,num_cells_detected,fraction,empty,label\n"run, 1",7275,0.875,,1e-3x\n')

        metrics = read_metrics(ArtifactFile(path))

        assert metrics == {
            "run_name": "run, 1",
            "num_cells_detected": 7275,
            "fraction": 0.875,
            "empty": None,
            "label": "1e-3x",
        }
        assert isinstance(metrics["num_cells_detected"], int)

    @pytest.mark.parametrize(
        "text",
        [
            "a,b\n1\n",
            "a,a\n1,2\n",
            "a,b\n1,2\n3,4\n",
            "a,b\n",
            ",b\n1,2\n",
            'a,b\n"1,2\n',
        ],
        ids=["width", "duplicate", "two-records", "no-record", "empty-name", "quote"],
    )
    def test_rejects_malformed_tables(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text(text)

        with pytest.raises(ReaderFormatError, match="one unique header and one record"):
            read_metrics(ArtifactFile(path))

    def test_rejects_non_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_bytes(b"a\n\xff\n")
        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_metrics(ArtifactFile(path))


class TestReadGenePanel:
    """The gene panel contributes its identity only."""

    def test_identity_only(self, tmp_path: Path) -> None:
        panel = {
            "metadata": {"sha256_checksum": "0" * 64},
            "payload": {
                "panel": {
                    "identity": {"design_id": "hBreast_v1", "name": "Breast", "version": "1.0.0", "extra": [1]},
                    "num_gene_targets": 280,
                    "species": "Human",
                    "tissue": "Breast",
                },
                "spec_version": "0.1.0",
                "targets": [{"type": {"data": {"name": "GeneA"}}}],
            },
        }

        identity = read_gene_panel(_json(tmp_path / "gene_panel.json", panel))

        assert identity == {
            "design_id": "hBreast_v1",
            "name": "Breast",
            "version": "1.0.0",
            "num_gene_targets": 280,
            "species": "Human",
            "tissue": "Breast",
            "spec_version": "0.1.0",
        }

    def test_requires_payload_panel(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match=r"payload\.panel as a JSON object"):
            read_gene_panel(_json(tmp_path / "gene_panel.json", {"payload": {"panel": []}}))


class TestReadProteinPanel:
    """The protein panel maps each public ID to exactly one target."""

    @staticmethod
    def _panel(targets: list[object] | str) -> dict[str, object]:
        return {
            "payload": {
                "panel": {
                    "identity": {"design_id": "protein-panel", "name": "protein-panel"},
                    "num_targets": len(targets),
                    "panel_type": "PANEL_TYPE_PROTEIN",
                    "spec_version": "1.0.0",
                    "targets": targets,
                }
            }
        }

    def test_one_to_one_targets_and_identity(self, tmp_path: Path) -> None:
        target = {
            "public_id": "TXP000019",
            "short_name": "PD-1",
            "long_name": "PD 1",
            "clone": "n/a",
            "codeword_index": 16,
        }

        panel = read_protein_panel(_json(tmp_path / "protein_panel.json", self._panel([target])))

        assert panel.targets["TXP000019"].long_name == "PD 1"
        assert panel.targets["TXP000019"].codeword_index == 16
        assert panel.identity == {
            "design_id": "protein-panel",
            "name": "protein-panel",
            "num_targets": 1,
            "panel_type": "PANEL_TYPE_PROTEIN",
            "spec_version": "1.0.0",
        }

    @pytest.mark.parametrize(
        ("targets", "match"),
        [
            ("none", "targets as a list"),
            ([{"public_id": "TXP1", "long_name": "a", "clone": "c", "codeword_index": "16"}], "integer codeword_index"),
            (
                [
                    {"public_id": "TXP1", "long_name": "a", "clone": "c", "codeword_index": 1},
                    {"public_id": "TXP1", "long_name": "b", "clone": "c", "codeword_index": 2},
                ],
                "duplicate public_id",
            ),
        ],
        ids=["not-a-list", "codeword", "duplicate"],
    )
    def test_rejects_malformed_targets(self, tmp_path: Path, targets: list[object] | str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_protein_panel(_json(tmp_path / "protein_panel.json", self._panel(targets)))
