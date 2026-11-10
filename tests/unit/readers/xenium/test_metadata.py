"""Experiment, metrics and panel metadata become typed JSON-serializable records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
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


PANEL = ReaderErrorContext("xenium", "panel", "6.0.0")
METRICS = ReaderErrorContext("xenium", "metrics_summary.csv", "6.0.0")


def _json(path: Path, payload: object) -> ArtifactFile:
    path.write_text(json.dumps(payload))
    return ArtifactFile(path)


class TestReadExperiment:
    """experiment.xenium yields the version, family, calibration and kept attributes."""

    def test_version_family_calibration_and_attrs(self, tmp_path: Path) -> None:
        experiment = read_experiment(_json(tmp_path / "experiment.xenium", EXPERIMENT))

        assert experiment.version == "5.2.0"
        assert experiment.family == "polygon_sets"
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
        with pytest.raises(ReaderFormatError, match=match) as caught:
            read_experiment(_json(tmp_path / "experiment.xenium", EXPERIMENT | {key: value}))

        # The software version is checked before the format version is known.
        version = None if key == "analysis_sw_version" else "5.2.0"
        assert caught.value.context == ReaderErrorContext("xenium", "experiment.xenium", version)

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
    """metrics_summary.csv is one record whose metric types Arrow infers."""

    def test_inferred_values_and_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text('run_name,num_cells_detected,fraction,empty,label,flag\n"run, 1",7275,0.875,,1e-3x,true\n')

        metrics = read_metrics(ArtifactFile(path), context=METRICS)

        assert metrics == {
            "run_name": "run, 1",
            "num_cells_detected": 7275,
            "fraction": 0.875,
            "empty": None,
            "label": "1e-3x",
            "flag": True,
        }
        assert isinstance(metrics["num_cells_detected"], int)

    def test_quoted_newlines_and_dates_keep_their_text(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text('run_name,date,time,count\n"run\n1",2024-01-01,12:00:00,3\n')

        metrics = read_metrics(ArtifactFile(path), context=METRICS)

        assert metrics == {"run_name": "run\n1", "date": "2024-01-01", "time": "12:00:00", "count": 3}

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (b"a,b\n1\n", "Expected 2 columns, got 1"),
            (b"a,a\n1,2\n", "repeated column names"),
            (b"a,b\n1,2\n3,4\n", "2 records"),
            (b"a,b\n", "0 records"),
            (b",b\n1,2\n", "empty column name"),
            (b"a\n\xff\n", "not UTF-8"),
        ],
        ids=["width", "duplicate", "two-records", "no-record", "empty-name", "utf8"],
    )
    def test_rejects_malformed_tables(self, tmp_path: Path, payload: bytes, match: str) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_bytes(payload)

        with pytest.raises(ReaderFormatError, match=match) as caught:
            read_metrics(ArtifactFile(path), context=METRICS)

        assert caught.value.context == METRICS


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

        identity = read_gene_panel(_json(tmp_path / "gene_panel.json", panel), context=PANEL)

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
            read_gene_panel(_json(tmp_path / "gene_panel.json", {"payload": {"panel": []}}), context=PANEL)


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

        panel = read_protein_panel(_json(tmp_path / "protein_panel.json", self._panel([target])), context=PANEL)

        assert panel.targets["TXP000019"].long_name == "PD 1"
        assert panel.targets["TXP000019"].codeword_index == 16
        assert panel.identity == {
            "design_id": "protein-panel",
            "name": "protein-panel",
            "num_targets": 1,
            "panel_type": "PANEL_TYPE_PROTEIN",
            "spec_version": "1.0.0",
        }

    @pytest.mark.parametrize("value", [0, 2**63 + 1, 2**64 - 1])
    def test_codeword_index_unsigned_range(self, tmp_path: Path, value: int) -> None:
        target = {"public_id": "TXP1", "long_name": "a", "clone": "c", "codeword_index": value}
        panel = read_protein_panel(_json(tmp_path / "protein_panel.json", self._panel([target])), context=PANEL)
        assert panel.targets["TXP1"].codeword_index == value

    @pytest.mark.parametrize("value", [-1, 2**64, True])
    def test_codeword_index_rejects_invalid_domain(self, tmp_path: Path, value: int) -> None:
        target = {"public_id": "TXP1", "long_name": "a", "clone": "c", "codeword_index": value}
        with pytest.raises(ReaderFormatError, match="uint64 codeword_index"):
            read_protein_panel(_json(tmp_path / "protein_panel.json", self._panel([target])), context=PANEL)

    @pytest.mark.parametrize(
        ("targets", "match"),
        [
            ("none", "targets as a list"),
            ([{"public_id": "TXP1", "long_name": "a", "clone": "c", "codeword_index": "16"}], "uint64 codeword_index"),
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
            read_protein_panel(_json(tmp_path / "protein_panel.json", self._panel(targets)), context=PANEL)
