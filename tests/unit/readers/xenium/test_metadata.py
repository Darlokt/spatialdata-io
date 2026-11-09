"""Experiment, metrics and panel metadata become typed JSON-serializable records."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import _metadata
from spatialdata_io.readers.xenium._metadata import (
    read_experiment,
    read_gene_panel,
    read_metrics,
    read_protein_panel,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

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
    """metrics_summary.csv is one strict record of exact values."""

    def test_exact_values_and_missing_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text('run_name,num_cells_detected,fraction,empty,label\n"run, 1",7275,0.875,,1e-3x\n')

        metrics = read_metrics(ArtifactFile(path), context=ReaderErrorContext("xenium", "metrics_summary.csv"))

        assert metrics == {
            "run_name": "run, 1",
            "num_cells_detected": 7275,
            "fraction": 0.875,
            "empty": None,
            "label": "1e-3x",
        }
        assert isinstance(metrics["num_cells_detected"], int)

    def test_quoted_newlines_and_text_metric_spellings(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text('run_name,hex,plus,separator,empty\n"run\n1",0x10,+2,1_0,\n')

        metrics = read_metrics(ArtifactFile(path), context=ReaderErrorContext("xenium", "metrics_summary.csv"))

        assert metrics == {"run_name": "run\n1", "hex": "0x10", "plus": 2, "separator": 10.0, "empty": None}

    def test_excess_rows_fail_without_reading_the_whole_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        payload = b"a,b\n" + b"1,2\n" * 4_000_000
        source = io.BytesIO(payload)
        reads = mocker.spy(source, "read")
        mocker.patch("pathlib.Path.open", return_value=source)

        with pytest.raises(ReaderFormatError, match="one nonempty header and one record"):
            read_metrics(ArtifactFile(tmp_path / "metrics_summary.csv"), context=PANEL)

        assert sum(len(chunk) for chunk in reads.spy_return_list) < len(payload) // 2
        assert source.closed

    def test_oversized_header_reads_at_most_one_block(self, tmp_path: Path, mocker: MockerFixture) -> None:
        source = io.BytesIO(b"a" * (2 * 1024 * 1024) + b"\n1\n")
        reads = mocker.spy(source, "read")
        lines = mocker.spy(source, "readline")
        mocker.patch("pathlib.Path.open", return_value=source)
        with pytest.raises(ReaderFormatError, match="oversized header"):
            read_metrics(ArtifactFile(tmp_path / "metrics_summary.csv"), context=PANEL)
        assert sum(len(chunk) for chunk in lines.spy_return_list) == 64 * 1024 + 1
        reads.assert_not_called()
        assert source.closed

    def test_excess_row_in_another_batch_is_rejected(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(_metadata, "_CSV_BLOCK_BYTES", 4)
        path = tmp_path / "metrics_summary.csv"
        path.write_text("a,b\n1,2\n3,4\n")

        with pytest.raises(ReaderFormatError, match="more than one record"):
            read_metrics(ArtifactFile(path), context=PANEL)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("a,b\n1\n", "Expected 2 columns, got 1"),
            ("a,a\n1,2\n", "repeated column names"),
            ("a,b\n1,2\n3,4\n", "one nonempty header and one record"),
            ("a,b\n", "one nonempty header and one record"),
            (",b\n1,2\n", "one nonempty header and one record"),
            ('a,b\n"1,2\n', "malformed CSV"),
        ],
        ids=["width", "duplicate", "two-records", "no-record", "empty-name", "quote"],
    )
    def test_rejects_malformed_tables(self, tmp_path: Path, text: str, match: str) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_text(text)

        with pytest.raises(ReaderFormatError, match=match):
            read_metrics(ArtifactFile(path), context=ReaderErrorContext("xenium", "metrics_summary.csv"))

    def test_rejects_non_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_summary.csv"
        path.write_bytes(b"a\n\xff\n")
        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_metrics(ArtifactFile(path), context=ReaderErrorContext("xenium", "metrics_summary.csv"))


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
