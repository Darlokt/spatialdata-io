"""Exercise the seqFISH CLI through a real synthetic dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.testing import assert_array_equal
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys

from spatialdata_io._cli.commands.seqfish import seqfish_command
from tests.integration.readers.seqfish._fixtures import write_seqfish_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


class TestSeqfishCommand:
    """Validate public command conversion and inclusion semantics."""

    def test_complete_conversion(self, runner: CliRunner, tmp_path: Path) -> None:
        source = write_seqfish_dataset(tmp_path / "source", rois=("Roi1",))
        output = tmp_path / "output.zarr"
        result = runner.invoke(
            seqfish_command,
            ["--input", str(source), "--output", str(output), "--shape-geometry", "both", "--rois", "Roi1"],
        )
        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert isinstance(restored.tables["table_Roi1"].X, sparse.csr_array | sparse.csr_matrix)
        assert_array_equal(restored.tables["table_Roi1"].X.data, [3, 5])
        assert get_table_keys(restored.tables["table_Roi1"]) == ("Roi1_Segmentation", "region", "instance_id")
        assert set(restored.shapes) == {"Roi1_Boundaries", "Roi1_CellCoordinates"}
        assert "z_raw" in restored.points["Roi1_TranscriptList"].columns
        assert "z" not in restored.points["Roi1_TranscriptList"].columns

    def test_table_only_conversion(self, runner: CliRunner, tmp_path: Path) -> None:
        source = write_seqfish_dataset(tmp_path / "source", rois=("Roi1",))
        for artifact in source.iterdir():
            if artifact.name.endswith((".tiff", ".geojson", "_TranscriptList.csv")):
                artifact.unlink()
        output = tmp_path / "output.zarr"
        result = runner.invoke(
            seqfish_command,
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--no-include-images",
                "--no-include-labels",
                "--no-include-transcripts",
                "--no-include-shapes",
            ],
        )
        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        assert "spatialdata_attrs" not in restored.tables["table_Roi1"].uns
        assert not restored.images
        assert not restored.labels
        assert not restored.points
        assert not restored.shapes
