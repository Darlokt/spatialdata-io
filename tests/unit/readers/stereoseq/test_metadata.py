"""Bounded image-only pitch metadata and HDF5 resource ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq._layout import CellSource, SquareSource
from spatialdata_io.readers.stereoseq._metadata import read_image_pitch
from tests.integration.readers.stereoseq._fixtures import write_cell_gef, write_square_gef

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture


class TestReadImagePitch:
    """Use only selected sample metadata, never expression or coordinate arrays."""

    def test_no_companion_preserves_grid_fallback(self) -> None:
        assert read_image_pitch((), sample_id="sample") is None

    @pytest.mark.parametrize("kind", ["cells", "square"])
    def test_scalar_metadata_without_dataset_reads(
        self, tmp_path: Path, mocker: MockerFixture, kind: Literal["cells", "square"]
    ) -> None:
        path = ArtifactFile(
            write_cell_gef(tmp_path / "sample.cellbin.gef")
            if kind == "cells"
            else write_square_gef(tmp_path / "sample.tissue.gef")
        )
        source = CellSource(path, "unadjusted") if kind == "cells" else SquareSource(path, "all")
        read = mocker.spy(h5py.Dataset, "__getitem__")
        close = mocker.spy(h5py.File, "close")

        assert read_image_pitch((source,), sample_id="sample") == 500
        read.assert_not_called()
        assert close.call_count == 1

    @pytest.mark.parametrize("value", [0, -1, 1.5, np.nan])
    def test_invalid_present_pitch_closes_source(self, tmp_path: Path, mocker: MockerFixture, value: float) -> None:
        path = ArtifactFile(write_cell_gef(tmp_path / "sample.cellbin.gef"))
        with h5py.File(path, "r+") as handle:
            handle.attrs["resolution"] = value
        close = mocker.spy(h5py.File, "close")
        with pytest.raises(ReaderFormatError, match="resolution"):
            read_image_pitch((CellSource(path, "unadjusted"),), sample_id="sample")
        assert close.call_count == 1

    def test_conflicting_source_bins_fail(self, tmp_path: Path) -> None:
        path = ArtifactFile(write_square_gef(tmp_path / "sample.tissue.gef"))
        with h5py.File(path, "r+") as handle:
            handle["geneExp/bin50/expression"].attrs["resolution"] = 600
        with pytest.raises(ReaderFormatError, match="conflicting pitch"):
            read_image_pitch((SquareSource(path, "all"),), sample_id="sample")

    def test_conflicting_companion_pitches_fail(self, tmp_path: Path) -> None:
        square = ArtifactFile(write_square_gef(tmp_path / "sample.tissue.gef"))
        cells = ArtifactFile(write_cell_gef(tmp_path / "sample.cellbin.gef"))
        with h5py.File(cells, "r+") as handle:
            handle.attrs["resolution"] = 600
        with pytest.raises(ReaderFormatError, match="common grid pitch"):
            read_image_pitch((SquareSource(square, "all"), CellSource(cells, "unadjusted")), sample_id="sample")

    def test_declared_sample_disagreement_fails(self, tmp_path: Path) -> None:
        path = ArtifactFile(write_cell_gef(tmp_path / "sample.cellbin.gef"))
        with h5py.File(path, "r+") as handle:
            handle.attrs["sn"] = "other"
        with pytest.raises(ReaderFormatError, match="selected sample"):
            read_image_pitch((CellSource(path, "unadjusted"),), sample_id="sample")

    def test_no_square_bins_fails(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "sample.tissue.gef")
        with h5py.File(path, "w") as handle:
            handle.create_group("geneExp")
        with pytest.raises(ReaderFormatError, match="no stored square bins"):
            read_image_pitch((SquareSource(path, "all"),), sample_id="sample")

    def test_file_failure_keeps_path_exception_and_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "missing.gef")
        with pytest.raises(FileNotFoundError) as caught:
            read_image_pitch((CellSource(path, "unadjusted"),), sample_id="sample")
        assert any("image calibration" in note for note in caught.value.__notes__)
