"""Grid coverage and exact source-order sparse bin alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, spmatrix

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._bins import read_bin_table
from spatialdata_io.readers.visium_hd._layout import BinLayout
from spatialdata_io.readers.visium_hd._transforms import SlideFrame
from tests.integration.readers.visium_hd._fixtures import HOMOGRAPHY, write_bundle, write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _positions(root: Path) -> ArtifactFile:
    return ArtifactFile(root / "binned_outputs/square_002um/spatial/tissue_positions.parquet")


class TestReadBinTable:
    """Calibrated sparse bin matrices without densification."""

    @pytest.mark.parametrize("kind", ["raw", "filtered"])
    @pytest.mark.parametrize("unique", [False, True])
    def test_sparse_values_and_physical_centres(self, tmp_path: Path, kind: str, *, unique: bool) -> None:
        bundle = write_bundle(tmp_path, sizes=(2,), segmented=False)
        directory = tmp_path / "binned_outputs/square_002um"
        layout = BinLayout(
            2,
            ArtifactFile(directory / f"{kind}_feature_bc_matrix.h5"),
            _positions(tmp_path),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)
        table = read_bin_table(layout, (17, 17), frame, make_var_names_unique=unique)
        assert isinstance(table.X, csr_array)
        assert not isinstance(table.X, spmatrix)
        expected = bundle.raw[2] if kind == "raw" else bundle.raw[2][bundle.filtered_rows[2]]
        assert_array_equal(table.X.toarray(), expected)
        assert table.var_names.tolist() == (["Gene", "Gene-1", "Other"] if unique else ["Gene", "Gene", "Other"])
        assert_array_equal(
            table.obsm["spatial"],
            np.column_stack((table.obs["array_col"].to_numpy() * 2 + 1, table.obs["array_row"].to_numpy() * 2 + 1)),
        )

    def test_expression_stays_sparse(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        directory = tmp_path / "binned_outputs/square_002um"
        layout = BinLayout(
            2,
            ArtifactFile(directory / "raw_feature_bc_matrix.h5"),
            _positions(tmp_path),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("expression densification"))
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)
        table = read_bin_table(layout, (17, 17), frame, make_var_names_unique=True)
        assert isinstance(table.X, csr_array)
        assert table.X.nnz == 6

    def test_empty_matrix_rejected(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2,), segmented=False)
        directory = tmp_path / "binned_outputs/square_002um"
        count_path = directory / "filtered_feature_bc_matrix.h5"
        write_counts(count_path, np.zeros((0, 3), dtype=np.int32), [], "4.1.0")
        layout = BinLayout(
            2,
            ArtifactFile(count_path),
            _positions(tmp_path),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)
        with pytest.raises(ReaderFormatError, match="positive observation and feature dimensions"):
            read_bin_table(layout, (17, 17), frame, make_var_names_unique=True)
