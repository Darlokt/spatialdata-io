"""Shared count ingestion, HD population validation and independent linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, spmatrix

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd import _tables as tables
from spatialdata_io.readers.visium_hd._layout import BinLayout
from spatialdata_io.readers.visium_hd._transforms import SlideFrame
from tests.integration.readers.visium_hd._fixtures import HOMOGRAPHY, write_bundle, write_counts

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _source(tmp_path: Path) -> ArtifactFile:
    path = ArtifactFile(tmp_path / "matrix.h5")
    write_counts(path, np.array([[0, 1, 2], [3, 0, 4]], dtype=np.int32), ["first", "second"], "4.1.0")
    return path


def _replace(path: ArtifactFile, name: str, values: object) -> None:
    with h5py.File(path, "r+") as handle:
        del handle[f"matrix/{name}"]
        handle.create_dataset(f"matrix/{name}", data=values)


class TestReadCounts:
    """Preserve source populations through shared sparse count ingestion."""

    def test_values_dtype_and_order(self, tmp_path: Path) -> None:
        result = tables.read_counts(_source(tmp_path))
        assert isinstance(result.X, csr_array)
        assert result.X.dtype == np.float32
        assert_array_equal(result.X.toarray(), [[0, 1, 2], [3, 0, 4]])
        assert_array_equal(result.obs_names, ["first", "second"])

    def test_zero_counts(self, tmp_path: Path) -> None:
        path = _source(tmp_path)
        for name, values in (
            ("data", np.array([], dtype=np.int32)),
            ("indices", np.array([], dtype=np.int32)),
            ("indptr", np.array([0, 0, 0], dtype=np.int64)),
        ):
            _replace(path, name, values)
        zero = tables.read_counts(path)
        assert isinstance(zero.X, csr_array)
        assert zero.X.nnz == 0

    @pytest.mark.parametrize("entries", [[1, 1, 0, 2], [2, 1, 0, 2]])
    def test_normalized_source_coordinates(self, tmp_path: Path, entries: list[int]) -> None:
        path = _source(tmp_path)
        _replace(path, "indices", np.array(entries, dtype=np.int32))
        result = tables.read_counts(path)
        assert isinstance(result.X, csr_array)
        assert result.X.has_canonical_format
        expected = [[0, 3, 0], [3, 0, 4]] if entries[0] == 1 else [[0, 2, 1], [3, 0, 4]]
        assert_array_equal(result.X.toarray(), expected)

    @pytest.mark.parametrize("value", [-1.0, 0.5, np.nan, np.inf])
    def test_invalid_count_values(self, tmp_path: Path, value: float) -> None:
        path = _source(tmp_path)
        _replace(path, "data", np.array([value, 2, 3, 4], dtype=np.float64))
        with pytest.raises((ReaderFormatError, ValueError)):
            tables.read_counts(path)

    def test_duplicate_barcodes(self, tmp_path: Path) -> None:
        path = _source(tmp_path)
        _replace(path, "barcodes", np.array([b"same", b"same"]))
        with pytest.raises(ReaderFormatError, match="duplicate barcodes"):
            tables.read_counts(path)

    def test_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            tables.read_counts(ArtifactFile(tmp_path / "absent.h5"))


class TestLinkTable:
    """Link only against the independent materialized target IDs."""

    def test_linkage_and_serializable_categories(self, tmp_path: Path) -> None:
        result = tables.read_counts(_source(tmp_path))
        result.obs["instance_id"] = pd.Series([2, 5], index=result.obs_names, dtype="int64[pyarrow]")
        result = tables.link_table(result, "boundaries", pd.Index([2, 5]), position="vendor cell centroid")
        assert result.uns["spatialdata_attrs"]["region"] == "boundaries"
        assert result.obs["region"].cat.categories.tolist() == ["boundaries"]
        result.write_h5ad(tmp_path / "roundtrip.h5ad")

    def test_membership(self, tmp_path: Path) -> None:
        result = tables.read_counts(_source(tmp_path))
        result.obs["instance_id"] = [2, 5]
        with pytest.raises(ReaderFormatError):
            tables.link_table(result, "boundaries", pd.Index([2, 3]), position="vendor cell centroid")


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
            ArtifactFile(directory / "spatial/tissue_positions.parquet"),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)
        table = tables.read_bin_table(layout, (17, 17), frame, make_var_names_unique=unique)
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
            ArtifactFile(directory / "spatial/tissue_positions.parquet"),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("expression densification"))
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)
        table = tables.read_bin_table(layout, (17, 17), frame, make_var_names_unique=True)
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
            ArtifactFile(directory / "spatial/tissue_positions.parquet"),
            ArtifactFile(directory / "spatial/scalefactors_json.json"),
        )
        frame = SlideFrame(HOMOGRAPHY, np.eye(3), np.eye(3), 0.0, 0.0)
        with pytest.raises(ReaderFormatError, match="positive observation and feature dimensions"):
            tables.read_bin_table(layout, (17, 17), frame, make_var_names_unique=True)
