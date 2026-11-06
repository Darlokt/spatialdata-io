"""Verify DBiT table reading, barcode alignment and row sorting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from anndata._io.specs.registry import IORegistryError
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.dbit._barcodes import read_barcodes
from spatialdata_io.readers.dbit._tables import grid_order, read_table, sort_rows
from tests.integration.readers.dbit.test_reader import _write_layout

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _read(path: Path) -> ad.AnnData:
    """Resolve the two table artifacts and invoke their owning parsers."""
    context = ReaderErrorContext("dbit", "test table")
    return read_table(
        require_file(path / "sample.h5ad", context=context),
        read_barcodes(require_file(path / "barcode_list.txt", context=context)),
    )


class TestReadTable:
    """Preserve auxiliary annotations while validating expression and identifiers."""

    def test_source_order_positions_and_untouched_expression(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        source = ad.read_h5ad(tmp_path / "sample.h5ad")

        result = _read(tmp_path)

        assert result.obs_names.tolist() == source.obs_names.tolist()
        assert result.obs["array_A"].tolist() == [2, 1]
        assert result.obs["array_B"].tolist() == [2, 1]
        assert result.obs["array_A"].dtype == pd.ArrowDtype(pa.int64())
        grid = result.obsm["spatial_grid"]
        assert isinstance(grid, np.ndarray)
        assert grid.dtype == np.int64
        np.testing.assert_array_equal(grid, [[2, 2], [1, 1]])
        assert result.uns["spatialdata_io_positions"] == {
            "schema_version": 1,
            "coordinates": {
                "spatial_grid": {
                    "axes": ["x", "y"],
                    "units": "grid",
                    "frame": "grid_pixels",
                    "position": "one-based barcode channel ordinals (A, B)",
                }
            },
        }
        # Final linkage owns expression normalization.
        assert type(result.X) is type(source.X)
        assert type(result.layers["counts"]) is type(source.layers["counts"])

    def test_positions_metadata_is_not_shared_between_reads(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)

        first, second = _read(tmp_path), _read(tmp_path)

        assert first.uns["spatialdata_io_positions"] is not second.uns["spatialdata_io_positions"]

    @pytest.mark.parametrize("destination", ["spatial_source", "spatial_grid", "spatialdata_io_positions"])
    def test_coordinate_destinations_reject_collisions(self, tmp_path: Path, destination: str) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        source.obsm["spatial"] = np.ones((source.n_obs, 3), dtype=np.int16)
        if destination == "spatialdata_io_positions":
            source.uns[destination] = {"source": "retained"}
        else:
            source.obsm[destination] = np.ones((source.n_obs, 2))
        source.write_h5ad(path)
        with pytest.raises(ReaderFormatError, match=r"destination|spatial_source"):
            _read(tmp_path)

    def test_imported_embedding_moves_unchanged(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        native = np.arange(source.n_obs * 3, dtype=np.int16).reshape(source.n_obs, 3)
        source.obsm["spatial"] = native
        source.write_h5ad(path)

        table = _read(tmp_path)

        np.testing.assert_array_equal(table.obsm["spatial_source"], native)
        assert table.obsm["spatial_source"].dtype == np.int16
        assert "spatial" not in table.obsm

    @pytest.mark.parametrize("kind", ["missing", "no_obs", "no_var", "duplicate_obs", "duplicate_var", "empty_var"])
    def test_invalid_table(self, tmp_path: Path, kind: str) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        if kind == "missing":
            source.X = None
        elif kind == "no_obs":
            source = source[:0].copy()
        elif kind == "no_var":
            source = source[:, :0].copy()
        elif kind == "duplicate_obs":
            source.obs_names = [source.obs_names[0]] * 2
        elif kind == "duplicate_var":
            source.var_names = ["same", "same"]
        else:
            source.var_names = ["", "gene"]
        source.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match=r"expression|identifiers"):
            _read(tmp_path)

    @pytest.mark.parametrize(
        "names",
        [
            ["CCCCAAAATTGGCCA", "GGGGTTTTAACCGGTT"],
            ["CCCCAAAATTGGCCAX", "GGGGTTTTAACCGGTT"],
            ["CCCCAAAATTGGCCAA-one", "CCCCAAAATTGGCCAA-two"],
            ["AAAAAAAATTGGCCAA", "GGGGTTTTAACCGGTT"],
            ["CCCCAAAAGGGGGGGG", "GGGGTTTTAACCGGTT"],
        ],
    )
    def test_invalid_barcode_pair(self, tmp_path: Path, names: list[str]) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        source.obs_names = names
        source.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="barcode"):
            _read(tmp_path)

    def test_h5ad_read_error_has_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path)
        mocker.patch("spatialdata_io.readers.dbit._tables.ad.read_h5ad", side_effect=OSError("injected"))

        with pytest.raises(OSError, match="injected") as raised:
            _read(tmp_path)

        assert any("sample.h5ad" in note for note in raised.value.__notes__)

    def test_real_h5ad_owner_closes_on_parse_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        with h5py.File(path, "a") as handle:
            handle["X"].attrs["encoding-type"] = "invalid"
        opened = mocker.spy(h5py, "File")

        with pytest.raises(IORegistryError, match="encoding"):
            _read(tmp_path)

        assert not opened.spy_return.id.valid
        with h5py.File(path, "a") as handle:
            assert handle.id.valid


class TestGridOrder:
    """Sort rows by A, then B, independent of source order."""

    def test_lexicographic_order(self) -> None:
        coordinates = np.array([[2, 1], [1, 2], [2, 2], [1, 1]], dtype=np.int64)

        order = grid_order(coordinates)

        np.testing.assert_array_equal(order, [3, 1, 0, 2])
        np.testing.assert_array_equal(coordinates[order], [[1, 1], [1, 2], [2, 1], [2, 2]])


class TestSortRows:
    """Reorder every observation-aligned slot together, copying only when needed."""

    def test_sorted_table_is_returned_without_copy(self, mocker: MockerFixture) -> None:
        table = ad.AnnData(X=sparse.csr_array(np.eye(2)))
        copy = mocker.spy(ad.AnnData, "copy")

        assert sort_rows(table, np.array([0, 1], dtype=np.intp)) is table
        copy.assert_not_called()

    def test_reorders_aligned_slots_into_an_owned_table(self) -> None:
        table = ad.AnnData(
            X=sparse.csr_array(np.array([[1, 0], [0, 2], [3, 0]])),
            obs=pd.DataFrame({"label": ["a", "b", "c"]}, index=["r0", "r1", "r2"]),
        )
        table.obsm["spatial_grid"] = np.array([[3, 1], [1, 1], [2, 1]], dtype=np.int64)

        result = sort_rows(table, np.array([1, 2, 0], dtype=np.intp))

        assert not result.is_view
        assert result.obs_names.tolist() == ["r1", "r2", "r0"]
        assert result.obs["label"].tolist() == ["b", "c", "a"]
        np.testing.assert_array_equal(result.obsm["spatial_grid"], [[1, 1], [2, 1], [3, 1]])
        assert isinstance(result.X, sparse.csr_array)
        np.testing.assert_array_equal(result.X.toarray(), [[0, 2], [3, 0], [1, 0]])
