"""Verify owned DBiT expression and barcode alignment before final linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pytest
from anndata._io.specs.registry import IORegistryError
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.dbit._barcodes import _read_barcodes
from spatialdata_io.readers.dbit._tables import _expression_array, _read_table
from tests.integration.readers.dbit.test_reader import _write_layout

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _read(path: Path) -> ad.AnnData:
    """Resolve the two table artifacts and invoke their owning parsers."""
    context = ReaderErrorContext("dbit", "test table")
    return _read_table(
        require_file(path / "sample.h5ad", context=context),
        _read_barcodes(require_file(path / "barcode_list.txt", context=context)),
    )


class TestExpressionArray:
    """Require exact, representable duplicate sums before sparse canonicalization."""

    @pytest.mark.parametrize("storage", [sparse.csr_array, sparse.csr_matrix, sparse.csc_array, sparse.csc_matrix])
    @pytest.mark.parametrize("values", [[100, 100], [-100, -100]])
    def test_rejects_integer_overflow(
        self,
        storage: type[sparse.csr_array | sparse.csr_matrix | sparse.csc_array | sparse.csc_matrix],
        values: list[int],
    ) -> None:
        expression = storage((np.array(values, dtype=np.int8), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))

        with pytest.raises(ValueError, match="overflow source dtype int8"):
            _expression_array(expression)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_exact_large_integer_limits(self, dtype: type[np.int64 | np.uint64]) -> None:
        maximum = np.iinfo(dtype).max
        expression = sparse.csr_array(
            (np.array([maximum, 1], dtype=dtype), np.array([0, 0]), np.array([0, 2])), shape=(1, 1)
        )

        with pytest.raises(ValueError, match="overflow"):
            _expression_array(expression)

        # One less in the source makes the duplicate sum exactly representable.
        expression.data[0] -= 1
        result = _expression_array(expression)

        assert int(result.data[0]) == maximum
        assert result.dtype == dtype

    def test_distinct_unsorted_columns_do_not_aggregate(self) -> None:
        expression = sparse.csr_array(
            (np.array([100, 100], dtype=np.int8), np.array([1, 0]), np.array([0, 2])), shape=(1, 2)
        )

        result = _expression_array(expression)

        np.testing.assert_array_equal(result.data, [100, 100])
        np.testing.assert_array_equal(result.indices, [0, 1])
        assert result.has_canonical_format

    def test_float_duplicate_sum(self) -> None:
        expression = sparse.csr_array((np.array([0.125, 0.25]), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))

        np.testing.assert_array_equal(_expression_array(expression).data, [0.375])


class TestReadTable:
    """Preserve auxiliary annotations while validating expression and identifiers."""

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

    def test_imported_embedding_sorted_exactly_with_rows(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        native = np.arange(source.n_obs * 3, dtype=np.int16).reshape(source.n_obs, 3)
        source.obsm["spatial"] = native
        source.write_h5ad(path)
        table = _read(tmp_path)
        order = source.obs_names.get_indexer(table.obs_names)
        np.testing.assert_array_equal(table.obsm["spatial_source"], native[order])
        assert table.obsm["spatial_source"].dtype == np.int16
        assert table.obsm["spatial_grid"].dtype == np.int64
        assert "spatial" not in table.obsm
        assert table.uns["spatialdata_io_positions"]["coordinates"]["spatial_grid"]["units"] == "grid"

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

    @pytest.mark.parametrize("layer", ["X", "counts"])
    @pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
    def test_nonfinite_expression(self, tmp_path: Path, layer: str, value: float) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        expression = sparse.csr_array(np.array([[value, 0], [0, 1]], dtype=np.float64))
        if layer == "X":
            source.X = expression
        else:
            source.layers[layer] = expression
        source.write_h5ad(path)

        with pytest.raises(ValueError, match="finite"):
            _read(tmp_path)

    def test_sorted_table_needs_no_copy(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)[::-1].copy()
        del source.layers["counts"]
        source.write_h5ad(path)
        mocker.patch.object(ad.AnnData, "copy", side_effect=AssertionError("unnecessary copy"))

        result = _read(tmp_path)

        assert isinstance(result.X, sparse.csr_array)
        assert result.obs["array_A"].tolist() == [1, 2]

    def test_dense_expression_is_sparse_before_reordering(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path, sparse_source=False)
        original_copy = ad.AnnData.copy

        def checked_copy(table: ad.AnnData, filename: str | Path | None = None) -> ad.AnnData:
            """Fail if sorting materializes a second dense expression container."""
            assert isinstance(table.X, sparse.csr_array)
            assert isinstance(table.layers["counts"], sparse.csr_array)
            return original_copy(table, filename)

        mocker.patch.object(ad.AnnData, "copy", checked_copy)

        result = _read(tmp_path)

        assert isinstance(result.X, sparse.csr_array)
        np.testing.assert_array_equal(result.X.toarray(), [[1, 2], [3, 4]])

    def test_noncanonical_counts_sum_without_dtype_change(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        source.X = sparse.csr_array(
            (np.array([1, 2, 4], dtype=np.int64), np.array([0, 0, 1]), np.array([0, 2, 3])), shape=(2, 2)
        )
        source.write_h5ad(path)

        result = _read(tmp_path)

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.has_canonical_format
        assert result.X.dtype == np.int64
        np.testing.assert_array_equal(result.X.toarray(), [[0, 4], [3, 0]])

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
