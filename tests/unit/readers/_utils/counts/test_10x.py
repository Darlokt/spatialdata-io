"""Unit tests for the typed public-Scanpy 10x source adapters."""

from __future__ import annotations

import gzip
import traceback
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from spatialdata_io.readers._utils.counts import _10x, read_10x_h5, read_10x_mtx
from spatialdata_io.readers._utils.counts._10x import _finalize_counts, _validate_mtx_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile

_CONTEXT = ReaderErrorContext(reader="test_reader", artifact="count matrix", detected_version="3")


@pytest.fixture(autouse=True)
def source_validation(mocker: MockerFixture) -> None:
    """Keep MatrixMarket artifact inspection in the integration contract tests."""
    mocker.patch("spatialdata_io.readers._utils.counts._10x._validate_mtx_header")


def _artifact_file(path: Path) -> ArtifactFile:
    return cast("ArtifactFile", path)


def _artifact_directory(path: Path) -> ArtifactDirectory:
    return cast("ArtifactDirectory", path)


def _table() -> AnnData:
    # Scanpy decodes 10x int32 count storage to float32.
    return AnnData(sparse.csr_matrix(np.asarray([[1, 0], [0, 2]], dtype=np.float32)))


def _raise_preserved_error(error: Exception) -> None:
    raise error


class TestRead10xH5:
    def test_forwards_only_explicit_options_and_validates_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        source_table = _table()
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=source_table,
        )
        normalizer = mocker.spy(_10x, "normalize_owned_anndata")
        finalizer = mocker.spy(_10x, "finalize_count_arrays")

        result = read_10x_h5(path, context=_CONTEXT)

        assert result is source_table
        source.assert_called_once_with(path, gex_only=False, backup_url=None)
        normalizer.assert_called_once_with(source_table, duplicate_sums=[])
        finalizer.assert_called_once()

    def test_wraps_canonical_scanpy_csr_storage_without_copying_buffers(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        source_table = _table()
        source = cast("sparse.csr_matrix", source_table.X)
        data = source.data
        indices = source.indices
        indptr = source.indptr
        mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=source_table,
        )

        result = read_10x_h5(path, context=_CONTEXT)

        assert result is source_table
        expression_value: object = result.X
        expression = cast("sparse.csr_array", expression_value)
        assert isinstance(expression, sparse.csr_array)
        assert isinstance(expression, sparse.sparray)
        assert not isinstance(expression, sparse.spmatrix)
        assert np.shares_memory(expression.data, data)
        assert np.shares_memory(expression.indices, indices)
        assert np.shares_memory(expression.indptr, indptr)

    def test_source_failure_preserves_identity_cause_traceback_and_skips_normalization(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _artifact_file(tmp_path / "broken.h5")
        cause = OSError("low-level failure")
        error = FileNotFoundError("missing source")
        error.__cause__ = cause

        def raise_source_error(*_args: object, **_kwargs: object) -> None:
            _raise_preserved_error(error)

        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            side_effect=raise_source_error,
        )
        normalizer = mocker.patch("spatialdata_io.readers._utils.counts._10x.normalize_owned_anndata")

        with pytest.raises(FileNotFoundError) as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value is error
        assert caught.value.__cause__ is cause
        assert "_raise_preserved_error" in {frame.name for frame in traceback.extract_tb(caught.value.__traceback__)}
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        source.assert_called_once()
        normalizer.assert_not_called()

    def test_normalization_failure_is_preserved_with_source_context(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        error = TypeError("unsupported expression")
        mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=_table(),
        )
        normalizer = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x.normalize_owned_anndata",
            side_effect=error,
        )

        with pytest.raises(TypeError) as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value is error
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        normalizer.assert_called_once()

    def test_rejects_nonfinite_counts_with_source_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        malformed = AnnData(sparse.csr_matrix(np.asarray([[np.inf]], dtype=np.float64)))
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=malformed,
        )

        with pytest.raises(ReaderFormatError, match="finite values") as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value.context == _CONTEXT
        assert caught.value.path == path
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        source.assert_called_once()


class TestRead10xMtx:
    def test_forwards_only_explicit_options_and_validates_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _artifact_directory(tmp_path)
        source_table = _table()
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_mtx",
            return_value=source_table,
        )
        normalizer = mocker.spy(_10x, "normalize_owned_anndata")

        result = read_10x_mtx(path, context=_CONTEXT, make_unique=False, prefix="sample_", compressed=False)

        assert result is source_table
        source.assert_called_once_with(
            path,
            var_names="gene_symbols",
            make_unique=False,
            cache=False,
            gex_only=False,
            prefix="sample_",
            compressed=False,
        )
        normalizer.assert_called_once_with(source_table, duplicate_sums=[])

    def test_source_failure_preserves_identity_and_skips_normalization(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _artifact_directory(tmp_path)
        error = OSError("cannot read companion")
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_mtx",
            side_effect=error,
        )
        normalizer = mocker.patch("spatialdata_io.readers._utils.counts._10x.normalize_owned_anndata")

        with pytest.raises(OSError, match="cannot read companion") as caught:
            read_10x_mtx(path, context=_CONTEXT)

        assert caught.value is error
        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        source.assert_called_once()
        normalizer.assert_not_called()


def _compressed(
    values: list[float], indices: list[int], dtype: str, storage: str
) -> sparse.csr_array | sparse.csc_array:
    buffers = (np.array(values, dtype=dtype), np.array(indices, dtype=np.int32), np.array([0, len(values)]))
    shape = (1, max(indices) + 1)
    return sparse.csr_array(buffers, shape=shape) if storage == "csr" else sparse.csc_array(buffers, shape=shape[::-1])


class TestFinalizeCounts:
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
    def test_rejects_repeated_entries_with_source_context(self, tmp_path: Path, storage: str, dtype: str) -> None:
        path = tmp_path / "counts.h5"
        source = _compressed([2, 3, 4], [1, 0, 1], dtype, storage)

        with pytest.raises(ReaderFormatError, match=r"1 repeated \(barcode, feature\) entries") as caught:
            _finalize_counts(AnnData(source), path, _CONTEXT)

        assert caught.value.context == _CONTEXT
        assert caught.value.path == path
        np.testing.assert_array_equal(source.data, [2, 3, 4])

    @pytest.mark.parametrize("values", [(0.5, 0.5), (-1.0, 2.0)])
    def test_repeated_entries_cannot_become_valid_counts(self, tmp_path: Path, values: tuple[float, float]) -> None:
        source = _compressed(list(values), [0, 0], "float64", "csr")
        with pytest.raises(ReaderFormatError, match="repeated"):
            _finalize_counts(AnnData(source), tmp_path / "counts.h5", _CONTEXT)

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    def test_sorts_unique_entries_without_summing(self, tmp_path: Path, storage: str) -> None:
        table = AnnData(_compressed([2**30, 1], [1, 0], "float32", storage))

        result = _finalize_counts(table, tmp_path / "counts.h5", _CONTEXT)

        assert result is table
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.has_canonical_format
        assert result.X.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.X.data, [1, 2**30])

    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("bool", 1, "float32"),
            ("int32", 2**24, "float32"),
            ("int64", 2**24 + 1, "float64"),
            ("float32", 2**30, "float32"),
            ("float64", 7, "float32"),
            ("float64", 2**24 + 1, "float64"),
        ],
    )
    def test_stores_counts_as_float32_or_float64(self, tmp_path: Path, dtype: str, value: int, expected: str) -> None:
        result = _finalize_counts(
            AnnData(sparse.csr_array(np.array([[value]], dtype=dtype))), tmp_path / "c.h5", _CONTEXT
        )
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(expected)
        assert int(result.X.data[0]) == value

    def test_canonical_float32_counts_are_reused(self, tmp_path: Path) -> None:
        data = sparse.csr_array(np.asarray([[0, 1, 100_000]], dtype=np.float32))
        table = AnnData(data)
        assert _finalize_counts(table, tmp_path / "counts.h5", _CONTEXT).X is data

    @pytest.mark.parametrize("value", [-1, 0.5])
    def test_rejects_malformed_counts(self, tmp_path: Path, value: float) -> None:
        table = AnnData(sparse.csr_array(np.asarray([[value]], dtype=np.float32)))
        with pytest.raises(ReaderFormatError, match="negative or fractional"):
            _finalize_counts(table, tmp_path / "counts.h5", _CONTEXT)

    def test_rejects_malformed_structure(self, tmp_path: Path) -> None:
        source = sparse.csr_array([[1, 2]], dtype=np.float32)
        source.indices = np.asarray([0, 2], dtype=np.int32)
        with pytest.raises(ReaderFormatError, match="indices must be < 2"):
            _finalize_counts(AnnData(source), tmp_path / "counts.h5", _CONTEXT)

    def test_rejects_unsupported_values(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="real floating dtype"):
            _finalize_counts(AnnData(sparse.csr_array([[1j]])), tmp_path / "counts.h5", _CONTEXT)

    def test_unused_sparse_capacity_is_not_a_source_count(self, tmp_path: Path) -> None:
        source = sparse.csr_array([[1, 2]], dtype=np.int32)
        source.data = np.append(source.data, np.int32(-1))
        source.indices = np.append(source.indices, np.int32(99))

        result = _finalize_counts(AnnData(source), tmp_path / "counts.h5", _CONTEXT)

        assert isinstance(result.X, sparse.csr_array)
        np.testing.assert_array_equal(result.X.data, [1, 2])
        assert len(source.data) == 3

    def test_empty_counts(self, tmp_path: Path) -> None:
        result = _finalize_counts(AnnData(sparse.csr_array((2, 2), dtype=np.float32)), tmp_path / "c.h5", _CONTEXT)
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.shape == (2, 2)

    def test_missing_expression(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="missing count data"):
            _finalize_counts(AnnData(), tmp_path / "counts.h5", _CONTEXT)


class TestValidateMtxHeader:
    @pytest.mark.parametrize("compressed", [False, True])
    def test_integer_header(self, tmp_path: Path, *, compressed: bool) -> None:
        payload = b"%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 1\n"
        path = tmp_path / ("x_matrix.mtx.gz" if compressed else "x_matrix.mtx")
        path.write_bytes(gzip.compress(payload) if compressed else payload)
        _validate_mtx_header(_artifact_directory(tmp_path), context=_CONTEXT, prefix="x_", compressed=compressed)

    def test_legacy_ignores_compression(self, tmp_path: Path) -> None:
        (tmp_path / "genes.tsv").touch()
        (tmp_path / "matrix.mtx").write_bytes(b"%%MatrixMarket matrix coordinate integer general\n")
        _validate_mtx_header(_artifact_directory(tmp_path), context=_CONTEXT, prefix=None, compressed=True)

    @pytest.mark.parametrize("header", [b"real general", b"integer skew-symmetric", b"pattern general", b"x" * 1024])
    def test_rejects_non_count_declarations(self, tmp_path: Path, header: bytes) -> None:
        (tmp_path / "matrix.mtx").write_bytes(b"%%MatrixMarket matrix coordinate " + header + b"\n")
        with pytest.raises(ReaderFormatError, match="integer coordinate"):
            _validate_mtx_header(_artifact_directory(tmp_path), context=_CONTEXT, prefix=None, compressed=False)
