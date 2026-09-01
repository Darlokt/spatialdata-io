"""Unit tests for the typed public-Scanpy 10x source adapters."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx
from spatialdata_io.readers._utils.errors import ReaderErrorContext

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile

_CONTEXT = ReaderErrorContext(reader="test_reader", artifact="count matrix", detected_version="3")


def _artifact_file(path: Path) -> ArtifactFile:
    return cast("ArtifactFile", path)


def _artifact_directory(path: Path) -> ArtifactDirectory:
    return cast("ArtifactDirectory", path)


def _table() -> AnnData:
    return AnnData(sparse.csr_matrix(np.asarray([[1, 0], [0, 2]], dtype=np.int32)))


def _raise_preserved_error(error: Exception) -> None:
    raise error


class TestRead10xH5:
    def test_forwards_only_explicit_options_and_normalizes_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        source_table = _table()
        normalized = _table()
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=source_table,
        )
        normalizer = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x.normalize_owned_anndata",
            return_value=normalized,
        )

        result = read_10x_h5(path, context=_CONTEXT, genome="GRCh38", gex_only=False)

        assert result is normalized
        source.assert_called_once_with(path, genome="GRCh38", gex_only=False, backup_url=None)
        normalizer.assert_called_once_with(source_table)

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

    def test_real_normalizer_rejects_upstream_view_with_source_context(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        view = _table()[:, :]
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=view,
        )

        with pytest.raises(ValueError, match="owned AnnData, not a view") as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        source.assert_called_once()

    def test_real_normalizer_rejects_nonfinite_expression_with_source_context(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        path = _artifact_file(tmp_path / "counts.h5")
        malformed = AnnData(sparse.csr_matrix(np.asarray([[np.inf]], dtype=np.float64)))
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_h5",
            return_value=malformed,
        )

        with pytest.raises(ValueError, match="must contain only finite values") as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value.__notes__ == [
            "while normalizing adata.X",
            f"While handling {_CONTEXT.describe(path=path)}.",
        ]
        source.assert_called_once()


class TestRead10xMtx:
    def test_forwards_only_explicit_options_and_normalizes_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _artifact_directory(tmp_path)
        source_table = _table()
        normalized = _table()
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_mtx",
            return_value=source_table,
        )
        normalizer = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x.normalize_owned_anndata",
            return_value=normalized,
        )

        result = read_10x_mtx(
            path,
            context=_CONTEXT,
            var_names="gene_ids",
            make_unique=False,
            gex_only=False,
            prefix="sample_",
            compressed=False,
        )

        assert result is normalized
        source.assert_called_once_with(
            path,
            var_names="gene_ids",
            make_unique=False,
            cache=False,
            gex_only=False,
            prefix="sample_",
            compressed=False,
        )
        normalizer.assert_called_once_with(source_table)

    @pytest.mark.parametrize("prefix", [None, "", ".", "..", ".hidden_", "sample-1_", "sample name_", "样本_"])
    def test_accepts_lexical_filename_prefixes(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        prefix: str | None,
    ) -> None:
        path = _artifact_directory(tmp_path)
        source = mocker.patch(
            "spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_mtx",
            return_value=_table(),
        )

        read_10x_mtx(path, context=_CONTEXT, prefix=prefix)

        assert source.call_args.kwargs["prefix"] == prefix

    @pytest.mark.parametrize(
        "prefix",
        [
            "bad\0prefix",
            "bad\x1fprefix",
            "/absolute/",
            "folder/name_",
            "folder\\name_",
            "../escape_",
            "..\\escape_",
            "C:drive_",
            "sample:stream_",
            'bad"prefix',
            "bad<prefix",
            "bad>prefix",
            "bad|prefix",
            "bad?prefix",
            "bad*prefix",
            "CON.",
            "prn.",
            "AUX.",
            "nul.",
            "CONIN$.",
            "conout$.",
            "COM1.",
            "com¹.",
            "LPT9.",
            "lpt³.",
        ],
    )
    def test_rejects_path_syntax_before_source_or_normalization(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        prefix: str,
    ) -> None:
        path = _artifact_directory(tmp_path)
        source = mocker.patch("spatialdata_io.readers._utils.counts._10x._scanpy_read_10x_mtx")
        normalizer = mocker.patch("spatialdata_io.readers._utils.counts._10x.normalize_owned_anndata")

        with pytest.raises(ValueError, match="prefix must be portable filename text") as caught:
            read_10x_mtx(path, context=_CONTEXT, prefix=prefix)

        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        source.assert_not_called()
        normalizer.assert_not_called()

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
