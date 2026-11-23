"""On-disk 10x MEX triplets, literal annotations and sparse table assembly."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.counts import MtxFiles, read_10x_mtx, read_10x_mtx_fields
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from tests.integration.readers._utils.counts._fixtures import (
    BARCODES,
    CONTEXT,
    COUNTS,
    FEATURE_IDS,
    FEATURE_NAMES,
    FEATURE_TYPES,
    assert_canonical_sparse,
    assert_sparse_equal,
)

if TYPE_CHECKING:
    from pathlib import Path


def _files(path: Path, *, prefix: str = "", compressed: bool = False, legacy: bool = False) -> MtxFiles:
    suffix = ".gz" if compressed else ""
    features = "genes.tsv" if legacy else "features.tsv"
    return MtxFiles(
        ArtifactFile(path / f"{prefix}matrix.mtx{suffix}"),
        ArtifactFile(path / f"{prefix}barcodes.tsv{suffix}"),
        ArtifactFile(path / f"{prefix}{features}{suffix}"),
        compressed=compressed,
        legacy=legacy,
    )


def _write_lines(path: Path, lines: tuple[str, ...], *, compressed: bool) -> None:
    content = "".join(f"{line}\n" for line in lines)
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
            file.write(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")


def _write_mex(path: Path, *, prefix: str = "", compressed: bool = False, legacy: bool = False) -> MtxFiles:
    files = _files(path, prefix=prefix, compressed=compressed, legacy=legacy)
    _write_lines(
        files.matrix,
        (
            "%%MatrixMarket matrix coordinate integer general",
            "4 4 6",
            "1 1 1",
            "2 1 2",
            "2 2 3",
            "3 2 4",
            "1 3 5",
            "3 3 6",
        ),
        compressed=compressed,
    )
    columns = (FEATURE_IDS, FEATURE_NAMES) if legacy else (FEATURE_IDS, FEATURE_NAMES, FEATURE_TYPES)
    _write_lines(files.features, tuple("\t".join(row) for row in zip(*columns, strict=True)), compressed=compressed)
    _write_lines(files.barcodes, BARCODES, compressed=compressed)
    return files


class TestRead10xMtxFields:
    def test_projection_preserves_order_and_literal_utf8(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "features.tsv")
        path.write_text('007\t"样本"\tGene Expression\tunused\nNULL\tNA\tAntibody Capture\tother\n', encoding="utf-8")
        frame = read_10x_mtx_fields(path, {"symbol": 1, "id": 0}, compressed=False, context=CONTEXT)
        assert frame.columns.tolist() == ["symbol", "id"]
        assert frame["symbol"].tolist() == ['"样本"', "NA"]
        assert frame["id"].tolist() == ["007", "NULL"]
        assert all(dtype == pd.ArrowDtype(pa.large_string()) for dtype in frame.dtypes)


class TestRead10xMtx:
    def test_invalid_matrix_error_identifies_the_artifact(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path)
        _write_lines(
            files.matrix, ("%%MatrixMarket matrix coordinate integer general", "4 4 1", "1 1 1.25"), compressed=False
        )
        with pytest.raises(ReaderFormatError) as caught:
            read_10x_mtx(files, context=CONTEXT)
        assert caught.value.context == CONTEXT
        assert caught.value.path == files.matrix

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_duplicates_symmetry_and_exact_sum(self, tmp_path: Path, *, symmetric: bool) -> None:
        files = _files(tmp_path)
        symmetry = "symmetric" if symmetric else "general"
        files.matrix.write_text(
            f"%%MatrixMarket matrix coordinate integer {symmetry}\n% note\n2 2 3\n2 1 16777216\n2 1 1\n2 2 3\n"
        )
        files.features.write_text("a\tA\tGene Expression\nb\tB\tGene Expression\n")
        files.barcodes.write_text("x\ny\n")
        matrix = assert_canonical_sparse(read_10x_mtx(files, context=CONTEXT))
        assert matrix.dtype == np.float64
        assert matrix.nnz == 2 + symmetric
        assert matrix[0, 1] == 16777217
        assert matrix[1, 0] == (16777217 if symmetric else 0)
        assert matrix[1, 1] == 3

    def test_upper_triangle_in_symmetric_storage(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.features.write_text("a\tA\tGene Expression\nb\tB\tGene Expression\n")
        files.barcodes.write_text("x\ny\n")
        files.matrix.write_text("%%MatrixMarket matrix coordinate integer symmetric\n2 2 1\n1 2 7\n")
        with pytest.raises(ReaderFormatError, match="lower triangle"):
            read_10x_mtx(files, context=CONTEXT)

    def test_missing_feature_type_reports_feature_artifact(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.features.write_text("a\tA\n")
        files.barcodes.write_text("x\n")
        with pytest.raises(ReaderFormatError, match="MEX annotation fields") as caught:
            read_10x_mtx(files, context=CONTEXT)
        assert caught.value.path == files.features

    @pytest.mark.parametrize(("features", "barcodes"), [(0, 0), (1, 0), (0, 1)])
    @pytest.mark.parametrize("compressed", [False, True])
    def test_empty_axes(self, tmp_path: Path, features: int, barcodes: int, *, compressed: bool) -> None:
        files = _files(tmp_path, compressed=compressed)
        _write_lines(files.features, ("a\tA\tGene Expression",) * features, compressed=compressed)
        _write_lines(files.barcodes, ("007",) * barcodes, compressed=compressed)
        _write_lines(
            files.matrix,
            ("%%MatrixMarket matrix coordinate integer general", f"{features} {barcodes} 0"),
            compressed=compressed,
        )
        table = read_10x_mtx(files, context=CONTEXT)
        assert table.shape == (barcodes, features)
        assert assert_canonical_sparse(table).nnz == 0

    def test_reads_only_selected_triplet_and_preserves_literal_fields(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        (tmp_path / "genes.tsv").write_text("wrong\twrong\n")
        files.features.write_text('NULL\t"样本"\tGene Expression\n')
        files.barcodes.write_text("NA\n")
        files.matrix.write_bytes(
            b"%%MatrixMarket matrix coordinate integer general\r\n1 1 2\r\n  +1\t 1  +7  \r\n  % comment\r\n\r\n1 1 03"
        )
        table = read_10x_mtx(files, context=CONTEXT)
        assert table.obs_names.tolist() == ["NA"]
        assert table.var_names.tolist() == ['"样本"']
        assert table.var["gene_ids"].tolist() == ["NULL"]
        assert assert_canonical_sparse(table)[0, 0] == 10

    def test_no_stored_coordinates(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.matrix.write_text("%%MatrixMarket matrix coordinate integer general\n2 1 0\n")
        files.features.write_text("a\tA\tGene Expression\nb\tB\tGene Expression\n")
        files.barcodes.write_text("x\n")
        counts = assert_canonical_sparse(read_10x_mtx(files, context=CONTEXT))
        assert counts.shape == (1, 2)
        assert counts.nnz == 0
        assert counts.dtype == np.float32

    def test_lexical_ids_and_exact_counts(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.matrix.write_text("%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 16777217\n")
        files.features.write_text("007\t001\tGene Expression\n")
        files.barcodes.write_text("0001\n")
        table = read_10x_mtx(files, context=CONTEXT)
        assert table.obs_names.tolist() == ["0001"]
        assert table.var_names.tolist() == ["001"]
        assert table.var["gene_ids"].tolist() == ["007"]
        assert int(assert_canonical_sparse(table)[0, 0]) == 16777217

    def test_rejects_real_count_declaration_before_float_conversion(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path)
        source = files.matrix
        payload = source.read_text()
        source.write_text(payload.replace("coordinate integer", "coordinate real", 1))
        with pytest.raises(ReaderFormatError, match="integer coordinate"):
            read_10x_mtx(files, context=CONTEXT)

    @pytest.mark.parametrize(("compressed", "prefix"), [(False, ""), (False, "sample_"), (True, ""), (True, "样本_")])
    def test_v3_preserves_orientation_order_every_feature_type_and_unique_names(
        self,
        tmp_path: Path,
        *,
        compressed: bool,
        prefix: str,
    ) -> None:
        files = _write_mex(tmp_path, prefix=prefix, compressed=compressed)
        table = read_10x_mtx(files, context=CONTEXT)

        expression = assert_canonical_sparse(table)
        assert_sparse_equal(expression, COUNTS)
        assert tuple(table.obs_names) == BARCODES
        assert tuple(table.var_names) == ("duplicate", "protein", "duplicate-1", "empty-gene")
        assert tuple(table.var["gene_ids"]) == FEATURE_IDS
        assert tuple(table.var["feature_types"]) == FEATURE_TYPES
        assert "spatial" not in table.uns

        for artifact in (files.matrix, files.features, files.barcodes):
            artifact.unlink()
        assert_sparse_equal(expression, COUNTS)

    def test_v3_keeps_duplicate_symbols_without_make_unique(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path, compressed=True)
        table = read_10x_mtx(files, context=CONTEXT, make_unique=False)

        expression = assert_canonical_sparse(table)
        assert_sparse_equal(expression, COUNTS)
        assert tuple(table.var_names) == FEATURE_NAMES
        assert tuple(table.var["gene_ids"]) == FEATURE_IDS
        assert tuple(table.var["feature_types"]) == FEATURE_TYPES

    def test_legacy_uses_two_gene_fields(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path, prefix="legacy_", legacy=True)
        table = read_10x_mtx(files, context=CONTEXT)

        expression = assert_canonical_sparse(table)
        assert_sparse_equal(expression, COUNTS)
        assert tuple(table.obs_names) == BARCODES
        assert tuple(table.var_names) == ("duplicate", "protein", "duplicate-1", "empty-gene")
        assert tuple(table.var["gene_ids"]) == FEATURE_IDS
        assert "feature_types" not in table.var

    def test_missing_companion_preserves_context(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path, prefix="broken_", compressed=True)
        files.features.unlink()
        with pytest.raises(FileNotFoundError) as caught:
            read_10x_mtx(files, context=CONTEXT)
        assert caught.value.filename == str(files.features)
