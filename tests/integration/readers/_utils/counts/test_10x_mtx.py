"""On-disk 10x MEX triplets, literal annotations and sparse table assembly."""

from __future__ import annotations

import gzip
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.counts import MtxFiles, MtxMember, read_10x_mtx, read_10x_mtx_fields
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
        MtxMember(ArtifactFile(path / f"{prefix}matrix.mtx{suffix}"), compressed=compressed),
        MtxMember(ArtifactFile(path / f"{prefix}barcodes.tsv{suffix}"), compressed=compressed),
        MtxMember(ArtifactFile(path / f"{prefix}{features}{suffix}"), compressed=compressed),
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
        files.matrix.file,
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
    _write_lines(
        files.features.file, tuple("\t".join(row) for row in zip(*columns, strict=True)), compressed=compressed
    )
    _write_lines(files.barcodes.file, BARCODES, compressed=compressed)
    return files


class TestRead10xMtxFields:
    def test_projection_preserves_order_and_literal_utf8(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "features.tsv")
        path.write_text('007\t"样本"\tGene Expression\tunused\nNULL\tNA\tAntibody Capture\tother\n', encoding="utf-8")
        frame = read_10x_mtx_fields(MtxMember(path, compressed=False), {"symbol": 1, "id": 0}, context=CONTEXT)
        assert frame.columns.tolist() == ["symbol", "id"]
        assert frame["symbol"].tolist() == ['"样本"', "NA"]
        assert frame["id"].tolist() == ["007", "NULL"]
        assert all(dtype == pd.ArrowDtype(pa.large_string()) for dtype in frame.dtypes)


class TestRead10xMtx:
    @pytest.mark.parametrize("compression", tuple(product((False, True), repeat=3)))
    def test_decodes_each_member_using_its_declared_compression(
        self, tmp_path: Path, compression: tuple[bool, bool, bool]
    ) -> None:
        plain = _write_mex(tmp_path)
        members = []
        for original, compressed in zip((plain.matrix, plain.barcodes, plain.features), compression, strict=True):
            # A validated path may point to a symlink target with a different name.
            path = ArtifactFile(original.file.with_suffix(".payload"))
            data = original.file.read_bytes()
            path.write_bytes(gzip.compress(data) if compressed else data)
            members.append(MtxMember(path, compressed=compressed))

        table = read_10x_mtx(MtxFiles(members[0], members[1], members[2]), context=CONTEXT, make_unique=False)

        assert_sparse_equal(assert_canonical_sparse(table), COUNTS)
        assert tuple(table.obs_names) == BARCODES
        assert tuple(table.var_names) == FEATURE_NAMES
        assert tuple(table.var["gene_ids"]) == FEATURE_IDS
        assert tuple(table.var["feature_types"]) == FEATURE_TYPES

    def test_invalid_matrix_error_identifies_the_artifact(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path)
        _write_lines(
            files.matrix.file,
            ("%%MatrixMarket matrix coordinate integer general", "4 4 1", "1 1 1.25"),
            compressed=False,
        )
        with pytest.raises(ReaderFormatError) as caught:
            read_10x_mtx(files, context=CONTEXT)
        assert caught.value.context == CONTEXT
        assert caught.value.path == files.matrix.file

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_duplicates_symmetry_and_exact_sum(self, tmp_path: Path, *, symmetric: bool) -> None:
        files = _files(tmp_path)
        symmetry = "symmetric" if symmetric else "general"
        files.matrix.file.write_text(
            f"%%MatrixMarket matrix coordinate integer {symmetry}\n% note\n2 2 3\n2 1 16777216\n2 1 1\n2 2 3\n"
        )
        files.features.file.write_text("a\tA\tGene Expression\nb\tB\tGene Expression\n")
        files.barcodes.file.write_text("x\ny\n")
        matrix = assert_canonical_sparse(read_10x_mtx(files, context=CONTEXT))
        assert matrix.dtype == np.float64
        assert matrix.nnz == 2 + symmetric
        assert matrix[0, 1] == 16777217
        assert matrix[1, 0] == (16777217 if symmetric else 0)
        assert matrix[1, 1] == 3

    def test_upper_triangle_in_symmetric_storage(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.features.file.write_text("a\tA\tGene Expression\nb\tB\tGene Expression\n")
        files.barcodes.file.write_text("x\ny\n")
        files.matrix.file.write_text("%%MatrixMarket matrix coordinate integer symmetric\n2 2 1\n1 2 7\n")
        with pytest.raises(ReaderFormatError, match="lower triangle"):
            read_10x_mtx(files, context=CONTEXT)

    def test_missing_feature_type_reports_feature_artifact(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.features.file.write_text("a\tA\n")
        files.barcodes.file.write_text("x\n")
        with pytest.raises(ReaderFormatError, match="MEX annotation fields") as caught:
            read_10x_mtx(files, context=CONTEXT)
        assert caught.value.path == files.features.file

    @pytest.mark.parametrize(("features", "barcodes"), [(0, 0), (1, 0), (0, 1)])
    @pytest.mark.parametrize("compressed", [False, True])
    def test_empty_axes(self, tmp_path: Path, features: int, barcodes: int, *, compressed: bool) -> None:
        files = _files(tmp_path, compressed=compressed)
        _write_lines(files.features.file, ("a\tA\tGene Expression",) * features, compressed=compressed)
        _write_lines(files.barcodes.file, ("007",) * barcodes, compressed=compressed)
        _write_lines(
            files.matrix.file,
            ("%%MatrixMarket matrix coordinate integer general", f"{features} {barcodes} 0"),
            compressed=compressed,
        )
        table = read_10x_mtx(files, context=CONTEXT)
        assert table.shape == (barcodes, features)
        assert assert_canonical_sparse(table).nnz == 0

    def test_reads_only_selected_triplet_and_preserves_literal_fields(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        (tmp_path / "genes.tsv").write_text("wrong\twrong\n")
        files.features.file.write_text('NULL\t"样本"\tGene Expression\n')
        files.barcodes.file.write_text("NA\n")
        files.matrix.file.write_bytes(
            b"%%MatrixMarket matrix coordinate integer general\r\n1 1 2\r\n  +1\t 1  +7  \r\n  % comment\r\n\r\n1 1 03"
        )
        table = read_10x_mtx(files, context=CONTEXT)
        assert table.obs_names.tolist() == ["NA"]
        assert table.var_names.tolist() == ['"样本"']
        assert table.var["gene_ids"].tolist() == ["NULL"]
        assert assert_canonical_sparse(table)[0, 0] == 10

    def test_no_stored_coordinates(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.matrix.file.write_text("%%MatrixMarket matrix coordinate integer general\n2 1 0\n")
        files.features.file.write_text("a\tA\tGene Expression\nb\tB\tGene Expression\n")
        files.barcodes.file.write_text("x\n")
        counts = assert_canonical_sparse(read_10x_mtx(files, context=CONTEXT))
        assert counts.shape == (1, 2)
        assert counts.nnz == 0
        assert counts.dtype == np.float32

    def test_lexical_ids_and_exact_counts(self, tmp_path: Path) -> None:
        files = _files(tmp_path)
        files.matrix.file.write_text("%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 16777217\n")
        files.features.file.write_text("007\t001\tGene Expression\n")
        files.barcodes.file.write_text("0001\n")
        table = read_10x_mtx(files, context=CONTEXT)
        assert table.obs_names.tolist() == ["0001"]
        assert table.var_names.tolist() == ["001"]
        assert table.var["gene_ids"].tolist() == ["007"]
        assert int(assert_canonical_sparse(table)[0, 0]) == 16777217

    def test_rejects_real_count_declaration_before_float_conversion(self, tmp_path: Path) -> None:
        files = _write_mex(tmp_path)
        source = files.matrix.file
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

        for artifact in (files.matrix.file, files.features.file, files.barcodes.file):
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
        files.features.file.unlink()
        with pytest.raises(FileNotFoundError) as caught:
            read_10x_mtx(files, context=CONTEXT)
        assert caught.value.filename == str(files.features.file)
