"""Integration tests for public Scanpy 10x source composition."""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import pytest
from scipy import io as scipy_io
from scipy import sparse

from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import require_directory, require_file

if TYPE_CHECKING:
    from anndata import AnnData

_CONTEXT = ReaderErrorContext(reader="integration", artifact="10x counts", detected_version="synthetic")
_BARCODES = ("barcode-b", "barcode-a", "barcode-c", "barcode-empty")
_FEATURE_IDS = ("gene-id-a", "protein-id", "gene-id-b", "gene-id-empty")
_FEATURE_NAMES = ("duplicate", "protein", "duplicate", "empty-gene")
_FEATURE_TYPES = ("Gene Expression", "Antibody Capture", "Gene Expression", "Gene Expression")
_GENOMES = ("GRCh38", "GRCh38", "mm10", "GRCh38")
_COUNTS = sparse.csr_array(
    (
        np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
        np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int32),
        np.asarray([0, 2, 4, 6, 6], dtype=np.int32),
    ),
    shape=(4, 4),
)


def _write_h5_v3(path: Path) -> None:
    with h5py.File(path, "w") as file:
        file.attrs["library_ids"] = np.asarray([b"reader-owned-library"])
        file.attrs["protein_scaling_factor"] = 10.0
        matrix = file.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray(_BARCODES, dtype="S"))
        matrix.create_dataset("data", data=_COUNTS.data)
        matrix.create_dataset("indices", data=_COUNTS.indices)
        matrix.create_dataset("indptr", data=_COUNTS.indptr)
        matrix.create_dataset("shape", data=np.asarray([_COUNTS.shape[1], _COUNTS.shape[0]], dtype=np.int64))
        matrix.create_dataset("filtered_barcodes", data=np.asarray([1, 0, 1, 0], dtype=np.uint8))

        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray(_FEATURE_IDS, dtype="S"))
        features.create_dataset("name", data=np.asarray(_FEATURE_NAMES, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(_FEATURE_TYPES, dtype="S"))
        features.create_dataset("genome", data=np.asarray(_GENOMES, dtype="S"))
        features.create_dataset("target_set", data=np.asarray(["A", "protein", "B", "empty"], dtype="S"))


def _write_h5_legacy_genome(group: h5py.Group, *, offset: int) -> None:
    group.create_dataset("barcodes", data=np.asarray([b"legacy-b", b"legacy-a"]))
    group.create_dataset("data", data=np.asarray([1 + offset, 2 + offset, 3 + offset], dtype=np.int64))
    group.create_dataset("indices", data=np.asarray([0, 1, 0], dtype=np.int32))
    group.create_dataset("indptr", data=np.asarray([0, 2, 3], dtype=np.int32))
    group.create_dataset("shape", data=np.asarray([2, 2], dtype=np.int64))
    group.create_dataset("gene_names", data=np.asarray([b"legacy-z", b"legacy-a"]))
    group.create_dataset("genes", data=np.asarray([b"legacy-id-z", b"legacy-id-a"]))


def _write_h5_legacy(path: Path, *, genomes: tuple[str, ...]) -> None:
    with h5py.File(path, "w") as file:
        for offset, genome in enumerate(genomes):
            _write_h5_legacy_genome(file.create_group(genome), offset=offset * 10)


def _write_mtx(path: Path, matrix: sparse.sparray, *, compressed: bool) -> None:
    stream = io.BytesIO()
    scipy_io.mmwrite(stream, matrix)
    payload = stream.getvalue()
    if compressed:
        with gzip.open(path, "wb") as file:
            file.write(payload)
    else:
        path.write_bytes(payload)


def _write_lines(path: Path, lines: tuple[str, ...], *, compressed: bool) -> None:
    content = "".join(f"{line}\n" for line in lines)
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
            file.write(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")


def _write_mtx_v3(path: Path, *, prefix: str, compressed: bool, include_features: bool = True) -> tuple[Path, ...]:
    suffix = ".gz" if compressed else ""
    matrix_path = path / f"{prefix}matrix.mtx{suffix}"
    features_path = path / f"{prefix}features.tsv{suffix}"
    barcodes_path = path / f"{prefix}barcodes.tsv{suffix}"
    _write_mtx(matrix_path, _COUNTS.T, compressed=compressed)
    if include_features:
        _write_lines(
            features_path,
            tuple("\t".join(values) for values in zip(_FEATURE_IDS, _FEATURE_NAMES, _FEATURE_TYPES, strict=True)),
            compressed=compressed,
        )
    _write_lines(barcodes_path, _BARCODES, compressed=compressed)
    return matrix_path, features_path, barcodes_path


def _write_mtx_legacy(path: Path, *, prefix: str) -> tuple[Path, ...]:
    matrix_path = path / f"{prefix}matrix.mtx"
    genes_path = path / f"{prefix}genes.tsv"
    barcodes_path = path / f"{prefix}barcodes.tsv"
    _write_mtx(matrix_path, _COUNTS.T, compressed=False)
    _write_lines(
        genes_path,
        tuple("\t".join(values) for values in zip(_FEATURE_IDS, _FEATURE_NAMES, strict=True)),
        compressed=False,
    )
    _write_lines(barcodes_path, _BARCODES, compressed=False)
    return matrix_path, genes_path, barcodes_path


def _assert_canonical_sparse(table: AnnData) -> sparse.csr_array:
    assert not table.isbacked
    assert not table.is_view
    expression_value: object = table.X
    expression = cast("sparse.csr_array", expression_value)
    assert isinstance(expression, sparse.csr_array)
    assert isinstance(expression, sparse.sparray)
    assert not isinstance(expression, sparse.spmatrix)
    assert expression.has_canonical_format
    return expression


def _assert_sparse_equal(actual: sparse.csr_array, expected: sparse.csr_array) -> None:
    assert actual.shape == expected.shape
    difference = actual - expected
    difference.eliminate_zeros()
    assert difference.nnz == 0


def _open_hdf5_sources() -> set[Path]:
    paths: set[Path] = set()
    for identifier in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE):
        name = h5py.h5f.get_name(identifier)
        if name is not None:
            paths.add(Path(os.fsdecode(name)).resolve())
    return paths


class TestRead10xH5:
    @pytest.mark.parametrize(
        ("genome", "gex_only", "selected"),
        [
            (None, False, (0, 1, 2, 3)),
            (None, True, (0, 2, 3)),
            ("GRCh38", False, (0, 1, 3)),
            ("GRCh38", True, (0, 3)),
        ],
    )
    # Pytest injects the boolean parameter from the explicit case table.
    def test_v3_preserves_orientation_order_metadata_and_filtering(
        self,
        tmp_path: Path,
        *,
        genome: str | None,
        gex_only: bool,
        selected: tuple[int, ...],
    ) -> None:
        source = tmp_path / "matrix.h5"
        _write_h5_v3(source)
        path = require_file(source, context=_CONTEXT)

        table = read_10x_h5(path, context=_CONTEXT, genome=genome, gex_only=gex_only)

        expression = _assert_canonical_sparse(table)
        _assert_sparse_equal(expression, _COUNTS[:, selected])
        assert tuple(table.obs_names) == _BARCODES
        assert tuple(table.var_names) == tuple(_FEATURE_NAMES[index] for index in selected)
        assert tuple(table.var["gene_ids"]) == tuple(_FEATURE_IDS[index] for index in selected)
        assert tuple(table.var["feature_types"]) == tuple(_FEATURE_TYPES[index] for index in selected)
        assert tuple(table.var["genome"]) == tuple(_GENOMES[index] for index in selected)
        assert tuple(table.var["target_set"]) == tuple(("A", "protein", "B", "empty")[index] for index in selected)
        assert "spatial" not in table.uns
        assert source.resolve() not in _open_hdf5_sources()

    def test_v3_result_is_independent_from_removed_source(self, tmp_path: Path) -> None:
        source = tmp_path / "matrix.h5"
        _write_h5_v3(source)
        path = require_file(source, context=_CONTEXT)

        table = read_10x_h5(path, context=_CONTEXT, gex_only=False)
        source.unlink()

        expression = _assert_canonical_sparse(table)
        _assert_sparse_equal(expression, _COUNTS)
        assert tuple(table.obs_names) == _BARCODES

    def test_v3_supports_filtering_to_zero_features(self, tmp_path: Path) -> None:
        source = tmp_path / "no-gex.h5"
        _write_h5_v3(source)
        with h5py.File(source, "r+") as file:
            file["matrix/features/feature_type"][:] = np.asarray([b"Antibody Capture"] * 4)
        path = require_file(source, context=_CONTEXT)

        table = read_10x_h5(path, context=_CONTEXT, gex_only=True)

        expression = _assert_canonical_sparse(table)
        assert expression.shape == (4, 0)
        assert expression.nnz == 0
        assert tuple(table.obs_names) == _BARCODES
        assert len(table.var_names) == 0

    def test_legacy_selects_genome_and_preserves_order(self, tmp_path: Path) -> None:
        source = tmp_path / "legacy.h5"
        _write_h5_legacy(source, genomes=("GRCh38", "mm10"))
        path = require_file(source, context=_CONTEXT)

        table = read_10x_h5(path, context=_CONTEXT, genome="mm10", gex_only=True)

        expression = _assert_canonical_sparse(table)
        expected = sparse.csr_array(np.asarray([[11, 12], [13, 0]], dtype=np.int64))
        _assert_sparse_equal(expression, expected)
        assert tuple(table.obs_names) == ("legacy-b", "legacy-a")
        assert tuple(table.var_names) == ("legacy-z", "legacy-a")
        assert tuple(table.var["gene_ids"]) == ("legacy-id-z", "legacy-id-a")
        assert "feature_types" not in table.var
        assert source.resolve() not in _open_hdf5_sources()

    def test_legacy_single_genome_is_selected_implicitly(self, tmp_path: Path) -> None:
        source = tmp_path / "legacy-single.h5"
        _write_h5_legacy(source, genomes=("GRCh38",))
        path = require_file(source, context=_CONTEXT)

        table = read_10x_h5(path, context=_CONTEXT)

        expression = _assert_canonical_sparse(table)
        expected = sparse.csr_array(np.asarray([[1, 2], [3, 0]], dtype=np.int64))
        _assert_sparse_equal(expression, expected)
        assert tuple(table.obs_names) == ("legacy-b", "legacy-a")
        assert tuple(table.var_names) == ("legacy-z", "legacy-a")
        assert source.resolve() not in _open_hdf5_sources()

    def test_legacy_multiple_genomes_require_selection_and_close_source(self, tmp_path: Path) -> None:
        source = tmp_path / "legacy-multiple.h5"
        _write_h5_legacy(source, genomes=("GRCh38", "mm10"))
        path = require_file(source, context=_CONTEXT)

        with pytest.raises(ValueError, match="contains more than one genome") as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        assert source.resolve() not in _open_hdf5_sources()

    def test_malformed_v3_preserves_failure_context_and_closes_source(self, tmp_path: Path) -> None:
        source = tmp_path / "malformed.h5"
        with h5py.File(source, "w") as file:
            matrix = file.create_group("matrix")
            matrix.create_dataset("barcodes", data=np.asarray([b"barcode"]))
            matrix.create_dataset("data", data=np.asarray([1], dtype=np.int32))
            matrix.create_dataset("indices", data=np.asarray([0], dtype=np.int32))
            matrix.create_dataset("indptr", data=np.asarray([0, 1], dtype=np.int32))
            matrix.create_dataset("shape", data=np.asarray([1, 1], dtype=np.int64))
            matrix.create_dataset("id", data=np.asarray([b"gene-id"]))
            matrix.create_dataset("name", data=np.asarray([b"gene"]))
            matrix.create_dataset("feature_type", data=np.asarray([b"Gene Expression"]))
        path = require_file(source, context=_CONTEXT)

        with pytest.raises(ValueError, match="10x h5 has no features group") as caught:
            read_10x_h5(path, context=_CONTEXT)

        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
        assert source.resolve() not in _open_hdf5_sources()
        replacement = tmp_path / "replacement.h5"
        source.replace(replacement)
        replacement.replace(source)


class TestRead10xMtx:
    @pytest.mark.parametrize(("compressed", "prefix"), [(False, ""), (False, "sample_"), (True, ""), (True, "样本_")])
    # Pytest injects the boolean parameter from the explicit case table.
    def test_v3_preserves_orientation_order_filtering_and_unique_names(
        self,
        tmp_path: Path,
        *,
        compressed: bool,
        prefix: str,
    ) -> None:
        family = _write_mtx_v3(tmp_path, prefix=prefix, compressed=compressed)
        path = require_directory(tmp_path, context=_CONTEXT)

        table = read_10x_mtx(path, context=_CONTEXT, prefix=prefix, compressed=compressed)

        expression = _assert_canonical_sparse(table)
        _assert_sparse_equal(expression, _COUNTS[:, (0, 2, 3)])
        assert tuple(table.obs_names) == _BARCODES
        assert tuple(table.var_names) == ("duplicate", "duplicate-1", "empty-gene")
        assert tuple(table.var["gene_ids"]) == ("gene-id-a", "gene-id-b", "gene-id-empty")
        assert tuple(table.var["feature_types"]) == (
            "Gene Expression",
            "Gene Expression",
            "Gene Expression",
        )
        assert "spatial" not in table.uns

        for artifact in family:
            if artifact.exists():
                artifact.unlink()
        _assert_sparse_equal(expression, _COUNTS[:, (0, 2, 3)])

    def test_v3_gene_ids_and_all_feature_types(self, tmp_path: Path) -> None:
        _write_mtx_v3(tmp_path, prefix="all_", compressed=True)
        path = require_directory(tmp_path, context=_CONTEXT)

        table = read_10x_mtx(
            path,
            context=_CONTEXT,
            var_names="gene_ids",
            make_unique=False,
            gex_only=False,
            prefix="all_",
        )

        expression = _assert_canonical_sparse(table)
        _assert_sparse_equal(expression, _COUNTS)
        assert tuple(table.var_names) == _FEATURE_IDS
        assert tuple(table.var["gene_symbols"]) == _FEATURE_NAMES
        assert tuple(table.var["feature_types"]) == _FEATURE_TYPES

    def test_legacy_ignores_v3_compression_and_feature_filtering(self, tmp_path: Path) -> None:
        _write_mtx_legacy(tmp_path, prefix="legacy_")
        path = require_directory(tmp_path, context=_CONTEXT)

        table = read_10x_mtx(path, context=_CONTEXT, prefix="legacy_", compressed=True, gex_only=True)

        expression = _assert_canonical_sparse(table)
        _assert_sparse_equal(expression, _COUNTS)
        assert tuple(table.obs_names) == _BARCODES
        assert tuple(table.var_names) == ("duplicate", "protein", "duplicate-1", "empty-gene")
        assert tuple(table.var["gene_ids"]) == _FEATURE_IDS
        assert "feature_types" not in table.var

    def test_duplicate_matrix_market_coordinates_are_summed(self, tmp_path: Path) -> None:
        duplicated = sparse.coo_array(
            (
                np.asarray([2, 3], dtype=np.int32),
                (
                    np.asarray([0, 0], dtype=np.int32),
                    np.asarray([0, 0], dtype=np.int32),
                ),
            ),
            shape=(1, 1),
        )
        _write_mtx(tmp_path / "matrix.mtx", duplicated, compressed=False)
        _write_lines(tmp_path / "features.tsv", ("gene-id\tgene\tGene Expression",), compressed=False)
        _write_lines(tmp_path / "barcodes.tsv", ("barcode",), compressed=False)
        path = require_directory(tmp_path, context=_CONTEXT)

        table = read_10x_mtx(path, context=_CONTEXT, compressed=False)

        expression = _assert_canonical_sparse(table)
        assert expression.nnz == 1
        assert expression[0, 0] == 5

    def test_missing_companion_preserves_context(self, tmp_path: Path) -> None:
        _write_mtx_v3(
            tmp_path,
            prefix="broken_",
            compressed=True,
            include_features=False,
        )
        path = require_directory(tmp_path, context=_CONTEXT)

        with pytest.raises(FileNotFoundError, match=r"broken_features\.tsv\.gz") as caught:
            read_10x_mtx(path, context=_CONTEXT, prefix="broken_")

        assert caught.value.__notes__ == [f"While handling {_CONTEXT.describe(path=path)}."]
