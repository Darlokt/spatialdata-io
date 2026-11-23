"""On-disk 10x HDF5 layouts, standalone storage, and AnnData persistence."""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_h5_axes, read_10x_h5_components
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile, require_file
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

_GENOMES = ("GRCh38", "GRCh38", "mm10", "GRCh38")


def _write_h5_v3(path: Path) -> None:
    with h5py.File(path, "w") as file:
        file.attrs["library_ids"] = np.asarray([b"reader-owned-library"])
        file.attrs["protein_scaling_factor"] = 10.0
        matrix = file.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray(BARCODES, dtype="S"))
        matrix.create_dataset("data", data=COUNTS.data)
        matrix.create_dataset("indices", data=COUNTS.indices)
        matrix.create_dataset("indptr", data=COUNTS.indptr)
        matrix.create_dataset("shape", data=np.asarray([COUNTS.shape[1], COUNTS.shape[0]], dtype=np.int64))
        matrix.create_dataset("filtered_barcodes", data=np.asarray([1, 0, 1, 0], dtype=np.uint8))

        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray(FEATURE_IDS, dtype="S"))
        features.create_dataset("name", data=np.asarray(FEATURE_NAMES, dtype="S"))
        features.create_dataset("feature_type", data=np.asarray(FEATURE_TYPES, dtype="S"))
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


def _open_hdf5_sources() -> set[Path]:
    paths: set[Path] = set()
    for identifier in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE):
        name = h5py.h5f.get_name(identifier)
        if name is not None:
            paths.add(Path(os.fsdecode(name)).resolve())
    return paths


class TestRead10xH5Axes:
    def test_legacy_metadata_only_source(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "axes.h5")
        with h5py.File(path, "w") as source:
            group = source.create_group("GRCh38")
            group["gene_names"] = [b"001", b"NA"]
            group["barcodes"] = [b"007"]
        names, barcodes = read_10x_h5_axes(path, context=CONTEXT)
        assert names.tolist() == ["001", "NA"]
        assert barcodes.tolist() == ["007"]


class TestRead10xH5Components:
    def test_missing_source_preserves_io_error_and_adds_artifact_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "missing.h5")
        with pytest.raises(FileNotFoundError) as caught:
            read_10x_h5_components(path, context=CONTEXT)
        assert caught.value.__notes__ == [f"While handling {CONTEXT.describe(path=path)}."]

    def test_owned_sparse_buffers_and_aligned_annotations(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "source.h5")
        _write_h5_v3(path)
        counts, obs, var = read_10x_h5_components(path, context=CONTEXT)
        path.unlink()
        assert_sparse_equal(counts, COUNTS)
        assert counts.has_canonical_format
        assert tuple(obs.index) == BARCODES
        assert obs["filtered_barcodes"].tolist() == [True, False, True, False]
        assert tuple(var.index) == FEATURE_NAMES
        assert tuple(var["gene_ids"]) == FEATURE_IDS

    @pytest.mark.parametrize(
        ("key", "values", "expected"),
        [
            ("features", [1], "HDF5 group"),
            ("data", [[1]], "one-dimensional"),
            ("data", [1j], "real floating dtype"),
            ("shape", [4, 5], "sparse buffer lengths"),
            ("indices", [1.0], "must be integers"),
            ("indices", [0, 1, 1, 2, 0, 4], "outside its shape"),
            ("indices", [1], "sparse buffer lengths"),
            ("indptr", [0, 2, 4, 5, 5], "span exactly"),
            ("indptr", [0, 4, 3, 6, 6], "nondecreasing"),
            ("indptr", [1, 2, 4, 6, 6], "from zero"),
            ("features/name", [], "matrix shape"),
            ("barcodes", [], "matrix shape"),
            ("barcodes", [1, 2, 3, 4], "HDF5 strings"),
            ("shape", [[4, 4]], "one-dimensional"),
            ("shape", [4], "integer feature and barcode counts"),
            ("filtered_barcodes", [0, 2, 1, 0], "boolean flags"),
            ("features/gene_ids", [0, 1, 2, 3], "conflicts"),
        ],
    )
    def test_malformed_matrix_datasets(
        self, tmp_path: Path, key: str, values: list[int] | list[float] | list[complex] | list[list[int]], expected: str
    ) -> None:
        path = ArtifactFile(tmp_path / "matrix.h5")
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            matrix = handle["matrix"]
            if key in matrix:
                del matrix[key]
            matrix.create_dataset(key, data=values)
        with pytest.raises(ReaderFormatError, match=expected) as caught:
            read_10x_h5_components(path, context=CONTEXT)
        assert caught.value.path == path

    @pytest.mark.parametrize("storage", ["external_link", "soft_link", "external_data", "virtual"])
    def test_only_reads_standalone_datasets(self, tmp_path: Path, storage: str) -> None:
        path = ArtifactFile(tmp_path / "matrix.h5")
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            matrix = handle["matrix"]
            del matrix["data"]
            if storage == "external_link":
                matrix["data"] = h5py.ExternalLink("absent.h5", "/data")
            elif storage == "soft_link":
                matrix["data"] = h5py.SoftLink("/matrix/indices")
            elif storage == "external_data":
                matrix.create_dataset("data", shape=(6,), dtype="i4", external=[("absent.bin", 0, 24)])
            else:
                layout = h5py.VirtualLayout(shape=(6,), dtype="i4")
                layout[:] = h5py.VirtualSource("absent.h5", "/data", shape=(6,))
                matrix.create_virtual_dataset("data", layout)
        with pytest.raises(ReaderFormatError, match=r"(within the count file|stored in this HDF5 file)") as caught:
            read_10x_h5_components(path, context=CONTEXT)
        assert caught.value.path == path
        assert path.resolve() not in _open_hdf5_sources()

    def test_rejects_non_scalar_feature_values(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "matrix.h5")
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            handle["matrix/features/targets"] = np.zeros(4, dtype=[("code", "i4")])
        with pytest.raises(ReaderFormatError, match="numeric feature annotations"):
            read_10x_h5_components(path, context=CONTEXT)

    def test_nested_names_do_not_override_matrix_or_feature_fields(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "matrix.h5")
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            handle["matrix/features/target_sets/name"] = [b"unrelated"]
            handle["matrix/features/target_sets/data"] = [99]
        counts, _, var = read_10x_h5_components(path, context=CONTEXT)
        assert tuple(var.index) == FEATURE_NAMES
        assert_sparse_equal(counts, COUNTS)


class TestRead10xH5:
    @pytest.mark.parametrize("kind", ["h5ad", "zarr"])
    def test_big_endian_counts_and_utf8_survive_persistence(self, tmp_path: Path, kind: str) -> None:
        path = ArtifactFile(tmp_path / "matrix.h5")
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            del handle["matrix/data"], handle["matrix/barcodes"]
            handle["matrix/data"] = np.array([2**53 - 1, 2**24 + 1, 3, 4, 5, 6], dtype=">u8")
            handle["matrix/barcodes"] = np.array(["007", "NA", "样本", "NULL"], dtype=h5py.string_dtype("utf-8"))
        table = read_10x_h5(path, context=CONTEXT)
        dest = tmp_path / f"table.{kind}"
        if kind == "h5ad":
            table.write_h5ad(dest)
            restored = ad.read_h5ad(dest)
        else:
            table.write_zarr(dest)
            restored = ad.read_zarr(dest)
        assert restored.obs_names.tolist() == ["007", "NA", "样本", "NULL"]
        assert tuple(restored.var_names) == FEATURE_NAMES
        values = restored.X
        assert isinstance(values, sparse.csr_array | sparse.csr_matrix)
        assert int(values[0, 0]) == 2**53 - 1
        assert int(values[0, 1]) == 2**24 + 1

    def test_probe_matrix_feature_ids(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "probes.h5")
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            handle.create_dataset("matrix/features/gene_id", data=np.array([b"007", b"002", b"004", b"003"]))
        table = read_10x_h5(path, context=CONTEXT)
        assert table.var["probe_ids"].tolist() == list(FEATURE_IDS)
        assert table.var["gene_ids"].tolist() == ["007", "002", "004", "003"]

    def test_exact_int32_source(self, tmp_path: Path) -> None:
        path = tmp_path / "counts.h5"
        _write_h5_v3(path)
        with h5py.File(path, "r+") as handle:
            handle["matrix/data"][0] = 2**24 + 1
        table = read_10x_h5(require_file(path, context=CONTEXT), context=CONTEXT)
        assert int(assert_canonical_sparse(table)[0, 0]) == 2**24 + 1

    def test_rejects_repeated_barcode_feature_entries(self, tmp_path: Path) -> None:
        """The HDF5 format stores each (barcode, feature) entry at most once; repeats are never summed."""
        source = tmp_path / "duplicates.h5"
        _write_h5_v3(source)
        with h5py.File(source, "r+") as handle:
            handle["matrix/indices"][1] = 0

        with pytest.raises(ReaderFormatError, match=r"repeated \(barcode, feature\) entries"):
            read_10x_h5(require_file(source, context=CONTEXT), context=CONTEXT)
        assert source.resolve() not in _open_hdf5_sources()

    @pytest.mark.parametrize("value", [-1.0, 0.5])
    def test_rejects_negative_and_fractional_counts(self, tmp_path: Path, value: float) -> None:
        source = tmp_path / "matrix.h5"
        _write_h5_v3(source)
        with h5py.File(source, "r+") as handle:
            values = COUNTS.data.astype(np.float64)
            values[0] = value
            del handle["matrix/data"]
            handle.create_dataset("matrix/data", data=values)
        with pytest.raises(ReaderFormatError, match=r"negative|fractional"):
            read_10x_h5(require_file(source, context=CONTEXT), context=CONTEXT)

    def test_v3_preserves_orientation_order_metadata_and_every_feature_type(self, tmp_path: Path) -> None:
        source = tmp_path / "matrix.h5"
        _write_h5_v3(source)
        path = require_file(source, context=CONTEXT)

        table = read_10x_h5(path, context=CONTEXT)

        expression = assert_canonical_sparse(table)
        assert_sparse_equal(expression, COUNTS)
        assert expression.dtype == np.dtype(np.float32)
        assert tuple(table.obs_names) == BARCODES
        assert tuple(table.var_names) == FEATURE_NAMES
        assert tuple(table.var["gene_ids"]) == FEATURE_IDS
        assert tuple(table.var["feature_types"]) == FEATURE_TYPES
        assert tuple(table.var["genome"]) == _GENOMES
        assert tuple(table.var["target_set"]) == ("A", "protein", "B", "empty")
        assert "spatial" not in table.uns
        assert source.resolve() not in _open_hdf5_sources()

    def test_v3_result_is_independent_from_removed_source(self, tmp_path: Path) -> None:
        source = tmp_path / "matrix.h5"
        _write_h5_v3(source)
        path = require_file(source, context=CONTEXT)

        table = read_10x_h5(path, context=CONTEXT)
        source.unlink()

        expression = assert_canonical_sparse(table)
        assert_sparse_equal(expression, COUNTS)
        assert tuple(table.obs_names) == BARCODES

    def test_legacy_single_genome_is_selected_implicitly(self, tmp_path: Path) -> None:
        source = tmp_path / "legacy-single.h5"
        _write_h5_legacy(source, genomes=("GRCh38",))
        path = require_file(source, context=CONTEXT)

        table = read_10x_h5(path, context=CONTEXT)

        expression = assert_canonical_sparse(table)
        expected = sparse.csr_array(np.asarray([[1, 2], [3, 0]], dtype=np.int64))
        assert_sparse_equal(expression, expected)
        assert expression.dtype == np.dtype(np.float32)
        assert tuple(table.obs_names) == ("legacy-b", "legacy-a")
        assert tuple(table.var_names) == ("legacy-z", "legacy-a")
        assert source.resolve() not in _open_hdf5_sources()

    def test_legacy_multiple_genomes_are_rejected_and_close_source(self, tmp_path: Path) -> None:
        source = tmp_path / "legacy-multiple.h5"
        _write_h5_legacy(source, genomes=("GRCh38", "mm10"))
        path = require_file(source, context=CONTEXT)

        with pytest.raises(ReaderFormatError, match="contains more than one genome") as caught:
            read_10x_h5(path, context=CONTEXT)

        assert caught.value.path == path
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
        path = require_file(source, context=CONTEXT)

        with pytest.raises(ReaderFormatError, match="10x h5 has no features group") as caught:
            read_10x_h5(path, context=CONTEXT)

        assert caught.value.path == path
        assert source.resolve() not in _open_hdf5_sources()
