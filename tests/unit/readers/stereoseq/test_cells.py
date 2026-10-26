"""Native cell CSR, identity, exon alignment and resource contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, sparray, spmatrix

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq import _cells, _schemas
from spatialdata_io.readers.stereoseq._cells import read_cells
from spatialdata_io.readers.stereoseq._layout import CellSource
from tests.integration.readers.stereoseq._fixtures import write_cell_gef

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadCells:
    """Check semantic IDs independently of row position and alternate traversal."""

    def test_unsupported_modality_fails_before_cell_payload_discovery(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "protein.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["omics"] = "Proteomics"
        with pytest.raises(ReaderFormatError, match="omics"):
            read_cells(CellSource(path, "unadjusted"), read_borders=False)

    def test_native_transcriptomic_modality(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            handle.attrs["omics"] = np.array([b"Transcriptomics"], dtype="S32")

        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [11, 22])

    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    @pytest.mark.parametrize("read_borders", [False, True])
    def test_native_counts_metadata_and_layers(
        self, tmp_path: Path, profile: Literal["name_only", "id_and_name"], *, read_borders: bool
    ) -> None:
        path = write_cell_gef(tmp_path / "s.gef", profile=profile)
        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=read_borders)
        table = result.table
        assert isinstance(table.X, sparray)
        assert not isinstance(table.X, spmatrix)
        assert isinstance(table.X, csr_array)
        assert table.X.dtype == np.dtype(np.uint64)
        assert table.X.has_canonical_format
        assert table.obs_names.tolist() == ["200", "100"]
        assert table.var_names.tolist() == (["z", "a"] if profile == "name_only" else ["g_z", "g_a"])
        assert table.obs["cellExon"].tolist() == [5, 6]
        assert table.var["geneExon"].tolist() == [5, 6]
        assert table.uns["stereoseq"]["cell_type_names"] == ["unknown", "cell"]
        assert "cellBin_cell_gene_exon_exp" not in table.uns
        assert_array_equal(result.cell_ids, [200, 100])
        assert_array_equal(result.coordinates, [[10, 20], [30, 40]])
        assert_array_equal(table.X.data, [11, 22])
        assert_array_equal(table.layers["exon"].data, [5, 6])
        assert (result.borders is not None) is read_borders
        assert result.origin == (0, 0)
        assert result.resolution == 500

    def test_unsorted_features_keep_exon_alignment_and_uint64_counts(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            group = handle["cellBin"]
            cells = group["cell"][:]
            cells["offset"] = [0, 2]
            cells["geneCount"] = [2, 1]
            group["cell"][:] = cells
            genes = group["gene"][:]
            genes["offset"] = [0, 1]
            genes["cellCount"] = [1, 2]
            group["gene"][:] = genes
            del group["cellExp"]
            group["cellExp"] = np.array([(1, 2**64 - 1), (0, 12), (1, 22)], dtype=[("geneID", "u4"), ("count", "u8")])
            del group["cellExpExon"]
            group["cellExpExon"] = np.array([7, 4, 6], dtype="u8")

        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [12, 2**64 - 1, 22])
        assert_array_equal(result.table.layers["exon"].data, [4, 7, 6])
        assert isinstance(result.table.X, csr_array)
        assert result.table.X.has_canonical_format

    @pytest.mark.parametrize(
        "failure", ["duplicate_ids", "bad_span", "trailing_records", "feature_bounds", "duplicates"]
    )
    def test_malformed_native_records(self, tmp_path: Path, failure: str) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            group = handle["cellBin"]
            cells = group["cell"][:]
            expression = group["cellExp"][:]
            if failure == "duplicate_ids":
                cells["id"] = 100
            elif failure == "bad_span":
                cells["offset"][1] = 0
            elif failure == "trailing_records":
                cells["geneCount"][1] = 0
            elif failure == "feature_bounds":
                expression["geneID"][0] = 2
            else:
                cells["offset"] = [0, 2]
                cells["geneCount"] = [2, 0]
                expression["geneID"] = 0
            group["cell"][:] = cells
            group["cellExp"][:] = expression

        with pytest.raises(ReaderFormatError):
            read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)

    def test_centroids_ignore_unselected_borders_and_optional_exons(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            group = handle["cellBin"]
            for name in ("cellBorder", "cellExon", "geneExon", "cellExpExon", "cellTypeList", "geneExp", "geneExpExon"):
                del group[name]
            group["cellBorder"] = np.ones(1)

        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert result.borders is None
        assert "exon" not in result.table.layers
        with pytest.raises(ReaderFormatError, match="cellBorder"):
            read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=True)

    def test_empty_cells_and_features(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            group = handle["cellBin"]
            for name in ("cell", "gene", "cellExp", "cellExon", "geneExon", "cellExpExon"):
                dtype = group[name].dtype
                del group[name]
                group[name] = np.empty(0, dtype=dtype)

        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert result.table.shape == (0, 0)
        assert result.coordinates.shape == (0, 2)

    def test_empty_cell_and_zero_feature_alternate_offset(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            group = handle["cellBin"]
            cells = group["cell"][:]
            cells["offset"] = [0, 0]
            cells["geneCount"] = [0, 2]
            group["cell"][:] = cells
            # Native cell expression is independent of unused geneExp spans;
            # real files store zero for offsets of unexpressed features.
            genes = group["gene"][:]
            genes["offset"] = [0, 0]
            genes["cellCount"] = [0, 1]
            group["gene"][:] = genes

        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.indptr, [0, 0, 2])
        assert_array_equal(result.table.X.data, [11, 22])

    def test_semantic_uint64_ids_are_not_narrowed(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            original = handle["cellBin/cell"][:]
            names = original.dtype.names
            assert names is not None
            fields = [(name, "u8" if name == "id" else original.dtype[name]) for name in names]
            cells = np.empty(2, dtype=fields)
            for name in names:
                cells[name] = original[name]
            cells["id"] = [2**64 - 1, 100]
            del handle["cellBin/cell"]
            handle["cellBin/cell"] = cells

        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert result.table.obs_names.tolist() == [str(2**64 - 1), "100"]
        assert result.cell_ids.dtype == np.dtype(np.uint64)

    def test_bounded_reads_never_densify(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        mocker.patch.object(_cells, "_EXPRESSION_BUFFER_BYTES", 4)
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("densified"))
        getitem = h5py.Dataset.__getitem__
        sizes: list[int] = []

        def tracked(dataset: h5py.Dataset, key: object) -> object:
            if dataset.name.endswith("/cellExp"):
                assert isinstance(key, slice)
                sizes.append(key.stop - key.start)
            return getitem(dataset, key)

        mocker.patch.object(h5py.Dataset, "__getitem__", tracked)
        result = read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert sizes == [1, 1]
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [11, 22])

    def test_resource_closure_and_io_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        handle = h5py.File(path, "r")
        file_class = h5py.File
        mocker.patch.object(h5py, "File", return_value=handle)
        mocker.patch.object(_schemas, "cell_schema", side_effect=RuntimeError("injected"))

        with pytest.raises(RuntimeError, match="injected"):
            read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
        assert not handle.id.valid
        mocker.patch.object(h5py, "File", file_class)
        with pytest.raises(OSError, match="No such file") as caught:
            read_cells(CellSource(ArtifactFile(tmp_path / "missing.gef"), "unadjusted"), read_borders=False)
        assert "stereoseq" in caught.value.__notes__[0]

    def test_category_text_dtype_rejected(self, tmp_path: Path) -> None:
        path = write_cell_gef(tmp_path / "s.gef")
        with h5py.File(path, "a") as handle:
            del handle["cellBin/cellTypeList"]
            handle["cellBin/cellTypeList"] = np.ones(2)

        with pytest.raises(ReaderFormatError, match="category text"):
            read_cells(CellSource(ArtifactFile(path), "unadjusted"), read_borders=False)
