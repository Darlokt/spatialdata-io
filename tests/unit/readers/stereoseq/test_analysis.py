"""Derived analysis stays distinct and aligns through native source identity."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array, csr_matrix

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq._analysis import read_analysis
from spatialdata_io.readers.stereoseq._cells import CellCounts
from spatialdata_io.readers.stereoseq._layout import AnalysisSource
from spatialdata_io.readers.stereoseq._square import SquareCounts

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _analysis(path: Path, *, cell_ids: bool = True) -> ad.AnnData:
    table = ad.AnnData(
        X=csr_array([[2.5, 0], [1.25, 4.75]]),
        obs=pd.DataFrame(index=[str((30 << 32) | 40), str((10 << 32) | 20)]),
        var=pd.DataFrame({"real_gene_name": ["duplicate", "duplicate"]}, index=["g_a", "g_z"]),
    )
    if cell_ids:
        table.obs["id"] = np.array([100, 200], dtype=np.uint32)
    table.obs["leiden"] = pd.Categorical(["2", "1"])
    table.obsm["spatial"] = np.array([[30, 40], [10, 20]], dtype=np.uint32)
    table.obsm["X_umap"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    table.uns.update(bin_type="cell_bins", bin_size=1, resolution=500)
    table.uns["sn"] = pd.DataFrame({"batch": ["-1"], "sn": ["sample"]})
    table.layers["metadata"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    table.write_h5ad(path)
    return table


def _counts(*, square: bool = False) -> CellCounts | SquareCounts:
    table = ad.AnnData(
        X=csr_array([[11, 0], [0, 22], [1, 1]]),
        obs=pd.DataFrame(index=["0", "1", "2"] if square else ["200", "100", "300"]),
        var=pd.DataFrame(index=["g_z", "g_a"]),
    )
    coordinates = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)
    table.obsm["spatial"] = coordinates
    table.uns["stereoseq"] = {"origin": [5, 6], "resolution": 500}
    if square:
        return SquareCounts(ArtifactFile(Path("sample.gef")), table, coordinates, 50, (5, 6), 500, "sample", "gef-2")
    return CellCounts(
        table, coordinates, np.array([200, 100, 300], dtype=np.uint64), None, (5, 6), 500, "sample", "gef-2"
    )


def _source(path: Path, *, resolution: str | None = "0_50") -> AnalysisSource:
    return AnalysisSource(ArtifactFile(path), "cells", resolution, "sample")


class TestReadAnalysis:
    """Coordinate IDs and optional cell IDs establish membership without row replacement."""

    @pytest.mark.parametrize("cell_ids", [True, False])
    def test_filtered_cells_with_coincident_raw_centroids(self, tmp_path: Path, *, cell_ids: bool) -> None:
        path = tmp_path / "sample.cellbin_0.50.h5ad"
        table = _analysis(path, cell_ids=cell_ids)[:1].copy()
        table.write_h5ad(path)
        raw = _counts()
        raw.coordinates[:] = [[30, 40], [30, 40], [50, 60]]
        if cell_ids:
            result = read_analysis(_source(path), counts={"cells": raw})
            assert result.table.obs["id"].tolist() == [100]
            np.testing.assert_array_equal(result.coordinates, [[30, 40]])
        else:
            with pytest.raises(ReaderFormatError, match="ambiguous source centroids"):
                read_analysis(_source(path), counts={"cells": raw})

    def test_unreadable_h5ad_names_the_artifact(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cellbin_0.50.h5ad"
        path.write_bytes(b"not an HDF5 file")

        with pytest.raises(OSError, match="Unable to") as caught:
            read_analysis(_source(path), counts={})
        assert caught.value.__notes__ == [
            f"While handling reader 'stereoseq', artifact 'analysis table', path '{path}'."
        ]

    def test_square_id_annotation_is_not_a_semantic_cell_key(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.bin50_0.50.h5ad"
        table = _analysis(path)
        table.uns.update(bin_type="bins", bin_size=50)
        table.write_h5ad(path)
        raw = _counts(square=True)
        source = AnalysisSource(ArtifactFile(path), 50, "0_50", "sample")
        result = read_analysis(source, counts={"bin50": raw})
        np.testing.assert_array_equal(result.coordinates, [[30, 40], [10, 20]])

    def test_missing_semantic_cell_id_fails_before_coordinate_lookup(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cellbin_0.50.h5ad"
        table = _analysis(path)
        table.obs["id"] = np.array([999, 200], dtype=np.uint32)
        table.write_h5ad(path)
        with pytest.raises(ReaderFormatError, match="cell-ID membership"):
            read_analysis(_source(path), counts={"cells": _counts()})

    @pytest.mark.parametrize("cell_ids", [True, False])
    def test_matches_reordered_filtered_native_products(self, tmp_path: Path, *, cell_ids: bool) -> None:
        path = tmp_path / "sample.cellbin_0.50.h5ad"
        original = _analysis(path, cell_ids=cell_ids)
        raw = _counts()
        result = read_analysis(_source(path), counts={"cells": raw})
        assert result.name == "cells_analysis_r0_50"
        assert result.resolution == 500
        assert result.table.obs_names.tolist() == original.obs_names.tolist()
        assert result.table.var_names.tolist() == ["g_a", "g_z"]
        assert result.instance_ids.tolist() == original.obs_names.tolist()
        assert isinstance(result.table.X, csr_matrix)
        np.testing.assert_array_equal(result.table.X.data, [2.5, 1.25, 4.75])
        np.testing.assert_array_equal(result.coordinates, [[30, 40], [10, 20]])
        np.testing.assert_array_equal(result.table.obsm["X_umap"], original.obsm["X_umap"])
        np.testing.assert_array_equal(result.table.layers["metadata"], original.layers["metadata"])
        assert raw.table.obs_names.tolist() == ["200", "100", "300"]
        np.testing.assert_array_equal(raw.coordinates, [[10, 20], [30, 40], [50, 60]])
        assert isinstance(raw.table.X, csr_array)
        np.testing.assert_array_equal(raw.table.X.data, [11, 22, 1, 1])

    def test_analysis_only_historical_unknown_clustering_resolution(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cell.cluster.h5ad"
        _analysis(path)
        result = read_analysis(_source(path, resolution=None), counts={})
        assert result.name == "cells_analysis"
        assert result.resolution == 500

    def test_resolves_historical_square_population_from_native_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.spatial.cluster.h5ad"
        table = _analysis(path, cell_ids=False)
        table.uns.update(bin_type="bins", bin_size=50)
        table.write_h5ad(path)
        source = AnalysisSource(ArtifactFile(path), None, None, "sample")
        result = read_analysis(source, counts={})
        assert result.name == "bin50_analysis"

    def test_explicit_square_hint_accepts_missing_optional_population_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.bin50_1.h5ad"
        table = _analysis(path, cell_ids=False)
        del table.uns["bin_type"]
        del table.uns["bin_size"]
        table.write_h5ad(path)
        assert (
            read_analysis(AnalysisSource(ArtifactFile(path), 50, "1", "sample"), counts={}).name == "bin50_analysis_r1"
        )

    def test_matching_gef_supplies_missing_physical_pitch(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cellbin_1.h5ad"
        table = _analysis(path)
        del table.uns["resolution"]
        table.write_h5ad(path)
        assert read_analysis(_source(path), counts={"cells": _counts()}).resolution == 500

    def test_reindexed_source_names_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cellbin_1.h5ad"
        table = _analysis(path)
        table.obs_names = ["sample:30_40", "sample:10_20"]
        table.write_h5ad(path)
        assert (
            read_analysis(_source(path), counts={"cells": _counts()}).instance_ids.tolist() == table.obs_names.tolist()
        )

    def test_historical_analysis_without_semantic_ids_requires_coordinate_membership(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cell.cluster.h5ad"
        _analysis(path, cell_ids=False)
        raw = _counts()
        raw.coordinates[:] = [[11, 20], [30, 40], [50, 60]]
        with pytest.raises(ReaderFormatError, match="analysis coordinates"):
            read_analysis(_source(path, resolution=None), counts={"cells": raw})

    @pytest.mark.parametrize("mismatch", ["sample", "coordinates", "cell_id", "features", "pitch", "ambiguous"])
    def test_rejects_incompatible_raw_or_analysis_identity(self, tmp_path: Path, mismatch: str) -> None:
        path = tmp_path / "sample.cellbin_1.h5ad"
        table = _analysis(path)
        raw = _counts()
        if mismatch == "sample":
            table.uns["sn"] = pd.DataFrame({"sn": ["another"]})
        elif mismatch == "coordinates":
            raw.coordinates[0, 0] = 11
        elif mismatch == "cell_id":
            table.obs["id"] = np.array([200, 100])
        elif mismatch == "features":
            table.var_names = ["g_x", "g_z"]
        elif mismatch == "pitch":
            table.uns["resolution"] = 600
        else:
            raw.coordinates[:] = [[30, 40], [30, 40], [50, 60]]
        table.write_h5ad(path)
        with pytest.raises(ReaderFormatError):
            read_analysis(_source(path), counts={"cells": raw})

    @pytest.mark.parametrize(
        "case", ["unknown_population", "cell_as_square", "square_as_cell", "bin_size", "missing_pitch"]
    )
    def test_rejects_unresolved_or_conflicting_format_metadata(self, tmp_path: Path, case: str) -> None:
        path = tmp_path / "sample.h5ad"
        table = _analysis(path)
        source = _source(path)
        if case == "unknown_population":
            source = AnalysisSource(ArtifactFile(path), None, None, "sample")
        elif case == "cell_as_square":
            source = AnalysisSource(ArtifactFile(path), 50, None, "sample")
        elif case == "square_as_cell":
            table.uns.update(bin_type="bins", bin_size=50)
        elif case == "bin_size":
            source = AnalysisSource(ArtifactFile(path), 50, None, "sample")
            table.uns.update(bin_type="bins", bin_size=100)
        else:
            del table.uns["resolution"]
        table.write_h5ad(path)
        with pytest.raises(ReaderFormatError):
            read_analysis(source, counts={})

    def test_reader_owns_loaded_table_without_copying_caller_state(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "sample.cellbin_1.h5ad"
        loaded = _analysis(path)
        mocker.patch("spatialdata_io.readers.stereoseq._analysis.ad.read_h5ad", return_value=loaded)
        assert read_analysis(_source(path), counts={}).table is loaded

    def test_uns_keys_are_sanitized_with_one_warning(self, tmp_path: Path) -> None:
        """Rename invalid and reader-colliding top-level keys only, keeping values and stored order."""
        path = tmp_path / "sample.cellbin_0.50.h5ad"
        table = _analysis(path)
        table.uns["my key"] = {"inner key": "kept"}
        table.uns["Resolution"] = "clustering"
        table.uns["Spatialdata_io_positions"] = np.array([1, 2])
        table.write_h5ad(path)

        with pytest.warns(UserWarning, match="uns keys") as caught:
            result = read_analysis(_source(path), counts={"cells": _counts()})

        assert [str(warning.message) for warning in caught] == [
            (
                "stereoseq: the uns of sample.cellbin_0.50.h5ad has names that a SpatialData table does not accept as "
                "uns keys; using 'Resolution' -> 'Resolution_1', 'Spatialdata_io_positions' -> "
                "'Spatialdata_io_positions_1', 'my key' -> 'my_key'. Rename them in the source to choose the names "
                "explicitly."
            )
        ]
        # HDF5 stores group members by name, so the stored order is alphabetical.
        uns = result.table.uns
        assert list(uns) == [
            "Resolution_1",
            "Spatialdata_io_positions_1",
            "bin_size",
            "bin_type",
            "my_key",
            "resolution",
            "sn",
        ]
        assert uns["my_key"] == {"inner key": "kept"}
        assert uns["Resolution_1"] == "clustering"
        np.testing.assert_array_equal(uns["Spatialdata_io_positions_1"], [1, 2])
        assert (uns["bin_type"], uns["bin_size"], uns["resolution"]) == ("cell_bins", 1, 500)
        assert uns["sn"]["sn"].tolist() == ["sample"]
        assert result.resolution == 500

    def test_valid_uns_keys_are_untouched_without_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.cellbin_0.50.h5ad"
        _analysis(path)
        stored = ad.read_h5ad(path).uns

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = read_analysis(_source(path), counts={"cells": _counts()})

        assert list(result.table.uns) == list(stored)
        pd.testing.assert_frame_equal(result.table.uns["sn"], stored["sn"])
