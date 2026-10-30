"""Physical positions preserve independent source populations and native storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import h5py
import numpy as np
import pytest
from anndata.tests.helpers import assert_equal
from dask.callbacks import Callback
from scipy.sparse import csr_array
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

from spatialdata_io import stereoseq
from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.stereoseq import _reader
from tests.integration.readers.stereoseq._fixtures import write_analysis, write_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from tests.integration.readers.stereoseq._fixtures import Family, Profile


class TestStereoseqPositions:
    @pytest.mark.parametrize("family", ["compact_numbered", "numbered", "outs", "flat"])
    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    def test_all_supported_sources(
        self, tmp_path: Path, mocker: MockerFixture, family: Family, profile: Profile
    ) -> None:
        fixture = write_dataset(
            tmp_path,
            family=family,
            profile=profile,
            products=("square", "cells", "adjusted_cells", "analysis", "adjusted_analysis", "registered", "mask"),
        )
        expression_guard = mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("densification"))
        tasks: list[object] = []
        with Callback(pretask=lambda *args: tasks.append(args)):
            result = stereoseq(
                fixture.root,
                sample_id=fixture.sample_id,
                bin_sizes=(1, 50),
                cell_bins="both",
                square_counts_file=fixture.square,
                cell_counts_file=fixture.cells,
                adjusted_cell_counts_file=fixture.adjusted_cells,
                analysis_files=(fixture.analysis, fixture.adjusted_analysis),
                image_roles=("registered", "mask"),
                scale_factors=(),
            )
        assert not tasks
        expression_guard.assert_not_called()
        for table in result.tables.values():
            assert isinstance(table.X, csr_array)
            declaration = cast("PositionsMetadata", table.uns[POSITIONS_KEY])["coordinates"]
            assert declaration["spatial"]["axes"] == ["x", "y"]
            assert declaration["spatial"]["units"] == "micrometer"
            assert declaration["spatial_grid"]["units"] == "grid"
            assert table.obsm["spatial"].dtype == np.float64
            np.testing.assert_array_equal(table.obsm["spatial"], table.obsm["spatial_grid"] * 0.5)
        np.testing.assert_array_equal(result.tables["cells_table"].obsm["spatial"], [[5, 10], [15, 20]])
        np.testing.assert_array_equal(result.tables["bin50_table"].obsm["spatial"], [[2.5, 3], [0.5, 1]])
        analysis_key = "cells_analysis_r0_5_table" if family == "outs" else "cells_analysis_table"
        assert result.tables[analysis_key].obsm["spatial_grid"].dtype == np.uint32
        table = result.tables["cells_table"]
        points = result.points["cells_centroids"]
        native = points.compute().loc[table.obs["instance_id"], ["x", "y"]].to_numpy()
        transform = get_transformation(points, to_coordinate_system="global")
        assert not isinstance(transform, dict)
        matrix = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_array_equal(
            table.obsm["spatial"], (np.column_stack((native, np.ones(len(native)))) @ matrix.T)[:, :2]
        )

    def test_nonzero_origin_is_not_added(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("cells", "square"))
        with h5py.File(fixture.square, "r+") as handle:
            handle["geneExp/bin50/expression"].attrs.update(minX=7, minY=9)
        with h5py.File(fixture.cells, "r+") as handle:
            handle.attrs.update(offsetX=7, offsetY=9)
        result = stereoseq(fixture.root, cell_bins="unadjusted", bin_sizes=(50,))
        np.testing.assert_array_equal(result.tables["cells_table"].obsm["spatial"], [[5, 10], [15, 20]])
        assert result.tables["cells_table"].uns["stereoseq"]["origin"] == [7, 9]

    @pytest.mark.parametrize("with_raw", [False, True])
    def test_reordered_filtered_analysis_keeps_source(self, tmp_path: Path, *, with_raw: bool) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("cells", "analysis"))
        write_analysis(fixture.analysis, reordered=True)
        source = ad.read_h5ad(fixture.analysis)[[0]].copy()
        source.raw = source.copy()
        source.layers["counts"] = csr_array([[1, 2]])
        source.write_h5ad(fixture.analysis)
        original_bytes = fixture.analysis.read_bytes()
        result = stereoseq(
            fixture.root, cell_bins="unadjusted" if with_raw else "none", analysis_files=(fixture.analysis,)
        )
        table = result.tables["cells_analysis_r0_5_table"]
        np.testing.assert_array_equal(table.obsm["spatial_grid"], [[30, 40]], strict=False)
        assert table.obsm["spatial_grid"].dtype == source.obsm["spatial"].dtype
        np.testing.assert_array_equal(table.obsm["spatial"], [[15, 20]])
        assert table.obs_names.equals(source.obs_names)
        assert_equal(table.raw, source.raw, exact=True)
        np.testing.assert_array_equal(table.obsm["X_pca"], source.obsm["X_pca"])
        assert fixture.analysis.read_bytes() == original_bytes

    def test_geometry_drop_preserves_full_analysis_population(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("cells", "analysis"), invalid_cell_ids=(200,))
        result = stereoseq(
            fixture.root,
            cell_bins="unadjusted",
            cell_geometry="polygons",
            invalid_geometries="drop",
            analysis_files=(fixture.analysis,),
        )
        assert result.tables["cells_table"].obs_names.tolist() == ["100"]
        np.testing.assert_array_equal(result.tables["cells_table"].obsm["spatial"], [[15, 20]])
        assert result.tables["cells_analysis_r0_5_table"].obs["id"].tolist() == [200, 100]

    def test_pitch_conflict_precedes_positions_and_linkage(self, tmp_path: Path, mocker: MockerFixture) -> None:
        fixture = write_dataset(tmp_path, products=("cells", "analysis"))
        source = ad.read_h5ad(fixture.analysis)
        source.uns["resolution"] = 600
        source.write_h5ad(fixture.analysis)
        positions = mocker.spy(_reader, "finalize_positions")
        linkage = mocker.spy(_reader, "link_table")
        with pytest.raises(ReaderFormatError, match="physical pitch"):
            stereoseq(fixture.root, cell_bins="unadjusted", analysis_files=(fixture.analysis,))
        positions.assert_not_called()
        linkage.assert_not_called()

    @pytest.mark.parametrize("collision", ["spatial_grid", POSITIONS_KEY])
    def test_imported_destination_collisions(self, tmp_path: Path, collision: str) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("analysis",))
        source = ad.read_h5ad(fixture.analysis)
        if collision == "spatial_grid":
            source.obsm[collision] = np.zeros((2, 2))
        else:
            source.uns[collision] = {"unrelated": 1}
        source.write_h5ad(fixture.analysis)
        with pytest.raises(ReaderFormatError, match="reserved") as error:
            stereoseq(fixture.root, analysis_files=(fixture.analysis,))
        assert str(fixture.analysis) in "".join(error.value.__notes__)

    @pytest.mark.parametrize("storage", ["h5ad", "anndata-zarr", "spatialdata-zarr"])
    def test_actual_reader_persistence(self, tmp_path: Path, storage: str) -> None:
        fixture = write_dataset(tmp_path / "source", family="outs", products=("analysis",))
        result = stereoseq(fixture.root, analysis_files=(fixture.analysis,))
        table = result.tables["cells_analysis_r0_5_table"]
        destination = tmp_path / storage
        if storage == "h5ad":
            table.write_h5ad(destination)
            restored = ad.read_h5ad(destination)
        elif storage == "anndata-zarr":
            table.write_zarr(destination)
            restored = ad.read_zarr(destination)
        else:
            result.write(destination)
            restored = read_zarr(destination).tables["cells_analysis_r0_5_table"]
        for key in ("spatial", "spatial_grid"):
            np.testing.assert_array_equal(restored.obsm[key], table.obsm[key], strict=True)
        coordinates = restored.uns[POSITIONS_KEY]["coordinates"]
        assert coordinates["spatial"]["frame"] == "global"
        assert coordinates["spatial"]["units"] == "micrometer"
        assert coordinates["spatial"]["position"] == "vendor_cell_center"
        assert table.obs_names.equals(restored.obs_names)
