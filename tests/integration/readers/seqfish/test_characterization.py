"""Freeze valid legacy seqFISH output independently of documented defects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io.readers.seqfish import seqfish
from tests.integration.readers.seqfish._fixtures import write_seqfish_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData


def _assert_expression(table: AnnData) -> None:
    assert isinstance(table.X, sparse.csr_array)
    assert not isinstance(table.X, sparse.spmatrix)
    assert_array_equal(table.X.data, [3, 5])
    assert_array_equal(table.X.indices, [0, 1])
    assert_array_equal(table.X.indptr, [0, 1, 2])
    assert table.X.dtype == np.dtype("int64")
    assert table.var_names.tolist() == ["GeneB", "GeneA"]
    assert table.obs_names.tolist() == ["0", "1"]
    assert_array_equal(table.obs["instance_id"], [9, 2])


class TestSeqfishCharacterization:
    """Keep the numerical format contract explicit before restructuring."""

    @pytest.mark.parametrize("with_z", [True, False])
    @pytest.mark.parametrize("with_cell", [True, False])
    @pytest.mark.parametrize("singleton_axes", [True, False])
    def test_complete_layout(self, tmp_path: Path, *, with_z: bool, with_cell: bool, singleton_axes: bool) -> None:
        root = write_seqfish_dataset(
            tmp_path,
            physical_size_y=0.5,
            with_z=with_z,
            with_cell=with_cell,
            singleton_axes=singleton_axes,
        )
        result = seqfish(root, rois=["Roi2", "Roi1"], shape_geometry="both")
        assert list(result.tables) == ["table_Roi2", "table_Roi1"]
        assert list(result.images) == ["Roi2_DAPI", "Roi1_DAPI"]
        assert list(result.labels) == ["Roi2_Segmentation", "Roi1_Segmentation"]
        assert list(result.points) == ["Roi2_TranscriptList", "Roi1_TranscriptList"]
        assert set(result.shapes) == {
            "Roi1_Boundaries",
            "Roi2_Boundaries",
            "Roi1_CellCoordinates",
            "Roi2_CellCoordinates",
        }
        for roi in ("Roi2", "Roi1"):
            table = result.tables[f"table_{roi}"]
            _assert_expression(table)
            assert_array_equal(table.obs["area"], [np.pi, 4 * np.pi])
            assert_array_equal(table.obsm["spatial"], [[1.5, 0.5], [4.5, 1.25]])
            assert get_table_keys(table) == (f"{roi}_Segmentation", "region", "instance_id")
            image = result.images[f"{roi}_DAPI"]
            labels = result.labels[f"{roi}_Segmentation"]
            assert image.dims == ("c", "y", "x")
            assert labels.dims == ("y", "x")
            assert image.dtype == np.dtype("uint16")
            assert labels.dtype == np.dtype("uint32")
            assert isinstance(image.data, da.Array)
            assert isinstance(labels.data, da.Array)
            assert_array_equal(image.data.compute(), np.arange(96, dtype=np.uint16).reshape(1, 8, 12))
            expected_labels = np.zeros((8, 12), dtype=np.uint32)
            expected_labels[1:3, 2:5] = 9
            expected_labels[4:7, 8:11] = 2
            assert_array_equal(labels.data.compute(), expected_labels)
            assert get_transformation(image, to_coordinate_system=roi) == Scale([0.5, 0.5], axes=("y", "x"))
            points = result.points[f"{roi}_TranscriptList"]
            materialized = points.compute()
            assert materialized["name"].tolist() == ["GeneA", "GeneB", "GeneA"]
            assert materialized["name"].cat.categories.tolist() == ["GeneB", "GeneA"]
            assert_array_equal(materialized[["x", "y"]], [[4.5, 1.25], [1.5, 0.5], [4.75, 1.5]])
            assert "z" not in materialized
            assert ("z_raw" in materialized) is with_z
            assert ("cell" in materialized) is with_cell
            if with_z:
                assert_array_equal(materialized["z_raw"], [1, 2, 1])
            if with_cell:
                assert_array_equal(materialized["cell"], [2, 9, 0])
            assert get_transformation(points, to_coordinate_system=roi) == Identity()
            boundaries = result.shapes[f"{roi}_Boundaries"]
            circles = result.shapes[f"{roi}_CellCoordinates"]
            assert boundaries.index.tolist() == [9, 2]
            assert circles.index.tolist() == [9, 2]
            assert_array_equal(boundaries.bounds, [[2, 1, 5, 3], [8, 4, 11, 7]])
            assert_array_equal(circles["radius"], [1, 2])
            assert_array_equal(circles.geometry.x, [1.5, 4.5])
            assert_array_equal(circles.geometry.y, [0.5, 1.25])
            assert get_transformation(circles, to_coordinate_system=roi) == Identity()

    @pytest.mark.parametrize("circles", [True, False])
    def test_excluded_products(self, tmp_path: Path, *, circles: bool) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        for suffix in ("Segmentation.tiff", "TranscriptList.csv", "Boundaries.geojson"):
            (root / f"Roi1_{suffix}").unlink()
        result = seqfish(
            root,
            include_images=False,
            include_labels=False,
            include_transcripts=False,
            include_shapes=circles,
            shape_geometry="circles",
        )
        assert list(result.tables) == ["table_Roi1"]
        assert not result.images
        assert not result.labels
        assert not result.points
        assert list(result.shapes) == (["Roi1_CellCoordinates"] if circles else [])

    def test_missing_requested_artifact(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        (root / "Roi1_TranscriptList.csv").unlink()
        with pytest.raises(FileNotFoundError, match=r"Roi1_TranscriptList\.csv"):
            seqfish(root)

    def test_unknown_roi(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path)
        with pytest.raises(ValueError, match="unknown ROIs"):
            seqfish(root, rois=["Roi3"])
