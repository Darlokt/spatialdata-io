"""Public CosMx workflows: profiles, landmark geometry, laziness, and round trips."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
from click.testing import CliRunner
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray

from spatialdata_io import __version__
from spatialdata_io._cli.commands.cosmx import cosmx_command
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.cosmx import cosmx
from tests.integration.readers.cosmx._fixtures import LANDMARKS, POSITIONS, global_xy, local_xy, write_cosmx_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame as DaskDataFrame
    from spatialdata import SpatialData
    from xarray import DataTree

_AXES = ("x", "y")


def _expression(table: AnnData) -> sparse.csr_array:
    """Narrow AnnData's broad expression union to the reader's sparse contract."""
    assert isinstance(table.X, sparse.csr_array)
    return table.X


def _transform(element: DataArray | DataTree | DaskDataFrame, frame: str) -> np.ndarray:
    transform = get_transformation(element, frame)
    assert isinstance(transform, BaseTransformation)
    return transform.to_affine_matrix(input_axes=_AXES, output_axes=_AXES)


def _labels(result: SpatialData, fov: int) -> DataArray:
    labels = result.labels[f"{fov}_labels"]
    assert isinstance(labels, DataArray)
    return labels


class TestCosmx:
    """Assemble supported exports through the public reader."""

    @pytest.mark.parametrize("compressed", [False, True])
    def test_flat_2021_public_contract(self, tmp_path: Path, *, compressed: bool) -> None:
        fixture = write_cosmx_dataset(tmp_path, compressed=compressed)
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            result = cosmx(fixture.root)
        assert not tasks
        assert list(result.images) == ["2_image", "10_image"]
        assert list(result.labels) == ["2_labels", "10_labels"]
        assert list(result.points) == ["Run_1_transcripts"]
        assert result.attrs == {"spatialdata_io_reader": "cosmx", "spatialdata_io_software_version": __version__}

        table = result.tables["table"]
        np.testing.assert_array_equal(
            _expression(table).toarray(), [[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 0], [0, 0, 7, 1]]
        )
        assert table.var_names.tolist() == list(fixture.features)
        assert table.obs_names.tolist() == ["0", "1", "2", "3"]
        assert table.obs["cell_ID"].tolist() == [1, 2, 1, 2]
        assert table.obs["note"].tolist() == ["01", "02", "01", "02"]
        assert np.isnan(table.obs["Mean.DAPI"].iloc[3])
        assert get_table_keys(table) == (["2_labels", "10_labels"], "region", "cell_ID")

        points = result.points["Run_1_transcripts"].compute()
        assert len(points) == 6
        assert points["z_raw"].tolist() == [0, -1, 2] * 2
        assert points["cell_ID"].tolist() == [1, 0, 2] * 2

    def test_landmark_pixels_map_to_their_evidenced_local_and_global_coordinates(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root)
        for fov in POSITIONS:
            labels = _labels(result, fov)
            to_local, to_global = _transform(labels, str(fov)), _transform(labels, "global")
            pixels = labels.data.compute()
            for cell in LANDMARKS:
                row, column = np.argwhere(pixels == cell)[0]
                center = np.array([column + 0.5, row + 0.5, 1.0])
                np.testing.assert_array_equal((to_local @ center)[:2], local_xy(cell))
                np.testing.assert_array_equal((to_global @ center)[:2], global_xy(fov, cell, "flat_2021"))
            # The table's own coordinates agree with the raster geometry.
            rows = (table := result.tables["table"]).obs["fov"].to_numpy() == fov
            np.testing.assert_array_equal(table.obsm["spatial"][rows], [local_xy(cell) for cell in LANDMARKS])

    def test_points_share_every_fov_frame_with_rasters(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root)
        points = result.points["Run_1_transcripts"]
        for fov in POSITIONS:
            frame = points.compute()
            selected = frame[frame["fov"] == fov]
            xy = np.column_stack([selected["x"], selected["y"], np.ones(len(selected))])
            local = (_transform(points, str(fov)) @ xy.T).T[:, :2]
            np.testing.assert_array_equal(local, selected[["x_local", "y_local"]].to_numpy())

    def test_atomx_profile_reversed_axis_vendor_ids_and_unlinked_table(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, profile="atomx", compressed=True, rasters=False)
        result = cosmx(fixture.root)
        table = result.tables["table"]
        assert table.obs_names.tolist() == ["c_1_2_1", "c_1_2_2", "c_1_10_1", "c_1_10_2"]
        assert "cell_id" not in table.obs
        assert "spatialdata_attrs" not in table.uns
        assert table.var_names.tolist() == list(fixture.features)
        points = result.points["Run_1_transcripts"]
        np.testing.assert_array_equal(_transform(points, "2"), [[1, 0, -1000], [0, -1, 5000.5], [0, 0, 1]])

    def test_rasters_in_a_reversed_axis_export_fail(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, profile="atomx")
        with pytest.raises(ReaderFormatError, match="global axes are reversed"):
            cosmx(fixture.root)

    def test_labels_must_cover_every_fov_with_cells(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        (fixture.root / "CellLabels" / "CellLabels_F010.tif").unlink()
        with pytest.raises(ReaderFormatError, match=r"no labels for FOVs \[10\]"):
            cosmx(fixture.root)

    def test_without_rasters_or_transcripts(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, rasters=False, positions=False)
        result = cosmx(fixture.root, include_transcripts=False)
        assert not result.images
        assert not result.points
        assert "spatialdata_attrs" not in result.tables["table"].uns

    def test_include_transcripts_must_be_boolean(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="boolean"):
            cosmx(tmp_path, include_transcripts="yes")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_graphs_serialize_and_pyramids_stay_lazy(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root, scale_factors=(2,), chunks=(3, 4, 4))
        data = result.images["2_image"]["scale0"]["image"].data
        restored = pickle.loads(pickle.dumps(data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        np.testing.assert_array_equal(restored.compute(), data.compute())

    def test_zarr_round_trip_and_cli(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path / "input")
        output = tmp_path / "output.zarr"
        result = CliRunner().invoke(cosmx_command, ["--input", str(fixture.root), "--output", str(output)])
        assert result.exit_code == 0, result.output
        restored = read_zarr(output)
        table = restored.tables["table"]
        # Storage round trips return AnnData's own sparse type; compare exact values only.
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        np.testing.assert_array_equal(table.X.toarray(), [[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 0], [0, 0, 7, 1]])
        assert get_table_keys(table) == (["2_labels", "10_labels"], "region", "cell_ID")
        assert len(restored.points["Run_1_transcripts"].compute()) == 6
        np.testing.assert_array_equal(
            _transform(restored.labels["2_labels"], "global"),
            _transform(cosmx(fixture.root).labels["2_labels"], "global"),
        )
