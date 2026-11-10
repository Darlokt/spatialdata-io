"""Public CosMx workflows: profiles, landmark geometry, laziness, and round trips."""

from __future__ import annotations

import pickle
import re
import shutil
import warnings
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray

from spatialdata_io import __version__
from spatialdata_io._cli.commands.cosmx import cosmx_command
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.cosmx import _reader, cosmx
from tests.integration.readers.cosmx._fixtures import (
    LANDMARKS,
    POSITIONS,
    global_xy,
    local_xy,
    write_cosmx_dataset,
    write_extreme_cosmx_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame as DaskDataFrame
    from pytest_mock import MockerFixture
    from spatialdata import SpatialData
    from xarray import DataTree

_AXES = ("x", "y")
_FRAMES = ["fov_2_pixels", "fov_2_pixels", "fov_10_pixels", "fov_10_pixels"]


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
        # Without calibration nothing is in global, which always means micrometres.
        assert sorted(result.coordinate_systems) == ["fov_10_pixels", "fov_2_pixels", "global_pixels"]
        assert result.images["2_image"].coords["c"].to_numpy().tolist() == ["r", "g", "b"]

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

    @pytest.mark.parametrize("compressed", [False, True])
    @pytest.mark.parametrize("storage", ["h5ad", "anndata_zarr", "spatialdata"])
    def test_full_uint64_keys_link_and_persist(self, tmp_path: Path, storage: str, *, compressed: bool) -> None:
        fixture = write_cosmx_dataset(tmp_path, compressed=compressed)
        identities = {"1": str(2**63 + 1), "2": str(2**64 - 1)}
        fovs = {"2": str(2**64 - 1), "10": str(2**63 + 1)}
        suffix = ".csv.gz" if compressed else ".csv"
        for artifact in ("exprMat", "metadata", "tx", "fov_positions"):
            path = fixture.root / f"Run_1_{artifact}_file{suffix}"
            frame = pd.read_csv(path, dtype="str", keep_default_na=False)
            frame["fov"] = frame["fov"].replace(fovs)
            if "cell_ID" in frame:
                frame["cell_ID"] = frame["cell_ID"].replace(identities)
            frame.to_csv(path, index=False)
        for directory in ("CellLabels", "CellComposite"):
            for path in (fixture.root / directory).iterdir():
                old = "2" if "_F002" in path.name else "10"
                if directory == "CellLabels":
                    labels = np.zeros((6, 8), dtype=np.uint64)
                    for cell, (row, column) in LANDMARKS.items():
                        labels[row, column] = int(identities[str(cell)])
                    tifffile.imwrite(path, labels, photometric="minisblack")
                path.rename(path.with_name(path.name.replace(f"_F{int(old):03d}", f"_F{fovs[old]}")))

        result = cosmx(fixture.root)
        table = result.tables["table"]
        expected_ids = [2**63 + 1, 2**64 - 1] * 2
        assert table.obs["cell_ID"].dtype == table.obs["fov"].dtype == np.dtype(np.uint64)
        assert table.obs["cell_ID"].tolist() == expected_ids
        assert table.obs["fov"].tolist() == [2**64 - 1, 2**64 - 1, 2**63 + 1, 2**63 + 1]
        assert table.obs_names.name is None
        expected_regions = [f"{2**63 + 1}_labels", f"{2**64 - 1}_labels"]
        assert get_table_keys(table) == (expected_regions, "region", "cell_ID")
        assert table.obs["region"].cat.categories.tolist() == expected_regions
        for label in result.labels.values():
            assert label.dtype == np.dtype(np.uint64)
        if storage == "h5ad":
            target = tmp_path / "table.h5ad"
            table.write_h5ad(target)
            restored = ad.read_h5ad(target)
        elif storage == "anndata_zarr":
            target = tmp_path / "table.zarr"
            table.write_zarr(target)
            restored = ad.read_zarr(target)
        else:
            target = tmp_path / "spatialdata.zarr"
            result.write(target)
            stored = read_zarr(target)
            restored = stored.tables["table"]
            assert stored.points["Run_1_transcripts"].compute()["cell_ID"].tolist() == [
                2**63 + 1, 0, 2**64 - 1, 2**63 + 1, 0, 2**64 - 1
            ]  # fmt: skip
        assert restored.obs["cell_ID"].tolist() == expected_ids
        assert restored.obs["cell_ID"].dtype == np.dtype(np.uint64)
        assert restored.obs["fov"].tolist() == table.obs["fov"].tolist()
        regions, region_key, instance_key = get_table_keys(restored)
        assert list(regions) == expected_regions
        assert (region_key, instance_key) == ("region", "cell_ID")

    def test_landmark_pixels_map_to_their_evidenced_local_and_global_coordinates(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root)
        for fov in POSITIONS:
            labels = _labels(result, fov)
            to_local, to_global = _transform(labels, f"fov_{fov}_pixels"), _transform(labels, "global_pixels")
            pixels = labels.data.compute()
            for cell in LANDMARKS:
                row, column = np.argwhere(pixels == cell)[0]
                center = np.array([column + 0.5, row + 0.5, 1.0])
                np.testing.assert_array_equal((to_local @ center)[:2], local_xy(cell))
                np.testing.assert_array_equal((to_global @ center)[:2], global_xy(fov, cell, "flat_2021"))
            # The table's own coordinates agree with the raster geometry.
            rows = (table := result.tables["table"]).obs["fov"].to_numpy() == fov
            np.testing.assert_array_equal(table.obsm["spatial_pixels"][rows], [local_xy(cell) for cell in LANDMARKS])

    def test_points_share_every_fov_frame_with_rasters(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root)
        points = result.points["Run_1_transcripts"]
        for fov in POSITIONS:
            frame = points.compute()
            selected = frame[frame["fov"] == fov]
            xy = np.column_stack([selected["x"], selected["y"], np.ones(len(selected))])
            local = (_transform(points, f"fov_{fov}_pixels") @ xy.T).T[:, :2]
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
        np.testing.assert_array_equal(_transform(points, "fov_2_pixels"), [[1, 0, -1000], [0, -1, 5000.5], [0, 0, 1]])

    def test_rasters_in_a_reversed_axis_export_fail(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, profile="atomx")
        with pytest.raises(ReaderFormatError, match="global axes are reversed"):
            cosmx(fixture.root)

    def test_labels_must_cover_every_fov_with_cells(self, tmp_path: Path, mocker: MockerFixture) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        (fixture.root / "CellLabels" / "CellLabels_F010.tif").unlink()
        counts = mocker.spy(_reader, "read_counts")
        image = mocker.spy(_reader, "read_image")
        labels = mocker.spy(_reader, "read_labels")
        tasks: list[object] = []
        with (
            Callback(posttask=lambda *args: tasks.append(args)),
            pytest.raises(ReaderFormatError, match=r"no labels for FOVs \[10\]") as caught,
        ):
            cosmx(fixture.root)
        assert caught.value.path == fixture.root / "CellLabels"
        assert caught.value.context == ReaderErrorContext("cosmx", "cell labels")
        counts.assert_not_called()
        image.assert_not_called()
        labels.assert_not_called()
        assert not tasks

    @pytest.mark.parametrize("dotfile", [False, True])
    def test_present_empty_labels_fail_during_discovery(
        self, tmp_path: Path, mocker: MockerFixture, *, dotfile: bool
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        directory = fixture.root / "CellLabels"
        for path in directory.iterdir():
            path.unlink()
        if dotfile:
            (directory / ".hidden.tif").write_bytes(b"ignored")
        metadata = mocker.spy(_reader, "read_metadata")
        counts = mocker.spy(_reader, "read_counts")
        image = mocker.spy(_reader, "read_image")
        labels = mocker.spy(_reader, "read_labels")
        tasks: list[object] = []
        with (
            Callback(posttask=lambda *args: tasks.append(args)),
            pytest.raises(ReaderFormatError, match="CellLabels"),
        ):
            cosmx(fixture.root)
        metadata.assert_not_called()
        counts.assert_not_called()
        image.assert_not_called()
        labels.assert_not_called()
        assert not tasks

    def test_absent_labels_leave_images_and_table_unlinked(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        shutil.rmtree(fixture.root / "CellLabels")
        result = cosmx(fixture.root)
        assert list(result.images) == ["2_image", "10_image"]
        assert not result.labels
        assert result.tables["table"].obs["region"].tolist() == ["2_labels", "2_labels", "10_labels", "10_labels"]
        assert "spatialdata_attrs" not in result.tables["table"].uns

    def test_wholly_cell_empty_export_cannot_establish_dataset_signs(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, rasters=False)
        for suffix in ("metadata", "exprMat"):
            path = fixture.root / f"Run_1_{suffix}_file.csv"
            pd.read_csv(path).iloc[:0].to_csv(path, index=False)
        with pytest.raises(ReaderFormatError, match="one sign determined by cells"):
            cosmx(fixture.root, include_transcripts=False)

    def test_ids_must_fit_their_own_fov_label_dtype_without_reading_pixels(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        for suffix in ("metadata", "exprMat"):
            path = fixture.root / f"Run_1_{suffix}_file.csv"
            frame = pd.read_csv(path)
            frame.loc[(frame["fov"] == 2) & (frame["cell_ID"] == 1), "cell_ID"] = 70000
            frame.to_csv(path, index=False)
        tasks: list[object] = []
        with (
            Callback(posttask=lambda *args: tasks.append(args)),
            pytest.raises(ReaderFormatError, match=r"70000.*uint16"),
        ):
            cosmx(fixture.root)
        assert not tasks

    def test_without_rasters_or_transcripts(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, rasters=False, positions=False)
        result = cosmx(fixture.root, include_transcripts=False)
        assert not result.images
        assert not result.points
        assert "spatialdata_attrs" not in result.tables["table"].uns

    def test_nonfinite_placement_fails_before_element_construction(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root = write_extreme_cosmx_dataset(tmp_path)
        image = mocker.spy(_reader, "read_image")
        labels = mocker.spy(_reader, "read_labels")
        table = mocker.spy(_reader, "build_table")
        points = mocker.spy(_reader, "read_transcripts")
        tasks: list[object] = []
        with (
            np.errstate(all="raise"),
            Callback(posttask=lambda *args: tasks.append(args)),
            pytest.raises(ReaderFormatError, match="inconsistent with any translation"),
        ):
            cosmx(root)
        image.assert_not_called()
        labels.assert_not_called()
        table.assert_not_called()
        points.assert_not_called()
        assert not tasks

    @pytest.mark.parametrize("token", ["0e309", "0e-400", "0.0e-9223372036854775808"])
    def test_invalid_printed_width_fails_before_raster_construction(
        self, tmp_path: Path, mocker: MockerFixture, token: str
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        path = fixture.root / "Run_1_metadata_file.csv"
        path.write_text(path.read_text().replace("2.0,5.0", f"{token},5.0", 1))
        counts = mocker.spy(_reader, "read_counts")
        image = mocker.spy(_reader, "read_image")
        with pytest.raises(ReaderFormatError, match="rounding half-widths"):
            cosmx(fixture.root)
        counts.assert_not_called()
        image.assert_not_called()

    def test_include_transcripts_must_be_boolean(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="boolean"):
            cosmx(tmp_path, include_transcripts="yes")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_invalid_raster_option_fails_before_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="chunks"):
            cosmx(tmp_path / "missing", chunks=0)

    def test_graphs_serialize_and_pyramids_stay_lazy(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        result = cosmx(fixture.root, scale_factors=(2,), chunks=(3, 4, 4))
        data = result.images["2_image"]["scale0"]["image"].data
        restored = pickle.loads(pickle.dumps(data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        np.testing.assert_array_equal(restored.compute(), data.compute())

    @pytest.mark.parametrize("chunk_size", [16, 32])
    def test_compressed_tiff_composition_reads_only_selected_model_chunks(
        self, tmp_path: Path, mocker: MockerFixture, chunk_size: int
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        height, width = 64, 80
        yy, xx = np.indices((height, width), dtype=np.uint16)
        image = np.stack([(yy * 7 + xx * 3) % 256, xx % 256, yy % 256], axis=-1).astype(np.uint8)
        labels = np.zeros((height, width), dtype=np.uint16)
        for cell, (row, column) in LANDMARKS.items():
            labels[row, column] = cell
        for fov in POSITIONS:
            (fixture.root / "CellComposite" / f"CellComposite_F{fov:03d}.png").unlink()
            tifffile.imwrite(
                fixture.root / "CellComposite" / f"CellComposite_F{fov:03d}.tif",
                image,
                photometric="rgb",
                compression="deflate",
                tile=(16, 16),
            )
            tifffile.imwrite(
                fixture.root / "CellLabels" / f"CellLabels_F{fov:03d}.tif",
                labels,
                photometric="minisblack",
                compression="deflate",
                tile=(16, 16),
            )
        # Keep the same landmark row indices and the evidenced local Y = H - row convention.
        for suffix, columns in (
            ("metadata", ("CenterY_local_px", "CenterY_global_px")),
            ("tx", ("y_local_px", "y_global_px")),
        ):
            path = fixture.root / f"Run_1_{suffix}_file.csv"
            frame = pd.read_csv(path)
            for column in columns:
                frame[column] += height - 6
            frame.to_csv(path, index=False)

        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            result = cosmx(fixture.root, chunks=(3, chunk_size, chunk_size))
        assert not tasks
        actual_image = result.images["2_image"]
        assert isinstance(actual_image, DataArray)
        actual_labels = _labels(result, 2)
        assert actual_image.dims == ("c", "y", "x")
        assert actual_labels.dims == ("y", "x")
        assert actual_image.shape == (3, height, width)
        assert actual_labels.shape == (height, width)
        assert actual_image.dtype == np.dtype(np.uint8)
        assert actual_labels.dtype == np.dtype(np.uint16)
        assert actual_image.coords["c"].to_numpy().tolist() == ["r", "g", "b"]
        restored_image, restored_labels = pickle.loads(  # noqa: S301, RUF100 - Deserialize this test's own graphs.
            pickle.dumps((actual_image.data, actual_labels.data))
        )

        # Observe compressed payload reads, so the bound measures source I/O rather than graph size.
        payloads: dict[str, tuple[tuple[int, int], ...]] = {}
        for path in (
            fixture.root / "CellComposite" / "CellComposite_F002.tif",
            fixture.root / "CellLabels" / "CellLabels_F002.tif",
        ):
            with tifffile.TiffFile(path) as tiff:
                page = tiff.pages[0]
                payloads[path.name] = tuple(
                    (start, start + count) for start, count in zip(page.dataoffsets, page.databytecounts, strict=True)
                )
        reads: dict[str, list[tuple[int, int]]] = {name: [] for name in payloads}
        original_read = tifffile.FileHandle.read

        def tracked_read(handle: tifffile.FileHandle, size: int = -1, /) -> bytes:
            start = handle.tell()
            value = original_read(handle, size)
            if handle.name in reads:
                reads[handle.name].append((start, start + len(value)))
            return value

        mocker.patch.object(tifffile.FileHandle, "read", new=tracked_read)
        np.testing.assert_array_equal(
            restored_image[:, 1:5, 2:7].compute(scheduler="synchronous"), image[1:5, 2:7].transpose(2, 0, 1)
        )
        np.testing.assert_array_equal(restored_labels[1:5, 2:7].compute(scheduler="synchronous"), labels[1:5, 2:7])
        expected_tiles = {0} if chunk_size == 16 else {0, 1, 5, 6}
        for name, ranges in payloads.items():
            observed = {
                index
                for index, (start, stop) in enumerate(ranges)
                if any(max(start, low) < min(stop, high) for low, high in reads[name])
            }
            assert observed == expected_tiles
        center = np.array([2.5, 1.5, 1.0])
        for raster in (actual_image, actual_labels):
            np.testing.assert_array_equal(
                (_transform(raster, "global_pixels") @ center)[:2],
                result.tables["table"].obsm["spatial_global_pixels"][0],
            )

    @pytest.mark.cli
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
            _transform(restored.labels["2_labels"], "global_pixels"),
            _transform(cosmx(fixture.root).labels["2_labels"], "global_pixels"),
        )


class TestCosmxCalibration:
    """Use one supplied metric after the independently accepted placement mapping."""

    @pytest.mark.parametrize("compressed", [False, True])
    @pytest.mark.parametrize("raster_mode", ["none", "images", "labels", "both"])
    @pytest.mark.parametrize("include_transcripts", [False, True])
    def test_physical_positions(
        self, tmp_path: Path, raster_mode: str, *, compressed: bool, include_transcripts: bool
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path, compressed=compressed, rasters=raster_mode != "none")
        if raster_mode == "images":
            shutil.rmtree(tmp_path / "CellLabels")
        elif raster_mode == "labels":
            shutil.rmtree(tmp_path / "CellComposite")
        tasks: list[str] = []
        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = cosmx(fixture.root, pixel_size_um=0.5, include_transcripts=include_transcripts)
        assert not tasks
        table = result.tables["table"]
        np.testing.assert_array_equal(
            table.obsm["spatial"], [[501, 2502.75], [503, 2501.25], [1501.125, 3502.5], [1503.125, 3501]]
        )
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[2, 5], [6, 2], [2, 5], [6, 2]])
        np.testing.assert_array_equal(
            table.obsm["spatial_global_pixels"], [[1002, 5005.5], [1006, 5002.5], [3002.25, 7005], [3006.25, 7002]]
        )
        assert table.obs["spatial_pixel_frame"].tolist() == _FRAMES
        assert isinstance(table.obs["spatial_pixel_frame"].dtype, pd.CategoricalDtype)
        assert table.obs["spatial_pixel_frame"].cat.categories.tolist() == ["fov_2_pixels", "fov_10_pixels"]
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial_global_pixels"]["frame"] == "global_pixels"
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] == "global"
        # Calibrated elements are in global micrometres only, never also in the pixel frame.
        assert set(result.coordinate_systems) <= {"fov_2_pixels", "fov_10_pixels", "global"}
        for elements, key in ((result.images, "2_image"), (result.labels, "2_labels")):
            if key in elements:
                element = elements[key]
                transform = get_transformation(element, to_coordinate_system="global")
                assert isinstance(transform, BaseTransformation)
                # Source landmark at row 1, column 2; raster centers are corner-based.
                actual = transform.to_affine_matrix(("x", "y"), ("x", "y")) @ [2.5, 1.5, 1]
                np.testing.assert_array_equal(actual, [501, 2502.75, 1])
        if include_transcripts:
            points = result.points["Run_1_transcripts"]
            transform = get_transformation(points, to_coordinate_system="global")
            assert isinstance(transform, BaseTransformation)
            np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.5, 0.5, 1]))
            np.testing.assert_array_equal(points.compute()[["x", "y"]].to_numpy()[0], [1002, 5005.625])

    @pytest.mark.parametrize("pixel_size_um", [None, 0.5])
    def test_coarse_positions_compose_with_cell_bounds_and_point_frames(
        self, tmp_path: Path, pixel_size_um: float | None
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        # Whole-pixel origins straddle the true origins: above FOV 2 Y and below FOV 10 X.
        (fixture.root / "Run_1_fov_positions_file.csv").write_text("fov,x_px,y_px\n2,1000,5001\n10,3000,7000\n")
        result = cosmx(fixture.root, pixel_size_um=pixel_size_um)
        scale = 1.0 if pixel_size_um is None else pixel_size_um
        table = result.tables["table"]
        points = result.points["Run_1_transcripts"]
        events = points.compute()
        np.testing.assert_array_equal(
            table.obsm["spatial_global_pixels"],
            [global_xy(fov, cell, "flat_2021") for fov in POSITIONS for cell in LANDMARKS],
        )
        global_frame = "global_pixels" if pixel_size_um is None else "global"
        if pixel_size_um is not None:
            np.testing.assert_array_equal(table.obsm["spatial"], table.obsm["spatial_global_pixels"] * scale)
        for index, fov in enumerate(POSITIONS):
            # Bounds combine the half printed units of the local and global cell centers.
            half = np.array([0.1 if fov == 2 else 0.055, 0.1])
            for cell, (row, column) in LANDMARKS.items():
                center = np.array([column + 0.5, row + 0.5, 1.0])
                expected = table.obsm["spatial_global_pixels"][index * 2 + cell - 1]
                for raster in (result.images[f"{fov}_image"], result.labels[f"{fov}_labels"]):
                    observed = (_transform(raster, global_frame) @ center)[:2] / scale
                    assert np.all(np.abs(observed - expected) <= half)
                    np.testing.assert_array_equal(
                        (_transform(raster, f"fov_{fov}_pixels") @ center)[:2], local_xy(cell)
                    )
            selected = events[events["fov"] == fov]
            native = np.column_stack([selected["x"], selected["y"], np.ones(len(selected))])
            local = (_transform(points, f"fov_{fov}_pixels") @ native.T).T[:, :2]
            assert np.all(np.abs(local - selected[["x_local", "y_local"]].to_numpy()) <= half)
            global_points = (_transform(points, global_frame) @ native.T).T[:, :2]
            np.testing.assert_array_equal(global_points, native[:, :2] * scale)

    @pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_scale(self, tmp_path: Path, value: float) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            cosmx(tmp_path, pixel_size_um=value)

    @pytest.mark.parametrize("storage", ["h5ad", "zarr", "spatialdata"])
    def test_position_persistence(self, tmp_path: Path, storage: str) -> None:
        result = cosmx(write_cosmx_dataset(tmp_path / "source").root, pixel_size_um=0.5)
        table = result.tables["table"]
        destination = tmp_path / f"positions.{storage}"
        if storage == "h5ad":
            table.write_h5ad(destination)
            loaded = ad.read_h5ad(destination)
        elif storage == "zarr":
            table.write_zarr(destination)
            loaded = ad.read_zarr(destination)
        else:
            result.write(destination)
            loaded = read_zarr(destination).tables["table"]
        for key in ("spatial", "spatial_pixels", "spatial_global_pixels"):
            np.testing.assert_array_equal(loaded.obsm[key], table.obsm[key])
        assert loaded.uns[POSITIONS_KEY]["coordinates"]["spatial_pixels"]["frame_key"] == "spatial_pixel_frame"
        assert loaded.obs["spatial_pixel_frame"].tolist() == _FRAMES
        assert set(loaded.uns[POSITIONS_KEY]["coordinates"]["spatial"]["calibration"]) == {"global"}
        np.testing.assert_array_equal(loaded[[2, 0]].obsm["spatial"], [[1501.125, 3502.5], [501, 2502.75]])

    def test_wrong_scale_type(self, tmp_path: Path) -> None:
        invalid: object = True
        with pytest.raises(TypeError, match="positive finite"):
            cosmx(tmp_path, pixel_size_um=cast("float", invalid))

    def test_current_pancreas_grid_is_not_a_translation(self, tmp_path: Path) -> None:
        # Original manufacturer S0_metadata_file.csv.gz first ten XY rows:
        # https://objects.liquidweb.services/smi-public/wtx_manuscript/pancreas_discovery/index_pancreas.html
        fixture = write_cosmx_dataset(tmp_path, rasters=False, positions=False)
        coordinates = [
            (705, 23, 58294, 80841),
            (995, 25, 58583, 80839),
            (779, 45, 58368, 80819),
            (4056, 38, 61637, 80826),
            (885, 28, 58474, 80836),
            (1060, 30, 58648, 80834),
            (1674, 42, 59261, 80822),
            (4097, 64, 61678, 80800),
            (4164, 70, 61745, 80794),
            (850, 50, 58439, 80814),
        ]
        metadata = pd.DataFrame(
            coordinates, columns=["CenterX_local_px", "CenterY_local_px", "CenterX_global_px", "CenterY_global_px"]
        )
        metadata.insert(0, "cell_ID", range(1, 11))
        metadata.insert(0, "fov", 1)
        metadata.to_csv(fixture.root / "Run_1_metadata_file.csv", index=False)
        with pytest.raises(ReaderFormatError, match="inconsistent with any translation"):
            cosmx(fixture.root, pixel_size_um=0.120280945)

    def test_signed_table_only_and_stage_only_positions(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, profile="atomx", rasters=False)
        result = cosmx(fixture.root, pixel_size_um=0.5)
        np.testing.assert_array_equal(
            result.tables["table"].obsm["spatial"],
            [[501, 2497.75], [503, 2499.25], [1501.125, 3497.5], [1503.125, 3499]],
        )

    def test_physical_overflow(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, rasters=False)
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite physical"):
            cosmx(fixture.root, pixel_size_um=1e308)

    @pytest.mark.parametrize("raster_mode", ["images", "labels"])
    def test_raster_affine_overflow_fails_even_with_finite_physical_cells(
        self, tmp_path: Path, raster_mode: str
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path, positions=False)
        shutil.rmtree(fixture.root / ("CellLabels" if raster_mode == "images" else "CellComposite"))
        path = fixture.root / "Run_1_metadata_file.csv"
        metadata = pd.read_csv(path)
        local = np.array([-1e308, -9e307] * 2)
        global_ = np.array([0, 1e307] * 2)
        # Both rows determine a positive finite offset near 1e308. Scaling
        # the small global cell positions is safe, but scaling that offset is not.
        assert np.isfinite(global_ * 2).all()
        metadata["CenterX_local_px"] = [f"{value:.1f}" for value in local]
        metadata["CenterX_global_px"] = [f"{value:.1f}" for value in global_]
        metadata.to_csv(path, index=False)
        tasks: list[object] = []
        with (
            np.errstate(all="raise"),
            Callback(posttask=lambda *args: tasks.append(args)),
            pytest.raises(ReaderFormatError, match="nonfinite global raster affine"),
        ):
            cosmx(fixture.root, pixel_size_um=2)
        assert not tasks


class TestCosmxPathPolicy:
    """Discovered artifacts follow the dataset root's symlink policy."""

    def test_out_of_root_symlinks_require_the_relaxed_policy(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path]
    ) -> None:
        fixture = write_cosmx_dataset(tmp_path)
        expected = cosmx(fixture.root)
        link_outside_root(fixture.root / f"{fixture.file_prefix}_exprMat_file.csv")
        link_outside_root(fixture.root / "CellComposite" / "CellComposite_F002.png")

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            cosmx(fixture.root)

        assert_spatial_data_objects_are_identical(cosmx(fixture.root, path_policy="relaxed"), expected)


class TestCosmxNames:
    """The file prefix selects files; dataset_id only names the transcripts element."""

    def test_prefix_with_a_space_is_readable_and_repaired_for_element_names(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path / "input", file_prefix="Lung 5")
        message = (
            "cosmx: the file prefix 'Lung 5' is not a usable SpatialData element name; using 'Lung_5'. "
            "Pass dataset_id to choose the name explicitly."
        )
        with pytest.warns(UserWarning, match=re.escape(message)) as caught:
            result = cosmx(fixture.root)
        assert [str(warning.message) for warning in caught] == [message]
        assert list(result.points) == ["Lung_5_transcripts"]
        result.write(tmp_path / "output.zarr")
        assert list(read_zarr(tmp_path / "output.zarr").points) == ["Lung_5_transcripts"]

    def test_explicit_names_select_files_and_name_points_independently(self, tmp_path: Path) -> None:
        write_cosmx_dataset(tmp_path, file_prefix="Lung 5")
        write_cosmx_dataset(tmp_path, file_prefix="Run_2", rasters=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            explicit = cosmx(tmp_path, file_prefix="Lung 5", dataset_id="lung5")
            default = cosmx(tmp_path, file_prefix="Run_2")
        assert list(explicit.points) == ["lung5_transcripts"]
        assert list(default.points) == ["Run_2_transcripts"]
        with pytest.raises(ReaderFormatError, match="2 count matrices"):
            cosmx(tmp_path)

    def test_no_name_is_derived_without_transcripts(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, file_prefix="Lung 5")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cosmx(fixture.root, include_transcripts=False)
        assert not result.points

    @pytest.mark.parametrize(
        ("dataset_id", "error", "match"),
        [
            (3, TypeError, "dataset_id must be a string; found int"),
            ("my sample", ValueError, "dataset_id must be a valid SpatialData element name"),
            ("_", ValueError, "the transcripts element name '<dataset_id>_transcripts' must be a valid"),
        ],
    )
    def test_invalid_dataset_id_fails_before_io(
        self, tmp_path: Path, dataset_id: object, error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            cosmx(tmp_path / "missing", dataset_id=cast("str", dataset_id), include_transcripts=False)

    def test_derived_underscore_prefix_cannot_name_points(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path, file_prefix="_", rasters=False)
        with pytest.raises(ValueError, match=r"element name.*found '__transcripts'"):
            cosmx(fixture.root)
        assert list(cosmx(fixture.root, dataset_id="run").points) == ["run_transcripts"]

    def test_options_are_validated_in_signature_order_before_io(self, tmp_path: Path) -> None:
        invalid: list[tuple[str, object, type[Exception]]] = [
            ("file_prefix", "a/b", ValueError),
            ("dataset_id", "a b", ValueError),
            ("include_transcripts", "yes", TypeError),
            ("pixel_size_um", -1.0, ValueError),
            ("scale_factors", (1,), ValueError),
            ("chunks", 0, ValueError),
            ("path_policy", "lenient", ValueError),
        ]
        options = {name: value for name, value, _ in invalid}
        for name, _, error in invalid:
            # Every later option is invalid too, so the first error proves the order.
            with pytest.raises(error, match=name.split("_")[0]):
                cosmx(tmp_path / "missing", **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
            del options[name]


class TestCosmxMetadataColumns:
    """Metadata header names SpatialData cannot store are sanitized once, with one warning."""

    def test_unusable_names_are_repaired_and_round_trip(self, tmp_path: Path) -> None:
        fixture = write_cosmx_dataset(tmp_path / "input")
        path = fixture.root / "Run_1_metadata_file.csv"
        metadata = pd.read_csv(path)
        metadata["Mean.CD45 RO"] = [1.5, 2.5, 3.5, 4.5]
        metadata["Region"] = ["a", "b", "c", "d"]
        metadata.to_csv(path, index=False)
        message = (
            "cosmx: the cell metadata CSV header has column names that a SpatialData table does not accept; using "
            "'Mean.CD45 RO' -> 'Mean.CD45_RO', 'Region' -> 'Region_1'. Rename the source columns to choose the "
            "names explicitly."
        )
        with pytest.warns(UserWarning, match=re.escape(message)) as caught:
            result = cosmx(fixture.root)
        assert [str(warning.message) for warning in caught] == [message]
        table = result.tables["table"]
        assert table.obs["Mean.CD45_RO"].tolist() == [1.5, 2.5, 3.5, 4.5]
        assert table.obs["Region_1"].tolist() == ["a", "b", "c", "d"]
        assert get_table_keys(table) == (["2_labels", "10_labels"], "region", "cell_ID")
        result.write(tmp_path / "output.zarr")
        restored = read_zarr(tmp_path / "output.zarr").tables["table"]
        assert restored.obs["Mean.CD45_RO"].tolist() == [1.5, 2.5, 3.5, 4.5]
        assert restored.obs["Region_1"].tolist() == ["a", "b", "c", "d"]
