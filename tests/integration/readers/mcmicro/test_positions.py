"""Producer regionprops centroids mapped into independently measured raster frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from anndata.tests.helpers import assert_equal
from dask.callbacks import Callback
from scipy import sparse
from skimage.measure import regionprops_table
from spatialdata import get_centroids, read_zarr

import spatialdata_io.readers.mcmicro._tables as table_owner
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers.mcmicro import mcmicro
from tests.integration.readers.mcmicro._fixtures import write_mcmicro_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _analytic_dataset(root: Path, *, legacy_directory: bool = False) -> np.ndarray:
    """Write MCQuant's direct centroid-1/0 mapping for three exact mask shapes."""
    write_mcmicro_dataset(root, legacy_directory=legacy_directory)
    image = np.zeros((2, 64, 80), dtype=np.uint16)
    tifffile.imwrite(
        root / "registration" / "sample_part_7.ome.tif",
        image,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 2,
            "PhysicalSizeY": 3,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )
    labels = np.zeros((64, 80), dtype=np.uint32)
    labels[1, 2] = 2
    labels[3:5, 4:7] = 7
    labels[5, 8] = labels[6, 8] = labels[6, 9] = 11
    tifffile.imwrite(
        root / "segmentation" / ("sample_part_7" if legacy_directory else "method_a-sample_part_7") / "cells.ome.tif",
        labels,
        metadata={"axes": "YX"},
        photometric="minisblack",
    )
    measured = pd.DataFrame(regionprops_table(labels, properties=("label", "centroid", "area"))).set_index("label")
    measured = measured.loc[[11, 2, 7]]
    frame = pd.DataFrame(
        {
            "CellID": [11, 2, 7],
            "CD3": [1, 2, 3],
            "DAPI": [0, 4, 5],
            "X_centroid": measured["centroid-1"].to_numpy(),
            "Y_centroid": measured["centroid-0"].to_numpy(),
            "Area": measured["area"].to_numpy(),
            "Orientation": [0.125, -0.25, np.nan],
            "MajorAxisLength": [2.0, 0.0, 4.0],
            "Eccentricity": [0.0, 0.5, 1.0],
        }
    )
    frame.to_csv(root / "quantification" / "sample_part_7--method_a_cells.csv", index=False)
    return np.asarray([[25 / 3, 17 / 3], [2, 1], [5, 3.5]])


def _calibrate(path: Path, spacing: tuple[float, float | None]) -> None:
    pixels = tifffile.imread(path)
    metadata: dict[str, str | float] = {
        "axes": "CYX" if pixels.ndim == 3 else "YX",
        "PhysicalSizeX": spacing[0],
        "PhysicalSizeXUnit": "µm",
    }
    if spacing[1] is not None:
        metadata.update(PhysicalSizeY=spacing[1], PhysicalSizeYUnit="µm")
    tifffile.imwrite(path, pixels, ome=True, photometric="minisblack", metadata=metadata)


class TestMcmicroPositions:
    @pytest.mark.parametrize("legacy_directory", [False, True])
    def test_independent_mcquant_regionprops_origin(self, tmp_path: Path, *, legacy_directory: bool) -> None:
        native = _analytic_dataset(tmp_path, legacy_directory=legacy_directory)
        result = mcmicro(tmp_path)
        table = result.tables["table"]
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], native)
        np.testing.assert_array_equal(table.obsm["spatial"], (native + 0.5) * [2, 3])
        np.testing.assert_array_equal(table.obs["Area"], [3, 1, 6])
        np.testing.assert_array_equal(table.obs["Orientation"], [0.125, -0.25, np.nan])
        np.testing.assert_array_equal(table.obs["MajorAxisLength"], [2, 0, 4])
        np.testing.assert_array_equal(table.obs["Eccentricity"], [0, 0.5, 1])
        element = result.labels["sample_part_7_method_a_cells"]
        np.testing.assert_array_equal(element.coords["x"], np.arange(80) + 0.5)
        np.testing.assert_array_equal(element.coords["y"], np.arange(64) + 0.5)
        independent = get_centroids(element, coordinate_system="global").compute()
        matched = independent.loc[[11, 2, 7], ["x", "y"]].to_numpy()
        np.testing.assert_allclose(matched, [[53 / 3, 37 / 2], [5, 4.5], [11, 12]], rtol=0, atol=4e-15)
        np.testing.assert_allclose((native + 0.5) * [2, 3], matched, rtol=0, atol=4e-15)

    @pytest.mark.parametrize("mode", ["wsi", "tma"])
    @pytest.mark.parametrize("calibrated", [False, True])
    def test_frames_multiple_methods_and_independent_masks(
        self, tmp_path: Path, mode: Literal["wsi", "tma"], *, calibrated: bool
    ) -> None:
        write_mcmicro_dataset(tmp_path, mode=mode, multiple_methods=True)
        image_paths = (
            [tmp_path / "registration" / "sample_part_7.ome.tif"]
            if mode == "wsi"
            else [tmp_path / "dearray" / f"{core}.ome.tif" for core in (1, 2)]
        )
        spacings = [(2.0, 3.0)] if mode == "wsi" else [(2.0, 3.0), (5.0, 7.0)]
        if calibrated:
            for path, spacing in zip(image_paths, spacings, strict=True):
                _calibrate(path, spacing)
        if mode == "tma":
            _calibrate(tmp_path / "dearray" / "masks" / "1_mask.tif", (9.0, 11.0))
        tasks: list[str] = []
        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = mcmicro(tmp_path)
        assert not tasks
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(
            table.X.data, [10.125, 30.125, 20.5, 40.5, 11.125, 31.125, 21.5, 41.5] * len(image_paths)
        )
        assert table.obs["area"].tolist() == ["12.0"] * table.n_obs
        assert table.obs["CellID"].tolist() == [1, 2, 11, 12] * len(image_paths)
        declaration = table.uns[POSITIONS_KEY]["coordinates"]
        if mode == "wsi":
            assert declaration["spatial_pixels"]["frame"] == "global_regionprops_indices"
            assert "spatial_pixel_frame" not in table.obs
        else:
            assert (
                table.obs["spatial_pixel_frame"].tolist()
                == ["core_1_regionprops_indices"] * 4 + ["core_2_regionprops_indices"] * 4
            )
            assert isinstance(table.obs["spatial_pixel_frame"].dtype, pd.CategoricalDtype)
        if calibrated:
            physical = table.obsm["spatial"]
            assert isinstance(physical, np.ndarray)
            assert physical.dtype == np.dtype(np.float64)
            if mode == "tma":
                assert table.obs["spatial_frame"].tolist() == ["core_1"] * 4 + ["core_2"] * 4
                assert table.obs["spatial_frame"].cat.categories.tolist() == ["core_1", "core_2"]
            else:
                assert declaration["spatial"]["frame"] == "global"
            for region in table.obs["region"].unique():
                frame = "global" if mode == "wsi" else str(region).split("_method")[0]
                independent = get_centroids(result.labels[region], coordinate_system=frame).compute()
                rows = np.flatnonzero(table.obs["region"].to_numpy() == region)
                matched = independent.loc[table.obs["CellID"].iloc[rows], ["x", "y"]].to_numpy()
                np.testing.assert_allclose(physical[rows], matched, rtol=0, atol=4e-15)
        else:
            assert "spatial" not in table.obsm
            assert "spatial" not in declaration

    def test_mixed_calibration_omits_physical_for_entire_aggregate(self, tmp_path: Path) -> None:
        write_mcmicro_dataset(tmp_path, mode="tma")
        _calibrate(tmp_path / "dearray" / "1.ome.tif", (2.0, 3.0))
        _calibrate(tmp_path / "dearray" / "masks" / "2_mask.tif", (5.0, 7.0))
        result = mcmicro(tmp_path)
        table = result.tables["table"]
        assert table.n_obs == 4
        assert "spatial" not in table.obsm
        assert "spatial_frame" not in table.obs
        assert set(table.uns[POSITIONS_KEY]["coordinates"]) == {"spatial_pixels"}
        # An unmeasured, uncalibrated source image does not decide the table's metric.
        (tmp_path / "quantification" / "2--method_a_cells.csv").unlink()
        measured = mcmicro(tmp_path).tables["table"]
        np.testing.assert_array_equal(measured.obsm["spatial"], [[8, 10.5], [72, 79.5]])
        assert measured.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] == "core_1"

    @pytest.mark.parametrize("storage", ["h5ad", "anndata_zarr", "spatialdata_zarr"])
    def test_physical_and_native_positions_persist(self, tmp_path: Path, storage: str) -> None:
        root = tmp_path / "input"
        write_mcmicro_dataset(root, mode="tma", multiple_methods=True)
        for core, spacing in ((1, (2.0, 3.0)), (2, (5.0, 7.0))):
            _calibrate(root / "dearray" / f"{core}.ome.tif", spacing)
        result = mcmicro(root)
        table = result.tables["table"]
        path = tmp_path / storage
        if storage == "h5ad":
            table.write_h5ad(path)
            restored = ad.read_h5ad(path)
        elif storage == "anndata_zarr":
            table.write_zarr(path)
            restored = ad.read_zarr(path)
        else:
            result.write(path)
            restored = read_zarr(path).tables["table"]
        assert isinstance(restored.obs, pd.DataFrame)
        assert isinstance(table.obs, pd.DataFrame)
        for key in ("spatial", "spatial_pixels"):
            np.testing.assert_array_equal(restored.obsm[key], table.obsm[key], strict=True)
        assert_equal(restored.uns[POSITIONS_KEY], table.uns[POSITIONS_KEY], exact=True)
        for name in ("spatial_frame", "spatial_pixel_frame", "region"):
            pd.testing.assert_series_equal(restored.obs[name], table.obs[name], check_exact=True)
        for number, selection in enumerate(([6, 0, 3], np.asarray([True, False] * 4), np.empty(0, dtype=np.int64))):
            selected = restored[selection].copy()
            for key in ("spatial", "spatial_pixels"):
                np.testing.assert_array_equal(selected.obsm[key], table.obsm[key][selection], strict=True)
            assert_equal(selected.uns[POSITIONS_KEY], table.uns[POSITIONS_KEY], exact=True)
            selected.write_h5ad(tmp_path / f"slice-{number}.h5ad")
            sliced = ad.read_h5ad(tmp_path / f"slice-{number}.h5ad")
            assert_equal(sliced.obsm, selected.obsm, exact=True)
            assert_equal(sliced.uns[POSITIONS_KEY], selected.uns[POSITIONS_KEY], exact=True)

    def test_label_metadata_does_not_complete_partial_source_image(self, tmp_path: Path) -> None:
        write_mcmicro_dataset(tmp_path)
        _calibrate(tmp_path / "registration" / "sample_part_7.ome.tif", (2.0, None))
        _calibrate(tmp_path / "segmentation" / "method_a-sample_part_7" / "cells.ome.tif", (2.0, 3.0))
        table = mcmicro(tmp_path).tables["table"]
        assert "spatial" not in table.obsm
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[3.5, 3], [75.5, 60]])

    def test_physical_positions_finish_before_one_linkage_without_densification(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        _analytic_dataset(tmp_path)
        linkage = mocker.spy(table_owner, "parse_linked_table")
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("expression densified"))
        mocker.patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("expression densified"))
        result = mcmicro(tmp_path)
        table = result.tables["table"]
        linkage.assert_called_once()
        assert linkage.call_args.args[0] is table
        assert set(table.obsm) == {"spatial", "spatial_pixels"}
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(table.X.data, [1, 2, 4, 3, 5])
