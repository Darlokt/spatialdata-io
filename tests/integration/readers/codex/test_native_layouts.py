"""CODEX workflows using paths and images produced by Processor and Uploader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Affine, BaseTransformation, get_transformation
from xarray import DataArray

from spatialdata_io import codex
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from tests.integration.readers.codex._fixtures import (
    cell_frame,
    write_csv,
    write_fcs,
    write_image,
    write_processor_markers,
    write_processor_table,
    write_uploader_montage,
)
from tests.integration.readers.phenocycler._fixtures import CIM_PROFILE, write_qptiff

if TYPE_CHECKING:
    from pathlib import Path


class TestCodexNativeLayouts:
    """Verify discovery, optional cells, registration, and channel semantics."""

    def test_processor_markers_and_tables_across_regions(self, tmp_path: Path) -> None:
        pixels = write_processor_markers(tmp_path, "reg002")
        write_processor_markers(tmp_path, "reg001")
        write_processor_table(tmp_path, 1)
        write_processor_table(tmp_path, 2)
        write_processor_table(tmp_path, 3)
        directory = tmp_path / "processed_2019-05-01" / "stitched" / "reg002"
        (directory / "reg002_mask.tiff").touch()
        (directory / "reg002_cyc001_ch001_mask.tif").touch()
        tasks: list[str] = []

        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = codex(tmp_path, scale_factors=())

        assert tasks == []
        assert list(result.images) == ["reg001_image", "reg002_image"]
        assert list(result.tables) == ["reg001_table", "reg002_table", "reg003_table"]
        assert list(result.shapes) == ["reg001", "reg002", "reg003"]
        image = result.images["reg002_image"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.c.values.tolist() == ["DAPI", "CD45", "CD3"]
        assert image.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image.data.compute(), pixels)
        image_frames = get_transformation(image, get_all=True)
        assert isinstance(image_frames, dict)
        assert list(image_frames) == ["reg002_image_pixels"]
        cell_frames = get_transformation(result.shapes["reg002"], get_all=True)
        assert isinstance(cell_frames, dict)
        assert list(cell_frames) == ["reg002_table_pixels"]
        table = result.tables["reg002_table"]
        assert get_table_keys(table) == ("reg002", "region", "cell_id")
        assert table.obs["region"].tolist() == ["reg002"] * 3
        assert table.obs["cell_id"].tolist() == [7, 2, 11]
        assert table.var_names.tolist() == ["cyc002_ch001", "cyc001_ch002"]
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(table.X.data, [2, 3.25, -1.5, 5.5])

    def test_uploader_needs_explicit_montage_registration(self, tmp_path: Path) -> None:
        names = ["DAPI", "CD3", "CD45", "DAPI2", "CD8", "CD4"]
        _, pixels = write_uploader_montage(tmp_path, "reg001", labels=names)
        write_processor_table(tmp_path, 1)

        result = codex(tmp_path, pixel_size_um=0.4, scale_factors=())

        image = result.images["reg001_image"]
        assert isinstance(image, DataArray)
        assert image.c.values.tolist() == names
        np.testing.assert_array_equal(image.data.compute(), pixels)
        image_frames = get_transformation(image, get_all=True)
        assert isinstance(image_frames, dict)
        assert list(image_frames) == ["image_pixels"]
        cell_frames = get_transformation(result.shapes["reg001"], get_all=True)
        assert isinstance(cell_frames, dict)
        assert list(cell_frames) == ["global"]

        registration = Affine(np.diag([2.0, 2.0, 1.0]), input_axes=("x", "y"), output_axes=("x", "y"))
        explicit = codex(tmp_path, pixel_size_um=0.4, image_to_table=registration, scale_factors=())
        transform = get_transformation(explicit.images["reg001_image"], to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.8, 0.8, 1]))

    def test_uploader_uses_sidecar_and_accepts_one_image(self, tmp_path: Path) -> None:
        path, _ = write_uploader_montage(tmp_path, "reg002")
        names = ["A", "B", "C", "D", "E", "F"]
        (path.parent / "channelNames.txt").write_text("\n".join(names), encoding="utf-8")

        result = codex(path, scale_factors=())

        assert not result.tables
        image = result.images["reg002_image"]
        assert isinstance(image, DataArray)
        assert image.c.values.tolist() == names

    def test_image_only_bypasses_ambiguous_tables(self, tmp_path: Path) -> None:
        write_processor_markers(tmp_path, "reg001")
        selected = write_processor_table(tmp_path, 1)
        write_processor_table(tmp_path, 1, compensation="uncompensated")

        with pytest.raises(ValueError, match="2 candidates"):
            codex(tmp_path)
        no_tables = codex(tmp_path, read_tables=False)
        assert list(no_tables.images) == ["reg001_image"]
        assert not no_tables.tables
        assert not no_tables.shapes
        explicit = codex(tmp_path, cell_table_file=selected)
        assert list(explicit.tables) == ["reg001_table"]

    def test_compensation_selects_one_signal_product_across_regions(self, tmp_path: Path) -> None:
        paired = write_processor_table(tmp_path, 1)
        write_fcs(paired.with_suffix(".fcs"))
        write_processor_table(tmp_path, 2)
        for number in (1, 2):
            raw = write_processor_table(tmp_path, number, compensation="uncompensated")
            write_csv(raw, cell_frame().assign(region=number, cyc002_ch001=[1.0, 4.0, 2.0]))
        write_processor_markers(tmp_path, "reg003")

        compensated = codex(tmp_path, compensation="compensated")
        uncompensated = codex(tmp_path, compensation="uncompensated")

        assert list(compensated.tables) == list(uncompensated.tables) == ["reg001_table", "reg002_table"]
        assert list(uncompensated.images) == ["reg003_image"]
        for name in uncompensated.tables:
            assert compensated.tables[name].uns["spatialdata_io_compensation"] == "compensated"
            assert uncompensated.tables[name].uns["spatialdata_io_compensation"] == "uncompensated"
            compensated_x = compensated.tables[name].X
            uncompensated_x = uncompensated.tables[name].X
            assert isinstance(compensated_x, sparse.csr_array)
            assert isinstance(uncompensated_x, sparse.csr_array)
            assert compensated_x.dtype == np.dtype(np.float64)
            np.testing.assert_array_equal(compensated_x.data, [2, 3.25, -1.5, 5.5])
            np.testing.assert_array_equal(uncompensated_x.data, [1, 2, 4, 2, 5.5])

    def test_missing_compensation_variant_does_not_skip_a_region(self, tmp_path: Path) -> None:
        write_processor_table(tmp_path, 1)
        write_processor_table(tmp_path, 2, compensation="uncompensated")

        with pytest.raises(ReaderFormatError, match=r"'compensated'.*reg002_uncompensated"):
            codex(tmp_path, compensation="compensated")

        selected = codex(tmp_path, regions=("reg001",), compensation="compensated")
        assert list(selected.tables) == ["reg001_table"]

    def test_cim_scan_uses_its_own_calibration(self, tmp_path: Path) -> None:
        path = tmp_path / "example_reg002.qptiff"
        pixels = write_qptiff(path, software="CODEX Instrument Manager", scan_profile=CIM_PROFILE)
        tasks: list[str] = []
        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = codex(path, scale_factors=())
        assert tasks == []
        assert list(result.images) == ["reg002_image"]
        assert not result.tables
        image = result.images["reg002_image"]
        assert isinstance(image, DataArray)
        assert image.c.values.tolist() == ["DAPI", "CD3"]
        np.testing.assert_array_equal(image.data.compute(), pixels)
        transform = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.5, 0.5, 1]))

    def test_single_native_image_keeps_an_unnamed_flat_table(self, tmp_path: Path) -> None:
        write_processor_markers(tmp_path, "reg001")
        write_csv(tmp_path / "cells.csv")
        (tmp_path / "channels.csv").write_text("description\nnot a cell table\n", encoding="utf-8")

        result = codex(tmp_path)
        assert list(result.tables) == ["reg001_table"]
        assert get_table_keys(result.tables["reg001_table"]) == ("reg001", "region", "cell_id")
        assert result.shapes["reg001"].index.tolist() == [7, 2, 11]

        write_processor_markers(tmp_path, "reg002")
        with pytest.raises(ReaderFormatError, match="one selected region for unnamed cell tables"):
            codex(tmp_path)
        selected = codex(tmp_path, regions=("reg001",))
        assert list(selected.tables) == ["reg001_table"]

    def test_multiple_named_flat_tables_are_separate_regions(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "reg001_compensated.csv")
        write_csv(tmp_path / "export_reg002_uncompensated.csv", cell_frame().assign(region="002"))

        result = codex(tmp_path)

        assert list(result.tables) == ["reg001_table", "reg002_table"]
        assert list(result.shapes) == ["reg001", "reg002"]
        assert not result.images
        assert result.tables["reg002_table"].obs["region"].tolist() == ["reg002"] * 3

    def test_cim_scan_variants_are_validated_and_prefer_extended_range(self, tmp_path: Path) -> None:
        standard = tmp_path / "example_reg002.qptiff"
        extended = tmp_path / "example_reg002.er.qptiff"
        write_qptiff(standard, software="CODEX Instrument Manager", scan_profile=CIM_PROFILE, dtype="uint8")
        pixels = write_qptiff(extended, software="CODEX Instrument Manager", scan_profile=CIM_PROFILE)
        result = codex(tmp_path)
        image = result.images["reg002_image"]
        assert isinstance(image, DataArray)
        assert image.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image.data.compute(), pixels)

        write_qptiff(
            standard, software="CODEX Instrument Manager", scan_profile=CIM_PROFILE, dtype="uint8", shape=(16, 32)
        )
        with pytest.raises(ReaderFormatError, match="same-generation variants with matching channels, XY shape"):
            codex(tmp_path)

    def test_codex_restricts_qptiff_to_cim(self, tmp_path: Path) -> None:
        path = tmp_path / "fusion.qptiff"
        write_qptiff(path)
        with pytest.raises(ReaderFormatError, match="a CODEX Instrument Manager scan"):
            codex(path)

    def test_native_options_need_matching_artifacts(self, tmp_path: Path) -> None:
        path = tmp_path / "reg002.qptiff"
        write_qptiff(path, software="CODEX Instrument Manager", scan_profile=CIM_PROFILE)
        affine = Affine(np.diag([2.0, 2.0, 1.0]), input_axes=("x", "y"), output_axes=("x", "y"))
        with pytest.raises(ValueError, match="data_axes and tiff_series cannot be used with a CIM QPTIFF"):
            codex(path, tiff_series=0)
        with pytest.raises(ValueError, match="one selected region with both a table and an image"):
            codex(path, image_to_table=affine)
        with pytest.raises(ValueError, match="experiment_file require a CODEX cell table"):
            codex(path, pixel_size_um=0.5)

    def test_cim_and_native_cell_table_require_explicit_registration(self, tmp_path: Path) -> None:
        path = tmp_path / "reg001.qptiff"
        write_qptiff(path, software="CODEX Instrument Manager", scan_profile=CIM_PROFILE)
        write_processor_table(tmp_path, 1)
        separate = codex(tmp_path, pixel_size_um=0.25)
        transform = get_transformation(separate.images["reg001_image"], to_coordinate_system="image_um")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.5, 0.5, 1]))

        affine = Affine(np.diag([2.0, 2.0, 1.0]), input_axes=("x", "y"), output_axes=("x", "y"))
        aligned = codex(tmp_path, pixel_size_um=0.25, image_to_table=affine)
        transform = get_transformation(aligned.images["reg001_image"], to_coordinate_system="global")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.5, 0.5, 1]))

    def test_selected_native_table_verifies_declared_region(self, tmp_path: Path) -> None:
        write_processor_markers(tmp_path, "reg001")
        table = write_processor_table(tmp_path, 2)

        with pytest.raises(ReaderFormatError, match="the selected artifact region 'reg001'"):
            codex(tmp_path, regions=("reg001",), cell_table_file=table)

    def test_native_table_only_calibration_keeps_regional_origins(self, tmp_path: Path) -> None:
        for region in (2, 1):
            write_processor_table(tmp_path, region)
        result = codex(tmp_path, pixel_size_um=0.4, regions=("reg002", "reg001"))
        assert list(result.tables) == ["reg002_table", "reg001_table"]
        assert list(result.shapes) == ["reg002", "reg001"]
        assert not result.images
        for region in ("reg001", "reg002"):
            frame = f"{region}_table_um"
            element = result.shapes[region]
            transform = get_transformation(element, to_coordinate_system=frame)
            assert isinstance(transform, BaseTransformation)
            np.testing.assert_allclose(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.4, 0.4, 1]))
            table = result.tables[f"{region}_table"]
            assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] == frame
            positions = table.obsm["spatial"]
            assert isinstance(positions, np.ndarray)
            np.testing.assert_allclose(positions, [[8, 0.6], [1, 12], [5.2, 3.2]])


class TestLabelLinkage:
    def test_flat_mask_owns_instances_after_round_trip(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        write_image(tmp_path / "stitched.tif")
        pixels = np.zeros((16, 20), dtype=np.uint32)
        pixels[2, 1:4] = [11, 2, 7]
        tifffile.imwrite(tmp_path / "instances.tif", pixels)
        xy = ("x", "y")
        tasks: list[str] = []
        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = codex(
                tmp_path,
                labels="instances.tif",
                labels_to_image=Affine(np.diag([2, 2, 1]), xy, xy),
                image_to_table=Affine(np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]]), xy, xy),
                pixel_size_um=0.5,
                scale_factors=(2,),
            )
        assert not tasks
        output = tmp_path / "result.zarr"
        result.write(output)
        stored = read_zarr(output)
        assert list(stored.labels) == ["labels"]
        assert not stored.shapes
        table = stored.tables["table"]
        assert table.uns["spatialdata_io_source_region"] == "1"
        assert list(table.obs["region"].cat.categories) == ["labels"]
        assert table.obs["region"].tolist() == ["labels"] * 3
        assert table.obs["cell_id"].tolist() == [7, 2, 11]
        assert table.uns["spatialdata_attrs"]["instance_key"] == "cell_id"
        mask = stored.labels["labels"]["scale0"]["image"]
        np.testing.assert_array_equal(mask.values, pixels)
        matrix = get_transformation(mask, "global")
        assert not isinstance(matrix, dict)
        np.testing.assert_allclose(matrix.to_affine_matrix(xy, xy), [[1, 0, 5], [0, 1, 10], [0, 0, 1]])

    @pytest.mark.parametrize("bad_id", [0, 256])
    def test_link_requires_nonbackground_representable_ids(self, tmp_path: Path, bad_id: int) -> None:
        cells = cell_frame()
        cells.loc[0, "cell_id"] = bad_id
        write_csv(tmp_path / "cells.csv", cells)
        write_image(tmp_path / "stitched.tif")
        tifffile.imwrite(tmp_path / "mask.tif", np.zeros((32, 40), dtype=np.uint8))
        with pytest.raises(ReaderFormatError, match="nonzero cell IDs representable"):
            codex(tmp_path, labels="mask.tif")

    def test_mixed_mask_and_circle_targets_remain_regional(self, tmp_path: Path) -> None:
        for index in (1, 2):
            write_processor_markers(tmp_path, f"reg{index:03d}")
            write_processor_table(tmp_path, index)
        path = tmp_path / "reg001_labels.tif"
        # This mask contains the same IDs used by both regional tables.
        pixels = np.zeros((32, 40), dtype=np.uint16)
        pixels[0, :3] = [2, 11, 7]
        tifffile.imwrite(path, pixels)
        result = codex(tmp_path, labels={"reg001": path})
        assert list(result.labels) == ["reg001_labels"]
        assert list(result.shapes) == ["reg002"]
        assert result.tables["reg001_table"].obs["region"].tolist() == ["reg001_labels"] * 3
        assert result.tables["reg002_table"].obs["region"].tolist() == ["reg002"] * 3
        assert get_transformation(result.labels["reg001_labels"], get_all=True) == get_transformation(
            result.images["reg001_image"], get_all=True
        )
