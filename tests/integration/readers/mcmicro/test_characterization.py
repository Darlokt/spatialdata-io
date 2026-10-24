"""Preserve valid MCMICRO values while testing explicit scientific corrections."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from scipy import sparse
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.mcmicro import mcmicro
from tests.integration.readers.mcmicro._fixtures import write_mcmicro_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from spatialdata import SpatialData


def _expression(table: AnnData) -> sparse.csr_array:
    """Narrow AnnData's broad expression union to the reader's sparse contract."""
    assert isinstance(table.X, sparse.csr_array)
    return table.X


def _frames(result: SpatialData, name: str) -> dict[str, object]:
    """Return every coordinate system of one image or labels element."""
    element = result.images[name] if name in result.images else result.labels[name]
    frames = get_transformation(element, get_all=True)
    assert isinstance(frames, dict)
    return dict(frames)


class TestMcmicroCharacterization:
    """Values that the legacy reader produced correctly remain exactly unchanged."""

    @pytest.mark.parametrize("legacy_directory", [False, True])
    def test_wsi_values(self, tmp_path: Path, *, legacy_directory: bool) -> None:
        fixture = write_mcmicro_dataset(tmp_path, legacy_directory=legacy_directory)
        result = mcmicro(fixture.root)

        assert list(result.images) == ["sample_part_7_image"]
        assert list(result.labels) == ["sample_part_7_method_a_cells"]
        assert list(result.tables) == ["table"]

        image = result.images["sample_part_7_image"]
        assert image.dims == ("c", "y", "x")
        assert image.shape == (2, 64, 80)
        assert image.dtype == np.dtype(np.uint16)
        assert image.coords["c"].to_numpy().tolist() == ["CD3", "DAPI"]
        np.testing.assert_array_equal(image.data[:, :1, :2].compute(), [[[0, 1]], [[5120, 5121]]])
        assert isinstance(_frames(result, "sample_part_7_image")["global"], Identity)

        labels = result.labels["sample_part_7_method_a_cells"]
        assert labels.dims == ("y", "x")
        assert labels.dtype == np.dtype(np.uint32)
        np.testing.assert_array_equal(labels.data[2:4, 2:4].compute(), np.ones((2, 2), dtype=np.uint32))

        table = result.tables["table"]
        np.testing.assert_array_equal(_expression(table).toarray(), [[10.125, 30.125], [20.5, 40.5]])
        assert table.var_names.tolist() == ["CD3", "DAPI"]
        assert table.var["marker_name"].tolist() == ["CD3", "DAPI"]
        assert table.var["cycle"].tolist() == ["1", "1"]
        assert table.obs["CellID"].tolist() == [1, 2]
        np.testing.assert_array_equal(table.obsm["spatial"], [[3.5, 3], [75.5, 60]])
        # MCQuant defines no "custom_note", "area" or suffixed intensity fields: they stay exact text.
        assert table.obs["custom_note"].tolist() == ["first", "second"]
        assert table.obs["area"].tolist() == ["12.0", "12.0"]
        assert table.obs["CD3_intensity_max"].tolist() == ["100.0", "200.0"]
        assert get_table_keys(table) == ("sample_part_7_method_a_cells", "region", "CellID")

    def test_tma_values(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode="tma")
        result = mcmicro(fixture.root)

        assert list(result.images) == ["tma_map", "core_1_image", "core_2_image"]
        assert list(result.labels) == ["core_1_mask", "core_2_mask", "core_1_method_a_cells", "core_2_method_a_cells"]
        assert result.images["tma_map"].shape == (2, 64, 80)
        assert result.images["core_1_image"].shape == (2, 30, 40)
        assert result.images["core_2_image"].shape == (2, 21, 31)
        assert result.labels["core_1_mask"].shape == (3, 4)
        assert result.labels["core_2_mask"].shape == (2, 3)
        np.testing.assert_array_equal(
            result.images["core_2_image"].data[:, :1, :2].compute(), [[[2000, 2001]], [[2651, 2652]]]
        )

        table = result.tables["table"]
        assert table.obs_names.tolist() == ["0", "1", "2", "3"]
        assert table.obs["region"].tolist() == ["core_1_method_a_cells"] * 2 + ["core_2_method_a_cells"] * 2
        assert table.obs["CellID"].tolist() == [1, 2, 1, 2]
        np.testing.assert_array_equal(_expression(table).toarray(), [[10.125, 30.125], [20.5, 40.5]] * 2)
        assert get_table_keys(table) == (["core_1_method_a_cells", "core_2_method_a_cells"], "region", "CellID")


class TestMcmicroCorrections:
    """Each legacy defect is replaced by an independently specified contract."""

    def test_cores_and_masks_keep_exact_local_frames(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode="tma")
        result = mcmicro(fixture.root)
        assert set(_frames(result, "tma_map")) == {"global"}
        for core in ("core_1", "core_2"):
            assert set(_frames(result, f"{core}_image")) == {core}
            assert set(_frames(result, f"{core}_method_a_cells")) == {core}
            assert isinstance(_frames(result, f"{core}_image")[core], Identity)
            mask_frames = _frames(result, f"{core}_mask")
            assert set(mask_frames) == {f"{core}_mask"}
            assert isinstance(mask_frames[f"{core}_mask"], Identity)

    def test_pipeline_inputs_and_qc_artifacts_are_not_read(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode="tma")
        (fixture.root / "raw" / "cycle_02.czi").write_bytes(b"not an image")
        (fixture.root / "qc" / "coreograph" / "centroidsY-X.txt").unlink()
        (fixture.root / "qc" / "coreograph" / "TMA_MAP.tif").unlink()
        result = mcmicro(fixture.root)
        assert not {"cycle_01", "cycle_01-ffp", "cycle_01-dfp"} & set(result.images)

    def test_methods_and_full_compartments_have_distinct_linked_labels(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, multiple_methods=True, underscored_compartment=True)
        result = mcmicro(fixture.root)
        keys = ["sample_part_7_method_a_cell_mask_ring", "sample_part_7_method_b_cell_mask_ring"]
        assert list(result.labels) == keys
        table = result.tables["table"]
        assert get_table_keys(table) == (keys, "region", "CellID")
        assert table.obs["CellID"].tolist() == [1, 2, 11, 12]

    def test_mcquant_duplicate_markers_are_canonical_features(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, duplicate_markers=True)
        result = mcmicro(fixture.root)
        assert result.images["sample_part_7_image"].coords["c"].to_numpy().tolist() == ["DAPI_1", "CD3", "DAPI_2"]
        table = result.tables["table"]
        assert table.var_names.tolist() == ["DAPI_1", "CD3", "DAPI_2"]
        assert table.var["marker_name"].tolist() == ["DAPI", "CD3", "DAPI"]
        np.testing.assert_array_equal(_expression(table).toarray(), [[10.125, 30.125, 50.125], [20.5, 40.5, 60.5]])

    def test_marker_only_panel_is_supported(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, marker_only=True)
        assert mcmicro(fixture.root).tables["table"].var.columns.tolist() == ["marker_name", "cycle"]

    def test_missing_registered_image_fails_explicitly(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        (fixture.root / "registration" / "sample_part_7.ome.tif").unlink()
        with pytest.raises(ReaderFormatError, match="exactly one registered image"):
            mcmicro(fixture.root)

    def test_string_mode_is_not_a_boolean(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        (fixture.root / "qc" / "params.yml").write_text('workflow:\n  tma: "false"\n', encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=r"workflow\.tma"):
            mcmicro(fixture.root)

    def test_unpaired_quantification_fails(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        source = fixture.root / "quantification" / "sample_part_7--method_a_cells.csv"
        shutil.copy(source, fixture.root / "quantification" / "sample_part_7--method_z_cells.csv")
        with pytest.raises(ReaderFormatError, match="no label"):
            mcmicro(fixture.root)

    def test_label_capacity_and_grid_are_validated_without_pixels(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        labels = fixture.root / "segmentation" / "method_a-sample_part_7" / "cells.ome.tif"
        tifffile.imwrite(labels, np.zeros((64, 80), np.uint8), metadata={"axes": "YX"})
        source = fixture.root / "quantification" / "sample_part_7--method_a_cells.csv"
        source.write_text(source.read_text().replace("\n2,", "\n300,", 1), encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="label contract"):
            mcmicro(fixture.root)
        tifffile.imwrite(labels, np.zeros((10, 10), np.uint32), metadata={"axes": "YX"})
        with pytest.raises(ReaderFormatError, match="image grid"):
            mcmicro(fixture.root)

    @pytest.mark.parametrize(
        ("old", "new", "match"),
        [("10.125", "inf", "nonfinite"), ("custom_note", "CD3", "unique nonempty")],
    )
    def test_invalid_quantification_values_fail(self, tmp_path: Path, old: str, new: str, match: str) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        source = fixture.root / "quantification" / "sample_part_7--method_a_cells.csv"
        source.write_text(source.read_text().replace(old, new, 1), encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=match):
            mcmicro(fixture.root)
