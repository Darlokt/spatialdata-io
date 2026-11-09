"""Preserve valid MCMICRO values while testing explicit scientific corrections."""

from __future__ import annotations

import csv
import shutil
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.mcmicro import mcmicro
from tests.integration.readers.mcmicro._fixtures import write_mcmicro_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from spatialdata import SpatialData

    from tests.integration.readers.mcmicro._fixtures import McmicroFixture


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


def _write_large_ids(fixture: McmicroFixture, identifiers: tuple[int, int]) -> None:
    """Write exact source IDs to uint64 TIFF labels and their matching textual quantification."""
    for sample in fixture.samples:
        height, width = (64, 80) if fixture.mode == "wsi" else ((30, 40) if sample == "1" else (21, 31))
        labels = np.zeros((height, width), dtype=np.uint64)
        labels[2, 2], labels[3, 3] = identifiers
        tifffile.imwrite(
            fixture.root / "segmentation" / f"method_a-{sample}" / "cells.ome.tif",
            labels,
            ome=False,
            metadata={"axes": "YX"},
            photometric="minisblack",
        )
        source = fixture.root / "quantification" / f"{sample}--method_a_cells.csv"
        source.write_text(
            f"CellID,CD3,DAPI,X_centroid,Y_centroid\n{identifiers[0]},1,0,2,2\n{identifiers[1]},0,2,3,3\n",
            encoding="utf-8",
        )


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
        # Without OME sizes the image is in pixels, so it is not placed in the micrometre frame "global".
        assert _frames(result, "sample_part_7_image") == {"image_pixels": Identity()}
        assert _frames(result, "sample_part_7_method_a_cells") == {"image_pixels": Identity()}

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
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[3.5, 3], [75.5, 60]])
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

    @pytest.mark.parametrize("mode", ["wsi", "tma"])
    @pytest.mark.parametrize("legacy_directory", [False, True])
    def test_csv_roles_and_panel_order(
        self, tmp_path: Path, mode: Literal["wsi", "tma"], *, legacy_directory: bool
    ) -> None:
        """Reordered source columns keep panel order, nullable morphology and literal annotations."""
        fixture = write_mcmicro_dataset(tmp_path, mode=mode, duplicate_markers=True, legacy_directory=legacy_directory)
        panel = fixture.root / "markers.csv"
        panel.write_text("marker_name,note\nDAPI,007\nCD3,NA\nDAPI,\n", encoding="utf-8")
        for sample in fixture.samples:
            source = fixture.root / "quantification" / f"{sample}--method_a_cells.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["note", "DAPI_2", "Y_centroid", "CellID", "Area", "CD3", "X_centroid", "DAPI_1"])
                writer.writerow(["007,NA\nNULL", "0.10000000000000002", "2.5", "7", "", "-0.25", "1.5", "0"])
                writer.writerow(["", "0", "4", "3", "12.5", "0", "3.5", "2"])

        tasks: list[object] = []
        with Callback(posttask=lambda key, *_: tasks.append(key)):
            result = mcmicro(fixture.root)

        assert tasks == []
        table = result.tables["table"]
        assert isinstance(table.obs, pd.DataFrame)
        assert table.var_names.tolist() == ["DAPI_1", "CD3", "DAPI_2"]
        assert table.var["note"].tolist() == ["007", "NA", ""]
        assert table.obs["note"].tolist() == ["007,NA\nNULL", ""] * len(fixture.samples)
        assert table.obs["CellID"].tolist() == [7, 3] * len(fixture.samples)
        assert table.obs["Area"].isna().tolist() == [True, False] * len(fixture.samples)
        assert table.obs["Area"].dropna().tolist() == [12.5] * len(fixture.samples)
        np.testing.assert_array_equal(
            _expression(table).toarray(), [[0, -0.25, float("0.10000000000000002")], [2, 0, 0]] * len(fixture.samples)
        )
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[1.5, 2.5], [3.5, 4]] * len(fixture.samples))
        for image in result.images.values():
            assert image.coords["c"].to_numpy().tolist() == table.var_names.tolist()

    @pytest.mark.parametrize("mode", ["wsi", "tma"])
    @pytest.mark.parametrize("largest", [2**63 - 1, 2**63, 2**64 - 1])
    def test_large_label_ids_remain_exact_without_pixel_tasks(
        self, tmp_path: Path, mode: Literal["wsi", "tma"], largest: int
    ) -> None:
        """IDs above float precision retain source values through public linkage to uint64 labels."""
        fixture = write_mcmicro_dataset(tmp_path, mode=mode)
        identifiers = (2**53 + 1, largest)
        _write_large_ids(fixture, identifiers)

        tasks: list[object] = []
        with Callback(posttask=lambda key, *_: tasks.append(key)):
            result = mcmicro(fixture.root)

        assert tasks == []
        assert result.tables["table"].obs["CellID"].tolist() == list(identifiers) * len(fixture.samples)
        for name, labels in result.labels.items():
            if name.endswith("_cells"):
                assert labels.dtype == np.dtype(np.uint64)

    @pytest.mark.parametrize("mode", ["wsi", "tma"])
    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_full_uint64_linkage_persists(self, tmp_path: Path, mode: Literal["wsi", "tma"], storage: str) -> None:
        fixture = write_mcmicro_dataset(tmp_path / "run", mode=mode)
        identifiers = (2**63, 2**64 - 1)
        _write_large_ids(fixture, identifiers)
        result = mcmicro(fixture.root)
        table = result.tables["table"]
        destination = tmp_path / f"roundtrip.{storage}"
        if storage == "h5ad":
            table.write_h5ad(destination)
            restored = ad.read_h5ad(destination)
        else:
            result.write(destination)
            stored = read_zarr(destination)
            restored = stored.tables["table"]
            for name, labels in stored.labels.items():
                if name.endswith("_cells"):
                    assert labels.dtype == np.dtype(np.uint64)

        assert restored.obs["CellID"].dtype == np.dtype(np.uint64)
        assert restored.obs["CellID"].tolist() == list(identifiers) * len(fixture.samples)
        actual_linkage, expected_linkage = get_table_keys(restored), get_table_keys(table)
        np.testing.assert_array_equal(actual_linkage[0], expected_linkage[0])
        assert actual_linkage[1:] == expected_linkage[1:]
        assert restored.var_names.tolist() == table.var_names.tolist()
        np.testing.assert_array_equal(restored.obsm["spatial_pixels"], table.obsm["spatial_pixels"])
        actual = restored.X
        assert isinstance(actual, sparse.csr_array | sparse.csr_matrix)
        expected = _expression(table)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)


class TestMcmicroCorrections:
    """Each legacy defect is replaced by an independently specified contract."""

    def test_cores_and_masks_keep_exact_local_pixel_frames(self, tmp_path: Path) -> None:
        fixture = write_mcmicro_dataset(tmp_path, mode="tma")
        result = mcmicro(fixture.root)
        assert _frames(result, "tma_map") == {"tma_map_pixels": Identity()}
        for core in ("core_1", "core_2"):
            assert _frames(result, f"{core}_image") == {f"{core}_pixels": Identity()}
            assert _frames(result, f"{core}_method_a_cells") == {f"{core}_pixels": Identity()}
            assert _frames(result, f"{core}_mask") == {f"{core}_mask_pixels": Identity()}
        assert "global" not in result.coordinate_systems

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
        with pytest.raises(ReaderFormatError, match="value 300 above 255"):
            mcmicro(fixture.root)
        tifffile.imwrite(labels, np.zeros((10, 10), np.uint32), metadata={"axes": "YX"})
        with pytest.raises(ReaderFormatError, match="image grid"):
            mcmicro(fixture.root)

    @pytest.mark.parametrize(
        ("old", "new", "match"),
        [("10.125", "inf", "nonfinite"), ("custom_note", "CD3", "repeated column")],
    )
    def test_invalid_quantification_values_fail(self, tmp_path: Path, old: str, new: str, match: str) -> None:
        fixture = write_mcmicro_dataset(tmp_path)
        source = fixture.root / "quantification" / "sample_part_7--method_a_cells.csv"
        source.write_text(source.read_text().replace(old, new, 1), encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=match):
            mcmicro(fixture.root)
