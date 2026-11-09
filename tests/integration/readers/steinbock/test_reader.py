"""Exercise supported Steinbock layouts through the public reader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from anndata.tests.helpers import assert_equal
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import Identity, Scale, get_transformation
from xarray import DataTree

from spatialdata_io import __version__
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture

    from tests.integration.readers.steinbock.conftest import SteinbockDataset


def _write_layout(path: Path, name: str = "sample") -> tuple[np.ndarray, np.ndarray]:
    image_directory = path / "ome"
    mask_directory = path / "masks"
    image_directory.mkdir()
    mask_directory.mkdir()

    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.asarray(
        [[[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 2, 2], [0, 0, 0, 2, 2]]],
        dtype=np.uint16,
    )
    tifffile.imwrite(
        image_directory / f"{name}.ome.tiff", image, photometric="minisblack", ome=True, metadata={"axes": "CYX"}
    )
    tifffile.imwrite(mask_directory / f"{name}.tiff", labels, photometric="minisblack")

    obs = pd.DataFrame(
        {
            "image": [f"{name}.tiff", f"{name}.tiff"],
            "Image": [f"{name}.tiff", f"{name}.tiff"],
            "centroid-0": [1.0, 3.0],
            "centroid-1": [1.5, 2.5],
        },
        index=[f"Object 1 in {name}.tiff", f"Object 2 in {name}.tiff"],
    )
    var = pd.DataFrame({"Channel Name": ["A", "B"]}, index=["marker-a", "marker-b"])
    table = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 2], [3, 4]], dtype=np.float32)),
        obs=obs,
        var=var,
    )
    table.write_h5ad(path / "cells.h5ad")
    return image, labels[0]


class TestSteinbock:
    def test_h5ad_observation_column_names_are_sanitized(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        source = ad.read_h5ad(tmp_path / "cells.h5ad")
        source.obs["my note"] = ["x", "y"]
        source.obs["Cell_ID"] = ["c", "d"]
        source.write_h5ad(tmp_path / "cells.h5ad")

        with pytest.warns(UserWarning, match="observation metadata") as caught:
            result = steinbock(tmp_path)

        messages = [str(warning.message) for warning in caught]
        assert (
            "steinbock: the observation metadata of the cell table has column names that a SpatialData table does "
            "not accept; using 'my note' -> 'my_note', 'Cell_ID' -> 'Cell_ID_1'. Rename the source columns to "
            "choose the names explicitly."
        ) in messages
        table = result.tables["table"]
        assert table.obs.columns.tolist() == [
            "image",
            "centroid-0",
            "centroid-1",
            "my_note",
            "Cell_ID_1",
            "cell_id",
            "region",
        ]
        assert table.obs["my_note"].tolist() == ["x", "y"]
        assert table.obs["cell_id"].tolist() == [1, 2]
        result.write(tmp_path / "result.zarr")
        assert read_zarr(tmp_path / "result.zarr").tables["table"].obs.columns.tolist() == table.obs.columns.tolist()

    def test_invalid_h5ad_uns_key_is_sanitized_and_linked(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        source = ad.read_h5ad(tmp_path / "cells.h5ad")
        source.uns["my key"] = {"nested key": "kept"}
        source.write_h5ad(tmp_path / "cells.h5ad")

        with pytest.warns(UserWarning, match="SpatialData table does not accept") as caught:
            result = steinbock(tmp_path)

        assert (
            "steinbock: the unstructured metadata of the cell table has names that a SpatialData table does not "
            "accept as uns keys; using 'my key' -> 'my_key'. Rename them in the source to choose the names explicitly."
        ) in [str(warning.message) for warning in caught]

        table = result.tables["table"]
        assert "my key" not in table.uns
        assert table.uns["my_key"] == {"nested key": "kept"}
        assert get_table_keys(table) == ("sample_labels", "region", "cell_id")
        result.write(tmp_path / "result.zarr")
        assert read_zarr(tmp_path / "result.zarr").tables["table"].uns["my_key"] == {"nested key": "kept"}

    def test_file_derived_sample_and_feature_column_names_are_sanitized(self, tmp_path: Path) -> None:
        _write_layout(tmp_path, name="sample 1")
        source = ad.read_h5ad(tmp_path / "cells.h5ad")
        source.var["CD45 RO"] = ["x", "y"]
        source.write_h5ad(tmp_path / "cells.h5ad")
        # A macOS AppleDouble companion is not a sample.
        (tmp_path / "ome" / "._sample 1.ome.tiff").write_bytes(b"\x00\x05\x16\x07")

        with pytest.warns(UserWarning, match="sample_1") as caught:
            result = steinbock(tmp_path)

        assert [str(warning.message) for warning in caught] == [
            (
                "steinbock: the sample file name 'sample 1' is not a usable SpatialData element name; using "
                "'sample_1'. Rename the source files to choose the name explicitly."
            ),
            (
                "steinbock: the feature metadata of the cell table has column names that a SpatialData table does not "
                "accept; using 'Channel Name' -> 'Channel_Name', 'CD45 RO' -> 'CD45_RO'. Rename the source columns to "
                "choose the names explicitly."
            ),
        ]
        assert list(result.images) == ["sample_1_image"]
        assert list(result.labels) == ["sample_1_labels"]
        assert result.coordinate_systems == ["sample_1_pixels"]
        table = result.tables["table"]
        assert table.obs["image"].tolist() == ["sample 1.tiff", "sample 1.tiff"]
        assert table.obs["region"].tolist() == ["sample_1_labels", "sample_1_labels"]
        assert table.obs["cell_id"].tolist() == [1, 2]
        assert table.var.columns.tolist() == ["Channel_Name", "CD45_RO"]
        assert table.var["CD45_RO"].tolist() == ["x", "y"]
        assert table.uns["spatialdata_io_positions"]["coordinates"]["spatial_pixels"]["frame"] == (
            "sample_1_regionprops_indices"
        )

        result.write(tmp_path / "result.zarr")
        restored = read_zarr(tmp_path / "result.zarr")
        assert sorted(restored.images) == ["sample_1_image"]
        assert restored.tables["table"].var.columns.tolist() == ["Channel_Name", "CD45_RO"]

    """Characterize sample pairing, lazy rasters, cleanup, and linkage."""

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_inherited_slots_and_raw_survive_persistence(self, tmp_path: Path, storage: str) -> None:
        """Keep inherited values aligned, including raw's independent feature vocabulary."""
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        rows, features = source.shape
        source.obs["nullable_annotation"] = pd.array([1] + [None] * (rows - 1), dtype="Int16")
        source.layers["counts"] = sparse.csc_matrix(np.arange(rows * features).reshape(rows, features))
        source.layers["dense_metadata"] = np.arange(rows * features, dtype=np.float32).reshape(rows, features)
        source.layers["sparse_metadata"] = sparse.csc_matrix(source.layers["dense_metadata"])
        source.obsm["inherited"] = np.arange(rows * 2).reshape(rows, 2)
        source.varm["inherited"] = np.arange(features * 2).reshape(features, 2)
        source.obsp["inherited"] = sparse.csr_matrix(np.arange(rows * rows).reshape(rows, rows))
        source.varp["inherited"] = sparse.csr_matrix(np.arange(features * features).reshape(features, features))
        source.uns["inherited"] = {"nested": {"meaning": "unchanged", "number": 3}}
        raw = ad.AnnData(
            sparse.csr_matrix(np.arange(rows * 4, dtype=np.int32).reshape(rows, 4)),
            obs=pd.DataFrame(index=source.obs_names),
            var=pd.DataFrame({"annotation": [1.0, np.nan, 3.0, 4.0]}, index=["raw-z", "raw-a", "raw-b", "raw-c"]),
        )
        source.raw = raw
        source.write_h5ad(path)
        original = ad.read_h5ad(path)

        result = steinbock(tmp_path)
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert isinstance(table.layers["counts"], sparse.csr_array)
        assert isinstance(table.layers["sparse_metadata"], sparse.csc_matrix)
        destination = tmp_path / f"inherited.{storage}"
        if storage == "h5ad":
            table.write_h5ad(destination)
            restored = ad.read_h5ad(destination)
        else:
            result.write(destination)
            restored = read_zarr(destination).tables["table"]

        order = [0, 1]
        expected = original[order].copy()
        for actual in (table, restored):
            assert actual.obs_names.tolist() == expected.obs_names.tolist()
            assert isinstance(actual.obs, pd.DataFrame)
            assert isinstance(expected.obs, pd.DataFrame)
            pd.testing.assert_series_equal(actual.obs["nullable_annotation"], expected.obs["nullable_annotation"])
            for name in ("counts", "dense_metadata", "sparse_metadata"):
                assert_equal(actual.layers[name], expected.layers[name], exact=True)
            for mapping in ("obsm", "varm", "obsp", "varp"):
                assert_equal(getattr(actual, mapping)["inherited"], getattr(expected, mapping)["inherited"], exact=True)
            assert_equal(actual.raw, expected.raw, exact=True)
            assert actual.raw is not None
            assert actual.raw.obs_names.tolist() == expected.obs_names.tolist()
            assert actual.uns["inherited"] == expected.uns["inherited"]
            assert get_table_keys(actual) == get_table_keys(table)
            np.testing.assert_array_equal(actual.obsm["spatial_pixels"], table.obsm["spatial_pixels"])
        unchanged = ad.read_h5ad(path)
        assert unchanged.obs_names.equals(original.obs_names)
        assert "region" not in unchanged.obs

    @pytest.mark.parametrize("masks_dir", ["masks", "masks_ilastik"])
    def test_imc_sidecar_exact_sample_calibration_and_native_centroids(
        self, steinbock_dataset: SteinbockDataset, masks_dir: str
    ) -> None:
        root = steinbock_dataset.root
        # This producer-shaped sidecar includes an unrelated acquisition and
        # deliberately differs from TIFF and table row order.
        (root / "images.csv").write_text(
            "extra,acquisition_height_um,image,width_px,height_px,acquisition_width_um\n"
            "unused,99,other.tiff,12,8,99\n"
            "kept,,sample-b.tiff,12,8,\n"
            "kept,20,sample-a.tiff,12,8,6\n",
            encoding="utf-8",
        )
        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            result = steinbock(root, masks_dir=masks_dir)
        assert not tasks
        for element in (result.images["sample-a_image"], result.labels["sample-a_labels"]):
            transform = get_transformation(element, to_coordinate_system="sample-a")
            assert isinstance(transform, Scale)
            np.testing.assert_array_equal(transform.scale, [0.5, 2.5])
        assert get_transformation(result.images["sample-b_image"], get_all=True) == {"sample-b_pixels": Identity()}
        assert sorted(result.coordinate_systems) == ["sample-a", "sample-b_pixels"]
        np.testing.assert_array_equal(
            result.tables["table"].obsm["spatial_pixels"], [[3, 1.5], [8.5, 5], [8.5, 5], [3, 1.5]]
        )
        np.testing.assert_array_equal(
            result.images["sample-a_image"].data.compute(), steinbock_dataset.images["sample-a"]
        )

    @pytest.mark.parametrize(
        "row",
        [
            "sample-a.tiff,12,8,6,",
            "sample-a.tiff,12,8,,20",
            "sample-a.tiff,12,8,0,20",
            "sample-a.tiff,12,8,nan,20",
            "sample-a.tiff,12,8,inf,20",
            "sample-a.tiff,12,8,-1,20",
            "sample-a.tiff,11,8,6,20",
            "sample-a.tiff,12,9,6,20",
            "sample-a.tiff,12.5,8,6,20",
            "sample-a.tiff,0,8,6,20",
            "sample-a.tiff,12,8,1e999,20",
            "sample-a.tiff,12,8,1e-999,20",
        ],
    )
    def test_malformed_selected_imc_metadata_fails(self, steinbock_dataset: SteinbockDataset, row: str) -> None:
        root = steinbock_dataset.root
        (root / "images.csv").write_text(
            "image,width_px,height_px,acquisition_width_um,acquisition_height_um\n" + row + "\n", encoding="utf-8"
        )
        with pytest.raises(ReaderFormatError, match=r"images\.csv"):
            steinbock(root)

    @pytest.mark.parametrize(
        "text",
        [
            "image,width_px,height_px,image\nsample-a.tiff,12,8,sample-a.tiff\n",
            "image,width_px,height_px\nsample-a.tiff,12,8\nsample-a.tiff,12,8\n",
            "image,width_px\nsample-a.tiff,12\n",
            "image,width_px,height_px,acquisition_width_um\nsample-a.tiff,12,8,6\n",
            "image,width_px,height_px\nsample-a.tiff,12,8,extra\n",
        ],
    )
    def test_ambiguous_or_malformed_sidecar_schema(self, steinbock_dataset: SteinbockDataset, text: str) -> None:
        (steinbock_dataset.root / "images.csv").write_text(text, encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=r"images\.csv"):
            steinbock(steinbock_dataset.root)

    def test_missing_sidecar_row_and_physical_columns_keep_pixels(self, steinbock_dataset: SteinbockDataset) -> None:
        root = steinbock_dataset.root
        (root / "images.csv").write_text("image,width_px,height_px\nother.tiff,12,8\n", encoding="utf-8")
        result = steinbock(root)
        for name in ("sample-a", "sample-b"):
            assert get_transformation(result.images[f"{name}_image"], get_all=True) == {f"{name}_pixels": Identity()}

    @pytest.mark.parametrize("physical_x", [0.5, 0.4])
    def test_sidecar_and_ome_agreement(self, steinbock_dataset: SteinbockDataset, physical_x: float) -> None:
        root = steinbock_dataset.root
        (root / "images.csv").write_text(
            "image,width_px,height_px,acquisition_width_um,acquisition_height_um\nsample-a.tiff,12,8,6,20\n",
            encoding="utf-8",
        )
        tifffile.imwrite(
            root / "ome" / "sample-a.ome.tiff",
            steinbock_dataset.images["sample-a"],
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "PhysicalSizeX": physical_x, "PhysicalSizeY": 2.5},
        )
        if physical_x == 0.4:
            with pytest.raises(ReaderFormatError, match="calibration"):
                steinbock(root)
        else:
            result = steinbock(root)
            assert isinstance(
                get_transformation(result.labels["sample-a_labels"], to_coordinate_system="sample-a"), Scale
            )

    def test_single_ome_calibrated_sample_uses_global(self, tmp_path: Path) -> None:
        image, _ = _write_layout(tmp_path)
        tifffile.imwrite(
            tmp_path / "ome" / "sample.ome.tiff",
            image,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "PhysicalSizeX": 300, "PhysicalSizeXUnit": "nm", "PhysicalSizeY": 0.7},
        )
        result = steinbock(tmp_path)
        assert result.coordinate_systems == ["global"]
        for element in (result.images["sample_image"], result.labels["sample_labels"]):
            transforms = get_transformation(element, get_all=True)
            assert isinstance(transforms, dict)
            assert list(transforms) == ["global"]
            transform = transforms["global"]
            assert isinstance(transform, Scale)
            np.testing.assert_array_equal(transform.scale, [0.3, 0.7])

    def test_partial_ome_image_and_mask_conflict(self, steinbock_dataset: SteinbockDataset) -> None:
        root = steinbock_dataset.root
        tifffile.imwrite(
            root / "ome" / "sample-a.ome.tiff",
            steinbock_dataset.images["sample-a"],
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "PhysicalSizeX": 0.3},
        )
        tifffile.imwrite(
            root / "masks" / "sample-a.tiff",
            steinbock_dataset.masks["sample-a"],
            ome=True,
            metadata={"axes": "YX", "PhysicalSizeX": 0.4},
        )
        with pytest.raises(ReaderFormatError, match="calibration"):
            steinbock(root)

    def test_synthetic_layout_preserves_public_contract(self, tmp_path: Path) -> None:
        image, labels = _write_layout(tmp_path)

        result = steinbock(tmp_path)

        assert result.coordinate_systems == ["sample_pixels"]
        for element in (result.images["sample_image"], result.labels["sample_labels"]):
            assert get_transformation(element, get_all=True) == {"sample_pixels": Identity()}
        assert list(result.images) == ["sample_image"]
        assert list(result.labels) == ["sample_labels"]
        assert list(result.tables) == ["table"]
        assert not result.points
        assert not result.shapes
        assert result.attrs == {
            "spatialdata_io_reader": "steinbock",
            "spatialdata_io_software_version": __version__,
        }

        image_element = result.images["sample_image"]
        assert image_element.shape == (2, 4, 5)
        assert image_element.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image_element.data[:, 1, 2].compute(), image[:, 1, 2])

        labels_element = result.labels["sample_labels"]
        assert labels_element.shape == (4, 5)
        assert labels_element.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(labels_element.data[1:3, 1:4].compute(), labels[1:3, 1:4])

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        np.testing.assert_array_equal(table.X.toarray(), np.asarray([[1, 2], [3, 4]], dtype=np.float32))
        assert table.obs_names.tolist() == ["Object 1 in sample.tiff", "Object 2 in sample.tiff"]
        assert table.obs["cell_id"].tolist() == [1, 2]
        assert table.obs["region"].tolist() == ["sample_labels", "sample_labels"]
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], np.asarray([[1.5, 1.0], [2.5, 3.0]]))
        assert table.var.columns.tolist() == ["Channel_Name"]
        assert get_table_keys(table) == ("sample_labels", "region", "cell_id")

    @pytest.mark.parametrize("masks_dir", [None, "masks_ilastik"])
    def test_two_samples_preserve_compound_ids_and_cleanup(
        self, steinbock_dataset: SteinbockDataset, masks_dir: str | None
    ) -> None:
        """The default reads masks/; masks_dir selects another segmentation."""
        if masks_dir is not None:
            # This segmentation fixture shifts each object down by one pixel.
            # Its cell table must describe the same selected segmentation.
            path = steinbock_dataset.root / "cells.h5ad"
            source = ad.read_h5ad(path)
            assert isinstance(source.obs, pd.DataFrame)
            source.obs["centroid-0"] += 1
            with ad.settings.override(disallow_forward_slash_in_h5ad=False):
                source.write_h5ad(path)

        result = (
            steinbock(steinbock_dataset.root)
            if masks_dir is None
            else steinbock(steinbock_dataset.root, masks_dir=masks_dir)
        )

        assert list(result.images) == ["sample-a_image", "sample-b_image"]
        assert list(result.labels) == ["sample-a_labels", "sample-b_labels"]
        masks = steinbock_dataset.masks if masks_dir is None else steinbock_dataset.ilastik
        for sample in ("sample-a", "sample-b"):
            image = result.images[f"{sample}_image"]
            labels = result.labels[f"{sample}_labels"]
            assert image.dims == ("c", "y", "x")
            assert image.shape == (2, 8, 12)
            assert image.dtype == np.dtype(np.float32)
            assert image.coords["c"].to_numpy().tolist() == ["DNA", "CD3"]
            assert labels.dims == ("y", "x")
            assert labels.dtype == np.dtype(np.uint16)
            np.testing.assert_array_equal(image.data.compute(), steinbock_dataset.images[sample])
            np.testing.assert_array_equal(labels.data.compute(), masks[sample])
            assert get_transformation(image, get_all=True) == {f"{sample}_pixels": Identity()}
            assert get_transformation(labels, get_all=True) == {f"{sample}_pixels": Identity()}

        table = result.tables["table"]
        assert table.obs_names.tolist() == [
            "Object 2 in sample-b.tiff",
            "Object 7 in sample-a.tiff",
            "Object 7 in sample-b.tiff",
            "Object 2 in sample-a.tiff",
        ]
        assert table.obs["cell_id"].tolist() == [2, 7, 7, 2]
        assert table.obs["region"].tolist() == ["sample-b_labels", "sample-a_labels"] * 2
        assert table.var_names.tolist() == ["DNA", "CD3"]
        assert table.var.columns.tolist() == ["Channel_Name", "Final_Concentration"]
        assert table.var["Final_Concentration"].tolist() == ["1:100", "1:200"]
        assert "Image" not in table.obs
        assert table.obs["image"].tolist() == ["sample-b.tiff", "sample-a.tiff"] * 2
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        np.testing.assert_array_equal(table.X.toarray(), [[1, 0], [0, 4], [5, 6], [7, 8]])

        # Upstream measurements are (row, column). The linked table stores XY.
        expected_centroids = np.asarray([[3.0, 1.5], [8.5, 5.0], [8.5, 5.0], [3.0, 1.5]])
        if masks_dir is not None:
            expected_centroids[:, 1] += 1
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], expected_centroids)
        assert get_table_keys(table) == (["sample-b_labels", "sample-a_labels"], "region", "cell_id")

    def test_unmatched_image_fails(self, steinbock_dataset: SteinbockDataset) -> None:
        (steinbock_dataset.root / "masks" / "sample-a.tiff").unlink()

        with pytest.raises(ReaderFormatError, match=r"paired|match|mask"):
            steinbock(steinbock_dataset.root)

    @pytest.mark.parametrize("channels", [3, 4])
    def test_arbitrary_ome_channels_preserve_scientific_names(
        self, steinbock_dataset: SteinbockDataset, channels: int
    ) -> None:
        image = np.arange(channels * 8 * 12, dtype=np.float32).reshape(channels, 8, 12)
        names = ["DNA", "CD3", "CD20", "CD45"][:channels]
        tifffile.imwrite(
            steinbock_dataset.root / "ome" / "sample-a.ome.tiff",
            image,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": names}},
        )

        result = steinbock(steinbock_dataset.root)
        element = result.images["sample-a_image"]
        assert element.coords["c"].to_numpy().tolist() == names
        np.testing.assert_array_equal(element.data.compute(), image)

    def test_rgb_image_is_rejected(self, steinbock_dataset: SteinbockDataset) -> None:
        """An interleaved RGB OME-TIFF is not a Steinbock channel stack and gets no colour channels."""
        path = steinbock_dataset.root / "ome" / "sample-a.ome.tiff"
        tifffile.imwrite(path, np.zeros((8, 12, 3), dtype=np.uint8), ome=True, photometric="rgb")

        with pytest.raises(ReaderFormatError, match="Axis 's' has length 3") as caught:
            steinbock(steinbock_dataset.root)

        assert caught.value.context.artifact == "OME image"
        assert caught.value.path == path

    def test_unmatched_mask_fails(self, steinbock_dataset: SteinbockDataset) -> None:
        tifffile.imwrite(
            steinbock_dataset.root / "masks" / "orphan.tiff",
            np.zeros((8, 12), dtype=np.uint16),
        )

        with pytest.raises(ReaderFormatError, match=r"paired|match|image"):
            steinbock(steinbock_dataset.root)

    def test_missing_optional_duplicate_image_column_is_supported(self, steinbock_dataset: SteinbockDataset) -> None:
        table_file = steinbock_dataset.root / "cells.h5ad"
        table = ad.read_h5ad(table_file)
        assert isinstance(table.obs, pd.DataFrame)
        del table.obs["Image"]
        with ad.settings.override(disallow_forward_slash_in_h5ad=False):
            table.write_h5ad(table_file)

        result = steinbock(steinbock_dataset.root)

        assert "Image" not in result.tables["table"].obs

    @pytest.mark.parametrize(
        ("dtype", "identifier"),
        [("int64", 7), ("int64", 2**53 + 1), ("int64", 2**63 - 1), ("uint64", 2**63), ("uint64", 2**64 - 1)],
    )
    def test_large_ids_and_written_store_remain_exact(self, tmp_path: Path, dtype: str, identifier: int) -> None:
        _write_layout(tmp_path)
        identifiers = [identifier, 2]
        mask = np.zeros((4, 5), dtype=dtype)
        mask[1, 1], mask[3, 2] = identifiers
        tifffile.imwrite(tmp_path / "masks" / "sample.tiff", mask, photometric="minisblack")
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs_names = [f"Object {identifier} in sample.tiff" for identifier in identifiers]
        source.write_h5ad(path)

        tasks: list[object] = []
        with Callback(posttask=lambda *args: tasks.append(args)):
            result = steinbock(tmp_path)
        assert tasks == []
        table = result.tables["table"]
        assert isinstance(table.obs, pd.DataFrame)
        assert table.obs["cell_id"].tolist() == identifiers
        assert table.obs["cell_id"].dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(result.labels["sample_labels"].data.compute(), mask)

        table.write_h5ad(tmp_path / "linked.h5ad")
        restored_table = ad.read_h5ad(tmp_path / "linked.h5ad")
        assert isinstance(restored_table.obs, pd.DataFrame)
        pd.testing.assert_series_equal(restored_table.obs["cell_id"], table.obs["cell_id"], check_exact=True)
        assert get_table_keys(restored_table) == get_table_keys(table)
        target = tmp_path / "result.zarr"
        result.write(target)
        restored = read_zarr(target)
        restored_table = restored.tables["table"]
        assert isinstance(restored_table.obs, pd.DataFrame)
        pd.testing.assert_series_equal(restored_table.obs["cell_id"], table.obs["cell_id"], check_exact=True)
        assert get_table_keys(restored_table) == get_table_keys(table)
        np.testing.assert_array_equal(restored.labels["sample_labels"].data.compute(), mask)

    def test_sparse_expression_layers_and_owned_metadata(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        rows, columns = 1000, 20_000
        expression = sparse.csr_array(
            (np.ones(rows, dtype=np.float32), (np.arange(rows), np.arange(rows))), shape=(rows, columns)
        )
        source = ad.AnnData(
            X=expression,
            obs=pd.DataFrame(
                {"image": ["sample.tiff"] * rows, "centroid-0": np.ones(rows), "centroid-1": np.ones(rows)},
                index=[f"Object {index + 1} in sample.tiff" for index in range(rows)],
            ),
            var=pd.DataFrame(index=[f"marker-{index}" for index in range(columns)]),
        )
        source.layers["counts"] = expression
        metadata = np.zeros((rows, columns), dtype=np.uint8)
        source.layers["metadata"] = metadata
        source.obsp["neighbors"] = sparse.eye_array(rows, format="csr")
        source.write_h5ad(path)
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("expression densified"))
        mocker.patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("expression densified"))
        before = path.read_bytes()

        result = steinbock(tmp_path)

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.shape == (rows, columns)
        assert table.X.nnz == rows
        assert table.X.has_canonical_format
        assert table.X.dtype == np.dtype(np.float32)
        assert isinstance(table.layers["counts"], sparse.csr_array)
        assert isinstance(table.layers["metadata"], np.ndarray)
        np.testing.assert_array_equal(table.layers["metadata"], metadata)
        assert "neighbors" in table.obsp
        assert path.read_bytes() == before

    @pytest.mark.parametrize("invalid", ["duplicate", "nonfinite", "nonfinite_counts"])
    def test_final_linkage_and_expression_reject_invalid_values(self, tmp_path: Path, invalid: str) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        table = ad.read_h5ad(path)
        if invalid == "duplicate":
            table.obs_names = ["Object 1 in sample.tiff", "Object 01 in sample.tiff"]
        elif invalid == "nonfinite":
            table.X = np.asarray([[1, np.nan], [2, 3]])
        else:
            table.layers["counts"] = np.asarray([[1, np.inf], [2, 3]])
        table.write_h5ad(path)

        with pytest.raises(ValueError, match=r"duplicate|finite"):
            steinbock(tmp_path)

    @pytest.mark.parametrize("factors", [[], [2]])
    def test_image_and_mask_share_pyramids_and_chunks(self, tmp_path: Path, factors: list[int]) -> None:
        image, labels = _write_layout(tmp_path)

        result = steinbock(tmp_path, scale_factors=factors, chunks=(1, 2, 3))
        image_element = result.images["sample_image"]
        label_element = result.labels["sample_labels"]
        assert isinstance(image_element, DataTree) == bool(factors)
        assert isinstance(label_element, DataTree) == bool(factors)
        if isinstance(image_element, DataTree) and isinstance(label_element, DataTree):
            assert list(image_element.children) == list(label_element.children)
            for level in image_element.children:
                assert image_element[level]["image"].shape[1:] == label_element[level]["image"].shape
        image_base = image_element["scale0"]["image"] if isinstance(image_element, DataTree) else image_element
        labels_base = label_element["scale0"]["image"] if isinstance(label_element, DataTree) else label_element
        np.testing.assert_array_equal(image_base.data.compute(), image)
        np.testing.assert_array_equal(labels_base.data.compute(), labels)
        assert image_base.data.chunksize == (1, 2, 3)
        assert labels_base.data.chunksize == (2, 3)

    def test_imported_annotation_types_and_missingness_are_preserved(self, tmp_path: Path) -> None:
        """Retain imported extension metadata while enriching the owned table."""
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs["nullable_count"] = pd.array([2**53 + 1, None], dtype="Int64")
        source.obs["nullable_flag"] = pd.array([True, None], dtype="boolean")
        source.obs["classification"] = pd.Categorical(["T", "B"], categories=["B", "T"], ordered=True)
        source.var["confidence"] = np.asarray([0.25] * source.n_vars, dtype=np.float32)
        source.var["nullable_rank"] = pd.array([1] + [None] * (source.n_vars - 1), dtype="UInt16")
        source.write_h5ad(path)
        original = ad.read_h5ad(path)

        result = steinbock(tmp_path)

        table = result.tables["table"]
        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(table.var, pd.DataFrame)
        assert isinstance(original.obs, pd.DataFrame)
        assert isinstance(original.var, pd.DataFrame)
        expected_obs = original.obs.iloc[[0, 1]]
        for column in ("nullable_count", "nullable_flag", "classification"):
            pd.testing.assert_series_equal(table.obs[column], expected_obs[column], check_exact=True)
        for column in ("confidence", "nullable_rank"):
            pd.testing.assert_series_equal(table.var[column], original.var[column], check_exact=True)
        unchanged = ad.read_h5ad(path).obs
        assert isinstance(unchanged, pd.DataFrame)
        pd.testing.assert_frame_equal(unchanged, original.obs, check_exact=True)

        table.write_h5ad(tmp_path / "roundtrip.h5ad")
        restored = ad.read_h5ad(tmp_path / "roundtrip.h5ad")
        assert isinstance(restored.obs, pd.DataFrame)
        assert isinstance(restored.var, pd.DataFrame)
        pd.testing.assert_frame_equal(restored.obs, table.obs, check_exact=True)
        pd.testing.assert_frame_equal(restored.var, table.var, check_exact=True)

        result.write(tmp_path / "metadata.zarr")
        restored = read_zarr(tmp_path / "metadata.zarr").tables["table"]
        assert isinstance(restored.obs, pd.DataFrame)
        assert isinstance(restored.var, pd.DataFrame)
        pd.testing.assert_frame_equal(restored.obs, table.obs, check_exact=True)
        pd.testing.assert_frame_equal(restored.var, table.var, check_exact=True)

    @pytest.mark.parametrize("field", ["region", "cell_id"])
    @pytest.mark.parametrize("conflicting", [False, True])
    def test_reserved_output_columns(self, tmp_path: Path, field: str, *, conflicting: bool) -> None:
        """Reserved output columns."""
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        if field == "region":
            source.obs[field] = ["other" if conflicting else "sample_labels"] * 2
        else:
            source.obs[field] = [1, 1 if conflicting else 2]
        source.write_h5ad(path)

        before = path.read_bytes()
        with pytest.raises(ReaderFormatError, match=f"occupied.*{field}"):
            steinbock(tmp_path)
        assert path.read_bytes() == before

    @pytest.mark.parametrize("slot", ["X", "counts"])
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize("case", [("int8", 100), ("uint8", 150)])
    def test_integer_duplicate_overflow_promotes_before_reduction(
        self, tmp_path: Path, mocker: MockerFixture, slot: str, storage: str, case: tuple[str, int]
    ) -> None:
        """Preserve actual H5AD duplicate sums by promoting each affected expression slot."""
        dtype, value = case
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        constructor = sparse.csr_array if storage == "csr" else sparse.csc_array
        expression = constructor(
            (
                np.asarray([value, value, 7], dtype=dtype),
                np.asarray([0, 0, 1], dtype=np.int64),
                np.asarray([0, 2, 3], dtype=np.int64),
            ),
            shape=(2, 2),
        )
        if slot == "X":
            source.X = expression
        else:
            source.layers[slot] = expression
        source.write_h5ad(path)
        imported = ad.read_h5ad(path)
        actual = imported.X if slot == "X" else imported.layers[slot]
        assert isinstance(actual, (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array))
        assert not actual.has_canonical_format
        np.testing.assert_array_equal(actual.data, [value, value, 7])
        before = path.read_bytes()
        for container in (sparse.csr_array, sparse.csc_array, sparse.csr_matrix, sparse.csc_matrix):
            mocker.patch.object(container, "toarray", side_effect=AssertionError("expression densified"))

        with pytest.warns(UserWarning, match="combined.*promoted") as reported:
            table = steinbock(tmp_path).tables["table"]
        actual = table.X if slot == "X" else table.layers[slot]
        assert isinstance(actual, sparse.csr_array)
        assert actual.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(actual.data, [2 * value, 7])
        # The fixture's "Channel Name" feature column adds the column-name warning.
        (summed,) = (warning for warning in reported if "combined" in str(warning.message))
        assert slot in str(summed.message)
        assert path.name in str(summed.message)
        assert summed.filename == __file__
        assert path.read_bytes() == before

    @pytest.mark.parametrize("slot", ["X", "counts"])
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(("dtype", "value"), [("int16", 100), ("uint64", 2**53 + 1)])
    def test_safe_integer_duplicates_preserve_exact_values_and_dtype(
        self, tmp_path: Path, slot: str, storage: str, dtype: str, value: int
    ) -> None:
        """Canonicalize safe imported duplicates without float promotion or widening."""
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        constructor = sparse.csr_array if storage == "csr" else sparse.csc_array
        expression = constructor(
            (
                np.asarray([value, value, 7], dtype=dtype),
                np.asarray([0, 0, 1], dtype=np.int64),
                np.asarray([0, 2, 3], dtype=np.int64),
            ),
            shape=(2, 2),
        )
        if slot == "X":
            source.X = expression
        else:
            source.layers[slot] = expression
        source.write_h5ad(path)
        imported = ad.read_h5ad(path)
        actual = imported.X if slot == "X" else imported.layers[slot]
        assert isinstance(actual, (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array))
        assert not actual.has_canonical_format

        table = steinbock(tmp_path).tables["table"]

        result = table.X if slot == "X" else table.layers[slot]
        assert isinstance(result, sparse.csr_array)
        assert not isinstance(result, sparse.spmatrix)
        assert result.has_canonical_format
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.data, np.asarray([2 * value, 7], dtype=dtype))
        np.testing.assert_array_equal(result.indices, [0, 1])
        np.testing.assert_array_equal(result.indptr, [0, 1, 2])

    @pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
    def test_nullable_numeric_centroid_annotations(self, tmp_path: Path, dtype: str) -> None:
        """Normalize genuine nullable H5AD coordinates while retaining source fields."""
        _write_layout(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs["centroid-0"] = pd.Series([1, 3], dtype=dtype).array
        source.obs["centroid-1"] = pd.Series([2, 4], dtype=dtype).array
        source.write_h5ad(path)
        original = ad.read_h5ad(path)

        table = steinbock(tmp_path).tables["table"]

        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(original.obs, pd.DataFrame)
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[2, 1], [4, 3]])
        assert table.obsm["spatial_pixels"].dtype == np.dtype(dtype.lower())
        for column in ("centroid-0", "centroid-1"):
            pd.testing.assert_series_equal(table.obs[column], original.obs[column], check_exact=True)


class TestSteinbockPathPolicy:
    """Discovered artifacts follow the dataset root's symlink policy."""

    def test_out_of_root_symlinks_require_the_relaxed_policy(
        self, steinbock_dataset: SteinbockDataset, link_outside_root: Callable[[Path], Path]
    ) -> None:
        root = steinbock_dataset.root
        expected = steinbock(root)
        link_outside_root(root / "cells.h5ad")
        link_outside_root(root / "ome" / "sample-a.ome.tiff")
        link_outside_root(root / "masks" / "sample-a.tiff")

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            steinbock(root)

        assert_spatial_data_objects_are_identical(steinbock(root, path_policy="relaxed"), expected)

    @pytest.mark.parametrize("policy", ["strict", "relaxed"])
    def test_symlinked_table_file_follows_the_policy(
        self,
        steinbock_dataset: SteinbockDataset,
        link_outside_root: Callable[[Path], Path],
        policy: Literal["strict", "relaxed"],
    ) -> None:
        root = steinbock_dataset.root
        expected = steinbock(root)
        (root / "cells.h5ad").rename(root / "objects.h5ad")
        link_outside_root(root / "objects.h5ad")

        if policy == "strict":
            with pytest.raises(ReaderFormatError, match="path_policy='relaxed'") as caught:
                steinbock(root, table_file="objects.h5ad")
            assert caught.value.context.artifact == "cell table"
        else:
            result = steinbock(root, table_file="objects.h5ad", path_policy=policy)
            assert_spatial_data_objects_are_identical(result, expected)


class TestSteinbockSources:
    """``masks_dir`` selects a contained mask directory; ``table_file`` names the table."""

    def test_custom_masks_dir(self, steinbock_dataset: SteinbockDataset) -> None:
        root = steinbock_dataset.root
        (root / "masks").rename(root / "segmentation")

        with pytest.raises(FileNotFoundError) as caught:
            steinbock(root)
        assert caught.value.filename == str(root / "masks")

        result = steinbock(root, masks_dir="segmentation")
        for sample, mask in steinbock_dataset.masks.items():
            np.testing.assert_array_equal(result.labels[f"{sample}_labels"].data.compute(), mask)

    def test_nested_masks_dir(self, steinbock_dataset: SteinbockDataset) -> None:
        root = steinbock_dataset.root
        (root / "segmentations").mkdir()
        (root / "masks").rename(root / "segmentations" / "deepcell")

        result = steinbock(root, masks_dir="segmentations/deepcell")

        np.testing.assert_array_equal(
            result.labels["sample-a_labels"].data.compute(), steinbock_dataset.masks["sample-a"]
        )

    def test_masks_dir_symlink_outside_root_follows_the_policy(
        self, steinbock_dataset: SteinbockDataset, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        root = steinbock_dataset.root
        expected = steinbock(root)
        outside = tmp_path_factory.mktemp("external") / "masks"
        (root / "masks").rename(outside)
        (root / "masks").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'") as caught:
            steinbock(root)
        assert caught.value.context.artifact == "masks"
        assert_spatial_data_objects_are_identical(steinbock(root, path_policy="relaxed"), expected)

    @pytest.mark.parametrize("absolute", [False, True])
    def test_custom_table_file(self, steinbock_dataset: SteinbockDataset, tmp_path: Path, *, absolute: bool) -> None:
        root = steinbock_dataset.root
        expected = steinbock(root)
        table_file: Path | str = tmp_path / "objects.h5ad" if absolute else "exports/objects.h5ad"
        (root / "exports").mkdir()
        (root / "cells.h5ad").rename(root / table_file)

        with pytest.raises(FileNotFoundError):
            steinbock(root)
        assert_spatial_data_objects_are_identical(steinbock(root, table_file=table_file), expected)

    def test_relative_table_file_cannot_leave_the_root(self, steinbock_dataset: SteinbockDataset) -> None:
        with pytest.raises(ReaderFormatError, match="root"):
            steinbock(steinbock_dataset.root, table_file="../cells.h5ad")
