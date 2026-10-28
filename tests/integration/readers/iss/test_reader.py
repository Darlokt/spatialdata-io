"""Exercise the real ISS TIFF/h5ad workflow and exact scientific output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io import __version__
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.iss import iss
from tests.integration.readers.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from tests.integration.readers.iss.conftest import IssDataset


class TestIss:
    """Read genuine artifacts without compatibility shims or mocked parsers."""

    @pytest.mark.parametrize("instance_key", [None, "segmentation_id"])
    def test_artifacts_preserve_values_order_transforms_and_linkage(
        self, iss_dataset: IssDataset, instance_key: str | None
    ) -> None:
        result = iss(
            iss_dataset.root,
            raw_image_file="raw.tiff",
            labels_file="labels.tiff",
            table_file="cells.h5ad",
            instance_key=instance_key,
            dataset_id="fov",
        )

        assert list(result.images) == ["fov_raw_image"]
        assert list(result.labels) == ["fov_labels_image"]
        assert list(result.tables) == ["table"]
        assert not result.points
        assert not result.shapes
        assert result.attrs == {
            "spatialdata_io_reader": "iss",
            "spatialdata_io_software_version": __version__,
        }
        image, labels = result.images["fov_raw_image"], result.labels["fov_labels_image"]
        assert image.dims == ("c", "y", "x")
        assert image.shape == (2, 32, 40)
        assert image.dtype == np.dtype(np.uint16)
        assert image.coords["c"].to_numpy().tolist() == [0, 1]
        assert labels.dims == ("y", "x")
        assert labels.shape == (32, 40)
        assert labels.dtype == np.dtype(np.uint16)
        # Default chunks hold one channel and up to 1024 x 1024 pixels.
        assert image.chunks == ((1, 1), (32,), (40,))
        assert labels.chunks == ((32,), (40,))
        np.testing.assert_array_equal(image.data.compute(), iss_dataset.raw)
        np.testing.assert_array_equal(labels.data.compute(), iss_dataset.labels)
        assert get_transformation(image, get_all=True) == {"global": Identity()}
        assert get_transformation(labels, get_all=True) == {"global": Identity()}

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert isinstance(table.layers["counts"], sparse.csr_array)
        assert table.X.has_canonical_format
        np.testing.assert_array_equal(table.X.toarray(), [[0, 3, 1], [5, 0, 2]])
        assert table.X.dtype == np.dtype(np.float32)
        assert table.obs_names.tolist() == ["7", "2"]
        assert table.var_names.tolist() == ["GeneB", "GeneA", "GeneB-1"]
        assert table.var["feature_id"].tolist() == ["f2", "f1", "f3"]
        assert table.obs[instance_key or "instance_id"].tolist() == [7, 2]
        assert table.obs["region"].tolist() == ["fov_labels_image", "fov_labels_image"]
        assert get_table_keys(table) == ("fov_labels_image", "region", instance_key or "instance_id")
        np.testing.assert_array_equal(table.obsm["spatial"], [[2.0, 4.0], [26.0, 9.0]])

    @pytest.mark.parametrize("channels", [1, 2, 3, 4])
    @pytest.mark.parametrize("factors", [(), (2, 2)])
    def test_pyramid_modes_preserve_scientific_channels(
        self, tmp_path: Path, channels: int, factors: tuple[int, ...]
    ) -> None:
        dataset = write_iss_dataset(tmp_path, channels=channels)

        result = iss(
            dataset.root,
            raw_image_file="raw.tiff",
            labels_file="labels.tiff",
            table_file="cells.h5ad",
            scale_factors=factors,
            chunks=(1, 16, 16),
        )
        image, labels = result.images["region_raw_image"], result.labels["region_labels_image"]
        assert isinstance(image, DataTree if factors else DataArray)
        assert isinstance(labels, DataTree if factors else DataArray)
        if isinstance(image, DataTree) and isinstance(labels, DataTree):
            # The image and its labels share every pyramid level.
            assert list(image.children) == list(labels.children) == ["scale0", "scale1", "scale2"]
            for level in image.children:
                assert image[level]["image"].shape[1:] == labels[level]["image"].shape
            assert image["scale2"]["image"].shape == (channels, 8, 10)
            for child in labels.children.values():
                assert set(np.unique(child["image"].data.compute())) <= {0, 2, 7}
            image, labels = image["scale0"]["image"], labels["scale0"]["image"]
        assert image.coords["c"].to_numpy().tolist() == list(range(channels))
        np.testing.assert_array_equal(image.data.compute(), dataset.raw)
        np.testing.assert_array_equal(labels.data.compute(), dataset.labels)

    @pytest.mark.parametrize("height", [1, 32])
    def test_semantic_singleton_axes_survive(self, tmp_path: Path, height: int) -> None:
        dataset = write_iss_dataset(tmp_path, channels=1, height=height)

        result = iss(dataset.root, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

        assert result.images["region_raw_image"].shape == (1, height, 40)
        assert result.labels["region_labels_image"].shape == (height, 40)
        np.testing.assert_array_equal(result.labels["region_labels_image"].data.compute(), dataset.labels)

    @pytest.mark.parametrize("instance_key", [None, "segmentation_id"])
    @pytest.mark.parametrize("large", [2**53 + 1, 2**63 + 1])
    def test_large_integer_ids_remain_exact(
        self, iss_dataset: IssDataset, instance_key: str | None, large: int
    ) -> None:
        table_file = iss_dataset.root / "cells.h5ad"
        table = ad.read_h5ad(table_file)
        identifiers = np.asarray([large, large + 2], dtype=np.uint64)
        assert isinstance(table.obs, pd.DataFrame)
        table.obs["segmentation_id"] = identifiers
        table.obs_names = [str(value) for value in identifiers]
        table.write_h5ad(table_file)
        labels = np.zeros_like(iss_dataset.labels, dtype=np.uint64)
        labels[:, 1:4], labels[:, 24:28] = identifiers[0], identifiers[1]
        tifffile.imwrite(iss_dataset.root / "labels.tiff", labels, photometric="minisblack")

        result = iss(
            iss_dataset.root,
            raw_image_file="raw.tiff",
            labels_file="labels.tiff",
            table_file="cells.h5ad",
            instance_key=instance_key,
        )
        np.testing.assert_array_equal(result.tables["table"].obs[instance_key or "instance_id"], identifiers)
        np.testing.assert_array_equal(result.labels["region_labels_image"].data.compute(), labels)

    def test_explicit_instance_column_preserves_opaque_names(self, iss_dataset: IssDataset) -> None:
        table_file = iss_dataset.root / "cells.h5ad"
        table = ad.read_h5ad(table_file)
        table.obs_names = ["cell-b", "cell-a"]
        table.write_h5ad(table_file)

        result = iss(
            iss_dataset.root,
            raw_image_file="raw.tiff",
            labels_file="labels.tiff",
            table_file="cells.h5ad",
            instance_key="segmentation_id",
        )

        assert result.tables["table"].obs_names.tolist() == ["cell-b", "cell-a"]
        assert result.tables["table"].obs["segmentation_id"].tolist() == [7, 2]

    def test_write_roundtrip(self, iss_dataset: IssDataset, tmp_path: Path) -> None:
        result = iss(iss_dataset.root, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
        output = tmp_path / "out.zarr"
        result.write(output)
        loaded = read_zarr(output)
        assert set(loaded.images) == {"region_raw_image"}
        assert loaded.tables["table"].obs["instance_id"].tolist() == [7, 2]
        np.testing.assert_array_equal(loaded.labels["region_labels_image"].data.compute(), iss_dataset.labels)

    def test_imported_annotation_types_and_missingness_are_preserved(self, tmp_path: Path) -> None:
        """Retain imported extension metadata while enriching the owned table."""
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs["nullable_count"] = pd.array([2**53 + 1, None], dtype="Int64")
        source.obs["nullable_flag"] = pd.array([True, None], dtype="boolean")
        source.obs["classification"] = pd.Categorical(["T", "B"], categories=["B", "T"], ordered=True)
        source.var["confidence"] = np.asarray([0.25] * source.n_vars, dtype=np.float32)
        source.var["nullable_rank"] = pd.array([1] + [None] * (source.n_vars - 1), dtype="UInt16")
        source.write_h5ad(path)
        original = ad.read_h5ad(path)

        result = iss(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")

        table = result.tables["table"]
        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(table.var, pd.DataFrame)
        assert isinstance(original.obs, pd.DataFrame)
        assert isinstance(original.var, pd.DataFrame)
        expected_obs = original.obs.iloc[[0, 1]]
        expected_var = original.var.copy()
        expected_var.index = ["GeneB", "GeneA", "GeneB-1"]
        for column in ("nullable_count", "nullable_flag", "classification"):
            pd.testing.assert_series_equal(table.obs[column], expected_obs[column], check_exact=True)
        for column in ("confidence", "nullable_rank"):
            pd.testing.assert_series_equal(table.var[column], expected_var[column], check_exact=True)
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

    @pytest.mark.parametrize("field", ["region", "instance_id"])
    @pytest.mark.parametrize("conflicting", [False, True])
    def test_reserved_output_columns(self, tmp_path: Path, field: str, *, conflicting: bool) -> None:
        """Reserved output columns."""
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        if field == "region":
            source.obs[field] = ["other" if conflicting else "region_labels_image"] * 2
        else:
            source.obs[field] = [7, 7 if conflicting else 2]
        source.write_h5ad(path)

        if conflicting:
            with pytest.raises(ReaderFormatError, match="conflicting reserved"):
                iss(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
        else:
            result = iss(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
            assert result.tables["table"].obs[field].tolist() == source.obs[field].tolist()
