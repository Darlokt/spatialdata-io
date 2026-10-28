"""Exercise supported Steinbock layouts through the public reader."""

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
from xarray import DataTree

from spatialdata_io import __version__
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture

    from tests.integration.readers.steinbock.conftest import SteinbockDataset


def _write_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image_directory = path / "ome"
    mask_directory = path / "masks_deepcell"
    image_directory.mkdir()
    mask_directory.mkdir()

    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.asarray(
        [[[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 2, 2], [0, 0, 0, 2, 2]]],
        dtype=np.uint16,
    )
    tifffile.imwrite(
        image_directory / "sample.ome.tiff", image, photometric="minisblack", ome=True, metadata={"axes": "CYX"}
    )
    tifffile.imwrite(mask_directory / "sample.tiff", labels, photometric="minisblack")

    obs = pd.DataFrame(
        {
            "image": ["sample.tiff", "sample.tiff"],
            "Image": ["sample.tiff", "sample.tiff"],
            "centroid-0": [1.0, 3.0],
            "centroid-1": [1.5, 2.5],
        },
        index=["Object 1 in sample.tiff", "Object 2 in sample.tiff"],
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
    """Characterize sample pairing, lazy rasters, cleanup, and linkage."""

    def test_synthetic_layout_preserves_public_contract(self, tmp_path: Path) -> None:
        image, labels = _write_layout(tmp_path)

        result = steinbock(tmp_path)

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
        np.testing.assert_array_equal(table.obsm["spatial"], np.asarray([[1.5, 1.0], [2.5, 3.0]]))
        assert table.var.columns.tolist() == ["Channel_Name"]
        assert get_table_keys(table) == ("sample_labels", "region", "cell_id")

    @pytest.mark.parametrize("labels_kind", ["deepcell", "ilastik"])
    def test_two_samples_preserve_compound_ids_and_cleanup(
        self, steinbock_dataset: SteinbockDataset, labels_kind: Literal["deepcell", "ilastik"]
    ) -> None:
        if labels_kind == "ilastik":
            # This segmentation fixture shifts each object down by one pixel.
            # Its cell table must describe the same selected segmentation.
            path = steinbock_dataset.root / "cells.h5ad"
            source = ad.read_h5ad(path)
            assert isinstance(source.obs, pd.DataFrame)
            source.obs["centroid-0"] += 1
            with ad.settings.override(disallow_forward_slash_in_h5ad=False):
                source.write_h5ad(path)

        result = steinbock(steinbock_dataset.root, labels_kind=labels_kind)

        assert list(result.images) == ["sample-a_image", "sample-b_image"]
        assert list(result.labels) == ["sample-a_labels", "sample-b_labels"]
        masks = steinbock_dataset.deepcell if labels_kind == "deepcell" else steinbock_dataset.ilastik
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
            assert get_transformation(image, get_all=True) == {sample: Identity()}
            assert get_transformation(labels, get_all=True) == {sample: Identity()}

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
        if labels_kind == "ilastik":
            expected_centroids[:, 1] += 1
        np.testing.assert_array_equal(table.obsm["spatial"], expected_centroids)
        assert get_table_keys(table) == (["sample-b_labels", "sample-a_labels"], "region", "cell_id")

    def test_unmatched_image_fails(self, steinbock_dataset: SteinbockDataset) -> None:
        (steinbock_dataset.root / "masks_deepcell" / "sample-a.tiff").unlink()

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

    def test_unmatched_mask_fails(self, steinbock_dataset: SteinbockDataset) -> None:
        tifffile.imwrite(
            steinbock_dataset.root / "masks_deepcell" / "orphan.tiff",
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

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_large_ids_and_written_store_remain_exact(self, tmp_path: Path, dtype: type[np.int64 | np.uint64]) -> None:
        _write_layout(tmp_path)
        identifiers = [2**53 + 1, np.iinfo(dtype).max]
        mask = np.zeros((4, 5), dtype=dtype)
        mask[1, 1], mask[3, 2] = identifiers
        tifffile.imwrite(tmp_path / "masks_deepcell" / "sample.tiff", mask, photometric="minisblack")
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs_names = [f"Object {identifier} in sample.tiff" for identifier in identifiers]
        source.write_h5ad(path)

        result = steinbock(tmp_path)
        np.testing.assert_array_equal(result.tables["table"].obs["cell_id"], np.asarray(identifiers, dtype=dtype))
        np.testing.assert_array_equal(result.labels["sample_labels"].data.compute(), mask)

        target = tmp_path / "result.zarr"
        result.write(target)
        restored = read_zarr(target)
        np.testing.assert_array_equal(restored.tables["table"].obs["cell_id"], np.asarray(identifiers, dtype=dtype))
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

        if conflicting:
            with pytest.raises(ReaderFormatError, match="conflicting reserved"):
                steinbock(tmp_path)
        else:
            result = steinbock(tmp_path)
            assert result.tables["table"].obs[field].tolist() == source.obs[field].tolist()

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
        np.testing.assert_array_equal(table.obsm["spatial"], [[2, 1], [4, 3]])
        assert table.obsm["spatial"].dtype == np.dtype(dtype.lower())
        for column in ("centroid-0", "centroid-1"):
            pd.testing.assert_series_equal(table.obs[column], original.obs[column], check_exact=True)
