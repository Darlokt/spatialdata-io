"""Steinbock owned expression, exact object IDs, XY centroids, and metadata cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._schemas import LabelGrid
from spatialdata_io.readers.steinbock._tables import (
    _centroids,
    _clean_metadata,
    _read_table,
    _set_instance_ids,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _sample(*, shape: tuple[int, int] = (8, 12), dtype: str = "uint16") -> LabelGrid:
    """Provide independently inspected grid metadata without constructing models."""
    return LabelGrid(shape, np.dtype(dtype))


def _write_table(path: Path) -> ArtifactFile:
    """Write interleaved samples with exact IDs and unrelated owned metadata."""
    images = ["sample-b.tiff", "sample-a.tiff", "sample-b.tiff", "sample-a.tiff"]
    table = ad.AnnData(
        X=sparse.csr_array(np.asarray([[1, 0], [0, 4], [5, 6], [7, 8]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "image": images,
                "Image": images,
                "centroid-0": [1.5, 5.0, 5.0, 1.5],
                "centroid-1": [3.0, 8.5, 8.5, 3.0],
                "cell_type": ["B", "T", "T", "B"],
            },
            index=[f"Object {identifier} in {image}" for identifier, image in zip([2, 7, 7, 2], images, strict=True)],
        ),
        var=pd.DataFrame({"Channel Name": ["DNA", "CD3"]}, index=["DNA", "CD3"]),
    )
    table.layers["metadata"] = np.ones((4, 2), dtype=np.int16)
    table.obsp["adj"] = sparse.csr_array(np.eye(4, dtype=np.uint8))
    table.obsm["spatial"] = np.full((4, 2), -10.0)
    table.write_h5ad(path)
    return ArtifactFile(path)


class TestReadTable:
    """Validate the real H5AD boundary and preserve the owned source contract."""

    def test_values_order_metadata_and_source_ownership(self, tmp_path: Path) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        samples = {"sample-a": _sample(), "sample-b": _sample()}
        tasks: list[str] = []

        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            table = _read_table(path, samples)

        assert not tasks
        assert table.obs_names.tolist() == [
            "Object 2 in sample-b.tiff",
            "Object 7 in sample-a.tiff",
            "Object 7 in sample-b.tiff",
            "Object 2 in sample-a.tiff",
        ]
        assert table.var_names.tolist() == ["DNA", "CD3"]
        assert isinstance(table.X, (sparse.csr_array, sparse.csr_matrix))
        np.testing.assert_array_equal(table.X.data, [1, 4, 5, 6, 7, 8])
        assert table.X.dtype == np.dtype(np.float32)
        assert table.obs["cell_id"].tolist() == [2, 7, 7, 2]
        assert table.obs["region"].tolist() == ["sample-b_labels", "sample-a_labels"] * 2
        assert isinstance(table.obs["region"].dtype, pd.CategoricalDtype)
        np.testing.assert_array_equal(table.obsm["spatial"], [[3, 1.5], [8.5, 5], [8.5, 5], [3, 1.5]])
        assert table.obs["cell_type"].tolist() == ["B", "T", "T", "B"]
        assert "Image" not in table.obs
        assert table.var.columns.tolist() == ["Channel_Name"]
        np.testing.assert_array_equal(table.layers["metadata"], np.ones((4, 2), dtype=np.int16))
        assert isinstance(table.obsp["adj"], (sparse.csr_array, sparse.csr_matrix))
        np.testing.assert_array_equal(table.obsp["adj"].diagonal(), np.ones(4, dtype=np.uint8))

        original = ad.read_h5ad(path)
        assert "Image" in original.obs
        assert "cell_id" not in original.obs
        assert original.var.columns.tolist() == ["Channel Name"]
        np.testing.assert_array_equal(original.obsm["spatial"], np.full((4, 2), -10.0))

    @pytest.mark.parametrize("invalid", ["expression", "duplicate_observation", "duplicate_feature", "empty_feature"])
    def test_invalid_h5ad_schema(self, tmp_path: Path, invalid: str) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        table = ad.read_h5ad(path)
        if invalid == "expression":
            table.X = None
        elif invalid == "duplicate_observation":
            table.obs_names = ["same"] * 4
        elif invalid == "duplicate_feature":
            table.var_names = ["DNA", "DNA"]
        else:
            table.var_names = ["DNA", ""]
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError) as caught:
            _read_table(path, {"sample-a": _sample(), "sample-b": _sample()})

        assert caught.value.path == path
        assert caught.value.context.reader == "steinbock"

    @pytest.mark.parametrize("missing", ["image", "centroid-0", "centroid-1"])
    def test_missing_observation_columns(self, tmp_path: Path, missing: str) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        table = ad.read_h5ad(path)
        assert isinstance(table.obs, pd.DataFrame)
        del table.obs[missing]
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="columns"):
            _read_table(path, {"sample-a": _sample(), "sample-b": _sample()})

    def test_read_failure_has_artifact_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "cells.h5ad")
        path.write_bytes(b"not an HDF5 file")

        with pytest.raises(OSError, match="file signature") as caught:
            _read_table(path, {"sample-a": _sample()})

        assert any("steinbock" in note and str(path) in note for note in caught.value.__notes__)

    def test_third_party_non_dataframe_boundary(self, tmp_path: Path, mocker: MockerFixture) -> None:
        result = mocker.Mock(X=np.ones((1, 1)), obs=None, var=pd.DataFrame(index=["DNA"]))
        mocker.patch("spatialdata_io.readers.steinbock._tables.ad.read_h5ad", return_value=result)

        with pytest.raises(TypeError, match="eager DataFrames"):
            _read_table(ArtifactFile(tmp_path / "cells.h5ad"), {"sample-a": _sample()})


class TestSetInstanceIds:
    """Parse exact sample-local integers without float conversion or filename rewriting."""

    @pytest.mark.parametrize("name", ["sample", "sample.tiff fragment", "sample in tissue", "sample with spaces"])
    def test_complete_source_filename(self, tmp_path: Path, name: str) -> None:
        filename = f"{name}.tiff"
        obs = pd.DataFrame(
            {"image": [filename, filename]}, index=[f"Object 2 in {filename}", f"Object 7 in {filename}"]
        )

        _set_instance_ids(obs, {name: _sample()}, ArtifactFile(tmp_path / "cells.h5ad"))

        assert obs["cell_id"].tolist() == [2, 7]
        assert obs["region"].tolist() == [f"{name}_labels"] * 2

    @pytest.mark.parametrize("identifier", [2**53 + 1, 2**63 + 1, 2**64 - 1])
    def test_large_uint64_ids_exact(self, tmp_path: Path, identifier: int) -> None:
        """Large uint64 ids exact."""
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=[f"Object {identifier} in sample.tiff"])

        _set_instance_ids(obs, {"sample": _sample(dtype="uint64")}, ArtifactFile(tmp_path / "cells.h5ad"))

        assert int(obs["cell_id"].iloc[0]) == identifier
        assert str(obs["cell_id"].dtype) == ("uint64[pyarrow]" if identifier >= 2**63 else "int64[pyarrow]")


class TestCentroids:
    """Keep asymmetric XY order and validate each sample's inspected pixel frame."""

    def test_xy_order_and_frame_edges(self, tmp_path: Path) -> None:
        obs = pd.DataFrame({"image": ["sample.tiff"] * 3, "centroid-0": [1.5, 0, 7], "centroid-1": [8.5, 0, 11]})

        coordinates = _centroids(obs, {"sample": _sample()}, ArtifactFile(tmp_path / "cells.h5ad"))
        np.testing.assert_array_equal(coordinates, [[8.5, 1.5], [0, 0], [11, 7]])


class TestCleanMetadata:
    """Remove only redundant metadata and reject lossy name normalization."""

    @pytest.mark.parametrize("duplicate", [False, True])
    def test_optional_duplicate_and_legacy_feature_columns(self, tmp_path: Path, *, duplicate: bool) -> None:
        obs = pd.DataFrame({"image": ["sample.tiff"]})
        if duplicate:
            obs["Image"] = obs["image"]
        var = pd.DataFrame({"Channel Name": ["DNA"], "Final Concentration / Dilution": ["1:100"]})

        _clean_metadata(obs, var, ArtifactFile(tmp_path / "cells.h5ad"))

        assert obs.columns.tolist() == ["image"]
        assert var.columns.tolist() == ["Channel_Name", "Final_Concentration"]
        assert var["Final_Concentration"].tolist() == ["1:100"]
