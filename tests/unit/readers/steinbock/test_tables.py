"""Steinbock owned expression, exact object IDs, XY centroids, and metadata cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._images import Calibration, SampleGrid
from spatialdata_io.readers.steinbock._tables import (
    _set_instance_ids,
    read_table,
    set_positions,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _sample(name: str, *, shape: tuple[int, int] = (8, 12), dtype: str = "uint16") -> SampleGrid:
    """Provide independently inspected grid metadata without constructing models."""
    return SampleGrid(f"{name}.tiff", shape, np.dtype(dtype), "sample", None)


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
        samples = {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")}
        tasks: list[str] = []

        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            table = read_table(path, samples)

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
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[3, 1.5], [8.5, 5], [8.5, 5], [3, 1.5]])
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
            read_table(path, {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")})

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
            read_table(path, {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")})

    def test_read_failure_has_artifact_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "cells.h5ad")
        path.write_bytes(b"not an HDF5 file")

        with pytest.raises(OSError, match="file signature") as caught:
            read_table(path, {"sample-a": _sample("sample-a")})

        assert any("steinbock" in note and str(path) in note for note in caught.value.__notes__)

    def test_third_party_non_dataframe_boundary(self, tmp_path: Path, mocker: MockerFixture) -> None:
        result = mocker.Mock(X=np.ones((1, 1)), obs=None, var=pd.DataFrame(index=["DNA"]))
        mocker.patch("spatialdata_io.readers.steinbock._tables.ad.read_h5ad", return_value=result)

        with pytest.raises(TypeError, match="eager DataFrames"):
            read_table(ArtifactFile(tmp_path / "cells.h5ad"), {"sample-a": _sample("sample-a")})

    @pytest.mark.parametrize("field", ["cell_id", "region"])
    def test_occupied_generated_columns_fail_before_annotation_mutation(
        self, tmp_path: Path, mocker: MockerFixture, field: str
    ) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        source = ad.read_h5ad(path)
        source.obs[field] = ["reserved"] * 4
        assert isinstance(source.obs, pd.DataFrame)
        before = source.obs.copy()
        spatial = source.obsm["spatial"]
        mocker.patch("spatialdata_io.readers.steinbock._tables.ad.read_h5ad", return_value=source)

        with pytest.raises(ReaderFormatError, match=f"occupied.*{field}"):
            read_table(path, {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")})

        pd.testing.assert_frame_equal(source.obs, before, check_exact=True)
        assert source.obsm["spatial"] is spatial
        assert "spatial_pixels" not in source.obsm
        assert source.var.columns.tolist() == ["Channel Name"]

    def test_xy_order_and_frame_edges(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "cells.h5ad")
        ad.AnnData(
            X=np.ones((3, 1)),
            obs=pd.DataFrame(
                {"image": ["sample.tiff"] * 3, "centroid-0": [1.5, 0, 7], "centroid-1": [8.5, 0, 11]},
                index=[f"Object {number} in sample.tiff" for number in (1, 2, 3)],
            ),
        ).write_h5ad(path)

        table = read_table(path, {"sample": _sample("sample")})
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[8.5, 1.5], [0, 0], [11, 7]])

    @pytest.mark.parametrize("duplicate", [False, True])
    def test_optional_duplicate_and_legacy_feature_columns(self, tmp_path: Path, *, duplicate: bool) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        source = ad.read_h5ad(path)
        assert isinstance(source.obs, pd.DataFrame)
        if duplicate:
            source.obs["Image"] = source.obs["image"]
        else:
            del source.obs["Image"]
        source.var["Final Concentration / Dilution"] = ["1:100", "1:200"]
        with ad.settings.override(disallow_forward_slash_in_h5ad=False):
            source.write_h5ad(path)

        table = read_table(path, {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")})

        assert "Image" not in table.obs
        assert table.var.columns.tolist() == ["Channel_Name", "Final_Concentration"]
        assert table.var["Final_Concentration"].tolist() == ["1:100", "1:200"]

    def test_feature_column_names_are_sanitized_with_one_warning(self, tmp_path: Path) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        source = ad.read_h5ad(path)
        source.var["CD45 RO"] = ["a", "b"]
        source.var["cd45_ro"] = ["c", "d"]
        source.var["Final Concentration / Dilution"] = ["1:100", "1:200"]
        with ad.settings.override(disallow_forward_slash_in_h5ad=False):
            source.write_h5ad(path)

        with pytest.warns(UserWarning, match="column names") as caught:
            table = read_table(path, {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")})

        assert table.var.columns.tolist() == ["Channel_Name", "CD45_RO_1", "cd45_ro", "Final_Concentration"]
        assert table.var["CD45_RO_1"].tolist() == ["a", "b"]
        assert [str(warning.message) for warning in caught] == [
            (
                "steinbock: the feature metadata of the cell table has column names that a SpatialData table "
                "does not accept; using 'Channel Name' -> 'Channel_Name', 'CD45 RO' -> 'CD45_RO_1'. Rename the "
                "source columns to choose the names explicitly."
            )
        ]

    def test_uns_keys_are_sanitized_with_one_warning(self, tmp_path: Path) -> None:
        """Rename invalid and reader-colliding top-level keys only, keeping values and stored order."""
        path = _write_table(tmp_path / "cells.h5ad")
        source = ad.read_h5ad(path)
        source.uns["my key"] = {"inner key": "kept"}
        source.uns["valid"] = "v"
        source.uns["Spatialdata_IO_positions"] = np.asarray([1, 2])
        source.write_h5ad(path)
        samples = {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")}

        with pytest.warns(UserWarning, match="SpatialData table does not accept") as caught:
            table = read_table(path, samples)

        assert [str(warning.message) for warning in caught] == [
            (
                "steinbock: the feature metadata of the cell table has column names that a SpatialData table does not "
                "accept; using 'Channel Name' -> 'Channel_Name'. Rename the source columns to choose the names "
                "explicitly."
            ),
            (
                "steinbock: the unstructured metadata of the cell table has names that a SpatialData table does not "
                "accept as uns keys; using 'Spatialdata_IO_positions' -> 'Spatialdata_IO_positions_1', 'my key' -> "
                "'my_key'. Rename them in the source to choose the names explicitly."
            ),
        ]
        # HDF5 stores group members by name, so the stored order is alphabetical.
        assert list(table.uns) == ["Spatialdata_IO_positions_1", "my_key", "valid"]
        assert table.uns["my_key"] == {"inner key": "kept"}
        assert table.uns["valid"] == "v"
        np.testing.assert_array_equal(table.uns["Spatialdata_IO_positions_1"], [1, 2])

        set_positions(table, samples, path)
        assert list(table.uns) == ["Spatialdata_IO_positions_1", "my_key", "valid", POSITIONS_KEY]

    def test_valid_uns_keys_are_untouched(self, tmp_path: Path) -> None:
        path = _write_table(tmp_path / "cells.h5ad")
        source = ad.read_h5ad(path)
        source.uns["qc"] = {"inner key": 1}
        source.write_h5ad(path)

        with pytest.warns(UserWarning, match="SpatialData table does not accept") as caught:
            table = read_table(path, {"sample-a": _sample("sample-a"), "sample-b": _sample("sample-b")})

        # Only the fixture's feature column "Channel Name" warns.
        assert len(caught) == 1
        assert "feature metadata" in str(caught[0].message)
        assert list(table.uns) == ["qc"]
        assert table.uns["qc"] == {"inner key": 1}


class TestSetInstanceIds:
    """Parse exact sample-local integers without float conversion or filename rewriting."""

    @pytest.mark.parametrize("name", ["sample", "sample.tiff fragment", "sample in tissue", "sample with spaces"])
    def test_complete_source_filename(self, tmp_path: Path, name: str) -> None:
        filename = f"{name}.tiff"
        obs = pd.DataFrame(
            {"image": [filename, filename]}, index=[f"Object 2 in {filename}", f"Object 7 in {filename}"]
        )

        _set_instance_ids(obs, {name: _sample(name)}, ArtifactFile(tmp_path / "cells.h5ad"))

        assert obs["cell_id"].tolist() == [2, 7]
        assert str(obs["cell_id"].dtype) == "uint64[pyarrow]"
        assert obs["region"].tolist() == [f"{name}_labels"] * 2

    @pytest.mark.parametrize("identifier", [2**53 + 1, 2**63, 2**64 - 1])
    def test_large_uint64_ids_exact(self, tmp_path: Path, identifier: int) -> None:
        """Large uint64 ids exact."""
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=[f"Object {identifier} in sample.tiff"])

        _set_instance_ids(obs, {"sample": _sample("sample", dtype="uint64")}, ArtifactFile(tmp_path / "cells.h5ad"))

        assert int(obs["cell_id"].iloc[0]) == identifier
        assert str(obs["cell_id"].dtype) == "uint64[pyarrow]"


class TestSetPositions:
    """Declare native centroids and derive physical positions only from complete calibration."""

    @pytest.mark.parametrize(("native_x", "physical_x"), [(-0.5, 0.0), (0.5, 5e-324), (-1.5, -5e-324)])
    def test_shifted_zero_and_representable_subnormal_are_valid(
        self, tmp_path: Path, native_x: float, physical_x: float
    ) -> None:
        table = self._table([[native_x, 0.5]], ["sample.tiff"])
        set_positions(
            table, {"sample": self._grid("sample", "global", (5e-324, 3))}, ArtifactFile(tmp_path / "cells.h5ad")
        )
        np.testing.assert_array_equal(table.obsm["spatial"], [[physical_x, 3]])
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[native_x, 0.5]])
        assert table.obsm["spatial"].dtype == np.dtype(np.float64)

    def test_single_calibrated_sample_declares_fixed_frames(self, tmp_path: Path) -> None:
        table = self._table([[1, 2]], ["sample.tiff"])
        set_positions(table, {"sample": self._grid("sample", "global", (2, 4))}, ArtifactFile(tmp_path / "cells.h5ad"))
        np.testing.assert_array_equal(table.obsm["spatial"], [[3, 10]])
        assert table.uns[POSITIONS_KEY] == {
            "schema_version": 1,
            "coordinates": {
                "spatial_pixels": {
                    "axes": ["x", "y"],
                    "units": "pixel",
                    "frame": "sample_regionprops_indices",
                    "position": "source_regionprops_centroid_zero_based_pixel_index_centres",
                },
                "spatial": {
                    "axes": ["x", "y"],
                    "units": "micrometer",
                    "frame": "global",
                    "position": "source_regionprops_centroid_in_raster_corner_frame",
                    "calibration": {"global": "evidence"},
                },
            },
        }
        assert "spatial_frame" not in table.obs
        assert "spatial_pixel_frame" not in table.obs

    def test_several_samples_scale_rows_by_their_own_frame(self, tmp_path: Path) -> None:
        table = self._table([[1, 2], [1, 2], [0, 0]], ["b.tiff", "a.tiff", "b.tiff"])
        grids = {"a": self._grid("a", "a", (2, 4)), "b": self._grid("b", "b", (10, 100))}
        set_positions(table, grids, ArtifactFile(tmp_path / "cells.h5ad"))
        np.testing.assert_array_equal(table.obsm["spatial"], [[15, 250], [3, 10], [5, 50]])
        assert table.obs["spatial_frame"].tolist() == ["b", "a", "b"]
        assert table.obs["spatial_pixel_frame"].tolist() == [
            "b_regionprops_indices",
            "a_regionprops_indices",
            "b_regionprops_indices",
        ]
        coordinates = table.uns[POSITIONS_KEY]["coordinates"]
        assert coordinates["spatial"]["frame_key"] == "spatial_frame"
        assert coordinates["spatial"]["calibration"] == {"a": "evidence", "b": "evidence"}
        assert coordinates["spatial_pixels"]["frame_key"] == "spatial_pixel_frame"

    @pytest.mark.parametrize("calibrated", [(), ("a",)])
    def test_incomplete_calibration_keeps_native_positions_only(
        self, tmp_path: Path, calibrated: tuple[str, ...]
    ) -> None:
        table = self._table([[1, 2], [1, 2]], ["a.tiff", "b.tiff"])
        grids = {name: self._grid(name, name, (2, 4) if name in calibrated else None) for name in ("a", "b")}
        set_positions(table, grids, ArtifactFile(tmp_path / "cells.h5ad"))
        assert "spatial" not in table.obsm
        assert "spatial_frame" not in table.obs
        assert list(table.uns[POSITIONS_KEY]["coordinates"]) == ["spatial_pixels"]

    @staticmethod
    def _table(native: list[list[float]], images: list[str]) -> ad.AnnData:
        table = ad.AnnData(obs=pd.DataFrame({"image": images}, index=[str(row) for row in range(len(images))]))
        table.obsm["spatial_pixels"] = np.asarray(native, dtype=np.float64)
        return table

    @staticmethod
    def _grid(name: str, frame: str, spacing: tuple[float, float] | None) -> SampleGrid:
        calibration = None if spacing is None else Calibration(spacing, "evidence")
        return SampleGrid(f"{name}.tiff", (8, 12), np.dtype("uint16"), frame, calibration)
