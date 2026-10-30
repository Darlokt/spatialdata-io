"""Prove Steinbock regionprops-to-raster physical coordinates on analytic masks."""

from __future__ import annotations

from dataclasses import dataclass
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
from spatialdata.models import get_table_keys
from spatialdata.transformations import Scale, get_transformation
from xarray import DataArray

import spatialdata_io.readers.steinbock._images as image_owner
import spatialdata_io.readers.steinbock._reader as reader_owner
import spatialdata_io.readers.steinbock._tables as table_owner
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.steinbock import steinbock

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture


@dataclass(frozen=True, slots=True)
class AnalyticDataset:
    root: Path
    native_xy: NDArray[np.float64]
    labels: NDArray[np.uint16]
    source: ad.AnnData


def _write_dataset(
    root: Path,
    spacings: Mapping[str, tuple[float, float] | None],
    *,
    family: Literal["deepcell", "ilastik"] = "deepcell",
) -> AnalyticDataset:
    """Match the producer's actual regionprops and object-identity export."""
    (root / "ome").mkdir()
    for kind in ("deepcell", "ilastik"):
        (root / f"masks_{kind}").mkdir()
    base = np.zeros((10, 12), dtype=np.uint16)
    base[1, 2] = 2
    base[3:5, 4:7] = 7
    base[5, 8] = base[6, 8] = base[6, 9] = 11
    selected = np.roll(base, (1, 1), axis=(0, 1)) if family == "ilastik" else base
    measured = regionprops_table(selected, properties=("label", "centroid"))
    measured_xy = {
        int(identifier): (float(x), float(y))
        for identifier, x, y in zip(measured["label"], measured["centroid-1"], measured["centroid-0"], strict=True)
    }
    for name, spacing in spacings.items():
        metadata: dict[str, str | float | dict[str, list[str]]] = {"axes": "CYX", "Channel": {"Name": ["DNA", "CD3"]}}
        if spacing is not None:
            metadata.update(
                PhysicalSizeX=spacing[0], PhysicalSizeY=spacing[1], PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm"
            )
        tifffile.imwrite(
            root / "ome" / f"{name}.ome.tiff",
            np.arange(240, dtype=np.uint16).reshape(2, 10, 12),
            ome=True,
            metadata=metadata,
            photometric="minisblack",
        )
        for kind in ("deepcell", "ilastik"):
            mask = base if kind == "deepcell" else np.roll(base, (1, 1), axis=(0, 1))
            tifffile.imwrite(root / f"masks_{kind}" / f"{name}.tiff", mask, photometric="minisblack")
    names = list(reversed(list(spacings)))
    identities = [(name, identifier) for identifier in (11, 2, 7) for name in names]
    native = np.asarray([measured_xy[identifier] for _, identifier in identities], dtype=np.float64)
    rows = len(identities)
    expression = sparse.csr_array(
        (np.arange(1, rows + 1, dtype=np.float32), (np.arange(rows), np.arange(rows) % 2)), shape=(rows, 2)
    )
    table = ad.AnnData(
        expression,
        obs=pd.DataFrame(
            {
                "image": [f"{name}.tiff" for name, _ in identities],
                "centroid-0": native[:, 1],
                "centroid-1": native[:, 0],
                "annotation": pd.Categorical(["T", "B", "T"] * len(names)),
            },
            index=[f"Object {identifier} in {name}.tiff" for name, identifier in identities],
        ),
        var=pd.DataFrame({"Channel Name": ["DNA", "CD3"]}, index=["marker-z", "marker-a"]),
    )
    table.layers["counts"] = expression.copy()
    table.layers["metadata"] = np.arange(rows * 2, dtype=np.int16).reshape(rows, 2)
    table.obsm["other_embedding"] = np.arange(rows * 3, dtype=np.int16).reshape(rows, 3)
    table.raw = ad.AnnData(
        expression.copy(), obs=pd.DataFrame(index=table.obs_names), var=pd.DataFrame(index=["raw-a", "raw-b"])
    )
    table.write_h5ad(root / "cells.h5ad")
    return AnalyticDataset(root, native, selected, table)


class TestSteinbockPositions:
    @pytest.mark.parametrize("family", ["deepcell", "ilastik"])
    def test_producer_centroids_align_to_unchanged_raster_frame(
        self, tmp_path: Path, family: Literal["deepcell", "ilastik"]
    ) -> None:
        fixture = _write_dataset(tmp_path, {"sample-a": (2.0, 3.0)}, family=family)
        shift = 1 if family == "ilastik" else 0
        expected_native = np.asarray([[25 / 3, 17 / 3], [2, 1], [5, 3.5]]) + shift
        np.testing.assert_array_equal(fixture.native_xy, expected_native)
        result = steinbock(tmp_path, labels_kind=family)
        table = result.tables["table"]
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], expected_native)
        np.testing.assert_array_equal(table.obsm["spatial"], (expected_native + 0.5) * [2, 3])
        assert table.obsm["spatial"].dtype == np.dtype(np.float64)
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] == "global"
        assert "spatial_frame" not in table.obs
        assert "spatial_source" not in table.obsm
        labels = result.labels["sample-a_labels"]
        assert isinstance(labels, DataArray)
        np.testing.assert_array_equal(labels.data.compute(), fixture.labels)
        np.testing.assert_array_equal(labels.coords["x"], np.arange(12) + 0.5)
        np.testing.assert_array_equal(labels.coords["y"], np.arange(10) + 0.5)
        expected_transform = Scale([2.0, 3.0], axes=("x", "y"))
        assert get_transformation(labels, "global") == expected_transform
        raster_centroids = get_centroids(labels, coordinate_system="global").compute()
        matched = raster_centroids.loc[table.obs["cell_id"], ["x", "y"]].to_numpy()
        physical = table.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        # The affine and pixelwise centroid reduction can differ by one
        # float64 rounding at this bounded (<30 micrometre) fixture extent.
        np.testing.assert_allclose(physical, matched, rtol=0, atol=4e-15)
        single = table.obs["cell_id"].to_numpy() == 2
        np.testing.assert_array_equal(table.obsm["spatial"][single], [[5 + 2 * shift, 4.5 + 3 * shift]])

    def test_two_independent_frames_preserve_duplicate_ids_row_order_and_distances(self, tmp_path: Path) -> None:
        spacings = {"sample-a": (2.0, 3.0), "sample-b": (5.0, 7.0)}
        fixture = _write_dataset(tmp_path, spacings)
        result = steinbock(tmp_path)
        table = result.tables["table"]
        assert table.obs_names.tolist() == fixture.source.obs_names.tolist()
        assert table.var_names.tolist() == ["marker-z", "marker-a"]
        assert table.obs["region"].tolist() == ["sample-b_labels", "sample-a_labels"] * 3
        assert table.obs["cell_id"].tolist() == [11, 11, 2, 2, 7, 7]
        assert table.obs["spatial_frame"].tolist() == ["sample-b", "sample-a"] * 3
        assert isinstance(table.obs["spatial_frame"].dtype, pd.CategoricalDtype)
        declaration = table.uns[POSITIONS_KEY]["coordinates"]["spatial"]
        assert declaration["frame_key"] == "spatial_frame"
        assert set(declaration["calibration"]) == set(spacings)
        assert get_table_keys(table) == (["sample-b_labels", "sample-a_labels"], "region", "cell_id")
        physical = table.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        for name, spacing in spacings.items():
            rows = np.flatnonzero(table.obs["image"].to_numpy() == f"{name}.tiff")
            np.testing.assert_array_equal(table.obsm["spatial"][rows], (fixture.native_xy[rows] + 0.5) * spacing)
            assert np.linalg.norm(table.obsm["spatial"][rows[1]] - table.obsm["spatial"][rows[2]]) == pytest.approx(
                np.hypot(3 * spacing[0], 2.5 * spacing[1])
            )
            independent = get_centroids(result.labels[f"{name}_labels"], coordinate_system=name).compute()
            np.testing.assert_allclose(
                physical[rows], independent.loc[table.obs["cell_id"].iloc[rows], ["x", "y"]], rtol=0, atol=1e-14
            )

    @pytest.mark.parametrize(
        "spacings",
        [{"sample-a": None}, {"sample-a": None, "sample-b": None}, {"sample-a": (2.0, 3.0), "sample-b": None}],
    )
    def test_native_and_mixed_populations_do_not_fabricate_physical_coordinates(
        self, tmp_path: Path, spacings: dict[str, tuple[float, float] | None]
    ) -> None:
        fixture = _write_dataset(tmp_path, spacings)
        table = steinbock(tmp_path).tables["table"]
        assert "spatial" not in table.obsm
        assert "spatial" not in table.uns[POSITIONS_KEY]["coordinates"]
        assert "spatial_frame" not in table.obs
        assert table.n_obs == fixture.source.n_obs
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], fixture.native_xy)
        selected = table[table.obs["image"] == "sample-a.tiff"].copy()
        assert "spatial" not in selected.obsm

    def test_identical_numerical_origins_still_have_distinct_frames(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, {"sample-a": (2.0, 3.0), "sample-b": (2.0, 3.0)})
        table = steinbock(tmp_path).tables["table"]
        np.testing.assert_array_equal(table.obsm["spatial"][::2], table.obsm["spatial"][1::2])
        assert set(table.obs["spatial_frame"]) == {"sample-a", "sample-b"}

    def test_resolves_calibration_once_before_single_final_linkage(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_dataset(tmp_path, {"sample-a": (2.0, 3.0), "sample-b": (5.0, 7.0)})
        spacing = mocker.spy(image_owner, "_spacing")
        linkage = mocker.spy(reader_owner, "parse_linked_table")
        table = steinbock(tmp_path).tables["table"]
        assert spacing.call_count == 2
        linkage.assert_called_once()
        assert linkage.call_args.args[0] is table
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame_key"] == "spatial_frame"

    def test_no_eager_pixel_tasks_or_densification_and_owned_annotations(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        fixture = _write_dataset(tmp_path, {"sample-a": (2.0, 3.0)})
        source_path = tmp_path / "cells.h5ad"
        before = source_path.read_bytes()
        source = ad.read_h5ad(source_path)
        # H5AD currently restores sparse arrays as matrices. Normalize once at
        # this owned-input seam so identity probes isolate coordinate mutation.
        assert isinstance(source.X, sparse.csr_matrix | sparse.csr_array)
        source.X = sparse.csr_array(source.X)
        source.layers["counts"] = sparse.csr_array(source.layers["counts"])
        expression = source.X
        counts = source.layers["counts"]
        raw = source.raw
        source_copy = mocker.spy(source, "copy")
        mocker.patch.object(table_owner.ad, "read_h5ad", return_value=source)
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("expression densified"))
        mocker.patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("expression densified"))
        task = mocker.Mock()
        with Callback(pretask=task):
            table = steinbock(tmp_path).tables["table"]
        task.assert_not_called()
        source_copy.assert_not_called()
        assert table is source
        assert table.X is expression
        assert table.layers["counts"] is counts
        assert table.raw is raw
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert table.X.has_canonical_format
        assert source_path.read_bytes() == before
        np.testing.assert_array_equal(table.X.data, np.arange(1, 4, dtype=np.float32))
        np.testing.assert_array_equal(table.layers["metadata"], fixture.source.layers["metadata"])
        np.testing.assert_array_equal(table.obsm["other_embedding"], fixture.source.obsm["other_embedding"])
        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(fixture.source.obs, pd.DataFrame)
        for key in ("centroid-0", "centroid-1", "annotation"):
            pd.testing.assert_series_equal(table.obs[key], fixture.source.obs[key], check_exact=True)

    @pytest.mark.parametrize("spacing", [None, (2.0, 3.0)])
    def test_unknown_embedding_preserved_without_invented_semantics(
        self, tmp_path: Path, spacing: tuple[float, float] | None
    ) -> None:
        _write_dataset(tmp_path, {"sample-a": spacing})
        path = tmp_path / "cells.h5ad"
        table = ad.read_h5ad(path)
        unknown = np.arange(12, dtype=np.int16).reshape(3, 4)
        table.obsm["spatial"] = unknown
        table.write_h5ad(path)
        actual = steinbock(tmp_path).tables["table"]
        np.testing.assert_array_equal(actual.obsm["spatial_source"], unknown, strict=True)
        assert "spatial_source" not in actual.uns[POSITIONS_KEY]["coordinates"]

    def test_imc_calibration_is_resolved_once_and_conflicts_fail(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_dataset(tmp_path, {"sample-a": None})
        path = tmp_path / "images.csv"
        path.write_text(
            "image,width_px,height_px,acquisition_width_um,acquisition_height_um\nsample-a.tiff,12,10,24,30\n"
        )
        metadata = mocker.spy(reader_owner, "read_image_metadata")
        table = steinbock(tmp_path).tables["table"]
        metadata.assert_called_once()
        assert "images.csv" in table.uns[POSITIONS_KEY]["coordinates"]["spatial"]["calibration"]["global"]
        path.write_text(
            "image,width_px,height_px,acquisition_width_um,acquisition_height_um\nsample-a.tiff,12,10,24,\n"
        )
        with pytest.raises(ReaderFormatError, match="partial physical extents"):
            steinbock(tmp_path)

    def test_conflicting_ome_and_imc_calibration_does_not_fallback(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, {"sample-a": (2.0, 3.0)})
        (tmp_path / "images.csv").write_text(
            "image,width_px,height_px,acquisition_width_um,acquisition_height_um\nsample-a.tiff,12,10,24,40\n"
        )
        with pytest.raises(ReaderFormatError, match="spacing"):
            steinbock(tmp_path)

    def test_partial_ome_calibration_retains_native_coordinates(self, tmp_path: Path) -> None:
        fixture = _write_dataset(tmp_path, {"sample-a": None})
        tifffile.imwrite(
            tmp_path / "ome" / "sample-a.ome.tiff",
            np.zeros((2, 10, 12), dtype=np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "PhysicalSizeX": 2, "PhysicalSizeXUnit": "µm"},
        )
        table = steinbock(tmp_path).tables["table"]
        assert "spatial" not in table.obsm
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], fixture.native_xy)

    def test_finite_source_coordinates_cannot_silently_overflow_physical_xy(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, {"sample-a": (1e308, 3.0)})
        with pytest.raises(ReaderFormatError, match="nonfinite physical positions"):
            steinbock(tmp_path)

    @pytest.mark.parametrize("storage", ["h5ad", "anndata_zarr", "spatialdata_zarr"])
    def test_position_snapshots_persist_with_exact_source_and_categorical_frames(
        self, tmp_path: Path, storage: str
    ) -> None:
        root = tmp_path / "input"
        root.mkdir()
        _write_dataset(root, {"sample-a": (2.0, 3.0), "sample-b": (5.0, 7.0)})
        source = ad.read_h5ad(root / "cells.h5ad")
        unknown = np.arange(24, dtype=np.int16).reshape(6, 4)
        source.obsm["spatial"] = unknown
        source.write_h5ad(root / "cells.h5ad")
        result = steinbock(root)
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
        for key in ("spatial", "spatial_pixels", "spatial_source"):
            np.testing.assert_array_equal(restored.obsm[key], table.obsm[key], strict=True)
        np.testing.assert_array_equal(restored.obsm["spatial_source"], unknown, strict=True)
        assert_equal(restored.uns[POSITIONS_KEY], table.uns[POSITIONS_KEY], exact=True)
        for key in ("spatial_frame", "spatial_pixel_frame", "region"):
            pd.testing.assert_series_equal(restored.obs[key], table.obs[key], check_exact=True)
        for number, selection in enumerate(([4, 0, 3], np.asarray([True, False] * 3), np.empty(0, dtype=np.int64))):
            selected = restored[selection].copy()
            for key in ("spatial", "spatial_pixels", "spatial_source"):
                np.testing.assert_array_equal(selected.obsm[key], table.obsm[key][selection], strict=True)
            assert_equal(selected.uns[POSITIONS_KEY], table.uns[POSITIONS_KEY], exact=True)
            selected.write_h5ad(tmp_path / f"slice-{number}.h5ad")
            sliced = ad.read_h5ad(tmp_path / f"slice-{number}.h5ad")
            assert_equal(sliced.obsm, selected.obsm, exact=True)
            assert_equal(sliced.uns[POSITIONS_KEY], selected.uns[POSITIONS_KEY], exact=True)
