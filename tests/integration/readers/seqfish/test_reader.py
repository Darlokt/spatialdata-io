"""Verify seqFISH composition, scientific corrections and lazy ownership."""

from __future__ import annotations

import json
import pickle
from functools import partial
from itertools import product
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import tifffile
from anndata.tests.helpers import assert_equal
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from scipy import sparse
from shapely.geometry import Polygon
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import Scale, get_transformation

import spatialdata_io.readers.seqfish._rasters as raster_module
import spatialdata_io.readers.seqfish._reader as reader_module
from spatialdata_io.readers._utils.csv import read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers.seqfish import seqfish
from tests.integration.readers.seqfish._fixtures import write_seqfish_dataset

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture


class TestSeqfish:
    """Exercise the public transaction using actual on-disk vendor artifacts."""

    @pytest.mark.parametrize("geometry", ["polygons", "both"])
    def test_polygons_only_inspects_calibration_without_pixel_graphs(
        self, tmp_path: Path, mocker: MockerFixture, geometry: Literal["polygons", "both"]
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), reorder_boundaries=True)
        tifffile.imwrite(
            root / "Roi1_DAPI.tiff",
            np.zeros((256, 320), dtype=np.uint16),
            ome=True,
            tile=(16, 16),
            photometric="minisblack",
            metadata={"axes": "YX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.25},
        )
        with_raster = seqfish(root, include_labels=False, include_transcripts=False, shape_geometry=geometry)
        inspection = mocker.spy(raster_module, "inspect_tiff")
        graph = mocker.spy(raster_module, "read_tiff")
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            result = seqfish(
                root, include_images=False, include_labels=False, include_transcripts=False, shape_geometry=geometry
            )
        inspection.assert_called_once()
        graph.assert_not_called()
        tasks.assert_not_called()
        assert not result.images
        assert_equal(result.tables["table_Roi1"], with_raster.tables["table_Roi1"], exact=True)
        assert result.shapes.keys() == with_raster.shapes.keys()
        for name, shapes in result.shapes.items():
            pd.testing.assert_frame_equal(shapes, with_raster.shapes[name])
            assert shapes.geometry.to_wkb().tolist() == with_raster.shapes[name].geometry.to_wkb().tolist()
            assert get_transformation(shapes, get_all=True) == get_transformation(
                with_raster.shapes[name], get_all=True
            )
        boundaries = result.shapes["Roi1_Boundaries"]
        assert boundaries.index.tolist() == [9, 2]
        assert boundaries["objectType"].tolist() == ["cell", "cell"]
        polygon = boundaries.geometry.iloc[0]
        assert isinstance(polygon, Polygon)
        assert_array_equal(polygon.exterior.coords, [[2, 1], [5, 1], [5, 3], [2, 3], [2, 1]])
        assert get_transformation(boundaries, to_coordinate_system="global") == Scale([0.25, 0.5], axes=("y", "x"))

    def test_calibration_only_accepts_nonspatial_axis_excluded_from_products(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        tifffile.imwrite(
            root / "Roi1_DAPI.tiff",
            np.zeros((2, 8, 12), dtype=np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "ZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.25},
        )
        result = seqfish(root, include_images=False, include_labels=False, include_transcripts=False)
        assert result.shapes["Roi1_Boundaries"].index.tolist() == [9, 2]
        with pytest.raises(ReaderFormatError, match="singleton additional axes"):
            seqfish(root, include_labels=False, include_transcripts=False)

    def test_polygons_still_require_valid_calibration(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        tifffile.imwrite(
            root / "Roi1_DAPI.tiff",
            np.zeros((8, 12), dtype=np.uint16),
            ome=True,
            metadata={"axes": "YX", "PhysicalSizeX": 0.5},
        )
        with pytest.raises(ReaderFormatError, match="physical"):
            seqfish(root, include_images=False, include_labels=False, include_transcripts=False)

    def test_mixed_uint64_transcript_assignments_persist(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root = write_seqfish_dataset(tmp_path / "source", rois=("Roi1",))
        assignments = [0, 2, 2**53 + 1, 2**63 + 1, 2**64 - 1] * 5
        (root / "Roi1_TranscriptList.csv").write_text(
            "name,x,y,cell,z\n" + "".join(f"GeneA,{row},2,{cell},{cell}\n" for row, cell in enumerate(assignments)),
            encoding="utf-8",
        )
        mocker.patch(
            "spatialdata_io.readers.seqfish._transcripts.read_csv_blocks", partial(read_csv_blocks, block_bytes=96)
        )
        result = seqfish(root)
        source_points = result.points["Roi1_TranscriptList"]
        assert source_points.npartitions > 3
        assert source_points.dtypes["cell"] == pd.ArrowDtype(pa.uint64())
        destination = tmp_path / "mixed_assignments.zarr"
        result.write(destination)
        restored = read_zarr(destination).points["Roi1_TranscriptList"].compute()
        assert restored["cell"].dtype == pd.ArrowDtype(pa.uint64())
        assert restored["cell"].tolist() == assignments
        assert restored["z_raw"].dtype == pd.ArrowDtype(pa.uint64())
        assert restored["z_raw"].tolist() == assignments
        assert restored["x"].tolist() == list(range(len(assignments)))

    @pytest.mark.parametrize("rois", [("Roi1",), ("Roi1", "Roi2")])
    def test_physical_declarations_follow_selected_roi_frame(self, tmp_path: Path, rois: tuple[str, ...]) -> None:
        root = write_seqfish_dataset(tmp_path, rois=rois)
        result = seqfish(root, include_images=False, include_transcripts=False, include_labels=False)
        for roi in rois:
            table = result.tables[f"table_{roi}"]
            description = table.uns["spatialdata_io_positions"]["coordinates"]["spatial"]
            assert description["frame"] == ("global" if len(rois) == 1 else roi)
            assert description["units"] == "micrometer"
            np.testing.assert_array_equal(table.obsm["spatial"], [[1.5, 0.5], [4.5, 1.25]])

    def test_preserves_independent_property_ids(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), reorder_boundaries=True)
        path = root / "Roi1_Boundaries.geojson"
        document = json.loads(path.read_text())
        for feature in document["features"]:
            feature["properties"]["id"] = f"annotation-{feature['id']}"
        path.write_text(json.dumps(document))
        result = seqfish(root, shape_geometry="polygons")
        shapes = result.shapes["Roi1_Boundaries"]
        assert shapes.index.tolist() == [9, 2]
        assert shapes["id"].tolist() == ["annotation-9", "annotation-2"]
        assert result.tables["table_Roi1"].obs["instance_id"].tolist() == [9, 2]

    @pytest.mark.parametrize("labels", [False, True])
    @pytest.mark.parametrize("shapes", [False, True])
    @pytest.mark.parametrize("geometry", ["polygons", "circles", "both"])
    def test_drop_filters_only_selected_polygon_entities(
        self, tmp_path: Path, *, labels: bool, shapes: bool, geometry: Literal["polygons", "circles", "both"]
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), reorder_boundaries=True, invalid_cell_ids=(9,))
        completed: list[object] = []
        with Callback(pretask=lambda key, *_: completed.append(key)):
            result = seqfish(
                root, include_labels=labels, include_shapes=shapes, shape_geometry=geometry, invalid_geometries="drop"
            )
        assert not completed
        filtered = shapes and geometry in {"polygons", "both"}
        table = result.tables["table_Roi1"]
        assert table.obs["instance_id"].tolist() == ([2] if filtered else [9, 2])
        assert table.obs_names.tolist() == (["1"] if filtered else ["0", "1"])
        assert isinstance(table.X, sparse.csr_array)
        assert_array_equal(table.X.data, [5] if filtered else [3, 5])
        assert table.var_names.tolist() == ["GeneB", "GeneA"]
        if filtered:
            assert result.shapes["Roi1_Boundaries"].index.tolist() == [2]
        if shapes and geometry in {"circles", "both"}:
            assert result.shapes["Roi1_CellCoordinates"].index.tolist() == ([2] if filtered else [9, 2])
        if labels:
            assert 9 in np.unique(result.labels["Roi1_Segmentation"].data.compute())
        assert_array_equal(result.points["Roi1_TranscriptList"].compute()["cell"], [2, 9, 0])

    @pytest.mark.parametrize("labels", [False, True])
    def test_polygon_drop_preserves_full_uint64_membership_and_persistence(
        self, tmp_path: Path, *, labels: bool
    ) -> None:
        identifiers = (2**63, 2**64 - 1)
        root = write_seqfish_dataset(
            tmp_path / "source",
            rois=("Roi1",),
            cell_ids=identifiers,
            invalid_cell_ids=(identifiers[0],),
            reorder_cells=True,
            reorder_boundaries=True,
        )
        result = seqfish(root, include_labels=labels, shape_geometry="both", invalid_geometries="drop")
        destination = tmp_path / "dropped.zarr"
        result.write(destination)
        restored = read_zarr(destination)
        table = restored.tables["table_Roi1"]
        assert table.obs["instance_id"].tolist() == [identifiers[1]]
        assert table.obs["instance_id"].dtype == np.dtype(np.uint64)
        assert table.obs_names.tolist() == ["1"]
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert_array_equal(table.X.data, [5])
        for name in ("Roi1_Boundaries", "Roi1_CellCoordinates"):
            assert restored.shapes[name].index.tolist() == [identifiers[1]]
            assert result.shapes[name].index.dtype == pd.ArrowDtype(pa.uint64())
            # GeoParquet round trips retain exact IDs in native uint64 indexes.
            assert restored.shapes[name].index.dtype == np.dtype(np.uint64)
        assert restored.points["Roi1_TranscriptList"].compute()["cell"].tolist() == [identifiers[1], identifiers[0], 0]
        if labels:
            assert get_table_keys(table) == ("Roi1_Segmentation", "region", "instance_id")
            assert np.unique(restored.labels["Roi1_Segmentation"].data.compute()).tolist() == [0, *identifiers]
        else:
            assert "spatialdata_attrs" not in table.uns

    @pytest.mark.parametrize("retain_wide", [False, True])
    def test_polygon_drop_selects_retained_count_dtype(self, tmp_path: Path, *, retain_wide: bool) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), invalid_cell_ids=(9,))
        first, second = (3, 2**24 + 1) if retain_wide else (2**24 + 1, 5)
        (root / "Roi1_CellxGene.csv").write_text(f",GeneB,GeneA\n9,{first},0\n2,0,{second}\n", encoding="utf-8")
        result = seqfish(
            root,
            include_images=False,
            include_transcripts=False,
            include_labels=False,
            shape_geometry="polygons",
            invalid_geometries="drop",
        )
        table = result.tables["table_Roi1"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.float64 if retain_wide else np.float32)
        assert_array_equal(table.X.data, [second])
        assert table.obs_names.tolist() == ["1"]
        assert table.obs["instance_id"].tolist() == [2]

    def test_polygon_drop_does_not_hide_source_identity_disagreement(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), invalid_cell_ids=(9,))
        source = root / "Roi1_Boundaries.geojson"
        document = json.loads(source.read_text())
        document["features"][0]["id"] = "99"
        source.write_text(json.dumps(document))
        with pytest.raises(ReaderFormatError, match="ID disagreement"):
            seqfish(root, invalid_geometries="drop")

    def test_all_polygons_dropped_within_one_roi_fails(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), invalid_cell_ids=(9, 2))
        write_seqfish_dataset(root, rois=("Roi2",))
        with pytest.raises(ReaderFormatError, match=r"all|empty"):
            seqfish(root, invalid_geometries="drop")

    def test_single_and_multiple_roi_frame_names(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path)
        single = seqfish(root, rois=("Roi1",), shape_geometry="both")
        multiple = seqfish(root)
        for elements in (single.images, single.labels, single.points, single.shapes):
            for element in elements.values():
                frames = get_transformation(element, get_all=True)
                assert isinstance(frames, dict)
                assert set(frames) == {"global"}
        for element in (multiple.images["Roi1_DAPI"], multiple.images["Roi2_DAPI"]):
            frames = get_transformation(element, get_all=True)
            assert isinstance(frames, dict)
            assert set(frames) in ({"Roi1"}, {"Roi2"})

    def test_invalid_polygon_policy_precedes_discovery(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid_geometries"):
            seqfish(tmp_path / "absent", invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_options_are_validated_in_signature_order_before_io(self, tmp_path: Path) -> None:
        absent = tmp_path / "absent"
        with pytest.raises(ValueError, match="shape_geometry"):
            seqfish(absent, shape_geometry="all", invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
        with pytest.raises(ValueError, match="rois"):
            seqfish(absent, rois=[], scale_factors=2)  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
        with pytest.raises(TypeError, match="scale_factors"):
            seqfish(absent, scale_factors=2, path_policy="lenient")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.
        with pytest.raises(ValueError, match="path_policy"):
            seqfish(absent, path_policy="lenient")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_file_derived_roi_names_are_repaired_once_and_persist(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path / "source", rois=("Roi 1", "Roi2"))
        with pytest.warns(UserWarning, match="ROI file prefix") as caught:
            result = seqfish(root, shape_geometry="both")
        assert [str(warning.message) for warning in caught] == [
            (
                "seqfish: the ROI file prefix 'Roi 1' is not a usable SpatialData element name; using 'Roi_1'. "
                "Rename the source files to choose the name explicitly."
            )
        ]
        assert caught[0].filename == __file__
        assert list(result.tables) == ["table_Roi_1", "table_Roi2"]
        assert set(result.images) == {"Roi_1_DAPI", "Roi2_DAPI"}
        assert set(result.shapes) == {
            "Roi_1_Boundaries",
            "Roi_1_CellCoordinates",
            "Roi2_Boundaries",
            "Roi2_CellCoordinates",
        }
        table = result.tables["table_Roi_1"]
        assert get_table_keys(table) == ("Roi_1_Segmentation", "region", "instance_id")
        assert table.uns["spatialdata_io_positions"]["coordinates"]["spatial"]["frame"] == "Roi_1"
        for element in (result.images["Roi_1_DAPI"], result.points["Roi_1_TranscriptList"]):
            frames = get_transformation(element, get_all=True)
            assert isinstance(frames, dict)
            assert set(frames) == {"Roi_1"}

        output = tmp_path / "output.zarr"
        result.write(output)
        restored = read_zarr(output)
        assert set(restored.tables) == {"table_Roi_1", "table_Roi2"}
        assert get_table_keys(restored.tables["table_Roi_1"]) == ("Roi_1_Segmentation", "region", "instance_id")

    def test_roi_names_equal_ignoring_case_fail_before_reading_tables(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1", "roi1"))
        tables = mocker.spy(reader_module, "read_table")
        with pytest.raises(ReaderFormatError, match="equal ignoring case"):
            seqfish(root)
        tables.assert_not_called()
        assert list(seqfish(root, rois=["roi1"]).tables) == ["table_roi1"]

    def test_sorted_discovery_and_explicit_roi_order(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path)
        discovered = seqfish(root)
        selected = seqfish(root, rois=("Roi2", "Roi1"))
        assert list(discovered.tables) == ["table_Roi1", "table_Roi2"]
        assert list(selected.tables) == ["table_Roi2", "table_Roi1"]
        assert list(seqfish(root, rois=("Roi2",)).tables) == ["table_Roi2"]

    def test_macos_apple_double_companions_are_ignored(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path)
        expected = seqfish(root)
        # macOS writes these metadata companions when copying to non-HFS volumes.
        for name in ("._Roi1_CellxGene.csv", "._Roi1_DAPI.tiff", "._Roi2_CellCoordinates.csv"):
            (root / name).write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X")

        result = seqfish(root)

        assert list(result.tables) == ["table_Roi1", "table_Roi2"]
        assert_spatial_data_objects_are_identical(result, expected)

    def test_cell_ids_exceeding_label_dtype_fail_without_pixel_tasks(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",), cell_ids=(70000, 2))
        labels = np.zeros((8, 12), dtype=np.uint16)
        labels[4:7, 8:11] = 2
        tifffile.imwrite(root / "Roi1_Segmentation.tiff", labels, rowsperstrip=2, compression="deflate")
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        completed: list[str] = []
        with (
            Callback(posttask=lambda key, *_: completed.append(str(key))),
            pytest.raises(ReaderFormatError, match="label dtype"),
        ):
            seqfish(root)
        assert completed == []
        decoder.assert_not_called()

    def test_large_ids_and_reordered_sources_align_by_id(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(
            tmp_path,
            rois=("Roi1",),
            cell_ids=(70000, 2),
            reorder_cells=True,
            reorder_boundaries=True,
        )
        result = seqfish(root, shape_geometry="both")
        table = result.tables["table_Roi1"]
        assert isinstance(table.X, sparse.csr_array)
        assert_array_equal(table.X.data, [3, 5])
        assert_array_equal(table.obs["instance_id"], [70000, 2])
        assert_array_equal(table.obsm["spatial"], [[1.5, 0.5], [4.5, 1.25]])
        boundaries = result.shapes["Roi1_Boundaries"]
        assert boundaries.index.tolist() == [70000, 2]
        assert_array_equal(boundaries.bounds, [[2, 1, 5, 3], [8, 4, 11, 7]])
        assert result.shapes["Roi1_CellCoordinates"].index.tolist() == [70000, 2]
        assert get_table_keys(table) == ("Roi1_Segmentation", "region", "instance_id")
        assert get_transformation(result.images["Roi1_DAPI"], to_coordinate_system="global") == Scale(
            [0.25, 0.5], axes=("y", "x")
        )
        assert get_transformation(boundaries, to_coordinate_system="global") == Scale([0.25, 0.5], axes=("y", "x"))

    @pytest.mark.parametrize("geometry", ["polygons", "circles", "both"])
    def test_geometry_selection(self, tmp_path: Path, geometry: Literal["polygons", "circles", "both"]) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        result = seqfish(root, shape_geometry=geometry)
        expected = {
            "polygons": {"Roi1_Boundaries"},
            "circles": {"Roi1_CellCoordinates"},
            "both": {"Roi1_Boundaries", "Roi1_CellCoordinates"},
        }
        assert set(result.shapes) == expected[geometry]
        assert get_table_keys(result.tables["table_Roi1"])[0] == "Roi1_Segmentation"
        disabled = seqfish(root, include_shapes=False, shape_geometry=geometry)
        assert not disabled.shapes

    def test_wide_expression_remains_sparse_without_densification(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        identifiers = np.arange(1024, dtype=np.int64) + 2**53 + 1
        values = np.zeros((1024, 128), dtype=np.int64)
        values[np.arange(1024), np.arange(1024) % 128] = np.arange(1024) + 1
        pd.DataFrame(values, index=pd.Index(identifiers, name=""), columns=[f"Gene{i}" for i in range(128)]).to_csv(
            root / "Roi1_CellxGene.csv"
        )
        pd.DataFrame(
            {"label": identifiers, "area": np.ones(1024), "center_x": np.arange(1024), "center_y": np.zeros(1024)}
        ).to_csv(root / "Roi1_CellCoordinates.csv", index=False)
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("expression densified"))
        mocker.patch.object(sparse.csr_array, "todense", side_effect=AssertionError("expression densified"))
        result = seqfish(
            root, include_images=False, include_labels=False, include_transcripts=False, include_shapes=False
        )
        table = result.tables["table_Roi1"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.shape == (1024, 128)
        assert table.X.nnz == 1024
        assert_array_equal(table.X.data, np.arange(1024) + 1)
        assert_array_equal(table.obs["instance_id"], identifiers)

    def test_table_only_requires_no_raster_transcripts_or_boundaries(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        for suffix in ("DAPI.tiff", "Segmentation.tiff", "TranscriptList.csv", "Boundaries.geojson"):
            (root / f"Roi1_{suffix}").unlink()
        result = seqfish(
            root, include_images=False, include_labels=False, include_transcripts=False, include_shapes=False
        )
        table = result.tables["table_Roi1"]
        assert isinstance(table.X, sparse.csr_array)
        assert "spatialdata_attrs" not in table.uns
        assert not result.images
        assert not result.labels
        assert not result.points
        assert not result.shapes

    @pytest.mark.parametrize("included", list(product((False, True), repeat=4)))
    def test_all_product_inclusions_resolve_only_dependencies(
        self, tmp_path: Path, included: tuple[bool, bool, bool, bool]
    ) -> None:
        images, labels, transcripts, shapes = included
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        if not (images or labels or shapes):
            (root / "Roi1_DAPI.tiff").unlink()
        for enabled, suffix in (
            (labels, "Segmentation.tiff"),
            (transcripts, "TranscriptList.csv"),
            (shapes, "Boundaries.geojson"),
        ):
            if not enabled:
                (root / f"Roi1_{suffix}").unlink()
        result = seqfish(
            root,
            include_images=images,
            include_labels=labels,
            include_transcripts=transcripts,
            include_shapes=shapes,
            shape_geometry="both",
        )
        assert bool(result.images) is images
        assert bool(result.labels) is labels
        assert bool(result.points) is transcripts
        assert bool(result.shapes) is shapes
        assert ("spatialdata_attrs" in result.tables["table_Roi1"].uns) is labels
        if shapes:
            assert set(result.shapes) == {"Roi1_Boundaries", "Roi1_CellCoordinates"}

    def test_labels_disabled_table_is_unlinked_without_shape_retargeting(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        result = seqfish(root, include_labels=False, shape_geometry="both")
        assert "spatialdata_attrs" not in result.tables["table_Roi1"].uns
        assert set(result.shapes) == {"Roi1_Boundaries", "Roi1_CellCoordinates"}

    def test_raster_and_transcript_graph_construction_executes_no_tasks(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        completed: list[str] = []
        with Callback(posttask=lambda key, *_: completed.append(str(key))):
            result = seqfish(root, chunks=(1, 2, 4))
        assert completed == []
        decoder.assert_not_called()
        image = result.images["Roi1_DAPI"]
        assert image.data.chunks == ((1,), (2, 2, 2, 2), (4, 4, 4))
        assert result.labels["Roi1_Segmentation"].data.chunks == ((2, 2, 2, 2), (4, 4, 4))
        restored_image = pickle.loads(pickle.dumps(image.data))  # noqa: S301, RUF100 - Test-owned lazy graph.
        assert_array_equal(restored_image[:, :1, :1].compute(), [[[0]]])
        restored_points = pickle.loads(  # noqa: S301, RUF100 - Test-owned lazy graph.
            pickle.dumps(result.points["Roi1_TranscriptList"])
        )
        points = restored_points.compute()
        assert points["name"].tolist() == ["GeneA", "GeneB", "GeneA"]
        assert "z" not in points
        assert_array_equal(points["z_raw"], [1, 2, 1])
        assert points["z_raw"].dtype.kind in "iu"

    def test_partial_image_reads_one_row_band(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source_read = mocker.spy(tiff_dask, "_read_tiff_selection")
        result = seqfish(root)
        source_read.assert_not_called()
        selected = result.images["Roi1_DAPI"].isel(y=slice(0, 1), x=slice(0, 1)).data.compute()
        assert_array_equal(selected, [[[0]]])
        source_read.assert_called_once()
        # The 2-row strips of this small image form one row band.
        assert source_read.call_args.args[1] == (slice(0, 8), slice(0, 12))

    def test_no_generated_pyramid_by_default_and_explicit_pyramid(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        result = seqfish(root)
        assert result.images["Roi1_DAPI"].dims == ("c", "y", "x")
        assert result.images["Roi1_DAPI"].data.chunks == ((1,), (8,), (12,))
        assert result.labels["Roi1_Segmentation"].data.chunks == ((8,), (12,))
        pyramid = seqfish(root, scale_factors=(2,))
        assert list(pyramid.images["Roi1_DAPI"].children) == ["scale0", "scale1"]
        assert list(pyramid.labels["Roi1_Segmentation"].children) == ["scale0", "scale1"]

    def test_invalid_raster_option_fails_before_io(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="scale_factors"):
            seqfish(tmp_path / "missing", scale_factors=2)  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_zero_count_control_is_part_of_known_transcript_vocabulary(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_CellxGene.csv"
        counts = pd.read_csv(source, index_col=0)
        counts["Blank control"] = [0, 0]
        counts.to_csv(source)
        source = root / "Roi1_TranscriptList.csv"
        events = pd.read_csv(source)
        events.loc[2, "name"] = "Blank control"
        events.to_csv(source, index=False)
        result = seqfish(root)
        table = result.tables["table_Roi1"]
        assert table.var_names.tolist() == ["GeneB", "GeneA", "Blank control"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.nnz == 2
        points = result.points["Roi1_TranscriptList"].compute()
        assert points["name"].cat.categories.tolist() == ["GeneB", "GeneA", "Blank control"]
        assert points["name"].tolist() == ["GeneA", "GeneB", "Blank control"]

    def test_unknown_transcript_feature_fails_on_compute(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_TranscriptList.csv"
        source.write_text(source.read_text(encoding="utf-8").replace("GeneA", "Unknown"), encoding="utf-8")
        result = seqfish(root)
        with pytest.raises(ReaderFormatError, match=r"Unknown|feature|vocabulary"):
            result.points["Roi1_TranscriptList"].compute()

    def test_projected_quoted_transcripts_validate_requested_partition_only(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        counts_source = root / "Roi1_CellxGene.csv"
        counts = pd.read_csv(counts_source, index_col=0)
        counts.columns = ["Gene,B", "GeneA"]
        counts.to_csv(counts_source)
        events = pd.DataFrame(
            {
                "name": ["Gene,B", "GeneA"] * 40,
                "x": np.arange(80, dtype=np.float64) / 4,
                "y": ["0.5"] * 79 + ["nan"],
                "cell": [9, 0] * 40,
                "ignored": ['a comma, and a "quote"'] * 80,
            }
        )
        events.to_csv(root / "Roi1_TranscriptList.csv", index=False)
        mocker.patch(
            "spatialdata_io.readers.seqfish._transcripts.read_csv_blocks", partial(read_csv_blocks, block_bytes=256)
        )
        constructed: list[str] = []
        with Callback(posttask=lambda key, *_: constructed.append(str(key))):
            result = seqfish(root)
        assert constructed == []
        points = result.points["Roi1_TranscriptList"]
        assert points.npartitions > 3
        assert "ignored" not in points.columns
        executed: list[str] = []
        with Callback(posttask=lambda key, *_: executed.append(str(key))):
            first = points.partitions[0].compute()
        assert first["name"].tolist() == ["Gene,B", "GeneA"] * (len(first) // 2) + (
            ["Gene,B"] if len(first) % 2 else []
        )
        assert len(executed) < points.npartitions
        with pytest.raises(ReaderFormatError, match="Roi1_TranscriptList"):
            points.partitions[-1].compute()

    @pytest.mark.parametrize("bad_record", ["GeneA,1,2,1,0,excess", "GeneA,1,2,1"])
    def test_transcript_record_width_is_validated_during_compute(self, tmp_path: Path, bad_record: str) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_TranscriptList.csv"
        with source.open("a", encoding="utf-8") as handle:
            handle.write(bad_record + "\n")
        result = seqfish(root)
        with pytest.raises(ReaderFormatError, match="Roi1_TranscriptList"):
            result.points["Roi1_TranscriptList"].compute()

    def test_missing_geojson_source_ids_fail(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_Boundaries.geojson"
        source.write_text(source.read_text(encoding="utf-8").replace('"id": "9",', ""), encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=r"id|ID"):
            seqfish(root)

    def test_property_id_cannot_replace_missing_feature_id(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_Boundaries.geojson"
        collection = json.loads(source.read_text(encoding="utf-8"))
        feature = collection["features"][0]
        feature["properties"]["id"] = feature.pop("id")
        source.write_text(json.dumps(collection), encoding="utf-8")
        with pytest.raises(ReaderFormatError, match=r"id|ID"):
            seqfish(root)

    def test_matching_property_id_does_not_narrow_large_source_id(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        identifier = 2**53 + 1
        counts_source = root / "Roi1_CellxGene.csv"
        counts = pd.read_csv(counts_source, index_col=0)
        counts.index = pd.Index([identifier, 2], name="")
        counts.to_csv(counts_source)
        cells_source = root / "Roi1_CellCoordinates.csv"
        cells = pd.read_csv(cells_source)
        cells.loc[0, "label"] = identifier
        cells.to_csv(cells_source, index=False)
        source = root / "Roi1_Boundaries.geojson"
        collection = json.loads(source.read_text(encoding="utf-8"))
        collection["features"][0]["id"] = str(identifier)
        collection["features"][0]["properties"]["id"] = identifier
        source.write_text(json.dumps(collection), encoding="utf-8")
        result = seqfish(root, include_labels=False)
        assert result.shapes["Roi1_Boundaries"].index.tolist() == [identifier, 2]
        assert result.shapes["Roi1_Boundaries"]["id"].iloc[0] == identifier
        assert pd.isna(result.shapes["Roi1_Boundaries"]["id"].iloc[1])
        assert_array_equal(result.tables["table_Roi1"].obs["instance_id"], [identifier, 2])

    def test_duplicate_cell_identifiers_fail(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_CellCoordinates.csv"
        cells = pd.read_csv(source)
        cells.loc[1, "label"] = 9
        cells.to_csv(source, index=False)
        with pytest.raises(ReaderFormatError, match=r"duplicate|unique"):
            seqfish(root)

    @pytest.mark.parametrize("identifiers", [(70000, 2), (2**63, 2**64 - 1), (2**64 - 1, 1)])
    def test_write_read_preserves_values_and_linkage(self, tmp_path: Path, identifiers: tuple[int, int]) -> None:
        root = write_seqfish_dataset(
            tmp_path / "source", rois=("Roi1",), cell_ids=identifiers, reorder_cells=True, reorder_boundaries=True
        )
        boundary_path = root / "Roi1_Boundaries.geojson"
        document = json.loads(boundary_path.read_text(encoding="utf-8"))
        # Cover JSON numeric IDs alongside the other feature's string ID.
        document["features"][0]["id"] = int(document["features"][0]["id"])
        boundary_path.write_text(json.dumps(document), encoding="utf-8")
        result = seqfish(root, shape_geometry="both")
        assert isinstance(result.tables["table_Roi1"].X, sparse.csr_array)
        output = tmp_path / "output.zarr"
        result.write(output)
        restored = read_zarr(output)
        table = restored.tables["table_Roi1"]
        assert_equal(
            table.uns["spatialdata_io_positions"],
            result.tables["table_Roi1"].uns["spatialdata_io_positions"],
            exact=True,
        )
        for storage in ("h5ad", "zarr"):
            table_path = tmp_path / f"positions_seqfish.{storage}"
            source_table = result.tables["table_Roi1"]
            if storage == "h5ad":
                source_table.write_h5ad(table_path)
                restored_positions = ad.read_h5ad(table_path)
            else:
                source_table.write_zarr(table_path)
                restored_positions = ad.read_zarr(table_path)
            assert_equal(
                restored_positions.uns["spatialdata_io_positions"],
                source_table.uns["spatialdata_io_positions"],
                exact=True,
            )
            assert_equal(restored_positions.obsm, source_table.obsm, exact=True)
            assert restored_positions.obs["instance_id"].tolist() == list(identifiers)
            assert restored_positions.obs["instance_id"].dtype == np.dtype(np.uint64)
            assert get_table_keys(restored_positions) == ("Roi1_Segmentation", "region", "instance_id")
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert_array_equal(table.X.data, [3, 5])
        assert table.obs["instance_id"].tolist() == list(identifiers)
        assert table.obs["instance_id"].dtype == np.dtype(np.uint64)
        assert get_table_keys(table) == ("Roi1_Segmentation", "region", "instance_id")
        for name in ("Roi1_Boundaries", "Roi1_CellCoordinates"):
            assert restored.shapes[name].index.tolist() == list(identifiers)
            assert result.shapes[name].index.dtype == pd.ArrowDtype(pa.uint64())
            # GeoParquet round trips retain exact IDs in native uint64 indexes.
            assert restored.shapes[name].index.dtype == np.dtype(np.uint64)
        assert np.unique(restored.labels["Roi1_Segmentation"].data.compute()).tolist() == sorted((0, *identifiers))
        assert restored.points["Roi1_TranscriptList"].compute()["cell"].tolist() == [identifiers[1], identifiers[0], 0]
        assert_array_equal(restored.points["Roi1_TranscriptList"].compute()["z_raw"], [1, 2, 1])

    @pytest.mark.data
    def test_managed_output_roundtrip(self, tmp_path: Path, require_test_dataset: Callable[[str], Path]) -> None:
        result = seqfish(require_test_dataset("seqfish"), shape_geometry="both")
        assert isinstance(result.tables["table_Roi1"].X, sparse.csr_array)
        output = tmp_path / "managed.zarr"
        result.write(output)
        restored = read_zarr(output)
        table = restored.tables["table_Roi1"]
        assert table.shape == (62, 12)
        assert_equal(
            table.uns["spatialdata_io_positions"],
            result.tables["table_Roi1"].uns["spatialdata_io_positions"],
            exact=True,
        )
        for storage in ("h5ad", "zarr"):
            table_path = tmp_path / f"positions_seqfish.{storage}"
            source_table = result.tables["table_Roi1"]
            if storage == "h5ad":
                source_table.write_h5ad(table_path)
                restored_positions = ad.read_h5ad(table_path)
            else:
                source_table.write_zarr(table_path)
                restored_positions = ad.read_zarr(table_path)
            assert_equal(
                restored_positions.uns["spatialdata_io_positions"],
                source_table.uns["spatialdata_io_positions"],
                exact=True,
            )
            assert_equal(restored_positions.obsm, source_table.obsm, exact=True)
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert table.var_names.tolist() == [
            "Arg1",
            "Arg2",
            "Tgfb1",
            "Ifngr1",
            "Rac1",
            "Stat6",
            "Ifngr2",
            "Col1a2",
            "P2ry12",
            "Tgfb2",
            "Ssh1",
            "Tmem119",
        ]
        assert get_table_keys(table) == ("Roi1_Segmentation", "region", "instance_id")
        assert_array_equal(table.X.data, result.tables["table_Roi1"].X.data)
        assert_array_equal(table.obsm["spatial"], result.tables["table_Roi1"].obsm["spatial"])
        assert len(restored.points["Roi1_TranscriptList"].compute()) == 9051
        assert restored.shapes["Roi1_Boundaries"].index.tolist() == result.shapes["Roi1_Boundaries"].index.tolist()


class TestSeqfishPathPolicy:
    """Discovered ROI artifacts follow the dataset root's symlink policy."""

    def test_out_of_root_symlinks_require_the_relaxed_policy(
        self, tmp_path: Path, link_outside_root: Callable[[Path], Path]
    ) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        expected = seqfish(root)
        link_outside_root(root / "Roi1_CellxGene.csv")
        link_outside_root(root / "Roi1_DAPI.tiff")

        with pytest.raises(ReaderFormatError, match="path_policy='relaxed'"):
            seqfish(root)

        assert_spatial_data_objects_are_identical(seqfish(root, path_policy="relaxed"), expected)
