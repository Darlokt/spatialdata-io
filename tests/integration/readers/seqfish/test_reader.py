"""Verify seqFISH composition, scientific corrections and lazy ownership."""

from __future__ import annotations

import json
import pickle
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import tifffile
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Scale, get_transformation

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

    def test_sorted_discovery_and_explicit_roi_order(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path)
        discovered = seqfish(root)
        selected = seqfish(root, rois=("Roi2", "Roi1"))
        assert list(discovered.tables) == ["table_Roi1", "table_Roi2"]
        assert list(selected.tables) == ["table_Roi2", "table_Roi1"]
        assert list(seqfish(root, rois=("Roi2",)).tables) == ["table_Roi2"]

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
        assert get_transformation(result.images["Roi1_DAPI"], to_coordinate_system="Roi1") == Scale(
            [0.25, 0.5], axes=("y", "x")
        )
        assert get_transformation(boundaries, to_coordinate_system="Roi1") == Scale([0.25, 0.5], axes=("y", "x"))

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

    def test_partial_image_reads_one_native_strip(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source_read = mocker.spy(tiff_dask, "_read_tiff_selection")
        result = seqfish(root)
        source_read.assert_not_called()
        selected = result.images["Roi1_DAPI"].isel(y=slice(0, 1), x=slice(0, 1)).data.compute()
        assert_array_equal(selected, [[[0]]])
        source_read.assert_called_once()
        assert source_read.call_args.args[1] == (slice(0, 2), slice(0, 12))

    def test_no_generated_pyramid_by_default_and_explicit_pyramid(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        result = seqfish(root)
        assert result.images["Roi1_DAPI"].dims == ("c", "y", "x")
        pyramid = seqfish(root, scale_factors=(2,))
        assert list(pyramid.images["Roi1_DAPI"].children) == ["scale0", "scale1"]
        assert list(pyramid.labels["Roi1_Segmentation"].children) == ["scale0", "scale1"]

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
        mocker.patch("spatialdata_io.readers.seqfish._transcripts._BLOCK_BYTES", 256)
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

    def test_conflicting_property_and_feature_ids_fail(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_Boundaries.geojson"
        collection = json.loads(source.read_text(encoding="utf-8"))
        collection["features"][0]["properties"]["id"] = "90000"
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
        assert_array_equal(result.tables["table_Roi1"].obs["instance_id"], [identifier, 2])

    def test_duplicate_cell_identifiers_fail(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path, rois=("Roi1",))
        source = root / "Roi1_CellCoordinates.csv"
        cells = pd.read_csv(source)
        cells.loc[1, "label"] = 9
        cells.to_csv(source, index=False)
        with pytest.raises(ReaderFormatError, match=r"duplicate|unique"):
            seqfish(root)

    def test_write_read_preserves_values_and_linkage(self, tmp_path: Path) -> None:
        root = write_seqfish_dataset(tmp_path / "source", rois=("Roi1",), cell_ids=(70000, 2))
        result = seqfish(root, shape_geometry="both")
        assert isinstance(result.tables["table_Roi1"].X, sparse.csr_array)
        output = tmp_path / "output.zarr"
        result.write(output)
        restored = read_zarr(output)
        table = restored.tables["table_Roi1"]
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert_array_equal(table.X.data, [3, 5])
        assert_array_equal(table.obs["instance_id"], [70000, 2])
        assert get_table_keys(table) == ("Roi1_Segmentation", "region", "instance_id")
        assert restored.shapes["Roi1_Boundaries"].index.tolist() == [70000, 2]
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
