"""Exercise scientific composition, lazy products, and MERSCOPE storage ownership."""

from __future__ import annotations

import pickle
import shutil
from typing import TYPE_CHECKING, Literal

import anndata as ad
import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import tifffile
from anndata.tests.helpers import assert_equal
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from scipy import sparse
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io import __version__
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers.merscope import merscope
from tests.integration.readers.merscope._fixtures import write_merscope_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestMerscope:
    """Read actual CSV, GeoParquet and TIFF artifacts through the public API."""

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_nullable_planes_preserve_xy_footprints(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile)
        vpt = None if profile == "standard" else fixture.analysis
        expected = merscope(fixture.root, vpt_directory=vpt)
        path = fixture.analysis / (
            "cell_boundaries.parquet" if profile == "standard" else f"{profile}_micron_space.parquet"
        )
        boundaries = gpd.read_parquet(path)
        boundaries["ZIndex"] = pd.array([None, 2, None, 0], dtype="Int64")
        boundaries.to_parquet(path)
        actual = merscope(fixture.root, vpt_directory=vpt)
        name = "slide_region_polygons"
        assert actual.shapes[name].index.equals(expected.shapes[name].index)
        assert actual.shapes[name].geometry.geom_equals_exact(expected.shapes[name].geometry, tolerance=0).all()
        actual_x = actual.tables["table"].X
        expected_x = expected.tables["table"].X
        assert isinstance(actual_x, sparse.csr_array)
        assert isinstance(expected_x, sparse.csr_array)
        assert_array_equal(actual_x.data, expected_x.data)

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    @pytest.mark.parametrize("current", [False, True])
    def test_drop_whole_cell_keeps_original_transcript_population(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"], *, current: bool
    ) -> None:
        fixture = write_merscope_dataset(
            tmp_path, profile=profile, invalid_first_cell=True, reassigned_transcripts=current
        )
        result = merscope(
            fixture.root, vpt_directory=None if profile == "standard" else fixture.analysis, invalid_geometries="drop"
        )

        shapes = result.shapes["slide_region_polygons"]
        table = result.tables["table"]
        assert shapes.index.tolist() == list(fixture.cell_ids[1:])
        assert table.obs_names.tolist() == list(fixture.cell_ids[1:])
        assert isinstance(table.X, sparse.csr_array)
        increment = 0 if profile == "standard" else 100
        assert_array_equal(table.X.data, [4 + increment, 3 + increment, 1 + increment])
        blank = table.obsm["blank"]
        assert isinstance(blank, sparse.csr_array)
        assert_array_equal(blank.data, [5 + increment])
        events = result.points["slide_region_transcripts"].compute()
        column = "cell_id" if profile == "standard" or current else "original_cell_id"
        assert events[column].tolist() == (
            [*fixture.cell_ids, "-1"] if column == "cell_id" else ["1001", "2002", "3003", "-1"]
        )

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_keep_marks_invalid_cell_source_components(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile, invalid_first_cell=True)
        result = merscope(fixture.root, vpt_directory=None if profile == "standard" else fixture.analysis)

        shapes = result.shapes["slide_region_polygons"]
        assert shapes.index.tolist() == list(fixture.cell_ids)
        assert shapes["geometry_representation"].tolist() == ["source_components", "union_footprint", "union_footprint"]
        geometry = shapes.geometry.iloc[0]
        assert isinstance(geometry, MultiPolygon)
        assert len(geometry.geoms) == 3

    def test_invalid_geometry_drop_does_not_hide_original_foreign_identity(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, invalid_first_cell=True)
        path = fixture.root / "cell_boundaries.parquet"
        frame = gpd.read_parquet(path)
        frame.loc[frame["EntityID"] == 1001, "EntityID"] = 9999
        frame.to_parquet(path)
        with pytest.raises(ReaderFormatError, match=r"original.*populations"):
            merscope(fixture.root, invalid_geometries="drop")

    def test_drop_cannot_hide_extra_invalid_foreign_boundary(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        path = fixture.root / "cell_boundaries.parquet"
        frame = gpd.read_parquet(path)
        extra = frame.iloc[:1].copy()
        extra["EntityID"] = 9999
        extra.geometry = gpd.GeoSeries([Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])], index=extra.index)
        gpd.GeoDataFrame(pd.concat([frame, extra], ignore_index=True)).to_parquet(path)
        with pytest.raises(ReaderFormatError, match=r"original.*populations"):
            merscope(fixture.root, invalid_geometries="drop")

    def test_invalid_components_and_marker_survive_zarr(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path / "source", invalid_first_cell=True)
        result = merscope(fixture.root)
        output = tmp_path / "output.zarr"
        result.write(output)
        restored = read_zarr(output).shapes["slide_region_polygons"]
        expected = result.shapes["slide_region_polygons"]
        assert restored.geometry.to_wkb().tolist() == expected.geometry.to_wkb().tolist()
        # Zarr restores the Arrow-backed string index with pandas' default string dtype.
        pd.testing.assert_series_equal(
            restored["geometry_representation"], expected["geometry_representation"], check_index_type=False
        )

    def test_geometry_drop_does_not_hide_conflicting_source_annotations(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, invalid_first_cell=True)
        path = fixture.root / "cell_metadata.csv"
        frame = pd.read_csv(path, index_col=0)
        frame["dataset_id"] = ["other", "slide_region", "slide_region"]
        frame.to_csv(path)
        with pytest.raises(ReaderFormatError, match=r"conflicting source.*dataset_id"):
            merscope(fixture.root, invalid_geometries="drop")

    def test_geometry_disabled_does_not_filter_or_require_polygons(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, invalid_first_cell=True)
        (fixture.root / "cell_boundaries.parquet").unlink()
        result = merscope(fixture.root, include_cell_boundaries=False, invalid_geometries="drop")
        assert result.tables["table"].obs_names.tolist() == list(fixture.cell_ids)
        assert not result.shapes

    def test_all_cells_dropped_fails_as_requested_element(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        path = fixture.root / "cell_boundaries.parquet"
        frame = gpd.read_parquet(path)
        invalid = MultiPolygon([Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])])
        frame.geometry = gpd.GeoSeries([invalid] * len(frame), index=frame.index)
        frame.to_parquet(path)
        with pytest.raises(ReaderFormatError, match="all cell boundaries dropped"):
            merscope(fixture.root, invalid_geometries="drop")

    def test_native_vectors_do_not_require_images_or_pixel_transform(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        shutil.rmtree(fixture.root / "images")
        result = merscope(fixture.root, include_images=False)
        for element in (result.shapes["slide_region_polygons"], result.points["slide_region_transcripts"]):
            assert get_transformation(element, to_coordinate_system="global") == Identity()

    @pytest.mark.parametrize("with_affine", [True, False])
    def test_table_and_image_frames_without_other_vectors(self, tmp_path: Path, *, with_affine: bool) -> None:
        fixture = write_merscope_dataset(tmp_path)
        if not with_affine:
            (fixture.root / "images" / "micron_to_mosaic_pixel_transform.csv").unlink()
        result = merscope(fixture.root, include_transcripts=False, include_cell_boundaries=False)
        frame = "global" if with_affine else "microns"
        description = result.tables["table"].uns["spatialdata_io_positions"]["coordinates"]["spatial"]
        assert description["frame"] == frame
        assert description["units"] == "micrometer"
        image_frames = get_transformation(result.images["slide_region_z0"], get_all=True)
        assert isinstance(image_frames, dict)
        assert set(image_frames) == ({"global"} if with_affine else {"mosaic_pixels"})
        np.testing.assert_array_equal(result.tables["table"].obsm["spatial"], [[2, 2], [10, 1], [21, 1]])

    def test_image_only_uses_full_inverse_affine(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        result = merscope(fixture.root, include_table=False, include_transcripts=False, include_cell_boundaries=False)
        transform = get_transformation(result.images["slide_region_z0"], to_coordinate_system="global")
        assert not isinstance(transform, dict)
        inverse = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_allclose(inverse @ [14.5, -1, 1], [2, 2, 1], rtol=1e-15, atol=1e-15)

    def test_uncalibrated_images_and_native_vectors_use_distinct_frames(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        (fixture.root / "images" / "micron_to_mosaic_pixel_transform.csv").unlink()
        result = merscope(fixture.root)
        image_frames = get_transformation(result.images["slide_region_z0"], get_all=True)
        vector_frames = get_transformation(result.shapes["slide_region_polygons"], get_all=True)
        assert isinstance(image_frames, dict)
        assert isinstance(vector_frames, dict)
        assert set(image_frames) == {"mosaic_pixels"}
        assert set(vector_frames) == {"microns"}

    def test_invalid_polygon_policy_precedes_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid_geometries"):
            merscope(tmp_path / "missing", invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_provenance_and_default_image_order(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        result = merscope(fixture.root)
        assert list(result.images) == ["slide_region_z0", "slide_region_z2"]
        assert result.images["slide_region_z0"].coords["c"].to_numpy().tolist() == ["Cyto", "DAPI"]
        assert result.attrs == {"spatialdata_io_reader": "merscope", "spatialdata_io_software_version": __version__}

    def test_invalid_raster_option_fails_before_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="chunks"):
            merscope(tmp_path / "missing", chunks=(1, 0, 16))

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_image_only_needs_no_counts_geometry_transcripts_or_affine(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile)
        for folder in {fixture.root, fixture.analysis}:
            for artifact in folder.iterdir():
                if artifact.is_file():
                    artifact.unlink()
        (fixture.root / "images" / "micron_to_mosaic_pixel_transform.csv").unlink()
        result = merscope(
            fixture.root,
            vpt_directory=None if profile == "standard" else fixture.analysis,
            include_transcripts=False,
            include_cell_boundaries=False,
            include_table=False,
        )
        assert list(result.images) == ["slide_region_z0", "slide_region_z2"]
        assert not result.points
        assert not result.shapes
        assert not result.tables

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_table_only_needs_no_affine_images_transcripts_or_geometry(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile)
        shutil.rmtree(fixture.root / "images")
        (fixture.root / "detected_transcripts.csv").unlink()
        for folder in {fixture.root, fixture.analysis}:
            for artifact in folder.glob("*.parquet"):
                artifact.unlink()
        if profile != "standard":
            (fixture.root / "cell_by_gene.csv").unlink()
            (fixture.root / "cell_metadata.csv").unlink()
        result = merscope(
            fixture.root,
            vpt_directory=None if profile == "standard" else fixture.analysis,
            include_transcripts=False,
            include_cell_boundaries=False,
            include_images=False,
        )
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.obs_names.tolist() == list(fixture.cell_ids)
        assert "spatialdata_attrs" not in table.uns
        assert not result.images
        assert not result.points
        assert not result.shapes

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_boundaries_only_needs_no_count_header_or_cell_metadata(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile, incomplete_z0=True)
        for folder in {fixture.root, fixture.analysis}:
            for artifact in folder.glob("*.csv"):
                artifact.unlink()
        result = merscope(
            fixture.root,
            vpt_directory=None if profile == "standard" else fixture.analysis,
            include_transcripts=False,
            include_table=False,
            include_images=False,
        )
        assert result.shapes["slide_region_polygons"].index.tolist() == list(fixture.cell_ids)
        assert_array_equal(result.shapes["slide_region_polygons"].geometry.area, [19, 4, 4])
        assert not result.tables

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    def test_points_only_reads_vocabulary_header_without_expression_body_or_metadata(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"]
    ) -> None:
        fixture = write_merscope_dataset(tmp_path, profile=profile)
        count_source = fixture.root / "cell_by_gene.csv"
        header = count_source.read_text(encoding="utf-8").splitlines()[0]
        count_source.write_text(header + "\nthis is not a count row\n", encoding="utf-8")
        for folder in {fixture.root, fixture.analysis}:
            (folder / "cell_metadata.csv").unlink()
            for artifact in folder.glob("*.parquet"):
                artifact.unlink()
        result = merscope(
            fixture.root,
            vpt_directory=None if profile == "standard" else fixture.analysis,
            include_cell_boundaries=False,
            include_table=False,
            include_images=False,
        )
        points = result.points["slide_region_transcripts"].compute()
        assert points["gene"].tolist() == ["GeneB", "GeneA", "Blank-1", "ZeroGene"]
        assert not result.tables
        assert not result.shapes
        assert not result.images

    def test_raw_vpt_fallback_requires_root_vocabulary_without_guessing_from_vpt(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, profile="cellpose")
        (fixture.root / "cell_by_gene.csv").unlink()
        with pytest.raises(FileNotFoundError) as error:
            merscope(fixture.root, vpt_directory="vpt", include_table=False)
        assert error.value.filename == str(fixture.root / "cell_by_gene.csv")
        result = merscope(fixture.root, vpt_directory="vpt", include_transcripts=False)
        assert result.tables["table"].obs_names.tolist() == list(fixture.cell_ids)

    def test_reassigned_vpt_transcripts_require_their_own_vocabulary(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, profile="watershed", reassigned_transcripts=True)
        (fixture.analysis / "cell_by_gene.csv").unlink()
        with pytest.raises(FileNotFoundError) as error:
            merscope(fixture.root, vpt_directory="vpt", include_table=False)
        assert error.value.filename == str(fixture.analysis / "cell_by_gene.csv")

    def test_footprints_are_independent_of_selected_image_z_layers(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path, incomplete_z0=True)
        at_zero = merscope(fixture.root, z_layers=(0,))
        at_two = merscope(fixture.root, z_layers=(2,))
        zero_shapes = at_zero.shapes["slide_region_polygons"]
        two_shapes = at_two.shapes["slide_region_polygons"]
        assert zero_shapes.index.tolist() == ["1001", "2002", "3003"]
        assert all(zero_shapes.geometry.geom_equals(two_shapes.geometry))
        assert_array_equal(zero_shapes.geometry.area, [19, 4, 4])

    def test_independent_geometry_population_is_validated_before_linkage(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        source = fixture.root / "cell_boundaries.parquet"
        boundaries = gpd.read_parquet(source)
        boundaries.loc[boundaries["EntityID"] == 2002, "EntityID"] = 9999
        boundaries.to_parquet(source)
        with pytest.raises(ReaderFormatError, match=r"2002|target|instance|population|identit"):
            merscope(fixture.root)

    @pytest.mark.parametrize("vpt", [False, True])
    def test_unknown_current_assignment_fails_when_its_partition_is_computed(
        self, tmp_path: Path, *, vpt: bool
    ) -> None:
        fixture = write_merscope_dataset(
            tmp_path, profile="cellpose" if vpt else "standard", reassigned_transcripts=vpt
        )
        source = fixture.analysis / "detected_transcripts.csv"
        events = pd.read_csv(source, index_col=0)
        events.loc[events.index[0], "cell_id"] = 999999
        events.to_csv(source)
        completed: list[str] = []
        with Callback(posttask=lambda key, *_: completed.append(str(key))):
            result = merscope(fixture.root, vpt_directory="vpt" if vpt else None)
        assert completed == []
        with pytest.raises(ReaderFormatError, match=r"cell|assignment|identit"):
            result.points["slide_region_transcripts"].compute()

    def test_projected_quoted_transcripts_validate_only_requested_partition(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        fixture = write_merscope_dataset(tmp_path)
        counts_source = fixture.root / "cell_by_gene.csv"
        counts = pd.read_csv(counts_source, index_col=0)
        counts = counts.rename(columns={"GeneB": "Gene,B"})
        counts.to_csv(counts_source)
        pd.DataFrame(
            {
                "global_x": np.arange(80, dtype=np.float64) / 4,
                "global_y": ["0.5"] * 79 + ["nan"],
                "global_z": np.zeros(80, dtype=np.int64),
                "gene": ["Gene,B", "GeneA"] * 40,
                "cell_id": [1001, -1] * 40,
                "x": np.full(80, 9999),
                "y": np.full(80, 8888),
                "ignored": ['a comma, and a "quote"'] * 80,
            }
        ).to_csv(fixture.root / "detected_transcripts.csv", index=False)
        mocker.patch("spatialdata_io.readers.merscope._transcripts._BLOCK_BYTES", 256)
        completed: list[str] = []
        with Callback(posttask=lambda key, *_: completed.append(str(key))):
            result = merscope(fixture.root)
        assert completed == []
        points = result.points["slide_region_transcripts"]
        assert points.npartitions > 3
        assert "ignored" not in points.columns
        executed: list[str] = []
        with Callback(posttask=lambda key, *_: executed.append(str(key))):
            first = points.partitions[0].compute()
        assert set(first["gene"]) <= {"Gene,B", "GeneA"}
        assert first["x"].max() < 100
        assert_array_equal(first["y"], np.full(len(first), 0.5))
        assert len(executed) < points.npartitions
        with pytest.raises(ReaderFormatError, match="detected_transcripts"):
            points.partitions[-1].compute()

    @pytest.mark.parametrize("record", ["1,2,0,GeneB,1001,excess", "1,2,0,GeneB"])
    def test_transcript_record_width_fails_even_for_projected_fields(self, tmp_path: Path, record: str) -> None:
        fixture = write_merscope_dataset(tmp_path)
        (fixture.root / "detected_transcripts.csv").write_text(
            "global_x,global_y,global_z,gene,cell_id\n1,2,0,GeneB,1001\n" + record + "\n", encoding="utf-8"
        )
        result = merscope(fixture.root)
        with pytest.raises(ReaderFormatError, match="detected_transcripts"):
            result.points["slide_region_transcripts"].compute()

    def test_partial_image_reads_one_row_band_and_graphs_serialize(self, tmp_path: Path, mocker: MockerFixture) -> None:
        fixture = write_merscope_dataset(tmp_path)
        source_read = mocker.spy(tiff_dask, "_read_tiff_selection")
        decoder = mocker.spy(tifffile.TiffPage, "asarray")
        completed: list[str] = []
        with Callback(posttask=lambda key, *_: completed.append(str(key))):
            result = merscope(fixture.root)
        source_read.assert_not_called()
        decoder.assert_not_called()
        assert completed == []
        image = result.images["slide_region_z0"]
        assert isinstance(image.data, da.Array)
        restored = pickle.loads(pickle.dumps(image.data))  # noqa: S301, RUF100 - Test-owned lazy graph.
        assert_array_equal(restored[1, :1, :1].compute(), [[0]])
        source_read.assert_called_once()
        # The 8-row strips of this small image form one row band.
        assert source_read.call_args.args[1] == (slice(0, 64), slice(0, 80))
        points = pickle.loads(pickle.dumps(result.points["slide_region_transcripts"]))  # noqa: S301, RUF100 - Test-owned lazy graph.
        assert points.compute()["cell_id"].tolist() == ["1001", "2002", "3003", "-1"]

    def test_transcript_coordinates_use_correctly_rounded_source_lexemes(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        (fixture.root / "detected_transcripts.csv").write_text(
            "global_x,global_y,global_z,gene,cell_id\n12.947243945462901,2,0,GeneB,1001\n",
            encoding="utf-8",
        )
        points = merscope(fixture.root).points["slide_region_transcripts"].compute()
        assert_array_equal(points["x"], [12.947243945462901])
        assert points["x"].iloc[0] != 12.9472439454629

    def test_large_integer_cell_id_is_exact_across_all_public_products(self, tmp_path: Path) -> None:
        fixture = write_merscope_dataset(tmp_path)
        identifier = 2**53 + 1
        for basename in ("cell_by_gene.csv", "cell_metadata.csv"):
            source = fixture.root / basename
            frame = pd.read_csv(source, index_col=0)
            frame.index = pd.Index([str(identifier), "2002", "3003"], name=frame.index.name)
            frame.to_csv(source)
        boundary_source = fixture.root / "cell_boundaries.parquet"
        boundaries = gpd.read_parquet(boundary_source)
        boundaries.loc[boundaries["EntityID"] == 1001, "EntityID"] = identifier
        boundaries.to_parquet(boundary_source)
        transcript_source = fixture.root / "detected_transcripts.csv"
        events = pd.read_csv(transcript_source, index_col=0)
        events.loc[events["cell_id"] == 1001, "cell_id"] = identifier
        events["transcript_id"] = ["ENST000001", "ENST000002", "control-1", "ENST000004"]
        events.to_csv(transcript_source)
        result = merscope(fixture.root)
        assert result.tables["table"].obs_names.tolist() == [str(identifier), "2002", "3003"]
        assert result.shapes["slide_region_polygons"].index.tolist() == [str(identifier), "2002", "3003"]
        points = result.points["slide_region_transcripts"].compute()
        assert points["cell_id"].tolist() == [str(identifier), "2002", "3003", "-1"]
        assert points["transcript_id"].tolist() == ["ENST000001", "ENST000002", "control-1", "ENST000004"]
        assert_array_equal(points["barcode_id"], [0, 1, 2, 3])
        assert_array_equal(points["fov"], [0, 0, 1, 1])

    def test_wide_counts_and_blank_measurements_never_densify(self, tmp_path: Path, mocker: MockerFixture) -> None:
        fixture = write_merscope_dataset(tmp_path)
        identifiers = [str(2**53 + i + 1) for i in range(1024)]
        values = np.zeros((1024, 130), dtype=np.int64)
        values[np.arange(1024), np.arange(1024) % 128] = np.arange(1024) + 1
        values[0, 128:] = [5, 7]
        pd.DataFrame(
            values,
            index=pd.Index(identifiers, name="cell"),
            columns=[f"Gene{i}" for i in range(128)] + ["Blank-1", "Blank-0"],
        ).to_csv(fixture.root / "cell_by_gene.csv")
        pd.DataFrame(
            {"center_x": np.arange(1024), "center_y": np.zeros(1024), "volume": np.ones(1024)},
            index=pd.Index(identifiers, name="EntityID"),
        ).to_csv(fixture.root / "cell_metadata.csv")
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("counts densified"))
        mocker.patch.object(sparse.csr_array, "todense", side_effect=AssertionError("counts densified"))
        result = merscope(fixture.root, include_images=False, include_transcripts=False, include_cell_boundaries=False)
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.shape == (1024, 129)
        assert table.X.nnz == 1025
        assert table.obs_names.tolist() == identifiers
        blanks = table.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert blanks.shape == (1024, 1)
        assert_array_equal(blanks.data, [5])
        assert table.uns["blank_names"] == ["Blank-1"]

    @pytest.mark.parametrize("profile", ["standard", "cellpose", "watershed"])
    @pytest.mark.parametrize("reassigned", [False, True])
    def test_zarr_roundtrip_preserves_values_footprints_and_linkage(
        self, tmp_path: Path, profile: Literal["standard", "cellpose", "watershed"], *, reassigned: bool
    ) -> None:
        fixture = write_merscope_dataset(tmp_path / "source", profile=profile, reassigned_transcripts=reassigned)
        result = merscope(fixture.root, vpt_directory=None if profile == "standard" else "vpt")
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert isinstance(table.obsm["blank"], sparse.csr_array)
        output = tmp_path / "output.zarr"
        result.write(output)
        restored = read_zarr(output)
        restored_table = restored.tables["table"]
        assert_equal(restored_table.uns["spatialdata_io_positions"], table.uns["spatialdata_io_positions"], exact=True)
        for storage in ("h5ad", "zarr"):
            table_path = tmp_path / f"positions_merscope.{storage}"
            source_table = table
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
        assert isinstance(restored_table.X, sparse.csr_array | sparse.csr_matrix)
        assert_array_equal(restored_table.X.data, table.X.data)
        assert_array_equal(restored_table.obsm["blank"].data, table.obsm["blank"].data)
        assert_array_equal(restored_table.uns["blank_names"], ["Blank-1"])
        assert get_table_keys(restored_table) == ("slide_region_polygons", "cells_region", "EntityID")
        assert restored_table.obs_names.tolist() == list(fixture.cell_ids)
        assert_array_equal(restored.shapes["slide_region_polygons"].geometry.area, [19, 4, 4])
        points = restored.points["slide_region_transcripts"].compute()
        assignment = "original_cell_id" if profile != "standard" and not reassigned else "cell_id"
        expected = (
            ["9001", "9002", "9003", "-1"] if profile != "standard" and reassigned else ["1001", "2002", "3003", "-1"]
        )
        assert points[assignment].tolist() == expected
        assert "z" not in points.columns
        assert_array_equal(points["z_raw"], [0, 2, 2, 0])
