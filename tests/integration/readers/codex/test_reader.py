"""Format-faithful CODEX cell-table and stitched-image workflows."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import readfcs
import tifffile
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Affine, BaseTransformation, get_transformation
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_tasks
from spatialdata_io import __version__
from spatialdata_io import codex as root_codex
from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.codex import codex
from tests.integration.readers.codex._fixtures import cell_frame, write_csv, write_fcs, write_image

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestCodex:
    """Prove actual CSV/FCS decoding, scientific linkage, and lazy TIFF composition."""

    def test_reordered_named_fields_and_metadata_extensions(self, tmp_path: Path) -> None:
        """Reordered source names retain metadata and distinguish complete cycle feature names."""
        frame = cell_frame()
        frame["cyc001_ch002_note"] = ["DNA", "DNA", "DNA"]
        frame["extra_metadata"] = ["NA", "NULL", "sample"]
        columns = ["extra_metadata", "cyc001_ch002", "y", "cell_id", "cyc001_ch002_note", "x", "region"]
        write_csv(tmp_path / "cells.csv", frame[columns])

        result = codex(tmp_path)
        table = result.tables["table"]

        assert table.var_names.tolist() == ["cyc001_ch002"]
        assert table.obs.columns.tolist() == ["extra_metadata", "y", "cell_id", "cyc001_ch002_note", "x", "region"]
        assert table.obs["extra_metadata"].tolist() == ["NA", "NULL", "sample"]
        assert table.obs["cyc001_ch002_note"].tolist() == ["DNA"] * 3
        assert table.obs["cell_id"].tolist() == [7, 2, 11]
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(table.X.data, [2, 5.5])
        assert "spatial" not in table.obsm
        np.testing.assert_array_equal(
            cast("pd.DataFrame", table.obs)[["x", "y"]].to_numpy(), [[20, 1.5], [2.5, 30], [13, 8]]
        )

    @pytest.mark.parametrize("large", [2**63, 2**64 - 1])
    def test_metadata_types_independent_of_csv_batches(self, tmp_path: Path, mocker: MockerFixture, large: int) -> None:
        """Arbitrary batch boundaries cannot round or reject exact unsigned metadata."""
        frame = cell_frame().iloc[:2].copy()
        frame["extra_integer"] = np.asarray([2**63 - 1, large], dtype=np.uint64)
        write_csv(tmp_path / "cells.csv", frame)
        whole = codex(tmp_path).tables["table"]

        mocker.patch("spatialdata_io.readers.codex._tables._CSV_BATCH_ROWS", 1)
        partitioned = codex(tmp_path).tables["table"]

        assert isinstance(whole.obs, pd.DataFrame)
        assert isinstance(partitioned.obs, pd.DataFrame)
        pd.testing.assert_frame_equal(partitioned.obs, whole.obs, check_exact=True)
        assert partitioned.obs["extra_integer"].dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(partitioned.obs["extra_integer"], [2**63 - 1, large])
        assert isinstance(whole.X, sparse.csr_array)
        assert isinstance(partitioned.X, sparse.csr_array)
        np.testing.assert_array_equal(partitioned.X.data, whole.X.data)
        np.testing.assert_array_equal(partitioned.X.indices, whole.X.indices)
        np.testing.assert_array_equal(partitioned.X.indptr, whole.X.indptr)

    @pytest.mark.parametrize("source_format", ["csv", "fcs"])
    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_annotation_roundtrip(self, tmp_path: Path, source_format: str, storage: str) -> None:
        """Persist primitive cell metadata, marker strings, exact linkage IDs, and sparse signals."""
        if source_format == "csv":
            frame = cell_frame()
            frame["optional"] = [0.25, np.nan, 0.75]
            frame["note"] = ["NA", "NULL", "sample"]
            frame["cell_id"] = [str(2**53 + 1), "2.0", "11"]
            write_csv(tmp_path / "cells.csv", frame)
        else:
            write_fcs(tmp_path / "cells.fcs")
        table = codex(tmp_path).tables["table"]
        destination = tmp_path / f"converted.{storage}"

        if storage == "h5ad":
            table.write_h5ad(destination)
            loaded = ad.read_h5ad(destination)
        else:
            table.write_zarr(destination)
            loaded = ad.read_zarr(destination)

        assert loaded.obs["cell_id"].tolist() == table.obs["cell_id"].tolist()
        assert loaded.var_names.tolist() == table.var_names.tolist()
        if source_format == "csv":
            np.testing.assert_array_equal(loaded.obs["optional"], [0.25, np.nan, 0.75])
            assert loaded.obs["note"].tolist() == ["NA", "NULL", "sample"]
        else:
            assert loaded.var["marker"].tolist() == ["CD45", "DNA"]
        assert isinstance(table.X, sparse.csr_array)
        assert isinstance(loaded.X, sparse.csr_matrix | sparse.csr_array)
        np.testing.assert_array_equal(loaded.X.data, table.X.data)

    @pytest.mark.parametrize("source_format", ["csv", "fcs"])
    @pytest.mark.parametrize("image_mode", ["none", "plain", "deflate"])
    def test_scientific_contract(self, tmp_path: Path, source_format: str, image_mode: str) -> None:
        """Verify exact table values, metadata order, coordinates, linkage, and lazy image composition."""
        if source_format == "csv":
            write_csv(tmp_path / "reg001_compensated.csv")
        else:
            write_fcs(tmp_path / "reg001_compensated.fcs")
        image = (
            None if image_mode == "none" else write_image(tmp_path / "stitched.tif", compressed=image_mode == "deflate")
        )
        tasks: list[str] = []

        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = codex(tmp_path)

        assert not tasks
        assert root_codex is codex
        assert list(result.tables) == ["table"]
        assert list(result.shapes) == ["1"]
        assert list(result.images) == (["images"] if image is not None else [])
        assert not result.labels
        assert not result.points
        assert result.attrs == {
            "spatialdata_io_reader": "codex",
            "spatialdata_io_software_version": __version__,
        }

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert table.X.has_canonical_format
        assert table.X.dtype == np.dtype(np.float32 if source_format == "fcs" else np.float64)
        np.testing.assert_array_equal(table.X.toarray(), [[0, 2], [3.25, 0], [-1.5, 5.5]])
        assert table.obs_names.tolist() == ["0", "1", "2"]
        assert table.var_names.tolist() == ["cyc002_ch001", "cyc001_ch002"]
        if source_format == "fcs":
            assert table.var["marker"].tolist() == ["CD45", "DNA"]
        assert table.obs.columns.tolist() == [
            "cell_id",
            "region",
            "tile_num",
            "x",
            "y",
            "z",
            "x_tile",
            "y_tile",
            "size",
            "tsne_x",
            "tsne_y",
            "homogeneity",
        ]
        assert table.obs["cell_id"].tolist() == [7, 2, 11]
        assert table.obs["cell_id"].dtype == np.dtype(np.int64)
        assert table.obs["region"].tolist() == ["1"] * 3
        assert isinstance(table.obs["region"].dtype, pd.CategoricalDtype)
        assert "spatial" not in table.obsm
        np.testing.assert_array_equal(
            cast("pd.DataFrame", table.obs)[["x", "y"]].to_numpy(), [[20, 1.5], [2.5, 30], [13, 8]]
        )
        assert get_table_keys(table) == ("1", "region", "cell_id")

        circles = result.shapes["1"]
        assert circles.index.tolist() == [7, 2, 11]
        np.testing.assert_array_equal(circles.geometry.x, [20, 2.5, 13])
        np.testing.assert_array_equal(circles.geometry.y, [1.5, 30, 8])
        np.testing.assert_array_equal(circles["radius"], [1, 1, 1])

        transform = get_transformation(circles, to_coordinate_system="table_pixels")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))

    @pytest.mark.parametrize("compressed", [False, True])
    def test_image_dtype_chunks_and_values(self, tmp_path: Path, *, compressed: bool) -> None:
        """Retain source dtype, channels, transformations, and serialized graph values in default chunks."""
        write_csv(tmp_path / "cells.csv")
        image = write_image(tmp_path / "stitched.tif", compressed=compressed)

        result = codex(tmp_path)
        raster = result.images["images"]

        # A single scale by default; 1024-pixel tiles hold one whole channel each.
        assert isinstance(raster, DataArray)
        assert raster.dims == ("c", "y", "x")
        assert raster.shape == (2, 32, 40)
        assert raster.dtype == np.dtype(np.uint16)
        assert raster.coords["c"].to_numpy().tolist() == [0, 1]
        assert raster.data.chunks == ((1, 1), (32,), (40,))

        restored = pickle.loads(pickle.dumps(raster.data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        np.testing.assert_array_equal(restored[:, 1:3, 2:5].compute(), image[:, 1:3, 2:5])

        transform = get_transformation(raster, to_coordinate_system="image_pixels")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))

    def test_scale_factors_build_a_pyramid(self, tmp_path: Path) -> None:
        """Forward public pyramid factors and chunks to the stitched image."""
        write_csv(tmp_path / "cells.csv")
        image = write_image(tmp_path / "stitched.tif")

        result = codex(tmp_path, scale_factors=(2,), chunks=(1, 16, 16))
        pyramid = result.images["images"]

        assert isinstance(pyramid, DataTree)
        assert list(pyramid.children) == ["scale0", "scale1"]
        assert pyramid["scale0"]["image"].data.chunksize == (1, 16, 16)
        np.testing.assert_array_equal(pyramid["scale0"]["image"].data.compute(), image)
        assert pyramid["scale1"]["image"].shape == (2, 16, 20)

    def test_fcs_fixture_is_independently_decoded(self, tmp_path: Path) -> None:
        """Verify that the fixture is actual FCS with the expected parameter and marker metadata."""
        path = tmp_path / "reg001_compensated.fcs"
        write_fcs(path)

        source = readfcs.ReadFCS(path)

        assert source.header["version"] == "3.1"
        assert source.channels["PnN"].tolist() == cell_frame().columns.tolist()
        assert source.channels["PnS"].tolist()[9:11] == ["CD45", "DNA"]
        np.testing.assert_array_equal(source.data, cell_frame().to_numpy(dtype=np.float32))

    def test_bounded_partial_tiff_decode(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Decode only the stored tile needed by a small lazy image selection."""
        write_csv(tmp_path / "cells.csv")
        image = write_image(tmp_path / "stitched.tif", compressed=True)
        decode = mocker.spy(tiff_tasks, "_read_tiff_selection")

        # Chunks equal to the 16-pixel stored tiles; default tiles would merge
        # every tile of a channel into one chunk.
        result = codex(tmp_path, chunks=(1, 16, 16))

        assert decode.call_count == 0
        raster = result.images["images"]
        np.testing.assert_array_equal(raster.data[0:1, 1:3, 2:5].compute(scheduler="synchronous"), image[0:1, 1:3, 2:5])
        assert decode.call_count == 1
        assert decode.call_args.args[1] == (slice(0, 1), slice(0, 16), slice(0, 16))

    def test_multiple_sources_require_explicit_selection(self, tmp_path: Path) -> None:
        """Require a caller selection when more than one cell table is present."""
        write_csv(tmp_path / "cells.csv")
        write_fcs(tmp_path / "cells.fcs")

        with pytest.raises(ReaderFormatError, match="2 candidates"):
            codex(tmp_path)

        result = codex(tmp_path, cell_table_file="cells.fcs")

        assert result.tables["table"].var["marker"].tolist() == ["CD45", "DNA"]

    def test_absolute_overrides_and_source_axes(self, tmp_path: Path) -> None:
        """Honor external file overrides, explicit axes, and requested output chunks."""
        root = tmp_path / "dataset"
        root.mkdir()
        write_csv(tmp_path / "cells.csv")
        image = write_image(tmp_path / "stack.tif")

        result = codex(
            root,
            cell_table_file=tmp_path / "cells.csv",
            image_file=tmp_path / "stack.tif",
            data_axes="CYX",
            chunks=(1, 8, 10),
        )
        raster = result.images["images"]

        assert raster.data.chunksize == (1, 8, 10)
        np.testing.assert_array_equal(raster.data[:, :2, :3].compute(), image[:, :2, :3])

    def test_multiseries_selection(self, tmp_path: Path) -> None:
        """Require an explicit series when multiple source images are available."""
        write_csv(tmp_path / "cells.csv")
        with tifffile.TiffWriter(tmp_path / "images.tif") as writer:
            writer.write(np.ones((8, 10), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.full((6, 7), 17, dtype=np.uint16), metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="tiff_series"):
            codex(tmp_path)

        result = codex(tmp_path, tiff_series=1, chunks=4)
        raster = result.images["images"]

        assert raster.shape == (1, 6, 7)
        np.testing.assert_array_equal(raster.data.compute(), np.full((1, 6, 7), 17, dtype=np.uint16))

    @pytest.mark.parametrize(
        ("names", "expected"),
        [(["marker-z", "marker-a"], ["marker-z", "marker-a"]), (["CD3", "CD3"], ["Channel:0:0", "Channel:0:1"])],
    )
    def test_ome_channel_labels_and_singletons(self, tmp_path: Path, names: list[str], expected: list[str]) -> None:
        """Retain OME channel labels while removing only named singleton dimensions."""
        write_csv(tmp_path / "cells.csv")
        image = np.arange(2 * 8 * 10, dtype=np.uint16).reshape(1, 1, 2, 8, 10)
        tifffile.imwrite(
            tmp_path / "stitched.ome.tif",
            image,
            photometric="minisblack",
            metadata={"axes": "TZCYX", "Channel": {"Name": names}},
        )

        result = codex(tmp_path)
        raster = result.images["images"]

        assert raster.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(raster.data.compute(), image[0, 0])

    def test_csv_multiple_batches_and_exact_identifiers(self, tmp_path: Path) -> None:
        """Keep sparse values and large lexical IDs exact across the production CSV batch boundary."""
        path = tmp_path / "cells.csv"
        with path.open("w", encoding="utf-8") as stream:
            stream.write("cell_id,region,x,y,cyc002_ch001,cyc001_ch002,note\n")
            for row in range(10_001):
                expression = "1.25" if row == 10_000 else "1"
                stream.write(f"{9007199254740993 + row},1,{row},0,{expression},0,{'fixture' * 20}\n")

        result = codex(tmp_path)
        table = result.tables["table"]

        assert isinstance(table.X, sparse.csr_array)
        assert table.X.shape == (10_001, 2)
        assert table.X.dtype == np.dtype(np.float64)
        assert table.obs_names[[0, 10_000]].tolist() == ["0", "10000"]
        assert table.obs["cell_id"].iloc[np.asarray([0, 10_000], dtype=np.int64)].tolist() == [
            9007199254740993,
            9007199254750993,
        ]
        np.testing.assert_array_equal(table.X[[0, 10_000], :].toarray(), [[1, 0], [1.25, 0]])
        assert table.X.nnz == 10_001

    def test_float_lexeme_does_not_round_other_cell_id(self, tmp_path: Path) -> None:
        """Retain a large integer ID when another source ID uses an integral decimal token."""
        (tmp_path / "cells.csv").write_text(
            "cell_id,region,x,y,cyc001_ch001\n9007199254740993,1,0,0,1\n2.0,1,1,1,2\n", encoding="utf-8"
        )

        result = codex(tmp_path)

        assert result.tables["table"].obs["cell_id"].tolist() == [9007199254740993, 2]


class TestCodexCalibration:
    """Calibrate the proved table frame and explicitly register selected images."""

    @pytest.mark.parametrize("source_format", ["csv", "fcs"])
    @pytest.mark.parametrize("factor", [1, 2])
    def test_calibrated_landmarks(self, tmp_path: Path, source_format: str, factor: int) -> None:
        (write_csv if source_format == "csv" else write_fcs)(tmp_path / f"cells.{source_format}")
        write_image(tmp_path / "stitched.tif")
        # Explicit caller mapping includes a crop origin and vertical reflection.
        mapping = Affine(
            np.asarray([[factor, 0, 10], [0, -factor, 40], [0, 0, 1]], dtype=np.float64),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        tasks: list[str] = []
        with Callback(pretask=lambda key, *_: tasks.append(str(key))):
            result = codex(tmp_path, pixel_size_um=0.5, image_to_table=mapping)
        assert not tasks
        table = result.tables["table"]
        np.testing.assert_array_equal(table.obsm["spatial"], [[10, 0.75], [1.25, 15], [6.5, 4]])
        np.testing.assert_array_equal(
            cast("pd.DataFrame", table.obs)[["x", "y"]].to_numpy(), [[20, 1.5], [2.5, 30], [13, 8]]
        )
        assert set(table.obsm) == {"spatial"}
        assert table.obsm["spatial"].dtype == np.float64
        circle_transform = get_transformation(result.shapes["1"], to_coordinate_system="global")
        image_transform = get_transformation(result.images["images"], to_coordinate_system="global")
        assert isinstance(circle_transform, BaseTransformation)
        assert isinstance(image_transform, BaseTransformation)
        np.testing.assert_array_equal(circle_transform.to_affine_matrix(("x", "y"), ("x", "y")), np.diag([0.5, 0.5, 1]))
        # Raster point (1.5, 2.5) maps independently to table (10+1.5*f, 40-2.5*f).
        actual = image_transform.to_affine_matrix(("x", "y"), ("x", "y")) @ [1.5, 2.5, 1]
        np.testing.assert_array_equal(actual, [5 + 0.75 * factor, 20 - 1.25 * factor, 1])
        assert result.shapes["1"]["radius"].tolist() == [1, 1, 1]

    @pytest.mark.parametrize("absolute", [False, True])
    def test_explicit_experiment_association(self, tmp_path: Path, *, absolute: bool) -> None:
        write_csv(tmp_path / "cells.csv")
        sidecar = tmp_path / "experiment.json"
        sidecar.write_text('{"version":"1.7.0.6","xyResolution":500}')
        selected = sidecar if absolute else Path("experiment.json")
        result = codex(tmp_path, experiment_file=selected)
        np.testing.assert_array_equal(result.tables["table"].obsm["spatial"], [[10, 0.75], [1.25, 15], [6.5, 4]])
        with pytest.raises(ValueError, match="conflicts"):
            codex(tmp_path, experiment_file=selected, pixel_size_um=0.6)
        codex(tmp_path, experiment_file=selected, pixel_size_um=0.5)

    def test_no_automatic_sidecar_or_image_association(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        write_image(tmp_path / "stitched.tif")
        (tmp_path / "experiment.json").write_text('{"version":"1.7.0.6","xyResolution":500}')
        result = codex(tmp_path, pixel_size_um=0.5)
        frames = get_transformation(result.images["images"], get_all=True)
        assert isinstance(frames, dict)
        assert set(frames) == {"image_pixels"}
        assert "spatial" not in codex(tmp_path).tables["table"].obsm

    def test_mapping_requires_image(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        with pytest.raises(ValueError, match="requires an image"):
            codex(tmp_path, image_to_table=Affine(np.eye(3), input_axes=("x", "y"), output_axes=("x", "y")))

    @pytest.mark.parametrize("storage", ["h5ad", "zarr", "spatialdata"])
    def test_position_persistence(self, tmp_path: Path, storage: str) -> None:
        source = tmp_path / "source"
        source.mkdir()
        write_csv(source / "cells.csv")
        result = codex(source, pixel_size_um=0.5)
        table = result.tables["table"]
        destination = tmp_path / f"positions.{storage}"
        if storage == "h5ad":
            table.write_h5ad(destination)
            loaded = ad.read_h5ad(destination)
        elif storage == "zarr":
            table.write_zarr(destination)
            loaded = ad.read_zarr(destination)
        else:
            result.write(destination)
            loaded = read_zarr(destination).tables["table"]
        np.testing.assert_array_equal(loaded.obsm["spatial"], table.obsm["spatial"])
        assert loaded.uns[POSITIONS_KEY]["coordinates"]["spatial"]["frame"] == "global"
        assert set(loaded.uns[POSITIONS_KEY]["coordinates"]["spatial"]["calibration"]) == {"global"}
        np.testing.assert_array_equal(loaded[[True, False, True]].obsm["spatial"], [[10, 0.75], [6.5, 4]])

    @pytest.mark.parametrize("scale", [None, 0.5])
    def test_real_akoya_qualified_header(self, tmp_path: Path, scale: float | None) -> None:
        # Original public Akoya sample, pinned to phenomenalist c35f531d0035ac6ea879bfe1b0340c6b5cc6b09e.
        # https://github.com/igordot/phenomenalist/blob/c35f531d0035ac6ea879bfe1b0340c6b5cc6b09e/inst/extdata/tonsil-akoya-2018-500.csv
        source = Path(__file__).parent / "data" / "akoya-tonsil-three-rows.csv"
        result = codex(tmp_path, cell_table_file=source, pixel_size_um=scale)
        table = result.tables["table"]
        assert table.obs["cell_id"].tolist() == [72993, 19306, 77026]
        assert table.var_names.tolist()[:5] == [
            "cyc001_ch001",
            "cyc001_ch002",
            "cyc001_ch003",
            "cyc001_ch004",
            "cyc002_ch001",
        ]
        assert table.var["marker"].tolist()[:5] == ["DAPI1", "Blank", "Blank", "Blank", "DAPI2"]
        assert table.obs.columns.tolist() == [
            "cell_id",
            "region",
            "tile_num",
            "x",
            "y",
            "z",
            "x_tile",
            "y_tile",
            "size",
            "homogeneity",
        ]
        native = np.asarray([[6447, 1556], [2047, 754], [5396, 3917]])
        np.testing.assert_array_equal(cast("pd.DataFrame", table.obs)[["x", "y"]].to_numpy(), native)
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.shape == (3, 32)
        expected = np.genfromtxt(source, delimiter=",", skip_header=1, usecols=range(9, 41))
        np.testing.assert_array_equal(table.X.toarray(), expected)
        if scale is None:
            assert "spatial" not in table.obsm
        else:
            np.testing.assert_array_equal(table.obsm["spatial"], [[3223.5, 778], [1023.5, 377], [2698, 1958.5]])

    def test_qualified_duplicate_short_names(self, tmp_path: Path) -> None:
        (tmp_path / "cells.csv").write_text(
            "cell_id:id,region:region,x:x,y:y,cyc001_ch001:A,cyc001_ch001:B\n1,1,1,1,1,2\n"
        )
        with pytest.raises(ReaderFormatError, match="duplicate column"):
            codex(tmp_path)

    def test_explicit_native_registration(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        write_image(tmp_path / "stitched.tif")
        matrix = np.asarray([[2, 0, 10], [0, -2, 40], [0, 0, 1]], dtype=np.float64)
        mapping = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
        result = codex(tmp_path, image_to_table=mapping)
        transform = get_transformation(result.images["images"], to_coordinate_system="table_pixels")
        assert isinstance(transform, BaseTransformation)
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), matrix)
        np.testing.assert_array_equal(mapping.to_affine_matrix(("x", "y"), ("x", "y")), matrix)
        assert "spatial" not in result.tables["table"].obsm

    def test_physical_overflow(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite physical"):
            codex(tmp_path, pixel_size_um=1e308)
