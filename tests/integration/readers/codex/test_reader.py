"""Format-faithful CODEX cell-table and stitched-image workflows."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
import readfcs
import tifffile
from dask.callbacks import Callback
from scipy import sparse
from spatialdata.models import get_table_keys
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_tasks
from spatialdata_io import __version__
from spatialdata_io import codex as root_codex
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.codex import codex
from tests.integration.readers.codex._fixtures import cell_frame, write_csv, write_fcs, write_image

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestCodex:
    """Prove actual CSV/FCS decoding, scientific linkage, and lazy TIFF composition."""

    @pytest.mark.parametrize("source_format", ["csv", "fcs"])
    @pytest.mark.parametrize("image_mode", ["none", "plain", "deflate"])
    def test_scientific_contract(self, tmp_path: Path, source_format: str, image_mode: str) -> None:
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
        np.testing.assert_array_equal(table.obsm["spatial"], [[20, 1.5], [2.5, 30], [13, 8]])
        assert get_table_keys(table) == ("1", "region", "cell_id")

        circles = result.shapes["1"]
        assert circles.index.tolist() == [7, 2, 11]
        np.testing.assert_array_equal(circles.geometry.x, [20, 2.5, 13])
        np.testing.assert_array_equal(circles.geometry.y, [1.5, 30, 8])
        np.testing.assert_array_equal(circles["radius"], [1, 1, 1])

        transform = cast("BaseTransformation", get_transformation(circles, to_coordinate_system="global"))
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))

    @pytest.mark.parametrize("compressed", [False, True])
    def test_image_dtype_chunks_and_values(self, tmp_path: Path, *, compressed: bool) -> None:
        write_csv(tmp_path / "cells.csv")
        image = write_image(tmp_path / "stitched.tif", compressed=compressed)

        result = codex(tmp_path)
        pyramid = result.images["images"]

        assert isinstance(pyramid, DataTree)
        assert list(pyramid.children) == ["scale0", "scale1", "scale2"]
        raster = pyramid["scale0"]["image"]
        assert raster.dims == ("c", "y", "x")
        assert raster.shape == (2, 32, 40)
        assert raster.dtype == np.dtype(np.uint16)
        assert raster.coords["c"].to_numpy().tolist() == [0, 1]
        assert raster.data.chunksize == (1, 16, 16)

        restored = pickle.loads(pickle.dumps(raster.data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        np.testing.assert_array_equal(restored[:, 1:3, 2:5].compute(), image[:, 1:3, 2:5])

        transform = cast("BaseTransformation", get_transformation(raster, to_coordinate_system="global"))
        np.testing.assert_array_equal(transform.to_affine_matrix(("x", "y"), ("x", "y")), np.eye(3))

    def test_fcs_fixture_is_independently_decoded(self, tmp_path: Path) -> None:
        path = tmp_path / "reg001_compensated.fcs"
        write_fcs(path)

        source = readfcs.ReadFCS(path)

        assert source.header["version"] == "3.1"
        assert source.channels["PnN"].tolist() == cell_frame().columns.tolist()
        assert source.channels["PnS"].tolist()[9:11] == ["CD45", "DNA"]
        np.testing.assert_array_equal(source.data, cell_frame().to_numpy(dtype=np.float32))

    def test_bounded_partial_tiff_decode(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_csv(tmp_path / "cells.csv")
        image = write_image(tmp_path / "stitched.tif", compressed=True)
        decode = mocker.spy(tiff_tasks, "_read_tiff_selection")

        # Native TIFF tasks are instrumented after construction to distinguish
        # actual source selection from model rechunking and pyramid graphs.
        result = codex(tmp_path)

        assert decode.call_count == 0
        raster = result.images["images"]["scale0"]["image"]
        np.testing.assert_array_equal(raster.data[0:1, 1:3, 2:5].compute(scheduler="synchronous"), image[0:1, 1:3, 2:5])
        assert decode.call_count == 1
        assert decode.call_args.args[1] == (slice(0, 1), slice(0, 16), slice(0, 16))

    def test_multiple_sources_require_explicit_selection(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        write_fcs(tmp_path / "cells.fcs")

        with pytest.raises(ReaderFormatError, match="2 candidates"):
            codex(tmp_path)

        result = codex(tmp_path, cell_table_file="cells.fcs")

        assert result.tables["table"].var["marker"].tolist() == ["CD45", "DNA"]

    def test_absolute_overrides_and_source_axes(self, tmp_path: Path) -> None:
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
        raster = result.images["images"]["scale0"]["image"]

        assert raster.data.chunksize == (1, 8, 10)
        np.testing.assert_array_equal(raster.data[:, :2, :3].compute(), image[:, :2, :3])

    def test_multiseries_selection(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        with tifffile.TiffWriter(tmp_path / "images.tif") as writer:
            writer.write(np.ones((8, 10), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.full((6, 7), 17, dtype=np.uint16), metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="tiff_series"):
            codex(tmp_path)

        result = codex(tmp_path, tiff_series=1, chunks=4)
        raster = result.images["images"]["scale0"]["image"]

        assert raster.shape == (1, 6, 7)
        np.testing.assert_array_equal(raster.data.compute(), np.full((1, 6, 7), 17, dtype=np.uint16))

    def test_ome_channel_labels_and_singletons(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        image = np.arange(2 * 8 * 10, dtype=np.uint16).reshape(1, 1, 2, 8, 10)
        tifffile.imwrite(
            tmp_path / "stitched.ome.tif",
            image,
            photometric="minisblack",
            metadata={"axes": "TZCYX", "Channel": {"Name": ["marker-z", "marker-a"]}},
        )

        result = codex(tmp_path)
        raster = result.images["images"]["scale0"]["image"]

        assert raster.coords["c"].to_numpy().tolist() == ["marker-z", "marker-a"]
        np.testing.assert_array_equal(raster.data.compute(), image[0, 0])

    def test_csv_multiple_batches_and_exact_identifiers(self, tmp_path: Path) -> None:
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
        (tmp_path / "cells.csv").write_text(
            "cell_id,region,x,y,cyc001_ch001\n9007199254740993,1,0,0,1\n2.0,1,1,1,2\n", encoding="utf-8"
        )

        result = codex(tmp_path)

        assert result.tables["table"].obs["cell_id"].tolist() == [9007199254740993, 2]
