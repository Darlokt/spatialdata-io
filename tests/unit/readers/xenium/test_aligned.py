"""Aligned H&E and IF images: declared axes, channel names and the micrometre transform."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata.transformations import Affine, Scale, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import xenium_aligned_image
from spatialdata_io.readers.xenium._aligned import read_alignment

if TYPE_CHECKING:
    from pathlib import Path

# Rows of a real Xenium Explorer H&E alignment (pancreas, XOA 1.6.0.8).
MATRIX = [[0.00184638, 0.64417243, 1433.27392], [-0.6441724, 0.00184638, 12184.7121], [0.0, 0.0, 1.0]]
PIXEL_SIZE = 0.2125


def _csv(path: Path, rows: list[list[float]] | str) -> ArtifactFile:
    path.write_text(rows if isinstance(rows, str) else "\n".join(",".join(repr(v) for v in row) for row in rows) + "\n")
    return ArtifactFile(path)


def _rgb(path: Path) -> np.ndarray:
    data = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    tifffile.imwrite(path, data, photometric="rgb", ome=True, metadata={"axes": "YXS"}, tile=(16, 16))
    return data


class TestReadAlignment:
    """Alignment files are finite, invertible 3 x 3 affine matrices."""

    def test_real_matrix(self, tmp_path: Path) -> None:
        np.testing.assert_array_equal(read_alignment(_csv(tmp_path / "a.csv", MATRIX)), MATRIX)

    @pytest.mark.parametrize(
        "text",
        [
            "1,0,0\n0,1,0\n",
            "1,0,0\n0,1,0\n0,0,2\n",
            "0,0,0\n0,0,0\n0,0,1\n",
            "1,0,x\n0,1,0\n0,0,1\n",
            "1,0,inf\n0,1,0\n0,0,1\n",
            "1,0\n0,1\n0,0,1\n",
            "1,0,0\n0,1,0\n0,0,1\n0,0,1\n",
            '"1,0,0\n',
        ],
        ids=["rows", "last-row", "singular", "text", "infinite", "width", "extra-row", "quote"],
    )
    def test_rejects_malformed_matrices(self, tmp_path: Path, text: str) -> None:
        with pytest.raises(ReaderFormatError, match="3 x 3"):
            read_alignment(_csv(tmp_path / "a.csv", text))


class TestXeniumAlignedImage:
    """Aligned images keep declared axes and channel names and map pixels to micrometres."""

    def test_rgb_samples_with_alignment(self, tmp_path: Path) -> None:
        data = _rgb(tmp_path / "x_he_image.ome.tif")
        alignment = _csv(tmp_path / "x_he_imagealignment.csv", MATRIX)

        image = xenium_aligned_image(tmp_path / "x_he_image.ome.tif", alignment, pixel_size=PIXEL_SIZE)

        assert image.dims == ("c", "y", "x")
        assert image.coords["c"].to_numpy().tolist() == ["r", "g", "b"]
        np.testing.assert_array_equal(image.data.compute(), np.moveaxis(data, -1, 0))
        transformation = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transformation, Affine)
        # Image pixels -> Xenium pixels (M) -> micrometres (pixel size).
        np.testing.assert_allclose(
            transformation.to_affine_matrix(("x", "y"), ("x", "y")),
            np.diag([PIXEL_SIZE, PIXEL_SIZE, 1.0]) @ np.asarray(MATRIX),
            rtol=0,
            atol=0,
        )

    def test_named_channels_without_alignment(self, tmp_path: Path) -> None:
        data = np.arange(2 * 12 * 16, dtype=np.uint16).reshape(2, 12, 16)
        path = tmp_path / "x_if_image.ome.tif"
        tifffile.imwrite(path, data, ome=True, metadata={"axes": "CYX", "Channel": {"Name": ["CD20", "DAPI"]}})

        image = xenium_aligned_image(path, pixel_size=PIXEL_SIZE)

        assert image.coords["c"].to_numpy().tolist() == ["CD20", "DAPI"]
        np.testing.assert_array_equal(image.data.compute(), data)
        transformation = get_transformation(image, to_coordinate_system="global")
        assert isinstance(transformation, Scale)
        np.testing.assert_array_equal(transformation.scale, [PIXEL_SIZE, PIXEL_SIZE])

    def test_three_channel_cyx_is_never_called_rgb(self, tmp_path: Path) -> None:
        # The Breast IDC H&E stores Hematoxylin/Eosin/Residual as CYX channels.
        path = tmp_path / "x_he_image.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((3, 12, 16), dtype=np.uint8),
            ome=True,
            metadata={"axes": "CYX", "Channel": {"Name": ["Hematoxylin", "Eosin", "Residual"]}},
        )

        image = xenium_aligned_image(path, pixel_size=PIXEL_SIZE)

        assert image.coords["c"].to_numpy().tolist() == ["Hematoxylin", "Eosin", "Residual"]

    def test_data_axes_override_and_options(self, tmp_path: Path) -> None:
        data = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
        path = tmp_path / "plain.tif"
        tifffile.imwrite(path, data.T)

        image = xenium_aligned_image(
            path, pixel_size=PIXEL_SIZE, data_axes=("x", "y"), chunks=(1, 4, 4), scale_factors=(2,)
        )

        scale0 = image["scale0"]["image"]
        np.testing.assert_array_equal(scale0.data.compute()[0], data)
        assert scale0.data.chunksize == (1, 4, 4)

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            ({"pixel_size": "0.2"}, TypeError, "number of micrometres"),
            ({"pixel_size": True}, TypeError, "number of micrometres"),
            ({"pixel_size": 0.0}, ValueError, "finite positive"),
            ({"pixel_size": float("nan")}, ValueError, "finite positive"),
            ({"pixel_size": 0.2, "data_axes": "yx"}, TypeError, "sequence of single-letter"),
            ({"pixel_size": 0.2, "scale_factors": [1]}, ValueError, "at least 2"),
        ],
    )
    def test_rejects_invalid_options(
        self, tmp_path: Path, options: dict[str, object], error: type[Exception], match: str
    ) -> None:
        _rgb(tmp_path / "x.ome.tif")

        with pytest.raises(error, match=match):
            xenium_aligned_image(tmp_path / "x.ome.tif", **options)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    def test_wrong_axes_and_missing_files(self, tmp_path: Path) -> None:
        _rgb(tmp_path / "x.ome.tif")

        with pytest.raises(ReaderFormatError, match="one axis per dimension"):
            xenium_aligned_image(tmp_path / "x.ome.tif", pixel_size=PIXEL_SIZE, data_axes=("y", "x"))
        with pytest.raises(FileNotFoundError):
            xenium_aligned_image(tmp_path / "missing.ome.tif", pixel_size=PIXEL_SIZE)
        with pytest.raises(FileNotFoundError):
            xenium_aligned_image(tmp_path / "x.ome.tif", tmp_path / "missing.csv", pixel_size=PIXEL_SIZE)
