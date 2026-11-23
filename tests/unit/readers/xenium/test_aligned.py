"""Aligned H&E and IF images: declared axes, channel names and the micrometre transform."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata.transformations import Affine, Scale, get_transformation

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import xenium_aligned_image
from spatialdata_io.readers.xenium._aligned import read_alignment, read_keypoint_alignment

if TYPE_CHECKING:
    from pathlib import Path


# Rows of a real Xenium Explorer H&E alignment (pancreas, XOA 1.6.0.8).
MATRIX = [[0.00184638, 0.64417243, 1433.27392], [-0.6441724, 0.00184638, 12184.7121], [0.0, 0.0, 1.0]]
PIXEL_SIZE = 0.2125
CONTEXT = ReaderErrorContext("xenium", "image alignment", "6.0.0")


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
        np.testing.assert_array_equal(read_alignment(_csv(tmp_path / "a.csv", MATRIX), context=CONTEXT), MATRIX)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("1,0,0\n0,1,0\n", "3 x 3"),
            ("1,0,0\n0,1,0\n0,0,2\n", "3 x 3"),
            ("0,0,0\n0,0,0\n0,0,1\n", "3 x 3"),
            ("1,0,x\n0,1,0\n0,0,1\n", "invalid value 'x'"),
            ("1,0,inf\n0,1,0\n0,0,1\n", "3 x 3"),
            ("1,0,0\n0,1,0\n0,0,1\n0,0,1\n", "3 x 3"),
        ],
        ids=["rows", "last-row", "singular", "text", "infinite", "extra-row"],
    )
    def test_rejects_malformed_matrices(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match) as caught:
            read_alignment(_csv(tmp_path / "a.csv", text), context=CONTEXT)

        assert caught.value.context == CONTEXT


class TestReadKeypointAlignment:
    """Keypoint pairs determine one similarity transform."""

    def test_projected_fields_and_least_squares_fit(self, tmp_path: Path) -> None:
        path = _csv(
            tmp_path / "sample_keypoints.csv",
            "alignmentY,note,fixedY,alignmentX,fixedX\n"
            "-1,first,-3,-1,5\n-1,second,1,1,3\n1,third,-3,-1,-1\n1,fourth,1,1,1\n",
        )

        matrix = read_keypoint_alignment(path, context=CONTEXT)

        # The points use x/y scales of 2 with an extra x residual of ±1.
        # Isotropic least squares projects the residual out of the similarity.
        np.testing.assert_allclose(matrix, [[0, -2, 2], [2, 0, -1], [0, 0, 1]], atol=1e-14)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("fixedX,fixedY,alignmentX,other\n", "missing columns"),
            ("fixedX,fixedY,alignmentX,alignmentX\n", "repeated column names"),
            ("fixedX,fixedY,alignmentX,alignmentY\n1,2,3,4\n", "too few"),
            ("fixedX,fixedY,alignmentX,alignmentY\n0,0,0,0\n0,1,0,1\n1,0,1,nan\n", "nonfinite"),
            ("fixedX,fixedY,alignmentX,alignmentY\n0,0,0,0\n0,1,0,1\n1,0,0,2\n", "degenerate"),
            ("fixedX,fixedY,alignmentX,alignmentY\n0,0,0,0\n0,1,0,1\n0,2,1,0\n", "degenerate"),
            ("fixedX,fixedY,alignmentX,alignmentY\n0,0,0,0\n0,1e200,0,1e200\n1e200,0,1e200,0\n", "overflow"),
        ],
        ids=["missing", "duplicate", "short", "nan", "source-line", "fixed-line", "overflow"],
    )
    def test_invalid_keypoints(self, tmp_path: Path, text: str, match: str) -> None:
        path = _csv(tmp_path / "sample_keypoints.csv", text)

        with pytest.raises(ReaderFormatError, match=match) as caught:
            read_keypoint_alignment(path, context=CONTEXT)

        assert caught.value.path == path
        assert caught.value.context == CONTEXT


class TestXeniumAlignedImage:
    """Aligned images keep declared axes and channel names and map pixels to micrometres."""

    @pytest.mark.parametrize("planar", [False, True])
    def test_rgb_samples_with_alignment(self, tmp_path: Path, *, planar: bool) -> None:
        path = tmp_path / "x_he_image.ome.tif"
        data = _rgb(path)
        if planar:
            tifffile.imwrite(
                path,
                np.moveaxis(data, -1, 0),
                photometric="rgb",
                planarconfig="separate",
                ome=True,
                metadata={"axes": "SYX"},
                tile=(16, 16),
            )
        alignment = _csv(tmp_path / "x_he_imagealignment.csv", MATRIX)

        image = xenium_aligned_image(path, alignment, pixel_size=PIXEL_SIZE)

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

        image = xenium_aligned_image(path, pixel_size=PIXEL_SIZE, data_axes="xy", chunks=(1, 4, 4), scale_factors=(2,))

        scale0 = image["scale0"]["image"]
        np.testing.assert_array_equal(scale0.data.compute()[0], data)
        assert scale0.data.chunksize == (1, 4, 4)
        assert scale0.coords["c"].to_numpy().tolist() == [0]

    def test_undeclared_samples_keep_their_ome_identity(self, tmp_path: Path) -> None:
        # Three MinIsBlack samples are not RGB, whatever their count.
        path = tmp_path / "x.ome.tif"
        data = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        tifffile.imwrite(
            path, data, photometric="minisblack", planarconfig="contig", ome=True, metadata={"axes": "YXS"}
        )

        image = xenium_aligned_image(path, pixel_size=PIXEL_SIZE)

        assert image.coords["c"].to_numpy().tolist() == [f"Channel:0:0:sample:{index}" for index in range(3)]
        np.testing.assert_array_equal(image.data.compute(), np.moveaxis(data, -1, 0))

    def test_samples_read_as_channels_keep_rgb_names(self, tmp_path: Path) -> None:
        data = _rgb(tmp_path / "x.ome.tif")

        image = xenium_aligned_image(tmp_path / "x.ome.tif", pixel_size=PIXEL_SIZE, data_axes="yxc")

        assert image.coords["c"].to_numpy().tolist() == ["r", "g", "b"]
        np.testing.assert_array_equal(image.data.compute(), np.moveaxis(data, -1, 0))

    def test_repeated_ome_names_fall_back_to_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "x.ome.tif"
        tifffile.imwrite(
            path, np.zeros((2, 12, 16), np.uint16), ome=True, metadata={"axes": "CYX", "Channel": {"Name": ["A", "A"]}}
        )

        image = xenium_aligned_image(path, pixel_size=PIXEL_SIZE)

        assert image.coords["c"].to_numpy().tolist() == ["Channel:0:0", "Channel:0:1"]

    def test_axis_reinterpreted_as_channels_gets_positions(self, tmp_path: Path) -> None:
        path = tmp_path / "x.ome.tif"
        data = np.arange(3 * 12 * 16, dtype=np.uint16).reshape(3, 12, 16)
        tifffile.imwrite(path, data, ome=True, metadata={"axes": "ZYX", "Channel": {"Name": ["DAPI"]}})

        # The single OME channel record describes neither the Z planes nor the X columns.
        planes = xenium_aligned_image(path, pixel_size=PIXEL_SIZE, data_axes="cyx")
        columns = xenium_aligned_image(path, pixel_size=PIXEL_SIZE, data_axes="ycx")

        assert planes.coords["c"].to_numpy().tolist() == [0, 1, 2]
        np.testing.assert_array_equal(planes.data.compute(), data)
        assert columns.coords["c"].to_numpy().tolist() == list(range(12))

    def test_declared_axes_that_cannot_become_cyx_are_malformed(self, tmp_path: Path) -> None:
        path = tmp_path / "x.ome.tif"
        tifffile.imwrite(path, np.zeros((2, 12, 16), np.uint16), ome=True, metadata={"axes": "ZYX"})

        with pytest.raises(ReaderFormatError, match="Axis 'z' has length 2") as caught:
            xenium_aligned_image(path, pixel_size=PIXEL_SIZE)

        assert caught.value.context == ReaderErrorContext("xenium", "aligned image")
        assert caught.value.path == path
        assert type(caught.value.__cause__) is ValueError

    def test_scale_factors_must_fit_the_image(self, tmp_path: Path) -> None:
        _rgb(tmp_path / "x.ome.tif")

        with pytest.raises(ValueError, match=r"exceed the spatial extent of the aligned image \(12 x 16 pixels\)"):
            xenium_aligned_image(tmp_path / "x.ome.tif", pixel_size=PIXEL_SIZE, scale_factors=(4, 4))

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            ({"pixel_size": "0.2"}, TypeError, "number of micrometres"),
            ({"pixel_size": True}, TypeError, "number of micrometres"),
            ({"pixel_size": 0.0}, ValueError, "finite positive"),
            ({"pixel_size": float("nan")}, ValueError, "finite positive"),
            (
                {"pixel_size": 0.2, "data_axes": ("y", "x", "s")},
                TypeError,
                "string such as 'CYX', or None; found tuple",
            ),
            ({"pixel_size": 0.2, "data_axes": "y x"}, ValueError, "distinct single-letter"),
            ({"pixel_size": 0.2, "data_axes": "yXy"}, ValueError, "distinct single-letter"),
            # Samples beside channels, or no X, fail before the image is read.
            ({"pixel_size": 0.2, "data_axes": "cyxs"}, ValueError, r"S \(samples\) counting as C"),
            ({"pixel_size": 0.2, "data_axes": "cys"}, ValueError, "including X and Y"),
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

        with pytest.raises(ValueError, match="rank 2 does not match array rank 3") as caught:
            xenium_aligned_image(tmp_path / "x.ome.tif", pixel_size=PIXEL_SIZE, data_axes="yx")
        assert type(caught.value) is ValueError
        context = ReaderErrorContext("xenium", "aligned image")
        assert caught.value.__notes__ == [f"While handling {context.describe(path=tmp_path / 'x.ome.tif')}."]
        with pytest.raises(FileNotFoundError):
            xenium_aligned_image(tmp_path / "missing.ome.tif", pixel_size=PIXEL_SIZE)
        with pytest.raises(FileNotFoundError):
            xenium_aligned_image(tmp_path / "x.ome.tif", tmp_path / "missing.csv", pixel_size=PIXEL_SIZE)
