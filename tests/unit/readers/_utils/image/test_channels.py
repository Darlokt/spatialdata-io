"""Tests for the shared rule labelling raster channels from declared colour or OME metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import (
    channel_source,
    inspect_tiff,
    ome_channel_names,
    raster_channel_names,
)
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.image._types import SampleColor

_CONTEXT = ReaderErrorContext("test", "raster channels")

# Every (declared, source) pair a 2D reader can accept after axis normalization:
# inspection keeps only non-singleton C, S, Z and T axes, and a CYX target keeps
# only c, y and x, with S read as c. Values: (samples, ome_samples).
_2D_CASES = [
    ("yx", "yx", False, False),
    ("yx", "xy", False, False),
    ("cyx", "cyx", False, False),
    ("cyx", "cxy", False, False),
    ("cyx", "syx", False, False),
    ("cyx", "sxy", False, False),
    ("cyx", "ycx", False, None),
    ("cyx", "yxc", False, None),
    ("cyx", "yxs", False, None),
    ("cyx", "ysx", False, None),
    ("cyx", "xcy", False, None),
    ("cyx", "xyc", False, None),
    ("cyx", "xys", False, None),
    ("cyx", "xsy", False, None),
    ("yxs", "yxc", True, True),
    ("yxs", "yxs", True, True),
    ("yxs", "xyc", True, True),
    ("yxs", "xys", True, True),
    ("yxs", "cyx", False, None),
    ("yxs", "cxy", False, None),
    ("yxs", "ycx", False, None),
    ("yxs", "ysx", False, None),
    ("yxs", "xcy", False, None),
    ("yxs", "xsy", False, None),
    ("yxs", "syx", False, None),
    ("yxs", "sxy", False, None),
    ("syx", "cyx", True, True),
    ("syx", "cxy", True, True),
    ("syx", "syx", True, True),
    ("syx", "sxy", True, True),
    ("syx", "ycx", False, None),
    ("syx", "yxc", False, None),
    ("syx", "yxs", False, None),
    ("syx", "ysx", False, None),
    ("syx", "xcy", False, None),
    ("syx", "xyc", False, None),
    ("syx", "xys", False, None),
    ("syx", "xsy", False, None),
    # A Z or T axis read as channels never corresponds to OME channel records.
    *(
        (declared, source, False, None)
        for declared in ("zyx", "tyx")
        for source in ("cyx", "cxy", "ycx", "yxc", "yxs", "ysx", "xcy", "xyc", "xys", "xsy", "syx", "sxy")
    ),
]


class TestRasterChannelNames:
    """Prefer a declared colour model for sample channels, then OME identifiers, then positions."""

    @pytest.mark.parametrize(
        ("color", "ome", "expected"),
        [
            ("rgb", None, ["r", "g", "b"]),
            ("rgba", None, ["r", "g", "b", "a"]),
            ("rgb", ["H&E:sample:0", "H&E:sample:1", "H&E:sample:2"], ["r", "g", "b"]),
            (None, ["H&E:sample:0", "H&E:sample:1", "H&E:sample:2"], ["H&E:sample:0", "H&E:sample:1", "H&E:sample:2"]),
            (None, None, None),
        ],
    )
    def test_sample_channels(
        self, color: SampleColor | None, ome: list[str] | None, expected: list[str] | None
    ) -> None:
        assert raster_channel_names(color, samples=True, ome=ome) == expected

    @pytest.mark.parametrize(("color", "ome"), [("rgb", ["DAPI", "CD3", "CD8"]), ("rgba", None), (None, ["DAPI"])])
    def test_other_channel_axes_ignore_the_sample_colour(
        self, color: SampleColor | None, ome: list[str] | None
    ) -> None:
        # The colour model describes only the samples axis, never a scientific channel axis.
        assert raster_channel_names(color, samples=False, ome=ome) == ome

    def test_returns_a_new_list(self) -> None:
        first = raster_channel_names("rgb", samples=True, ome=None)
        assert first is not None
        first.append("x")

        assert raster_channel_names("rgb", samples=True, ome=None) == ["r", "g", "b"]

    @pytest.mark.parametrize(
        ("photometric", "expected"),
        [("rgb", ["r", "g", "b"]), ("minisblack", ["H&E:sample:0", "H&E:sample:1", "H&E:sample:2"])],
    )
    def test_named_ome_samples_of_an_inspected_tiff(
        self, tmp_path: Path, photometric: str, expected: list[str]
    ) -> None:
        path = tmp_path / "image.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((8, 8, 3), dtype=np.uint8),
            photometric=photometric,
            planarconfig="contig",
            metadata={"axes": "YXS", "Channel": {"Name": ["H&E"]}},
        )
        metadata = inspect_tiff(ArtifactFile(path), context=_CONTEXT)
        level = metadata.series[0].levels[0]

        ome = ome_channel_names(metadata, series=0, count=3, samples=True, context=_CONTEXT)

        assert level.axes == ("Y", "X", "S")
        assert ome == ["H&E:sample:0", "H&E:sample:1", "H&E:sample:2"]
        assert raster_channel_names(level.color, samples=True, ome=ome) == expected


class TestChannelSource:
    """Map the raster's channel axis to the file's samples and OME channel records."""

    def test_covers_every_reachable_2d_case(self) -> None:
        assert len(_2D_CASES) == 62
        assert len({(declared, source) for declared, source, _, _ in _2D_CASES}) == 62

    @pytest.mark.parametrize(("declared", "source", "samples", "ome_samples"), _2D_CASES)
    def test_2d_cases(self, declared: str, source: str, *, samples: bool, ome_samples: bool | None) -> None:
        assert channel_source(tuple(declared), tuple(source)) == (samples, ome_samples)

    @pytest.mark.parametrize(
        ("declared", "source", "expected"),
        [
            # Samples beside a channel axis: colour may apply, OME records describe C instead.
            ("cyxs", "zyxc", (True, None)),
            ("cyxs", "cyxz", (False, False)),
            ("zyxs", "zyxc", (True, True)),
            ("czyx", "czyx", (False, False)),
            ("czyx", "zcyx", (False, None)),
            ("zyx", "zyx", (False, False)),
            ("cyx", "zyx", (False, None)),
            ("yxs", "yxz", (False, None)),
        ],
    )
    def test_volumes_and_channels_beside_samples(
        self, declared: str, source: str, expected: tuple[bool, bool | None]
    ) -> None:
        assert channel_source(tuple(declared), tuple(source)) == expected

    def test_rejects_axes_of_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="zip"):
            channel_source(("c", "y", "x"), ("y", "x"))
