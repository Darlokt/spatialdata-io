"""Tests for the shared rule labelling raster channels from declared colour or OME metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import inspect_tiff, ome_channel_names, raster_channel_names
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.image._types import SampleColor

_CONTEXT = ReaderErrorContext("test", "raster channels")


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
