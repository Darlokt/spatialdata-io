"""The metadata-only channel table of one MACSima stack."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from spatialdata.models import TableModel

from spatialdata_io.readers._utils.image import TiffMetadata
from spatialdata_io.readers.macsima._channels import NO_PADDING, PlannedChannel
from spatialdata_io.readers.macsima._metadata import ChannelMetadata, ChannelSource
from spatialdata_io.readers.macsima._schemas import CHANNEL_VAR_DTYPES
from spatialdata_io.readers.macsima._tables import make_channel_table

_STAIN = ChannelMetadata(
    name="CD3",
    cycle=1,
    image_type="stain",
    well="D01",
    roi=1,
    fluorophore="APC",
    clone="REA1151",
    exposure_ms=1212.52,
)
_BLEACH = ChannelMetadata(
    name="DAPI_background",
    cycle=15,
    image_type="bleach",
    well=None,
    roi=None,
    fluorophore=None,
    clone=None,
    exposure_ms=None,
)


def _channel(name: str, metadata: ChannelMetadata) -> PlannedChannel:
    source = ChannelSource(
        tiff=TiffMetadata(path=Path("/data/channel.tif"), series=(), ome=None),
        software_version="1.1.1",
        major=1,
        metadata=metadata,
        pixel_size_um=0.17,
        offset_yx=(0, 0),
        shape_yx=(4, 5),
        dtype=np.dtype(np.uint16),
    )
    return PlannedChannel(source=source, name=name, padding=NO_PADDING)


class TestMakeChannelTable:
    """One typed var row per image channel and no observations."""

    def test_exact_values_types_and_missingness(self) -> None:
        channels = [_channel("R1 CD3", _STAIN), _channel("R15 DAPI_background", _BLEACH)]

        table = make_channel_table(channels)

        assert table.n_obs == 0
        assert table.var.columns.tolist() == list(CHANNEL_VAR_DTYPES)
        assert table.X is None
        assert table.var_names.tolist() == ["R1 CD3", "R15 DAPI_background"]
        expected = pd.DataFrame(
            {
                "name": pd.array(["CD3", "DAPI_background"], dtype="string"),
                "cycle": np.array([1, 15], dtype=np.int64),
                "imagetype": pd.array(["stain", "bleach"], dtype="string"),
                "well": pd.array(["D01", pd.NA], dtype="string"),
                "ROI": pd.array([1, pd.NA], dtype="Int64"),
                "fluorophore": pd.array(["APC", pd.NA], dtype="string"),
                "clone": pd.array(["REA1151", pd.NA], dtype="string"),
                "exposure": np.array([1212.52, np.nan]),
            },
            index=pd.Index(["R1 CD3", "R15 DAPI_background"]),
        )
        assert isinstance(table.var, pd.DataFrame)
        pd.testing.assert_frame_equal(table.var, expected)

    def test_nullable_roi_preserves_large_integer_before_storage_inference(self) -> None:
        source = replace(_STAIN, roi=2**53 + 1)
        table = make_channel_table([_channel("stain", source), _channel("bleach", _BLEACH)])
        assert table.var["ROI"].iloc[0] == 2**53 + 1
        assert pd.isna(table.var["ROI"].iloc[1])
        assert str(table.var["ROI"].dtype) == "Int64"

    def test_annotates_no_element(self) -> None:
        table = make_channel_table([_channel("CD3", _STAIN)])

        assert TableModel.ATTRS_KEY not in table.uns or not table.uns[TableModel.ATTRS_KEY]
