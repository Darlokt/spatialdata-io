"""Describe the channels of one MACSima stack in a metadata-only table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from anndata import AnnData
from spatialdata.models import TableModel

from spatialdata_io.readers.macsima._schemas import CHANNEL_VAR_DTYPES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers.macsima._channels import PlannedChannel


def make_channel_table(channels: Sequence[PlannedChannel]) -> AnnData:
    """Return ``AnnData(X=None)`` with zero observations and one ``var`` row per channel.

    ``var_names`` are the image channel names. The table annotates no element
    and therefore has no region or instance keys.
    """
    metadata = [channel.source.metadata for channel in channels]
    values = {
        "name": [row.name for row in metadata],
        "cycle": [row.cycle for row in metadata],
        "imagetype": [row.image_type for row in metadata],
        "well": [row.well for row in metadata],
        "ROI": [row.roi for row in metadata],
        "fluorophore": [row.fluorophore for row in metadata],
        "clone": [row.clone for row in metadata],
        # A missing exposure becomes NaN in the float column.
        "exposure": [row.exposure_ms for row in metadata],
    }
    # Type source values before DataFrame inference can round nullable integer IDs.
    index = pd.Index([channel.name for channel in channels])
    columns = {name: pd.Series(values[name], index=index, dtype=dtype) for name, dtype in CHANNEL_VAR_DTYPES.items()}
    var = pd.DataFrame(columns, index=index)
    return TableModel.parse(AnnData(var=var))
