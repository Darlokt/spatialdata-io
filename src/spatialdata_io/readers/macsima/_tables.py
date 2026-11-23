"""Describe the channels of one MACSima stack in a metadata-only table.

The table has no observations and ``X=None``; each ``var`` row describes one
image channel, indexed by its channel name, with the columns and dtypes of
the channel metadata schema. It annotates no element.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from anndata import AnnData
from spatialdata.models import TableModel

from spatialdata_io.readers.macsima._schemas import channel_var_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers.macsima._channels import PlannedChannel


def make_channel_table(channels: Sequence[PlannedChannel]) -> AnnData:
    """Return ``AnnData(X=None)`` with zero observations and one ``var`` row per channel.

    ``var_names`` are the image channel names. The table annotates no element
    and therefore has no region or instance keys.
    """
    index = pd.Index([channel.name for channel in channels])
    var = pd.DataFrame(channel_var_columns([channel.source.metadata for channel in channels]), index=index)
    return TableModel.parse(AnnData(var=var))
