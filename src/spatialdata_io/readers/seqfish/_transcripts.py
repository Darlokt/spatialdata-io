"""Build projected path-backed seqFISH transcript partitions without a scan."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.csv import read_csv_blocks, read_csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers.seqfish._schemas import (
    STRING_DTYPE,
    normalize_transcript_partition,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    import dask.dataframe as dd

    from spatialdata_io.readers._utils.paths import ArtifactFile


def read_transcripts(path: ArtifactFile, *, features: tuple[str, ...], roi: str) -> dd.DataFrame:
    """Return lazy XY points with validated feature categories and raw z annotation.

    Only the CSV header is read at construction. Known categories
    and an explicit no-sort policy prevent the public model parser from
    executing a feature or index scan. Row validation belongs to each task.
    """
    context = ReaderErrorContext("seqfish", path.name)
    names = read_csv_header(path, context=context)
    columns = validate_transcript_columns(names, path)
    # Pandas projection ignores malformed record widths, and its Arrow engine
    # infers numbers before applying dtype, irreversibly rounding large IDs.
    # Arrow instead decodes the required fields as text inside each block task.
    frame = read_csv_blocks(path, header=names, columns=columns, context=context)
    # The empty meta frame gives the partition function the schema of a real
    # task without executing one.
    empty = pd.DataFrame({name: pd.Series(dtype=STRING_DTYPE) for name in columns})
    meta = normalize_transcript_partition(empty, features=features, path=path)
    normalized = frame.map_partitions(
        normalize_transcript_partition,
        features=features,
        path=path,
        meta=meta,
    )
    # Unknown divisions are intentional for independent CSV blocks. This exact
    # upstream warning does not indicate a failed sort, since none was requested.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            normalized,
            coordinates={"x": "x", "y": "y"},
            feature_key="name",
            instance_key="cell" if "cell" in columns else None,
            transformations={roi: Identity()},
            sort=False,
        )
