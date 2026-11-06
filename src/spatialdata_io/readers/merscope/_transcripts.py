"""Construct projected, path-backed MERSCOPE transcript partitions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.csv import read_csv_blocks, read_csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers.merscope._schemas import (
    STRING_DTYPE,
    normalize_transcript_partition,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    import dask.dataframe as dd

    from spatialdata_io.readers.merscope._layout import TranscriptSource


def read_transcripts(
    source: TranscriptSource,
    *,
    features: tuple[str, ...],
    frame: str,
    allowed_cell_ids: tuple[str, ...] | None,
) -> dd.DataFrame:
    """Return calibrated XY points without reading transcript rows eagerly.

    Lazy newline-aligned blocks decode projected lexical fields before
    reader-owned value normalization. CSV records must each occupy one
    physical line. Quoted commas and quotes remain supported.
    """
    path = source.path
    context = ReaderErrorContext("merscope", path.name)
    names = read_csv_header(path, context=context)
    columns = validate_transcript_columns(names, path)
    # Arrow decodes projected fields as text, so exact IDs are never rounded
    # through an inferred float, and checks every record's full width.
    dataframe = read_csv_blocks(path, header=names, columns=columns, context=context)
    empty = pd.DataFrame({name: pd.Series(dtype=STRING_DTYPE) for name in columns})

    meta = normalize_transcript_partition(
        empty, features=features, path=path, assignment=source.assignment, allowed_cell_ids=allowed_cell_ids
    )
    normalized = dataframe.map_partitions(
        normalize_transcript_partition,
        features=features,
        path=path,
        assignment=source.assignment,
        allowed_cell_ids=allowed_cell_ids,
        meta=meta,
    )

    # Unsorted independent partitions are intentional. Known feature categories
    # and sort=False avoid the public model's eager category/index scans.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            normalized,
            coordinates={"x": "x", "y": "y"},
            feature_key="gene",
            instance_key="cell_id" if source.assignment == "current" and "cell_id" in columns else None,
            transformations={frame: Identity()},
            sort=False,
        )
