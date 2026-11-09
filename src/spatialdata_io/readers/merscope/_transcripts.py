"""Build lazy transcript points from ``detected_transcripts.csv``.

The CSV is split into newline-aligned byte blocks; each partition decodes only
the projected columns with Arrow types (``_schemas.transcript_types``) and
normalizes them on compute, so building the points reads only the header.
Every record must fit one physical line: quoted commas and quotes are allowed,
quoted newlines are not. Records must have the full header width, even in
columns that are not read.

``global_x``/``global_y`` become float64 ``x``/``y`` in micrometres, in the
``global`` frame (identity). ``global_z`` is a stack index and stays the
annotation ``z_raw``. ``gene`` is categorical in the order of the paired count
header, which includes controls and features without counts; an unknown gene
fails when its partition is computed. Assignments to current cells are
``cell_id`` (the instance key, checked against the current cell population when
one is returned); events of the original instrument read beside VPT cells keep
their assignments as ``original_cell_id``, which refers to no returned element.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.csv import csv_header, read_csv_blocks
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers.merscope._schemas import (
    STRING_DTYPE,
    TRANSCRIPTS,
    normalize_transcript_partition,
    transcript_types,
)

if TYPE_CHECKING:
    import dask.dataframe as dd

    from spatialdata_io.readers.merscope._layout import TranscriptSource

_CONTEXT = ReaderErrorContext("merscope", TRANSCRIPTS)


def read_transcripts(
    source: TranscriptSource,
    *,
    features: tuple[str, ...],
    cell_population: pd.Index | None,
) -> dd.DataFrame:
    """Return projected lazy points; current assignments are checked against ``cell_population``.

    ``cell_population`` must be unique; ``None`` skips the assignment check.
    """
    path = source.path
    names = csv_header(path, context=_CONTEXT)
    types = transcript_types(names, path)
    columns = tuple(types)
    dataframe = read_csv_blocks(path, header=names, columns=columns, column_types=types, context=_CONTEXT)

    population = None if cell_population is None else pd.Index(cell_population, dtype=STRING_DTYPE)
    empty = pa.schema(list(types.items())).empty_table().to_pandas(types_mapper=pd.ArrowDtype)
    meta = normalize_transcript_partition(
        empty, features=features, path=path, assignment=source.assignment, cell_population=population
    )
    normalized = dataframe.map_partitions(
        normalize_transcript_partition,
        features=features,
        path=path,
        assignment=source.assignment,
        cell_population=population,
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
            transformations={"global": Identity()},
            sort=False,
        )
