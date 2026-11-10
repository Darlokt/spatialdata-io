"""Read ``<ROI>_TranscriptList.csv`` lazily into SpatialData points.

Each record is one decoded transcript (see :mod:`._schemas`):

.. code-block:: text

    name,x,y,z,cell
    GeneA,4.5,1.25,1,2
    GeneB,1.5,0.5,2,9

``x`` and ``y`` are µm in the ROI frame, so the points use the identity. ``z``
is an image-plane index without a physical calibration; it is kept as the
integer annotation ``z_raw`` and the points stay two-dimensional. ``cell`` is
the assigned cell ID or ``0`` and is stored as ``uint64[pyarrow]`` in every
partition, so IDs up to ``2**64 - 1`` stay exact however partitions differ.
``name`` is categorical with the CellxGene features as categories, in header
order, including features without counts.

Only the header is read when the points are built; every record must occupy
one physical line. Arrow parses each newline-aligned block with the declared
types when its partition is computed, so a malformed record, a line with an odd
number of double quotes, an unknown feature or a nonfinite coordinate fails then.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from spatialdata_io.readers._utils.csv import csv_header, read_csv_blocks
from spatialdata_io.readers.seqfish._schemas import (
    TRANSCRIPTS_CONTEXT,
    normalize_transcript_partition,
    transcript_types,
)

if TYPE_CHECKING:
    import dask.dataframe as dd

    from spatialdata_io.readers._utils.paths import ArtifactFile


def read_transcripts(path: ArtifactFile, *, features: tuple[str, ...], frame: str) -> dd.DataFrame:
    """Return lazy XY points with validated feature categories and raw z annotation."""
    header = csv_header(path, context=TRANSCRIPTS_CONTEXT)
    types = transcript_types(header, path)
    blocks = read_csv_blocks(path, header=header, columns=tuple(types), context=TRANSCRIPTS_CONTEXT, column_types=types)
    vocabulary = pd.Index(features)
    # The empty typed frame gives the partition function the schema of a real
    # task without executing one.
    empty = pa.schema(list(types.items())).empty_table().to_pandas(types_mapper=pd.ArrowDtype)
    meta = normalize_transcript_partition(empty, features=vocabulary, path=path)
    points = blocks.map_partitions(normalize_transcript_partition, features=vocabulary, path=path, meta=meta)
    # Known categories and sort=False keep the model parser from scanning the
    # features or the index. Unknown divisions are intentional for independent
    # CSV blocks, so this exact upstream warning does not indicate a failure.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The index of the dataframe is not monotonic increasing.*", category=UserWarning
        )
        return PointsModel.parse(
            points,
            coordinates={"x": "x", "y": "y"},
            feature_key="name",
            instance_key="cell" if "cell" in types else None,
            transformations={frame: Identity()},
            sort=False,
        )
