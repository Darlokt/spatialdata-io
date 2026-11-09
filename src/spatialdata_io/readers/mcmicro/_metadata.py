"""Read the run mode from ``qc/params.yml`` and the marker panel from ``markers.csv``.

MCMICRO writes ``qc/params.yml`` as a complete dump of the parameters a run
used (https://mcmicro.org/). Only one value is interpreted::

    workflow:
      tma: false        # YAML boolean: true for a tissue-microarray run
      ...               # every other key is ignored
    options: ...
    modules: ...

The document is composed with PyYAML's safe loader into an inert node graph,
so unrelated values are never constructed and their aliases never expand.
``workflow`` and ``tma`` must each occur exactly once in their mapping, and the
value must carry the YAML boolean tag (``"false"`` or ``1`` fail). No
pipeline version is read, so none is recorded in provenance.

``markers.csv`` is parsed by Arrow as exact text (its column contract is in
``_schemas``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import yaml

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers.mcmicro._schemas import (
    MARKERS_CONTEXT,
    MarkerPanel,
    normalize_markers,
    validate_columns,
)

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

_BOOL_TAG = "tag:yaml.org,2002:bool"
_WORKFLOW_CONTEXT = ReaderErrorContext("mcmicro", "workflow parameters")


def _member(node: yaml.Node, key: str, path: ArtifactFile) -> yaml.Node:
    """Return the single value of a mapping key, rejecting ambiguous duplicates."""
    if not isinstance(node, yaml.MappingNode):
        raise ReaderFormatError(
            context=_WORKFLOW_CONTEXT, path=path, found=f"{node.tag} node", expected=f"a mapping with {key!r}"
        )
    values = [value for name, value in node.value if isinstance(name, yaml.ScalarNode) and name.value == key]
    if len(values) != 1:
        raise ReaderFormatError(
            context=_WORKFLOW_CONTEXT,
            path=path,
            found=f"{len(values)} {key!r} entries",
            expected=f"exactly one {key!r} entry",
        )
    return values[0]


def read_workflow(path: ArtifactFile) -> Literal["wsi", "tma"]:
    """Return the run mode selected by the YAML boolean ``workflow.tma``."""
    expected = "YAML with a boolean workflow.tma"
    try:
        # Composition yields an inert node graph; no global loader state changes.
        with path.open(encoding="utf-8-sig") as source:
            root = yaml.compose(source, Loader=yaml.SafeLoader)
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=_WORKFLOW_CONTEXT, path=path, found="invalid UTF-8", expected=expected
        ) from error
    except (yaml.YAMLError, RecursionError) as error:
        raise ReaderFormatError(
            context=_WORKFLOW_CONTEXT, path=path, found=f"malformed YAML: {error}", expected=expected
        ) from error
    if root is None:
        raise ReaderFormatError(context=_WORKFLOW_CONTEXT, path=path, found="an empty document", expected=expected)

    tma = _member(_member(root, "workflow", path), "tma", path)
    if not isinstance(tma, yaml.ScalarNode) or tma.tag != _BOOL_TAG:
        raise ReaderFormatError(
            context=_WORKFLOW_CONTEXT, path=path, found=f"workflow.tma tagged {tma.tag}", expected=expected
        )
    return "tma" if yaml.SafeLoader.bool_values[tma.value.lower()] else "wsi"


def read_markers(path: ArtifactFile) -> MarkerPanel:
    """Parse the marker panel as exact text under one source owner."""
    try:
        with path.open("rb") as source, csv_errors(context=MARKERS_CONTEXT, path=path):
            columns = csv_header(path, context=MARKERS_CONTEXT, source=source)
            validate_columns(columns, path, context=MARKERS_CONTEXT)
            source.seek(0)
            table = pacsv.read_csv(
                source,
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(
                    column_types={name: pa.string() for name in columns},
                    include_columns=list(columns),
                    null_values=[],
                    strings_can_be_null=False,
                ),
            )
    except OSError as error:
        add_reader_context(error, context=MARKERS_CONTEXT, path=path)
        raise
    return normalize_markers(table.to_pandas(types_mapper=pd.ArrowDtype), path)
