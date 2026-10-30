"""MCMICRO workflow parameters and marker-panel parsing."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import yaml

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.mcmicro._schemas import MarkerPanel, normalize_markers, schema_error, validate_columns

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

_BOOL_TAG = "tag:yaml.org,2002:bool"


def _member(node: yaml.Node, key: str, path: ArtifactFile) -> yaml.Node:
    """Return the single value of a mapping key, rejecting ambiguous duplicates."""
    context = ReaderErrorContext("mcmicro", "workflow parameters")
    if not isinstance(node, yaml.MappingNode):
        raise ReaderFormatError(
            context=context, path=path, found=f"{node.tag} node", expected=f"a mapping with {key!r}"
        )
    values = [value for name, value in node.value if isinstance(name, yaml.ScalarNode) and name.value == key]
    if len(values) != 1:
        raise ReaderFormatError(
            context=context, path=path, found=f"{len(values)} {key!r} entries", expected=f"exactly one {key!r} entry"
        )
    return values[0]


def read_workflow(path: ArtifactFile) -> Literal["wsi", "tma"]:
    """Return the mode recorded by MCMICRO's complete ``qc/params.yml`` dump.

    Only the ``workflow.tma`` path is composed into nodes and interpreted; no
    other value is constructed, so unrelated aliases cannot expand.
    """
    expected = "YAML with a boolean workflow.tma"
    try:
        # Composition yields an inert node graph; no global loader state changes.
        with path.open(encoding="utf-8-sig") as source:
            root = yaml.compose(source, Loader=yaml.SafeLoader)
    except UnicodeDecodeError as error:
        raise schema_error(path, found="invalid UTF-8", expected=expected) from error
    except (yaml.YAMLError, RecursionError) as error:
        raise schema_error(path, found=f"malformed YAML: {error}", expected=expected) from error
    if root is None:
        raise schema_error(path, found="an empty document", expected=expected)

    tma = _member(_member(root, "workflow", path), "tma", path)
    if not isinstance(tma, yaml.ScalarNode) or tma.tag != _BOOL_TAG:
        raise schema_error(path, found=f"workflow.tma tagged {tma.tag}", expected=expected)
    return "tma" if yaml.SafeLoader.bool_values[tma.value.lower()] else "wsi"


def read_markers(path: ArtifactFile) -> MarkerPanel:
    """Parse the marker panel as exact text under one source owner."""
    expected = "a comma-delimited marker panel"
    try:
        with path.open("rb") as source:
            try:
                columns = next(csv.reader([source.readline().decode("utf-8-sig")], strict=True), [])
            except (UnicodeDecodeError, csv.Error) as error:
                raise schema_error(path, found="malformed CSV header", expected=expected) from error
            validate_columns(columns, path)
            source.seek(0)
            table = pacsv.read_csv(
                source,
                read_options=pacsv.ReadOptions(column_names=columns, skip_rows=1, use_threads=False),
                parse_options=pacsv.ParseOptions(newlines_in_values=True),
                convert_options=pacsv.ConvertOptions(column_types=dict.fromkeys(columns, pa.string()), null_values=[]),
            )
    except pa.ArrowInvalid as error:
        raise schema_error(path, found=f"malformed marker CSV: {error}", expected=expected) from error
    return normalize_markers(table.to_pandas(types_mapper=pd.ArrowDtype), path)
