"""Read Xenium run, metrics and panel metadata into JSON-serializable records."""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.xenium._version import detect_version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.xenium._version import Family

# Path blocks name files inside the bundle and are not dataset metadata. The Xenium
# Ranger command line can contain the producer's local paths.
_EXCLUDED_KEYS = frozenset({"images", "xenium_explorer_files"})
_RANGER_KEYS = ("version", "run_id", "original_analysis_uuid")
_PANEL_KEYS = ("num_gene_targets", "num_targets", "panel_type", "species", "tissue", "spec_version")
_INTEGER = re.compile(r"[+-]?[0-9]+")
_CSV_BLOCK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class Experiment:
    """Validated ``experiment.xenium`` facts used by discovery and assembly.

    ``version`` is the dotted format version, ``None`` for XOA 1.0/1.1.
    """

    version: str | None
    family: Family
    pixel_size: float
    images: Mapping[str, str]
    attrs: Mapping[str, object]

    def context(self, artifact: str) -> ReaderErrorContext:
        """Return the error context of a bundle artifact with the detected format version."""
        return ReaderErrorContext("xenium", artifact, self.version)


@dataclass(frozen=True, slots=True)
class ProteinTarget:
    """Annotation of one protein feature from ``protein_panel.json``."""

    long_name: str
    clone: str
    codeword_index: int


@dataclass(frozen=True, slots=True)
class ProteinPanel:
    """Protein panel identity and per-feature annotations keyed by public feature ID."""

    identity: Mapping[str, object]
    targets: Mapping[str, ProteinTarget]


def _positive(value: object, name: str, path: ArtifactFile, context: ReaderErrorContext) -> float:
    # JSON booleans are ints in Python but never a size.
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ReaderFormatError(
            context=context, path=path, found=f"{name} = {value!r}", expected=f"a finite positive {name}"
        )
    return float(value)


def _object(value: object, name: str, path: ArtifactFile, context: ReaderErrorContext) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{name} of type {type(value).__name__}",
            expected=f"{name} as a JSON object",
        )
    # read_json_object builds every JSON object, at any depth, as dict[str, object].
    return cast("dict[str, object]", value)


def read_experiment(path: ArtifactFile) -> Experiment:
    """Read ``experiment.xenium`` once: version, family, pixel size, image entries and attrs.

    Raises
    ------
    ReaderFormatError
        If the file is malformed, ``pixel_size`` is not finite and positive,
        ``analysis_sw_version`` is not text, ``z_step_size`` is not finite and
        positive when present, ``images`` is not an object of text paths, or the
        format version is unsupported.
    """
    context = ReaderErrorContext("xenium", "experiment.xenium")
    experiment = read_json_object(path, context=context)

    software = experiment.get("analysis_sw_version")
    if type(software) is not str:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"analysis_sw_version = {software!r}",
            expected="a software version string",
        )
    version, family = detect_version(experiment, software, path=path)
    context = replace(context, detected_version=version)
    pixel_size = _positive(experiment.get("pixel_size"), "pixel_size", path, context)
    if "z_step_size" in experiment:
        _positive(experiment["z_step_size"], "z_step_size", path, context)

    images = _object(experiment.get("images", {}), "images", path, context)
    if any(type(value) is not str or not value for value in images.values()):
        raise ReaderFormatError(
            context=context, path=path, found=f"images = {images!r}", expected="nonempty relative image paths"
        )

    attrs = {key: value for key, value in experiment.items() if key not in _EXCLUDED_KEYS}
    if "xenium_ranger" in attrs:
        ranger = _object(attrs["xenium_ranger"], "xenium_ranger", path, context)
        attrs["xenium_ranger"] = {key: ranger[key] for key in _RANGER_KEYS if key in ranger}
    return Experiment(version, family, pixel_size, {key: str(value) for key, value in images.items()}, attrs)


def _metric(token: str) -> object:
    """Parse one metric exactly: empty is missing, integers stay integers, text stays text."""
    if not token:
        return None
    if _INTEGER.fullmatch(token):
        return int(token)
    try:
        value = float(token)
    except ValueError:
        return token
    return value if math.isfinite(value) else token


def read_metrics(path: ArtifactFile, *, context: ReaderErrorContext) -> dict[str, object]:
    """Read the one-row ``metrics_summary.csv``; each value keeps the type its spelling states.

    Arrow reads every field as text, because the metric names and their types
    vary between releases. Each header and record must fit a 64 KiB parser
    block; excess records fail without materializing the rest of the file.

    Raises
    ------
    ReaderFormatError
        If the file is not UTF-8 CSV, or is not one unique, nonempty header and
        one record of equal width.
    """
    batches: list[pa.RecordBatch] = []
    rows = 0
    with path.open("rb") as handle, csv_errors(context=context, path=path):
        line = handle.readline(_CSV_BLOCK_BYTES + 1)
        if len(line) > _CSV_BLOCK_BYTES:
            raise ReaderFormatError(
                context=context,
                path=path,
                found="oversized header",
                expected=f"a header within {_CSV_BLOCK_BYTES} bytes",
            )
        with io.BytesIO(line) as header_stream:
            header = csv_header(path, context=context, source=header_stream)
        if "" in header:
            raise ReaderFormatError(
                context=context, path=path, found="empty column name", expected="one nonempty header and one record"
            )
        handle.seek(0)
        with pacsv.open_csv(
            handle,
            read_options=pacsv.ReadOptions(block_size=_CSV_BLOCK_BYTES, use_threads=False),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types=dict.fromkeys(header, pa.string()),
                include_columns=list(header),
                null_values=[],
                strings_can_be_null=False,
            ),
        ) as reader:
            for batch in reader:
                rows += batch.num_rows
                if rows > 1:
                    raise ReaderFormatError(
                        context=context,
                        path=path,
                        found="more than one record",
                        expected="one nonempty header and one record",
                    )
                batches.append(batch)
            table = pa.Table.from_batches(batches, schema=reader.schema)
    if rows != 1:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{table.num_rows} records with columns {list(header)!r}",
            expected="one nonempty header and one record",
        )
    return {name: _metric(table[name][0].as_py()) for name in header}


def _payload(path: ArtifactFile, context: ReaderErrorContext) -> tuple[dict[str, object], dict[str, object]]:
    """Return ``payload`` and ``payload.panel`` of a panel JSON document."""
    document = read_json_object(path, context=context)
    payload = _object(document.get("payload"), "payload", path, context)
    return payload, _object(payload.get("panel"), "payload.panel", path, context)


def _panel_identity(panel: dict[str, object], path: ArtifactFile, context: ReaderErrorContext) -> dict[str, object]:
    """Keep the scalar identity fields of a panel; per-target entries stay unread."""
    identity = _object(panel.get("identity", {}), "panel.identity", path, context)
    values: dict[str, object] = {key: value for key, value in identity.items() if isinstance(value, (str, int, float))}
    return values | {key: panel[key] for key in _PANEL_KEYS if isinstance(panel.get(key), (str, int, float))}


def read_gene_panel(path: ArtifactFile, *, context: ReaderErrorContext) -> dict[str, object]:
    """Return the identity of ``gene_panel.json``: name, design ID, version, species, tissue, targets, spec."""
    payload, panel = _payload(path, context)
    identity = _panel_identity(panel, path, context)
    # Gene panels record their specification version beside the panel.
    spec = payload.get("spec_version")
    return identity | {"spec_version": spec} if isinstance(spec, str) else identity


def read_protein_panel(path: ArtifactFile, *, context: ReaderErrorContext) -> ProteinPanel:
    """Return the protein panel identity and a 1:1 ``public_id`` target map.

    Raises
    ------
    ReaderFormatError
        If targets are malformed or a ``public_id`` repeats.
    """
    _, panel = _payload(path, context)
    targets_value = panel.get("targets")
    if type(targets_value) is not list:
        raise ReaderFormatError(
            context=context, path=path, found="no targets list", expected="payload.panel.targets as a list"
        )

    targets: dict[str, ProteinTarget] = {}
    for item in targets_value:
        target = _object(item, "target", path, context)
        public_id, long_name, clone, codeword = (
            target.get(key) for key in ("public_id", "long_name", "clone", "codeword_index")
        )
        if not (
            isinstance(public_id, str)
            and isinstance(long_name, str)
            and isinstance(clone, str)
            and type(codeword) is int
            and 0 <= codeword <= 2**64 - 1
        ):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"target {target!r}",
                expected="text public_id, long_name, clone and a uint64 codeword_index",
            )
        if public_id in targets:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"duplicate public_id {public_id!r}",
                expected="one target per protein",
            )
        targets[public_id] = ProteinTarget(long_name, clone, codeword)
    return ProteinPanel(_panel_identity(panel, path, context), targets)
