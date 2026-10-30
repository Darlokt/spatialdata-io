"""Read Xenium run, metrics and panel metadata into JSON-serializable records."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, cast

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.xenium._version import FormatVersion, detect_version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.xenium._version import Family

# Path blocks name files inside the bundle and are not dataset metadata. The Xenium
# Ranger command line can contain the producer's local paths.
_EXCLUDED_KEYS = frozenset({"images", "xenium_explorer_files"})
_RANGER_KEYS = ("version", "run_id", "original_analysis_uuid")
_PANEL_KEYS = ("num_gene_targets", "num_targets", "panel_type", "species", "tissue", "spec_version")
_METRICS_RECORDS = 2  # header and one value record
_INTEGER = re.compile(r"[+-]?[0-9]+")


@dataclass(frozen=True, slots=True)
class Experiment:
    """Validated ``experiment.xenium`` facts used by discovery and assembly."""

    version: FormatVersion | None
    family: Family
    software: str
    pixel_size: float
    images: Mapping[str, str]
    attrs: Mapping[str, object]


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


def _error(path: ArtifactFile, artifact: str, *, found: str, expected: str) -> ReaderFormatError:
    return ReaderFormatError(context=ReaderErrorContext("xenium", artifact), path=path, found=found, expected=expected)


def _positive(value: object, name: str, path: ArtifactFile) -> float:
    # JSON booleans are ints in Python but never a size.
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise _error(path, "experiment.xenium", found=f"{name} = {value!r}", expected=f"a finite positive {name}")
    return float(value)


def _object(value: object, name: str, path: ArtifactFile, artifact: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _error(
            path, artifact, found=f"{name} of type {type(value).__name__}", expected=f"{name} as a JSON object"
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
    experiment = read_json_object(path, context=ReaderErrorContext("xenium", "experiment.xenium"))

    software = experiment.get("analysis_sw_version")
    if type(software) is not str:
        raise _error(
            path, "experiment.xenium", found=f"analysis_sw_version = {software!r}", expected="a software version string"
        )
    version, family = detect_version(experiment, software, path=path)
    pixel_size = _positive(experiment.get("pixel_size"), "pixel_size", path)
    if "z_step_size" in experiment:
        _positive(experiment["z_step_size"], "z_step_size", path)

    images = _object(experiment.get("images", {}), "images", path, "experiment.xenium")
    if any(type(value) is not str or not value for value in images.values()):
        raise _error(path, "experiment.xenium", found=f"images = {images!r}", expected="nonempty relative image paths")

    attrs = {key: value for key, value in experiment.items() if key not in _EXCLUDED_KEYS}
    if "xenium_ranger" in attrs:
        ranger = _object(attrs["xenium_ranger"], "xenium_ranger", path, "experiment.xenium")
        attrs["xenium_ranger"] = {key: ranger[key] for key in _RANGER_KEYS if key in ranger}
    return Experiment(version, family, software, pixel_size, {key: str(value) for key, value in images.items()}, attrs)


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


def read_metrics(path: ArtifactFile) -> dict[str, object]:
    """Read the one-row ``metrics_summary.csv`` with strict CSV syntax.

    Raises
    ------
    ReaderFormatError
        If the file is not UTF-8, or is not one unique header and one record
        of equal width.
    """
    expected = "a UTF-8 CSV with one unique header and one record"
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = (row for row in csv.reader(handle, strict=True) if row)
            rows = list(islice(records, _METRICS_RECORDS + 1))
    except (UnicodeDecodeError, csv.Error) as error:
        raise _error(path, "metrics_summary.csv", found="malformed CSV", expected=expected) from error
    if (
        len(rows) != _METRICS_RECORDS
        or len(rows[0]) != len(rows[1])
        or len(set(rows[0])) != len(rows[0])
        or "" in rows[0]
    ):
        raise _error(path, "metrics_summary.csv", found=f"{len(rows)} records", expected=expected)
    return {name: _metric(token) for name, token in zip(rows[0], rows[1], strict=True)}


def _panel_identity(panel: dict[str, object], spec: object, path: ArtifactFile, artifact: str) -> dict[str, object]:
    """Keep the scalar identity fields of a panel; per-target entries stay unread."""
    identity = _object(panel.get("identity", {}), "panel.identity", path, artifact)
    values: dict[str, object] = {key: value for key, value in identity.items() if isinstance(value, (str, int, float))}
    values |= {key: panel[key] for key in _PANEL_KEYS if isinstance(panel.get(key), (str, int, float))}
    if isinstance(spec, str):
        values["spec_version"] = spec
    return values


def read_gene_panel(path: ArtifactFile) -> dict[str, object]:
    """Return the identity of ``gene_panel.json``: name, design ID, version, species, tissue, targets, spec."""
    document = read_json_object(path, context=ReaderErrorContext("xenium", "gene_panel.json"))
    payload = _object(document.get("payload"), "payload", path, "gene_panel.json")
    panel = _object(payload.get("panel"), "payload.panel", path, "gene_panel.json")
    return _panel_identity(panel, payload.get("spec_version"), path, "gene_panel.json")


def read_protein_panel(path: ArtifactFile) -> ProteinPanel:
    """Return the protein panel identity and a 1:1 ``public_id`` target map.

    Raises
    ------
    ReaderFormatError
        If targets are malformed or a ``public_id`` repeats.
    """
    artifact = "protein_panel.json"
    document = read_json_object(path, context=ReaderErrorContext("xenium", artifact))
    payload = _object(document.get("payload"), "payload", path, artifact)
    panel = _object(payload.get("panel"), "payload.panel", path, artifact)
    targets_value = panel.get("targets")
    if type(targets_value) is not list:
        raise _error(path, artifact, found="no targets list", expected="payload.panel.targets as a list")

    targets: dict[str, ProteinTarget] = {}
    for item in targets_value:
        target = _object(item, "target", path, artifact)
        public_id, long_name, clone, codeword = (
            target.get(key) for key in ("public_id", "long_name", "clone", "codeword_index")
        )
        if not (
            isinstance(public_id, str)
            and isinstance(long_name, str)
            and isinstance(clone, str)
            and type(codeword) is int
        ):
            raise _error(
                path,
                artifact,
                found=f"target {target!r}",
                expected="text public_id, long_name, clone and an integer codeword_index",
            )
        if public_id in targets:
            raise _error(path, artifact, found=f"duplicate public_id {public_id!r}", expected="one target per protein")
        targets[public_id] = ProteinTarget(long_name, clone, codeword)
    return ProteinPanel(_panel_identity(panel, panel.get("spec_version"), path, artifact), targets)
