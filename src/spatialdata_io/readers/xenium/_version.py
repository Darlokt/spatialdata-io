"""Map the ``experiment.xenium`` format version to one of three bundle families.

The ``major_version``/``minor_version``/``patch_version`` fields describe the
format of the analysis outputs, not the software release. Xenium Ranger rewrites
them to its own output format, except ``rename``, so they are the only reliable
layout discriminator. Software strings are provenance only.

XOA 1.0 and 1.1 predate the version fields. The documented release table maps
XOA 1.2 to format 1.1, 1.3-1.9 to formats 2.x-3.x, 2.0 to 4.0, 3.x to 5.x and
4.0 to 6.x; only a major version change breaks compatibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

type Family = Literal["integer_ids", "hex_ids", "polygon_sets"]

_FAMILIES: dict[int, Family] = {
    1: "integer_ids",
    2: "hex_ids",
    3: "hex_ids",
    4: "polygon_sets",
    5: "polygon_sets",
    6: "polygon_sets",
}
_FIELDS = ("major_version", "minor_version", "patch_version")
# XOA 1.0/1.1 software strings, e.g. "Xenium-1.0.1" or "xenium-1.0.2".
_UNVERSIONED_SOFTWARE = re.compile(r"[Xx]enium-1\.[01](?:\.\d+)*(?:-.*)?")


@dataclass(frozen=True, slots=True)
class FormatVersion:
    """A validated ``experiment.xenium`` format version; the patch can be absent."""

    major: int
    minor: int
    patch: int | None

    def __str__(self) -> str:
        """Return the dotted version, omitting an absent patch."""
        version = f"{self.major}.{self.minor}"
        return version if self.patch is None else f"{version}.{self.patch}"


def _field(experiment: Mapping[str, object], name: str, path: Path) -> int | None:
    value = experiment.get(name)
    if value is None:
        return None
    # bool is an int subclass but never a version number.
    if type(value) is not int or value < 0:
        raise ReaderFormatError(
            context=ReaderErrorContext("xenium", "experiment.xenium"),
            path=path,
            found=f"{name} = {value!r}",
            expected="a non-negative integer format version field",
        )
    return value


def detect_version(
    experiment: Mapping[str, object], software: str, *, path: Path
) -> tuple[FormatVersion | None, Family]:
    """Return the format version and bundle family of one parsed ``experiment.xenium``.

    Parameters
    ----------
    experiment
        Parsed ``experiment.xenium`` object.
    software
        Its validated ``analysis_sw_version``, used only to accept bundles that
        predate the version fields.
    path
        Source file for error context.

    Returns
    -------
    tuple[FormatVersion | None, Family]
        ``None`` for XOA 1.0/1.1 bundles without version fields, and the family:
        ``integer_ids`` (absent or major 1), ``hex_ids`` (2-3) or
        ``polygon_sets`` (4-6).

    Raises
    ------
    ReaderFormatError
        If the fields are malformed, partially present, absent for software other
        than XOA 1.0/1.1, or name an unsupported major version (0 or 7+).
    """
    major, minor, patch = (_field(experiment, name, path) for name in _FIELDS)
    context = ReaderErrorContext("xenium", "experiment.xenium")

    if major is None and minor is None and patch is None:
        if _UNVERSIONED_SOFTWARE.fullmatch(software) is None:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"no format version fields with analysis_sw_version {software!r}",
                expected="major_version and minor_version, which only XOA 1.0/1.1 bundles omit",
            )
        return None, "integer_ids"

    if major is None or minor is None:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"partial format version {dict(zip(_FIELDS, (major, minor, patch), strict=True))}",
            expected="both major_version and minor_version",
        )

    version = FormatVersion(major, minor, patch)
    family = _FAMILIES.get(major)
    if family is None:
        raise ReaderFormatError(
            context=ReaderErrorContext("xenium", "experiment.xenium", str(version)),
            path=path,
            found=f"unsupported format major version {major}",
            expected="a documented format major version 1-6 (XOA 1.2-4.x and Xenium Ranger outputs)",
        )
    return version, family
