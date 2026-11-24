"""Resolve processed QPI scans, bit-depth pairs and sidecars to regional paths.

A local scan may have three siblings, for example::

    Tonsil_Scan1.qptiff
    Tonsil_Scan1.er.qptiff
    Tonsil_Scan1.xpd

Directory discovery pairs ordinary and extended-range files by directory and
basename and prefers the extended-range image. An explicit file selects that
variant unless both bit depths are requested. A scan suffix is removed only
when the shortened region name is unique.

In a HuBMAP deposit, a scan under ``raw/images/`` can have channel annotations
under ``lab_processed/images/``. A scan-specific sidecar takes precedence. A
generic ``qptiff.channels.csv`` requires a single associated scan in the deposit,
even when the caller selects one region. Discovery skips the acquisition tree
``raw/images/phenocycler/`` and returns validated paths before TIFF inspection.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import colliding_element_names, sanitize_element_name
from spatialdata_io.readers._utils.paths import (
    list_directory,
    normalize_dataset_root,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
)
from spatialdata_io.readers.phenocycler._labels import MaskSource, bind_masks, resolve_mask_files

if TYPE_CHECKING:
    from spatialdata.transformations import Affine

    from spatialdata_io.readers._utils.paths import ArtifactFile, DatasetRoot, PathPolicy

_LAYOUT = ReaderErrorContext("phenocycler", "QPI scan layout")
_IMAGE = ReaderErrorContext("phenocycler", "QPI scan")
_XPD = ReaderErrorContext("phenocycler", "XPD protocol")
_CHANNELS = ReaderErrorContext("phenocycler", "channels.csv")
_EXTENSION = re.compile(r"(?:\.er\.qptiff|\.qptiff|\.ome\.tiff?|\.tiff?)$", re.IGNORECASE)
_SCAN_SUFFIX = re.compile(r"_Scan[0-9]+$", re.IGNORECASE)
_REGION = re.compile(r"(?<![a-z])reg([0-9]+)", re.IGNORECASE)
_ACQUISITION_DIR = ("raw", "images", "phenocycler")
_GENERIC_CHANNELS = "qptiff.channels.csv"


@dataclass(frozen=True, slots=True)
class ScanFiles:
    """One scan with the selected bit-depth variant and an optional ordinary counterpart."""

    region: str
    image: ArtifactFile
    lower_bit_depth: ArtifactFile | None
    name: str


@dataclass(frozen=True, slots=True)
class RegionFiles:
    """One scan and its optional metadata artifacts."""

    scan: ScanFiles
    xpd: ArtifactFile | None
    channels_csv: ArtifactFile | None
    mask: MaskSource | None = None


@dataclass(frozen=True, slots=True)
class QpiLayout:
    """Validated input paths in the requested region order."""

    root: DatasetRoot
    regions: tuple[RegionFiles, ...]


def _base(path: Path) -> str:
    return _EXTENSION.sub("", path.name)


def _extended(path: Path) -> bool:
    return path.name.lower().endswith(".er.qptiff")


def is_qptiff(path: Path) -> bool:
    """Check the filename of a processed QPTIFF, without accessing it."""
    return path.name.lower().endswith(".qptiff") and not path.name.lower().endswith(".raw.qptiff")


def _region_name(base: str) -> str | None:
    match = _REGION.search(base)
    if match is None:
        return None
    return "reg" + (match.group(1).lstrip("0") or "0").zfill(3)


def discover_scans(files: Mapping[Path, ArtifactFile], *, context: ReaderErrorContext) -> tuple[ScanFiles, ...]:
    """Pair QPI variants and name scans from their validated, lexically ordered candidates."""
    groups: dict[tuple[Path, str], dict[bool, Path]] = defaultdict(dict)
    for path in sorted(files):
        variants = groups[path.parent, _base(path).lower()]
        extended = _extended(path)
        if extended in variants:
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"two files at the same bit-depth rank ({variants[extended].name!r} and {path.name!r})",
                expected="one ordinary QPI and at most one extended-range companion per scan",
            )
        variants[extended] = path

    primary = tuple(
        (variants[True], variants.get(False)) if True in variants else (variants[False], None)
        for variants in groups.values()
    )
    raw_bases = (_base(path) for path, _ in primary)
    bases = tuple(
        _region_name(base) or sanitize_element_name(base, source="QPI filename", parameter=None) for base in raw_bases
    )
    shortened = tuple(_SCAN_SUFFIX.sub("", base) or base for base in bases)
    counts = Counter(name.lower() for name in shortened)
    regions = tuple(base if counts[short.lower()] > 1 else short for base, short in zip(bases, shortened, strict=True))
    if clashes := colliding_element_names(regions):
        paths = [path for (path, _), region in zip(primary, regions, strict=True) if region in clashes]
        raise ReaderFormatError(
            context=context,
            path=paths[0],
            found=f"multiple scans mapping to region names {clashes}",
            expected="distinct region names after removing the QPI extension and scan suffix",
        )
    return tuple(
        ScanFiles(region, files[path], files[other] if other is not None else None, path.name)
        for region, (path, other) in zip(regions, primary, strict=True)
    )


def _visible_entries(root: DatasetRoot) -> tuple[Path, ...]:
    entries: list[Path] = []
    directories: list[tuple[Path, frozenset[Path]]] = [(root.path, frozenset((root.path,)))]
    while directories:
        directory, ancestry = directories.pop()
        for path in list_directory(directory, context=_LAYOUT):
            name = path.name.lower()
            if name.startswith(".") or name == "__macosx" or name.endswith(".zarr"):
                continue
            parts = tuple(part.lower() for part in path.relative_to(root.path).parts)
            if parts[-len(_ACQUISITION_DIR) :] == _ACQUISITION_DIR:
                continue
            if is_qptiff(path) or name.endswith((".xpd", "channels.csv")):
                entries.append(path)
            elif path.is_dir():
                artifact = require_metadata_directory(root, path, context=_LAYOUT)
                target = artifact.resolve(strict=True)
                if target in ancestry:
                    raise ReaderFormatError(
                        context=_LAYOUT,
                        path=path,
                        found="a recursive directory symlink",
                        expected="a directory tree without ancestry cycles",
                    )
                directories.append((path, ancestry | {target}))
    return tuple(sorted(entries))


def _processed_dir(scan: Path, root: DatasetRoot) -> Path | None:
    for ancestor in scan.parents:
        if ancestor.name == "raw":
            candidate = ancestor.parent / "lab_processed" / "images"
            return candidate if candidate.is_relative_to(root.path) else None
        if ancestor == root.path:
            break
    return None


def _match_sidecar(
    root: DatasetRoot, image: Path, candidates: Mapping[Path, Sequence[Path]], context: ReaderErrorContext, suffix: str
) -> ArtifactFile | None:
    pattern = re.compile(rf"^{re.escape(_base(image))}(?![a-z0-9]).*{re.escape(suffix)}$", re.IGNORECASE)
    processed = _processed_dir(image, root)
    for directory in (image.parent, processed):
        if directory is None:
            continue
        matches = [
            path
            for path in candidates.get(directory, ())
            if pattern.fullmatch(path.name) and (directory != processed or path.name.lower() != _GENERIC_CHANNELS)
        ]
        if len(matches) > 1:
            raise ReaderFormatError(
                context=context,
                path=directory,
                found=f"multiple sidecars matching {image.name!r}: {[path.name for path in matches]}",
                expected="at most one matching sidecar in the nearest directory",
            )
        if matches:
            return require_metadata_file(root, matches[0], context=context)
    return None


def _generic_channels(
    root: DatasetRoot, image: Path, candidates: Mapping[Path, Sequence[Path]], scan_counts: Counter[Path]
) -> ArtifactFile | None:
    processed = _processed_dir(image, root)
    if processed is None:
        return None
    matches = [path for path in candidates.get(processed, ()) if path.name.lower() == _GENERIC_CHANNELS]
    if not matches:
        return None
    if len(matches) != 1 or scan_counts[processed] != 1:
        raise ReaderFormatError(
            context=_CHANNELS,
            path=processed,
            found=f"{len(matches)} generic channels CSV files for {scan_counts[processed]} scans",
            expected="one qptiff.channels.csv corresponding to one scan, or a scan-specific channels CSV",
        )
    return require_metadata_file(root, matches[0], context=_CHANNELS)


def _select_regions(scans: tuple[ScanFiles, ...], regions: tuple[str, ...] | None) -> tuple[ScanFiles, ...]:
    if regions is None:
        return scans
    sources = {scan.region.lower(): scan for scan in scans}
    if unknown := [region for region in regions if region.lower() not in sources]:
        message = f"Unknown regions {unknown}. Available regions: {[scan.region for scan in scans]}."
        raise ValueError(message)
    return tuple(sources[region.lower()] for region in regions)


def normalize_regions(regions: Sequence[str] | None) -> tuple[str, ...] | None:
    """Validate requested coordinate regions before inspecting any artifacts."""
    if regions is None:
        return None
    if isinstance(regions, str | bytes) or not isinstance(regions, Sequence):
        message = "regions must be a sequence of region names."
        raise TypeError(message)
    if not regions:
        message = "regions must contain at least one name."
        raise ValueError(message)
    if any(not isinstance(name, str) for name in regions):
        message = "regions must contain strings."
        raise TypeError(message)
    if any(not name.strip() for name in regions):
        message = "regions must contain nonempty names."
        raise ValueError(message)
    names = tuple(
        _region_name(name) or sanitize_element_name(name, source="QPI region", parameter=None) for name in regions
    )
    if clashes := colliding_element_names(names):
        message = f"Repeated region selections: {clashes}."
        raise ValueError(message)
    return names


def _single_source(
    source: Path, *, read_all_bit_depths: bool, path_policy: PathPolicy
) -> tuple[DatasetRoot, tuple[Path, ...], dict[Path, ArtifactFile]]:
    root = normalize_dataset_root(source.parent, context=_LAYOUT, path_policy=path_policy)
    lexical: Path = root.path / source.name
    selected = require_caller_file(root, source if source.is_absolute() else source.name, context=_IMAGE)
    if _EXTENSION.search(source.name) is None or source.name.lower().endswith(".raw.qptiff"):
        raise ReaderFormatError(
            context=_IMAGE, path=source, found=f"filename {source.name!r}", expected="a .qptiff or QPI .tif/.tiff"
        )
    neighbors = list_directory(lexical.parent, context=_LAYOUT)
    artifacts = {lexical: selected}
    if read_all_bit_depths and is_qptiff(lexical):
        for entry in neighbors:
            if entry != lexical and is_qptiff(entry) and _base(entry).lower() == _base(lexical).lower():
                artifacts[entry] = require_metadata_file(root, entry, context=_IMAGE)
    return root, neighbors, artifacts


def discover_layout(  # noqa: PLR0913 - Scan selection and explicit label registrations.
    path: str | Path,
    *,
    regions: tuple[str, ...] | None,
    read_all_bit_depths: bool,
    path_policy: PathPolicy,
    labels: str | Path | Mapping[str, str | Path] | None = None,
    labels_to_image: Affine | Mapping[str, Affine] | None = None,
) -> QpiLayout:
    """Find the selected QPI scans and their bounded sidecar inputs."""
    source = Path(path).expanduser()
    directory = source.is_dir()
    if directory:
        root = normalize_dataset_root(source, context=_LAYOUT, path_policy=path_policy)
        entries = _visible_entries(root)
        artifacts = {entry: require_metadata_file(root, entry, context=_IMAGE) for entry in entries if is_qptiff(entry)}
    else:
        root, entries, artifacts = _single_source(
            source, read_all_bit_depths=read_all_bit_depths, path_policy=path_policy
        )

    mask_files = resolve_mask_files(root, labels, context=ReaderErrorContext("phenocycler", "segmentation mask"))
    if mask_files:
        excluded = {file.resolve() for file in mask_files.values()}
        if not directory and source.resolve() in excluded:
            message = "The same file cannot supply the image and labels."
            raise ValueError(message)
        artifacts = {path: file for path, file in artifacts.items() if file.resolve() not in excluded}
    if not artifacts:
        raise ReaderFormatError(
            context=_LAYOUT, path=root.path, found="no processed QPTIFF scans", expected="at least one .qptiff scan"
        )
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for entry in entries:
        by_parent[entry.parent].append(entry)
    by_name = {(artifact, path.name): path for path, artifact in artifacts.items()}
    scans = discover_scans(artifacts, context=_LAYOUT)
    processed_counts = Counter(
        directory for scan in scans if (directory := _processed_dir(by_name[scan.image, scan.name], root)) is not None
    )
    selected_scans = _select_regions(scans, regions)
    if read_all_bit_depths and (
        clashes := colliding_element_names(
            [scan.region for scan in selected_scans]
            + [f"{scan.region}_8bit" for scan in selected_scans if scan.lower_bit_depth is not None]
        )
    ):
        raise ReaderFormatError(
            context=_LAYOUT,
            path=root.path,
            found=f"derived bit-depth names colliding with region names {clashes}",
            expected="distinct region names including the _8bit companion name",
        )
    masks = bind_masks(
        mask_files,
        labels_to_image,
        image_regions=[scan.region for scan in selected_scans],
        region_count=len(selected_scans),
    )
    result = []
    for scan in selected_scans:
        entry = by_name[scan.image, scan.name]
        xpd = _match_sidecar(root, entry, by_parent, _XPD, ".xpd")
        channels = _match_sidecar(root, entry, by_parent, _CHANNELS, "channels.csv") or _generic_channels(
            root, entry, by_parent, processed_counts
        )
        result.append(RegionFiles(scan, xpd, channels, masks.get(scan.region)))
    return QpiLayout(root, tuple(result))
