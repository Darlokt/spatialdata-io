"""Resolve flat CODEX regions and named Processor, Uploader, and CIM artifacts.

Flat directories may hold a cell table and an image. Native Processor
layouts group marker planes in ``stitched/regNNN`` and
regional tables in the segmentation output tree. Uploader montages can
appear in ``bestFocus`` or ``stitched``. CIM scans use the shared QPI
filenames and bit-depth variant rules. Discovery visits supported directory
names and binds each regional artifact to one normalized region.

For example, these two Processor artifacts belong to the same region.

.. code-block:: text

    stitched/reg001/reg001_cyc001_ch001_DAPI.tif
    segm/segm-1/fcs/compensated/reg001_compensated.csv

Processor/Uploader names are case-insensitive. Their numeric region index
uses at least three decimal digits in the output name. Named results follow
lexical order or the caller's explicit sequence. Compensated and uncompensated
tables are distinct signal products. The selected directory entry declares
both encoding and compensation, including when it links to a differently
named file. An equivalent CSV/FCS pair in the same folder is read from CSV.
QPI scan names follow the QPI layout rules.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import validate_element_name
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    list_directory,
    normalize_dataset_root,
    optional_unique_path,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
)
from spatialdata_io.readers.phenocycler._labels import MaskSource, bind_masks, resolve_mask_files
from spatialdata_io.readers.phenocycler._layout import ScanFiles, discover_scans, is_qptiff

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from spatialdata.transformations import Affine

    from spatialdata_io.readers._utils.paths import DatasetRoot, PathPolicy

_CELL_FORMATS: dict[str, Literal["csv", "fcs"]] = {".csv": "csv", ".fcs": "fcs"}
_TABLE_SIDECARS = frozenset({"channels.csv", "channelnames_report.csv", "tilemap.csv", "regionmap.csv"})
_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".btf"})
_REGION = re.compile(r"reg([0-9]+)", re.IGNORECASE)
_MARKER = re.compile(r"(reg[0-9]+)_cyc([0-9]+)_cha?([0-9]+)_(.+)\.tiff?", re.IGNORECASE)
_MONTAGE = re.compile(r"(reg[0-9]+)_montage\.tiff?", re.IGNORECASE)
_TABLE = re.compile(r"(?:^|_)(reg[0-9]+)_(?:un)?compensated\.(?:csv|fcs)$", re.IGNORECASE)
_MASK = re.compile(r"(?:^|[_-])(?:mask|labels)(?:\.ome|\.er)?\.(?:tiff?|btf|qptiff)$", re.IGNORECASE)
_PROCESSING_RUN = re.compile(r"processed_.+", re.IGNORECASE)
_SEGMENTATION_RUN = re.compile(r"segm-.+", re.IGNORECASE)
_LAYOUT_CONTEXT = ReaderErrorContext("codex", "dataset layout")
_CELL_CONTEXT = ReaderErrorContext("codex", "cell table")
_IMAGE_CONTEXT = ReaderErrorContext("codex", "stitched image")
_EXPERIMENT_CONTEXT = ReaderErrorContext("codex", "experiment calibration")
_CHANNEL_CONTEXT = ReaderErrorContext("codex", "montage channel names")


@dataclass(frozen=True, slots=True)
class CellTableSource:
    """A validated cell table with the encoding and signal product declared by its logical filename."""

    path: ArtifactFile
    kind: Literal["csv", "fcs"]
    compensation: Literal["compensated", "uncompensated"] | None = None


@dataclass(frozen=True, slots=True)
class MarkerImage:
    """A single marker plane identified by its cycle and channel numbers."""

    path: ArtifactFile
    cycle: int
    channel: int
    marker: str


@dataclass(frozen=True, slots=True)
class MarkerSet:
    """Validated one-plane filenames sorted by cycle and channel."""

    planes: tuple[MarkerImage, ...]


@dataclass(frozen=True, slots=True)
class Montage:
    """A Nolan Uploader hyperstack and its optional regional channel-name file."""

    path: ArtifactFile
    channel_names: ArtifactFile | None


type ImageSource = ArtifactFile | MarkerSet | Montage | ScanFiles


@dataclass(frozen=True, slots=True)
class RegionSource:
    """One region's optional cell table and raster; a flat region is unnamed."""

    name: str | None
    cell_table: CellTableSource | None
    image: ImageSource | None
    mask: MaskSource | None = None


@dataclass(frozen=True, slots=True)
class CodexLayout:
    """Validated source artifacts in output order, without loaded data."""

    root: DatasetRoot
    regions: tuple[RegionSource, ...]
    experiment: ArtifactFile | None


def _regional_name(raw: str) -> str:
    """Normalize the numeric part of a vendor regNNN name."""
    return f"reg{int(raw[3:]):03d}"


def _marker_id(path: Path, match: re.Match[str]) -> tuple[int, int, str]:
    cycle, channel = int(match[2]), int(match[3])
    marker = match[4]
    if cycle == 0 or channel == 0 or not marker.strip():
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT,
            path=path,
            found=f"cycle {cycle}, channel {channel}, marker {marker!r}",
            expected="positive cycle and channel numbers and a nonblank marker name",
        )
    return cycle, channel, marker


def _entries(root: DatasetRoot, top_level: tuple[Path, ...], *, read_tables: bool) -> tuple[Path, ...]:
    """Visit only known Processor/Uploader directories, checking containment before listing."""
    run = optional_unique_path(
        (path for path in top_level if _PROCESSING_RUN.fullmatch(path.name)),
        context=_LAYOUT_CONTEXT,
        search_path=root.path,
    )
    pending: list[Path] = [root.path]
    visited = {root.path}
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        entries = top_level if directory == root.path else list_directory(directory, context=_LAYOUT_CONTEXT)
        for path in entries:
            name = path.name.casefold()
            if (
                path == run
                or name in {"stitched", "bestfocus"}
                or (directory.name.casefold() == "stitched" and _REGION.fullmatch(name))
                or (
                    read_tables
                    and (name in {"segm", "fcs", "compensated", "uncompensated"} or _SEGMENTATION_RUN.fullmatch(name))
                )
            ):
                target = require_metadata_directory(root, path, context=_LAYOUT_CONTEXT).resolve()
                if target in visited:
                    raise ReaderFormatError(
                        context=_LAYOUT_CONTEXT,
                        path=path,
                        found="a repeated directory",
                        expected="one tree of artifacts",
                    )
                visited.add(target)
                pending.append(path)
            else:
                files.append(path)
    return tuple(files)


def _channel_file(root: DatasetRoot, image: Path, names_by_directory: dict[Path, list[Path]]) -> ArtifactFile | None:
    for directory in image.parents:
        if not directory.is_relative_to(root.path):
            break
        found = optional_unique_path(
            names_by_directory.get(directory, ()), context=_CHANNEL_CONTEXT, search_path=directory
        )
        if found is not None:
            return require_metadata_file(root, found, context=_CHANNEL_CONTEXT)
    return None


def _compensation(path: Path) -> Literal["compensated", "uncompensated"] | None:
    """Identify the Processor signal product from the declared filename suffix."""
    match path.stem.casefold().rsplit("_", maxsplit=1)[-1]:
        case "compensated" | "uncompensated" as product:
            return product
        case _:
            return None


def _table_source(path: ArtifactFile, logical_path: Path) -> CellTableSource:
    kind = _CELL_FORMATS.get(logical_path.suffix.lower())
    if kind is None:
        raise ReaderFormatError(
            context=_CELL_CONTEXT, path=path, found=logical_path.suffix, expected="a .csv or .fcs cell table"
        )
    return CellTableSource(path, kind, _compensation(logical_path))


def _is_cell_candidate(path: Path) -> bool:
    name = path.name.casefold()
    return (
        path.suffix.casefold() in _CELL_FORMATS
        and name not in _TABLE_SIDECARS
        and not name.endswith((".channels.csv", "_channels.csv"))
    )


def _group_tables(root: DatasetRoot, entries: Iterable[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    named: dict[str, list[Path]] = defaultdict(list)
    flat: list[Path] = []
    for path in entries:
        if _is_cell_candidate(path):
            if match := _TABLE.search(path.name):
                named[_regional_name(match[1])].append(path)
            elif path.parent == root.path:
                flat.append(path)
    return named, flat


def _table(
    root: DatasetRoot,
    candidates: Iterable[Path],
    override: str | Path | None,
    compensation: Literal["compensated", "uncompensated"] | None,
) -> CellTableSource | None:
    if override is not None:
        path = require_caller_file(root, override, context=_CELL_CONTEXT)
        return _table_source(path, Path(override))

    matches = tuple(candidates)
    if compensation is not None and matches:
        eligible = tuple(path for path in matches if _compensation(path) == compensation)
        if not eligible:
            raise ReaderFormatError(
                context=_CELL_CONTEXT,
                path=root.path,
                found=f"table candidates {[str(path) for path in sorted(matches)]!r}",
                expected=f"a cell table with the {compensation!r} filename suffix",
            )
        matches = eligible

    match matches:
        case (first, second) if (
            first.parent == second.parent
            and first.stem.casefold() == second.stem.casefold()
            and {first.suffix.casefold(), second.suffix.casefold()} == {".csv", ".fcs"}
        ):
            # Processor writes equivalent exports side by side. CSV can preserve precision lost in FCS floats.
            matches = (first if first.suffix.casefold() == ".csv" else second,)
    candidate = optional_unique_path(matches, context=_CELL_CONTEXT, search_path=root.path)
    if candidate is None:
        return None
    path = require_metadata_file(root, candidate, context=_CELL_CONTEXT)
    return _table_source(path, candidate)


def _image_override(
    root: DatasetRoot, override: str | Path, names_by_directory: dict[Path, list[Path]]
) -> tuple[str | None, ImageSource]:
    path = require_caller_file(root, override, context=_IMAGE_CONTEXT)
    if is_qptiff(path) and not _MASK.search(path.name):
        (scan,) = discover_scans({path: path}, context=_IMAGE_CONTEXT)
        return scan.region, scan
    if path.suffix.casefold() not in _IMAGE_SUFFIXES or _MASK.search(path.name):
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT,
            path=path,
            found=path.name,
            expected="a stitched TIFF, OME-TIFF, BigTIFF, BTF or CIM QPTIFF image",
        )
    if match := _MONTAGE.fullmatch(path.name):
        return _regional_name(match[1]), Montage(path, _channel_file(root, path, names_by_directory))
    if match := _MARKER.fullmatch(path.name):
        return _regional_name(match[1]), MarkerSet((MarkerImage(path, *_marker_id(path, match)),))
    return None, path


def _marker_set(root: DatasetRoot, candidates: list[tuple[Path, re.Match[str]]]) -> MarkerSet:
    planes: dict[tuple[int, int], MarkerImage] = {}
    directories = {path.parent for path, _ in candidates}
    if len(directories) > 1:
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT,
            path=root.path,
            found=f"marker planes in {[str(path) for path in sorted(directories)]}",
            expected="one directory of marker planes per region",
        )
    for candidate, match in candidates:
        cycle, channel, marker = _marker_id(candidate, match)
        key = (cycle, channel)
        if key in planes:
            raise ReaderFormatError(
                context=_IMAGE_CONTEXT,
                path=candidate,
                found=f"duplicate cycle {cycle}, channel {channel} also in {planes[key].path.name!r}",
                expected="one marker plane per cycle and channel",
            )
        path = require_metadata_file(root, candidate, context=_IMAGE_CONTEXT)
        planes[key] = MarkerImage(path, cycle, channel, marker)
    return MarkerSet(tuple(planes[key] for key in sorted(planes)))


def _selected_regions(available: set[str], regions: tuple[str, ...] | None, root: DatasetRoot) -> tuple[str, ...]:
    if regions is None:
        return tuple(sorted(available))
    if unknown := set(regions).difference(available):
        raise ReaderFormatError(
            context=_LAYOUT_CONTEXT,
            path=root.path,
            found=f"requested regions {sorted(unknown)!r}",
            expected=f"available regions {sorted(available)!r}",
        )
    return regions


def _flat_region(
    root: DatasetRoot,
    top_level: tuple[Path, ...],
    table: CellTableSource | None,
    image: ImageSource | None,
) -> RegionSource:
    if image is None:
        candidate = optional_unique_path(
            (path for path in top_level if path.suffix.casefold() in _IMAGE_SUFFIXES and not _MASK.search(path.name)),
            context=_IMAGE_CONTEXT,
            search_path=root.path,
        )
        if candidate is not None:
            image = require_metadata_file(root, candidate, context=_IMAGE_CONTEXT)
    if table is None and image is None:
        require_unique_path(
            (), context=ReaderErrorContext("codex", "cell table or stitched image"), search_path=root.path
        )
    return RegionSource(None, table, image)


def _names_by_directory(entries: Iterable[Path]) -> dict[Path, list[Path]]:
    files: dict[Path, list[Path]] = defaultdict(list)
    for entry in entries:
        if entry.name.casefold() == "channelnames.txt":
            files[entry.parent].append(entry)
    return files


def _group_images(
    root: DatasetRoot,
    entries: tuple[Path, ...],
) -> tuple[dict[str, list[tuple[Path, re.Match[str]]]], dict[str, list[Path]], dict[str, ScanFiles]]:
    markers: dict[str, list[tuple[Path, re.Match[str]]]] = defaultdict(list)
    montages: dict[str, list[Path]] = defaultdict(list)
    qptiffs: dict[Path, ArtifactFile] = {}
    for path in entries:
        if _MASK.search(path.name):
            continue
        if match := _MARKER.fullmatch(path.name):
            region = _regional_name(match[1])
            folder = _REGION.fullmatch(path.parent.name)
            if folder is not None and region != _regional_name(folder[0]):
                raise ReaderFormatError(
                    context=_IMAGE_CONTEXT,
                    path=path,
                    found=f"marker for {region!r}",
                    expected=f"a marker belonging to the parent region {folder[0]!r}",
                )
            markers[region].append((path, match))
        elif match := _MONTAGE.fullmatch(path.name):
            montages[_regional_name(match[1])].append(path)
        elif is_qptiff(path):
            qptiffs[path] = require_metadata_file(root, path, context=_IMAGE_CONTEXT)
    return markers, montages, {scan.region: scan for scan in discover_scans(qptiffs, context=_IMAGE_CONTEXT)}


def _native_image(  # noqa: PLR0913 - Each candidate group owns distinct file semantics.
    root: DatasetRoot,
    region: str,
    *,
    markers: list[tuple[Path, re.Match[str]]] | None,
    montages: list[Path] | None,
    scan: ScanFiles | None,
    names: dict[Path, list[Path]],
) -> ImageSource | None:
    if sum(candidate is not None for candidate in (markers, montages, scan)) > 1:
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT,
            path=root.path,
            found=f"multiple image layouts for {region!r}",
            expected="one marker set, montage or scan per region; select an image_file to override",
        )
    if markers is not None:
        return _marker_set(root, markers)
    if montages is not None:
        candidate = require_unique_path(montages, context=_IMAGE_CONTEXT, search_path=root.path)
        return Montage(
            require_metadata_file(root, candidate, context=_IMAGE_CONTEXT), _channel_file(root, candidate, names)
        )
    return scan


def _named_sources(  # noqa: PLR0913 - Each option selects a specific artifact.
    root: DatasetRoot,
    entries: tuple[Path, ...],
    *,
    regions: tuple[str, ...] | None,
    read_tables: bool,
    compensation: Literal["compensated", "uncompensated"] | None,
    cell_table_file: str | Path | None,
    image_override: tuple[str | None, ImageSource] | None,
) -> tuple[RegionSource, ...]:
    tables, flat_tables = _group_tables(root, entries if read_tables else ())
    override_name, override_image = (None, None) if image_override is None else image_override
    markers, montages, scans = _group_images(root, entries if image_override is None else ())
    image_regions = set(markers) | set(montages) | set(scans)
    if override_name is not None:
        image_regions.add(override_name)
    nested_tables = any(path.parent != root.path for candidates in tables.values() for path in candidates)
    if not image_regions and not nested_tables and len(tables) <= 1 and regions is None:
        return ()

    available = image_regions | set(tables)
    chosen = _selected_regions(available, regions, root)
    if (cell_table_file is not None or image_override is not None) and len(chosen) != 1:
        message = "cell_table_file and image_file overrides require exactly one selected CODEX region."
        raise ValueError(message)
    if flat_tables and len(chosen) != 1:
        raise ReaderFormatError(
            context=_CELL_CONTEXT,
            path=root.path,
            found=f"unnamed regional table candidates {[path.name for path in flat_tables]!r}",
            expected="one selected region for unnamed cell tables or named cell tables for multiple regions",
        )
    if override_name is not None and override_name not in chosen:
        raise ReaderFormatError(
            context=_IMAGE_CONTEXT,
            path=root.path,
            found=f"image for {override_name!r}",
            expected=f"an image for selected region {chosen[0]!r}",
        )

    names = _names_by_directory(entries) if montages else {}
    return tuple(
        RegionSource(
            region,
            _table(root, (*tables.get(region, ()), *flat_tables), cell_table_file, compensation)
            if read_tables
            else None,
            override_image
            or _native_image(
                root,
                region,
                markers=markers.get(region),
                montages=montages.get(region),
                scan=scans.get(region),
                names=names,
            ),
        )
        for region in chosen
    )


def discover_layout(  # noqa: PLR0913 - Public format choices select independent artifacts.
    path: str | Path,
    *,
    cell_table_file: str | Path | None,
    image_file: str | Path | None,
    experiment_file: str | Path | None = None,
    regions: tuple[str, ...] | None = None,
    read_tables: bool = True,
    compensation: Literal["compensated", "uncompensated"] | None = None,
    labels: str | Path | Mapping[str, str | Path] | None = None,
    labels_to_image: Affine | Mapping[str, Affine] | None = None,
    path_policy: PathPolicy = "strict",
) -> CodexLayout:
    """Resolve the selected directory or direct input file and its stable region order."""
    supplied = Path(path).expanduser()
    direct = supplied.is_file()
    root = normalize_dataset_root(supplied.parent if direct else path, context=_LAYOUT_CONTEXT, path_policy=path_policy)
    top_level = list_directory(root.path, context=_LAYOUT_CONTEXT)
    experiment = (
        None if experiment_file is None else require_caller_file(root, experiment_file, context=_EXPERIMENT_CONTEXT)
    )
    if direct:
        selected = supplied if supplied.is_absolute() else Path(supplied.name)
        if supplied.suffix.casefold() in _CELL_FORMATS:
            if cell_table_file is not None or not read_tables or compensation is not None:
                message = "A direct cell table input requires read_tables=True and no cell_table_file or compensation."
                raise ValueError(message)
            cell_table_file = selected
        else:
            if image_file is not None:
                message = "A direct image input cannot be combined with an image_file override."
                raise ValueError(message)
            image_file = selected
        top_level = tuple(
            entry for entry in top_level if entry.name == supplied.name or entry.name.casefold() == "channelnames.txt"
        )

    entries = top_level if direct else _entries(root, top_level, read_tables=read_tables)
    mask_files = resolve_mask_files(root, labels, context=ReaderErrorContext("codex", "segmentation mask"))
    if mask_files:
        excluded = {file.resolve() for file in mask_files.values()}
        if image_file is not None and (root.path / Path(image_file).expanduser()).resolve() in excluded:
            message = "The same file cannot supply the image and labels."
            raise ValueError(message)
        mask_entries = {path for path in entries if path.resolve() in excluded}
        top_level = tuple(path for path in top_level if path not in mask_entries)
        entries = tuple(path for path in entries if path not in mask_entries)
    image_override = None if image_file is None else _image_override(root, image_file, _names_by_directory(entries))
    named = _named_sources(
        root,
        entries,
        regions=regions,
        read_tables=read_tables,
        compensation=compensation,
        cell_table_file=cell_table_file,
        image_override=image_override,
    )
    if not named:
        table = (
            _table(root, (path for path in top_level if _is_cell_candidate(path)), cell_table_file, compensation)
            if read_tables
            else None
        )
        image = None if image_override is None else image_override[1]
        named = (_flat_region(root, top_level, table, image),)
    if compensation is not None and not any(source.cell_table is not None for source in named):
        message = "compensation requires a CODEX cell table in at least one selected region."
        raise ValueError(message)
    masks = bind_masks(
        mask_files,
        labels_to_image,
        image_regions=[source.name or "images" for source in named if source.image is not None],
        region_count=len(named),
    )
    return CodexLayout(
        root, tuple(replace(source, mask=masks.get(source.name or "images")) for source in named), experiment
    )


def validate_region_options(
    regions: Sequence[str] | None,
    *,
    read_tables: bool,
    table_selected: bool,
    compensation: Literal["compensated", "uncompensated"] | None = None,
) -> tuple[str, ...] | None:
    """Normalize selection once without interpreting a string as a sequence of regions."""
    if not isinstance(read_tables, bool):
        message = "read_tables must be a bool."
        raise TypeError(message)
    if not read_tables and table_selected:
        message = "cell_table_file requires read_tables=True."
        raise ValueError(message)
    if compensation is not None:
        if not isinstance(compensation, str):
            message = "compensation must be a string or None."
            raise TypeError(message)
        if compensation not in ("compensated", "uncompensated"):
            message = "compensation must be 'compensated' or 'uncompensated'."
            raise ValueError(message)
        if not read_tables or table_selected:
            message = "compensation requires read_tables=True and no cell_table_file override."
            raise ValueError(message)
    if regions is None:
        return None
    if isinstance(regions, (str, bytes)) or not isinstance(regions, Sequence):
        message = "regions must be a sequence of distinct region names or None."
        raise TypeError(message)
    selected = tuple(validate_element_name(region, parameter="regions") for region in regions)
    if not selected or len(set(selected)) != len(selected):
        message = "regions must be nonempty and contain distinct region names."
        raise ValueError(message)
    return selected
