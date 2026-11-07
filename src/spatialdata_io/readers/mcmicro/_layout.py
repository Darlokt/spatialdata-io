"""Discover one MCMICRO whole-slide or tissue-microarray run and pair its outputs.

The YAML boolean ``workflow.tma`` in ``qc/params.yml`` selects the layout
(https://mcmicro.org/). Only pipeline results are read; ``raw/``,
``illumination/``, Coreograph QC files and segmentation puncta tables are
ignored. A whole-slide run::

    run/
    ├── markers.csv
    ├── qc/params.yml
    ├── registration/<image>.ome.tif                 # exactly one TIFF-family image
    ├── segmentation/<method>-<image>/<mask>.ome.tif # optional
    └── quantification/<image>--<method>_<mask>.csv  # optional, one per label

A tissue-microarray run adds numbered Coreograph cores; segmentation and
quantification then refer to the core numbers::

    run/
    ├── markers.csv
    ├── qc/params.yml
    ├── registration/<tma>.ome.tif
    ├── dearray/<n>.ome.tif                       # n: positive integer, no leading zero
    ├── dearray/masks/<n>_mask.tif                # exactly one per core
    ├── segmentation/<method>-<n>/<mask>.ome.tif
    └── quantification/<n>--<method>_<mask>.csv

Names follow MCMICRO's own conventions:

- the image ID drops the extension and an ``.ome`` suffix (``getImageID``);
- the mask name is the label file name up to its first dot, as MCQuant names
  it (``cell_mask_ring.ome.tif`` is ``cell_mask_ring``);
- ``<image>--<method>_<mask>.csv`` is matched against whole image, method and
  mask names, so underscores in them are never split;
- a historical ``segmentation/<image>/`` directory without the method prefix
  takes the method of each mask from the one quantification table naming it.

Element names are ``<image>_image`` and ``<image>_<method>_<mask>`` for a
whole slide, and ``tma_map``, ``core_<n>_image``, ``core_<n>_mask`` and
``core_<n>_<method>_<mask>`` for a TMA. Image IDs, methods and masks come from
file names; one that is not a valid SpatialData element name is sanitized with
a warning. Names that collide, ignoring letter case as SpatialData does, fail
before any raster is opened. Images keep discovery order (cores numerically);
labels are ordered by image, then method and mask lexically, and every
quantification table follows its label.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import sanitize_element_name
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    optional_metadata_directory,
    require_metadata_directory,
    require_metadata_file,
)
from spatialdata_io.readers.mcmicro._metadata import read_workflow

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy

_TIFF_SUFFIXES = (".tif", ".tiff", ".btf")
_CORE = re.compile(r"(?P<core>[1-9][0-9]*)\.ome\.tif")
_CORE_MASK = re.compile(r"(?P<core>[1-9][0-9]*)_mask\.tif")
TABLE_KEY = "table"
TMA_MAP_KEY = "tma_map"


@dataclass(frozen=True, slots=True)
class Segmentation:
    """One label image: its image's element prefix, method, full mask stem and element name.

    ``method`` and ``compartment`` are the source names; ``key`` is the
    (sanitized) element and table-region name.
    """

    image_id: str
    method: str
    compartment: str
    path: ArtifactFile
    key: str


@dataclass(frozen=True, slots=True)
class Quantification:
    """Pair one MCQuant table with the exact label image it measured."""

    path: ArtifactFile
    target: Segmentation


@dataclass(frozen=True, slots=True)
class Core:
    """Keep a Coreograph core and its independently gridded tissue mask together."""

    number: int
    image: ArtifactFile
    mask: ArtifactFile

    @property
    def image_id(self) -> str:
        """Return the element prefix and grid name."""
        return f"core_{self.number}"

    @property
    def image_key(self) -> str:
        """Return the core image's element name."""
        return f"{self.image_id}_image"

    @property
    def mask_key(self) -> str:
        """Return the tissue mask's element name and grid name."""
        return f"{self.image_id}_mask"


@dataclass(frozen=True, slots=True)
class WsiLayout:
    """Describe a whole-slide run with one registered image; ``image_id`` is its element prefix."""

    markers: ArtifactFile
    image_id: str
    image: ArtifactFile
    segmentations: tuple[Segmentation, ...]
    quantifications: tuple[Quantification, ...]

    @property
    def image_key(self) -> str:
        """Return the registered image's element name."""
        return f"{self.image_id}_image"


@dataclass(frozen=True, slots=True)
class TmaLayout:
    """Describe a tissue-microarray run with its registered map and numbered cores."""

    markers: ArtifactFile
    tma_map: ArtifactFile
    cores: tuple[Core, ...]
    segmentations: tuple[Segmentation, ...]
    quantifications: tuple[Quantification, ...]


type McmicroLayout = WsiLayout | TmaLayout


def _error(path: Path, *, artifact: str, found: str, expected: str) -> ReaderFormatError:
    """Describe an inventory conflict with its reader-owned artifact role."""
    return ReaderFormatError(context=ReaderErrorContext("mcmicro", artifact), path=path, found=found, expected=expected)


def _listing(directory: ArtifactDirectory, artifact: str) -> tuple[Path, ...]:
    """List one known directory once, in lexical order, ignoring hidden entries."""
    context = ReaderErrorContext("mcmicro", artifact)
    entries = list_directory(directory, context=context)
    return tuple(entry for entry in entries if not entry.name.startswith("."))


def _file(root: DatasetRoot, relative: str, artifact: str) -> ArtifactFile:
    """Resolve a discovered artifact under the root's path policy."""
    return require_metadata_file(root, relative, context=ReaderErrorContext("mcmicro", artifact))


def _optional_output_directory(root: DatasetRoot, name: str) -> ArtifactDirectory | None:
    """Distinguish absent output from malformed or out-of-root output directories."""
    return optional_metadata_directory(root, name, context=ReaderErrorContext("mcmicro", name))


def _image_id(name: str) -> str:
    """Strip the extension and an ``.ome`` suffix, as MCMICRO's ``getImageID`` does."""
    stem = name.rsplit(".", 1)[0]
    return stem.removesuffix(".ome")


def _registered_image(root: DatasetRoot) -> tuple[str, ArtifactFile]:
    """Require exactly one registered TIFF-family image."""
    directory = require_metadata_directory(root, "registration", context=ReaderErrorContext("mcmicro", "registration"))
    images = [entry for entry in _listing(directory, "registration") if entry.name.lower().endswith(_TIFF_SUFFIXES)]
    if len(images) != 1:
        raise _error(
            directory,
            artifact="registration",
            found=f"{len(images)} TIFF images {[image.name for image in images]!r}",
            expected="exactly one registered image",
        )
    return _image_id(images[0].name), _file(root, f"registration/{images[0].name}", "registered image")


def _cores(root: DatasetRoot) -> tuple[Core, ...]:
    """Pair numbered Coreograph cores with their tissue masks in numeric order."""
    context = ReaderErrorContext("mcmicro", "dearray")
    directory = require_metadata_directory(root, "dearray", context=context)
    masks = require_metadata_directory(root, "dearray/masks", context=ReaderErrorContext("mcmicro", "core masks"))

    images: dict[int, str] = {}
    for entry in _listing(directory, "dearray"):
        if entry.name == "masks":
            continue
        match = _CORE.fullmatch(entry.name)
        if match is None:
            raise _error(
                entry, artifact="dearray", found=f"artifact {entry.name!r}", expected="<positive integer>.ome.tif cores"
            )
        images[int(match["core"])] = entry.name

    mask_names: dict[int, str] = {}
    for entry in _listing(masks, "core masks"):
        match = _CORE_MASK.fullmatch(entry.name)
        if match is None:
            raise _error(
                entry, artifact="core masks", found=f"artifact {entry.name!r}", expected="<core>_mask.tif masks"
            )
        mask_names[int(match["core"])] = entry.name

    if not images or set(images) != set(mask_names):
        raise _error(
            directory,
            artifact="dearray",
            found=f"cores {sorted(images)} and masks {sorted(mask_names)}",
            expected="a nonempty set of cores, each with exactly one mask",
        )
    return tuple(
        Core(
            number,
            _file(root, f"dearray/{images[number]}", "core image"),
            _file(root, f"dearray/masks/{mask_names[number]}", "core mask"),
        )
        for number in sorted(images)
    )


def _source_id(name: str, source_ids: tuple[str, ...], directory: Path) -> tuple[str, str | None]:
    """Match an official ``<method>-<image>`` or historical ``<image>`` directory."""
    candidates = [(source, name[: -len(source) - 1]) for source in source_ids if name.endswith(f"-{source}")]
    candidates = [(source, method) for source, method in candidates if method]
    historical = name in source_ids
    if len(candidates) + historical != 1:
        raise _error(
            directory,
            artifact="segmentation",
            found=f"directory {name!r} matching {len(candidates) + historical} images",
            expected=f"<method>-<image> or <image> for exactly one of {list(source_ids)!r}",
        )
    return (name, None) if historical else candidates[0]


def _quantification_sources(
    directory: ArtifactDirectory | None, source_ids: tuple[str, ...]
) -> dict[str, list[tuple[str, Path]]]:
    """Group ``<image>--<method>_<mask>.csv`` stems by their unique source image."""
    grouped: dict[str, list[tuple[str, Path]]] = {source: [] for source in source_ids}
    if directory is None:
        return grouped
    for entry in _listing(directory, "quantification"):
        if entry.suffix != ".csv":
            continue
        stem = entry.name.removesuffix(".csv")
        sources = [source for source in source_ids if stem.startswith(f"{source}--")]
        if len(sources) != 1:
            raise _error(
                entry,
                artifact="quantification",
                found=f"{len(sources)} matching images",
                expected=f"<image>--<method>_<mask>.csv for exactly one of {list(source_ids)!r}",
            )
        grouped[sources[0]].append((stem.removeprefix(f"{sources[0]}--"), entry))
    return grouped


def _segmentations(  # noqa: C901, RUF100 - one ordered pass resolves both segmentation directory grammars.
    root: DatasetRoot, image_ids: dict[str, str]
) -> tuple[tuple[Segmentation, ...], tuple[Quantification, ...]]:
    """Resolve labels and the exact label measured by every quantification table.

    ``image_ids`` maps MCMICRO source image IDs, in discovery order, to element prefixes.
    """
    source_ids = tuple(image_ids)
    context = ReaderErrorContext("mcmicro", "segmentation")
    segmentation_dir = _optional_output_directory(root, "segmentation")
    quantification_dir = _optional_output_directory(root, "quantification")
    quantified = _quantification_sources(quantification_dir, source_ids)

    labels: list[Segmentation] = []
    for directory in () if segmentation_dir is None else _listing(segmentation_dir, "segmentation"):
        source, method = _source_id(directory.name, source_ids, directory)
        label_dir = require_metadata_directory(root, f"segmentation/{directory.name}", context=context)
        masks: dict[str, Path] = {}
        for entry in _listing(label_dir, "labels"):
            # Segmentation directories may also hold puncta tables, which are not labels.
            if not entry.name.lower().endswith(_TIFF_SUFFIXES):
                continue
            compartment = entry.name.split(".", 1)[0]
            if compartment in masks:
                raise _error(
                    entry, artifact="labels", found=f"duplicate mask stem {compartment!r}", expected="one file per mask"
                )
            masks[compartment] = entry

        for compartment, entry in masks.items():
            if method is None:
                # Historical sample-only directories record the method only in MCQuant names.
                methods = {
                    rest.removesuffix(f"_{compartment}")
                    for rest, _ in quantified[source]
                    if rest.endswith(f"_{compartment}") and rest != f"_{compartment}"
                }
                if len(methods) != 1:
                    raise _error(
                        entry,
                        artifact="labels",
                        found=f"{len(methods)} quantified methods",
                        expected="exactly one MCQuant table identifying this historical mask's method",
                    )
                resolved = methods.pop()
            else:
                resolved = method
            relative = f"segmentation/{directory.name}/{entry.name}"
            key = sanitize_element_name(
                f"{image_ids[source]}_{resolved}_{compartment}",
                source="mcmicro: the label name",
                parameter=None,
            )
            labels.append(Segmentation(image_ids[source], resolved, compartment, _file(root, relative, "labels"), key))

    # Images keep discovery order; methods and masks are lexical within each image.
    order = {image_id: index for index, image_id in enumerate(image_ids.values())}
    labels.sort(key=lambda label: (order[label.image_id], label.method, label.compartment))
    by_name = {(label.image_id, f"{label.method}_{label.compartment}"): label for label in labels}

    quantifications: list[Quantification] = []
    for source, tables in quantified.items():
        for rest, entry in tables:
            target = by_name.get((image_ids[source], rest))
            if target is None:
                raise _error(
                    entry,
                    artifact="quantification",
                    found=f"no label for {rest!r}",
                    expected="<image>--<method>_<mask>.csv naming one discovered label",
                )
            quantifications.append(
                Quantification(_file(root, f"quantification/{entry.name}", "quantification"), target)
            )
    position = {label: index for index, label in enumerate(labels)}
    quantifications.sort(key=lambda table: position[table.target])
    return tuple(labels), tuple(quantifications)


def _require_unique_keys(layout: McmicroLayout, root: DatasetRoot) -> None:
    """Reject element names that collide ignoring letter case, as SpatialData compares them."""
    if isinstance(layout, WsiLayout):
        keys = [layout.image_key]
    else:
        keys = [TMA_MAP_KEY, *(core.image_key for core in layout.cores), *(core.mask_key for core in layout.cores)]
    keys += [label.key for label in layout.segmentations]
    if layout.quantifications:
        keys.append(TABLE_KEY)
    counts = Counter(key.lower() for key in keys)
    duplicates = sorted(key for key in keys if counts[key.lower()] > 1)
    if duplicates:
        raise _error(
            root.path,
            artifact="element names",
            found=f"colliding keys {duplicates!r}",
            expected="element names that are unique ignoring letter case",
        )


def discover_layout(path: str | Path, *, path_policy: PathPolicy = "strict") -> McmicroLayout:
    """Resolve the workflow mode and every supported output artifact once.

    Raises ``ReaderFormatError`` for an ambiguous, unpaired or colliding
    inventory and the shared path errors for missing or wrong-kind artifacts.
    """
    root = normalize_dataset_root(path, context=ReaderErrorContext("mcmicro", "dataset root"), path_policy=path_policy)
    mode = read_workflow(_file(root, "qc/params.yml", "workflow parameters"))
    markers = _file(root, "markers.csv", "markers")
    registered_id, registered = _registered_image(root)

    layout: McmicroLayout
    if mode == "wsi":
        image_id = sanitize_element_name(registered_id, source="mcmicro: the registered image ID", parameter=None)
        segmentations, quantifications = _segmentations(root, {registered_id: image_id})
        layout = WsiLayout(markers, image_id, registered, segmentations, quantifications)
    else:
        cores = _cores(root)
        sources = {str(core.number): core.image_id for core in cores}
        segmentations, quantifications = _segmentations(root, sources)
        layout = TmaLayout(markers, registered, cores, segmentations, quantifications)

    _require_unique_keys(layout, root)
    return layout
