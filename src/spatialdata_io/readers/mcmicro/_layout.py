"""Discover one deterministic MCMICRO whole-slide or tissue-microarray inventory."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    optional_directory,
    require_metadata_directory,
    require_metadata_file,
    sorted_paths,
)
from spatialdata_io.readers.mcmicro._metadata import read_workflow

if TYPE_CHECKING:
    from pathlib import Path

_TIFF_SUFFIXES = (".tif", ".tiff", ".btf")
_CORE = re.compile(r"(?P<core>[1-9][0-9]*)\.ome\.tif")
_CORE_MASK = re.compile(r"(?P<core>[1-9][0-9]*)_mask\.tif")
TABLE_KEY = "table"


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Identify one label image by its source image, method, and full mask stem."""

    image_id: str
    method: str
    compartment: str
    path: ArtifactFile

    @property
    def key(self) -> str:
        """Return the element and table-region name."""
        return f"{self.image_id}_{self.method}_{self.compartment}"


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
        """Return the element prefix and local coordinate-system name."""
        return f"core_{self.number}"


@dataclass(frozen=True, slots=True)
class WsiLayout:
    """Describe a whole-slide run with one registered image."""

    root: DatasetRoot
    markers: ArtifactFile
    image_id: str
    image: ArtifactFile
    segmentations: tuple[Segmentation, ...]
    quantifications: tuple[Quantification, ...]
    mode: Literal["wsi"] = field(default="wsi", init=False)


@dataclass(frozen=True, slots=True)
class TmaLayout:
    """Describe a tissue-microarray run with its registered map and numbered cores."""

    root: DatasetRoot
    markers: ArtifactFile
    tma_map: ArtifactFile
    cores: tuple[Core, ...]
    segmentations: tuple[Segmentation, ...]
    quantifications: tuple[Quantification, ...]
    mode: Literal["tma"] = field(default="tma", init=False)


type McmicroLayout = WsiLayout | TmaLayout


def _error(path: Path, *, artifact: str, found: str, expected: str) -> ReaderFormatError:
    """Describe an inventory conflict with its reader-owned artifact role."""
    return ReaderFormatError(context=ReaderErrorContext("mcmicro", artifact), path=path, found=found, expected=expected)


def _listing(directory: ArtifactDirectory, artifact: str) -> tuple[Path, ...]:
    """List one known directory once, in lexical order, ignoring hidden entries."""
    context = ReaderErrorContext("mcmicro", artifact)
    entries = sorted_paths(directory.iterdir(), context=context, search_path=directory)
    return tuple(entry for entry in entries if not entry.name.startswith("."))


def _file(root: DatasetRoot, relative: str, artifact: str) -> ArtifactFile:
    """Resolve a discovered artifact and keep it inside the dataset root."""
    return require_metadata_file(root, relative, context=ReaderErrorContext("mcmicro", artifact))


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
    segmentation_dir = optional_directory(root / "segmentation", context=context)
    quantification_dir = optional_directory(
        root / "quantification", context=ReaderErrorContext("mcmicro", "quantification")
    )
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
            labels.append(Segmentation(image_ids[source], resolved, compartment, _file(root, relative, "labels")))

    # Images keep discovery order; methods and masks are lexical within each image.
    order = {image_id: index for index, image_id in enumerate(image_ids.values())}
    labels.sort(key=lambda label: (order[label.image_id], label.method, label.compartment))
    by_name = {
        (source, f"{label.method}_{label.compartment}"): label
        for source, image_id in image_ids.items()
        for label in labels
        if label.image_id == image_id
    }

    quantifications: list[Quantification] = []
    for source, tables in quantified.items():
        for rest, entry in tables:
            target = by_name.get((source, rest))
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


def _require_unique_keys(layout: McmicroLayout) -> None:
    """Reject any final element-name collision before raster construction."""
    if isinstance(layout, WsiLayout):
        keys = [f"{layout.image_id}_image"]
    else:
        keys = ["tma_map", *(f"{core.image_id}_image" for core in layout.cores)]
        keys += [f"{core.image_id}_mask" for core in layout.cores]
    keys += [label.key for label in layout.segmentations]
    if layout.quantifications:
        keys.append(TABLE_KEY)
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise _error(
            layout.root, artifact="element names", found=f"colliding keys {duplicates!r}", expected="unique keys"
        )


def discover_layout(path: str | Path) -> McmicroLayout:
    """Resolve the workflow mode and every supported output artifact once."""
    root = normalize_dataset_root(path, context=ReaderErrorContext("mcmicro", "dataset root"))
    mode = read_workflow(_file(root, "qc/params.yml", "workflow parameters"))
    markers = _file(root, "markers.csv", "markers")
    registered_id, registered = _registered_image(root)

    layout: McmicroLayout
    if mode == "wsi":
        segmentations, quantifications = _segmentations(root, {registered_id: registered_id})
        layout = WsiLayout(root, markers, registered_id, registered, segmentations, quantifications)
    else:
        cores = _cores(root)
        sources = {str(core.number): core.image_id for core in cores}
        segmentations, quantifications = _segmentations(root, sources)
        layout = TmaLayout(root, markers, registered, cores, segmentations, quantifications)

    _require_unique_keys(layout)
    return layout
