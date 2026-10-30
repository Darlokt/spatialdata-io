"""Discover documented SAW product paths without inferring a producer release."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
)

if TYPE_CHECKING:
    from pathlib import Path

type Family = Literal["compact_numbered", "numbered", "outs", "flat"]
type ImageRole = Literal["registered", "mask", "tissue_cut", "distance_mask", "expression"]

_DIRECTORIES: dict[Family, tuple[str, ...]] = {
    "compact_numbered": (
        "02.count",
        "03.register",
        "04.tissuecut",
        "041.cellcut",
        "05.spatialcluster",
        "051.cellcluster",
    ),
    "numbered": ("03.count", "04.register", "05.tissuecut", "051.cellcut", "06.spatialcluster", "061.cellcluster"),
    "outs": ("feature_expression", "analysis", "image"),
    "flat": (),
}
_SQUARE_DIRECTORIES: dict[Family, tuple[str, ...]] = {
    "compact_numbered": ("02.count", "04.tissuecut"),
    "numbered": ("03.count", "05.tissuecut"),
    "outs": ("feature_expression",),
    "flat": (),
}
_CELL_DIRECTORIES: dict[Family, tuple[str, ...]] = {
    "compact_numbered": ("041.cellcut",),
    "numbered": ("051.cellcut",),
    "outs": ("feature_expression",),
    "flat": (),
}
_DECIMAL = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_BIN_ANALYSIS = re.compile(
    rf"(?P<sample>.+)\.bin(?P<bin>[1-9][0-9]*)_(?P<resolution>{_DECIMAL})(?:\.spatial\.cluster)?\.h5ad"
)
_CELL_ANALYSIS = re.compile(rf"(?P<sample>.+)\.cellbin_(?P<resolution>{_DECIMAL})(?P<adjusted>\.adjusted)?\.h5ad")
_IMAGE_SUFFIXES: tuple[tuple[str, ImageRole], ...] = (
    ("_regist.tif", "registered"),
    ("_tissue_cut.tif", "tissue_cut"),
    ("_mask.tif", "mask"),
)
_DISTANCE_MASK = re.compile(r"(?P<stem>.+)_mask_edm_dis_(?P<distance>[1-9][0-9]*)\.tif")


@dataclass(frozen=True, slots=True)
class SquareSource:
    """Select stored bins from one validated square GEF."""

    file: ArtifactFile
    bins: tuple[int, ...] | Literal["all"]


@dataclass(frozen=True, slots=True)
class CellSource:
    """Select one independently owned native cell population."""

    file: ArtifactFile
    kind: Literal["unadjusted", "adjusted"]


@dataclass(frozen=True, slots=True)
class AnalysisSource:
    """Hold filename evidence; historical names may omit bin or clustering resolution."""

    file: ArtifactFile
    population: Literal["cells", "adjusted_cells"] | int | None
    resolution: Decimal | None
    sample_id: str


@dataclass(frozen=True, slots=True)
class StainedImage:
    """An already registered stain and its derived image products."""

    file: ArtifactFile
    role: Literal["registered", "mask", "tissue_cut"]
    stain: str

    @property
    def name(self) -> str:
        """Return the output name, independent of the sample selector."""
        return f"{self.stain}_{self.role}"

    @property
    def is_mask(self) -> bool:
        """Return whether the product is a mask, read as labels."""
        return self.role != "registered"


@dataclass(frozen=True, slots=True)
class GridImage:
    """An expression image, tissue mask, or historical raster without a stain name."""

    file: ArtifactFile
    role: Literal["registered", "mask", "tissue_cut", "expression"]

    @property
    def name(self) -> str:
        """Return the distinct unstained product name."""
        return "expression_tissue_cut" if self.file.name == "bin1_img_tissue_cut.tif" else self.role

    @property
    def is_mask(self) -> bool:
        """Return whether the product is a mask, read as labels."""
        return self.role in ("mask", "tissue_cut")


@dataclass(frozen=True, slots=True)
class DistanceMask:
    """An expanded binary mask with its documented distance in pixels."""

    file: ArtifactFile
    stain: str
    distance: int

    @property
    def name(self) -> str:
        """Return a name retaining the expansion distance."""
        return f"{self.stain}_distance_mask_{self.distance}"

    @property
    def is_mask(self) -> bool:
        """Return ``True``: expanded masks are always read as labels."""
        return True


type ImageSource = StainedImage | GridImage | DistanceMask


@dataclass(frozen=True, slots=True)
class StereoseqLayout:
    """Carry selected paths only, with structural family separate from release."""

    root: DatasetRoot
    family: Family
    sample_id: str | None
    squares: tuple[SquareSource, ...]
    cells: tuple[CellSource, ...]
    analyses: tuple[AnalysisSource, ...]
    images: tuple[ImageSource, ...]
    calibration: tuple[SquareSource | CellSource, ...]


def _error(path: Path, found: str, expected: str) -> ReaderFormatError:
    return ReaderFormatError(
        context=ReaderErrorContext("stereoseq", "dataset layout"), path=path, found=found, expected=expected
    )


def _inventory(root: DatasetRoot) -> tuple[Family, tuple[Path, ...]]:
    """Scan each relevant immediate directory once and reject mixed structural families."""
    entries: dict[str, Path] = {entry.name: entry for entry in root.iterdir()}
    families = [family for family, names in _DIRECTORIES.items() if entries.keys() & set(names)]

    if "outs" in entries:
        families.append("outs")
    if len(families) > 1:
        raise _error(root, f"conflicting families {families}", "one SAW layout family")

    family: Family = families[0] if families else "flat"
    base: ArtifactDirectory = root
    if "outs" in entries:
        base = require_metadata_directory(root, "outs", context=ReaderErrorContext("stereoseq", "outs"))
        entries = {entry.name: entry for entry in base.iterdir()}

    if family == "flat":
        return family, tuple(sorted(entries.values()))

    files: list[Path] = []
    for name in _DIRECTORIES[family]:
        if name in entries:
            directory = require_metadata_directory(root, base / name, context=ReaderErrorContext("stereoseq", name))
            files.extend(sorted(directory.iterdir()))

    return family, tuple(files)


def _analysis_source(file: ArtifactFile) -> AnalysisSource:
    """Parse exact documented filename forms, retaining unknown historical facts."""
    name = file.name

    match = _BIN_ANALYSIS.fullmatch(name)
    if match:
        return AnalysisSource(
            file, int(match["bin"]), _clustering_resolution(match["resolution"], file), match["sample"]
        )

    match = _CELL_ANALYSIS.fullmatch(name)
    if match:
        population: Literal["cells", "adjusted_cells"] = "adjusted_cells" if match["adjusted"] else "cells"
        return AnalysisSource(file, population, _clustering_resolution(match["resolution"], file), match["sample"])

    historical: tuple[tuple[str, Literal["cells", "adjusted_cells"] | None], ...] = (
        (".adjusted.cell.cluster.h5ad", "adjusted_cells"),
        (".cell.cluster.h5ad", "cells"),
        (".spatial.cluster.h5ad", None),
    )
    for suffix, historical_population in historical:
        if name.endswith(suffix) and name != suffix:
            return AnalysisSource(file, historical_population, None, name.removesuffix(suffix))

    raise _error(file, f"analysis filename {name!r}", "a documented square/cell clustering H5AD filename")


def _clustering_resolution(value: str, file: ArtifactFile) -> Decimal:
    """Read a positive exact filename value with contextual exponent-limit failures."""
    try:
        resolution = Decimal(value)
    except InvalidOperation as error:
        raise _error(file, f"clustering resolution {value!r}", "a finite positive decimal resolution") from error
    if not resolution.is_finite() or resolution <= 0:
        raise _error(file, f"clustering resolution {value!r}", "a finite positive decimal resolution")
    return resolution


def _count_sample(path: Path, *, sample_id: str | None = None) -> str | None:
    for suffix in (".adjusted.cellbin.gef", ".cellbin.gef", ".tissue.gef", ".raw.gef"):
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix)

    if path.name.endswith(".gef"):
        identity = path.name.removesuffix(".gef")
        # A caller-selected square source can be <SN>.<stain>.gef or
        # <SN>.<label>[.raw].label.gef. Only an explicit sample selector
        # disambiguates those qualifiers from dots within the sample itself.
        if sample_id is not None and identity.startswith(f"{sample_id}."):
            qualifiers = identity.removeprefix(f"{sample_id}.").split(".")
            if all(qualifiers):
                return sample_id
        return identity
    return None


def _image_parts(path: Path) -> tuple[str, ImageRole, int] | None:
    """Separate full image suffixes before interpreting sample/stain ordering."""
    if path.name == "bin1_img.tif":
        return "", "expression", 0
    if path.name == "bin1_img_tissue_cut.tif":
        return "", "tissue_cut", 0

    match = _DISTANCE_MASK.fullmatch(path.name)
    if match:
        return match["stem"], "distance_mask", int(match["distance"])

    for suffix, role in _IMAGE_SUFFIXES:
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix), role, 0
    return None


def _image_identity(stem: str, family: Family, sample: str | None) -> tuple[str, str]:
    """Split sample and stain using the family-specific documented name order."""
    if sample is not None:
        if stem == sample:
            return sample, ""

        # Numbered SAW outputs put the stain before the sample; compact and
        # outs exports put it after. The explicit sample preserves dotted IDs.
        if family == "numbered" and stem.endswith(f"_{sample}"):
            return sample, stem.removesuffix(f"_{sample}")
        if family != "numbered" and stem.startswith(f"{sample}_"):
            return sample, stem.removeprefix(f"{sample}_")
    if family == "compact_numbered" or "_" not in stem:
        return stem, ""
    if family == "numbered":
        # Without a selector, use the documented family split point.
        stain, inferred = stem.split("_", 1)
    else:
        inferred, stain = stem.rsplit("_", 1)
    return inferred, stain


def _select_sample(
    root: DatasetRoot,
    family: Family,
    files: tuple[Path, ...],
    analyses: tuple[AnalysisSource, ...],
    requested: str | None,
) -> str | None:
    samples = {sample for path in files if (sample := _count_sample(path)) is not None}
    samples.update(source.sample_id for source in analyses)

    if not samples:
        for path in files:
            parts = _image_parts(path)
            if parts is not None and parts[0]:
                samples.add(_image_identity(parts[0], family, requested)[0])

    if requested is not None:
        return requested
    if len(samples) > 1:
        raise _error(root, f"samples {sorted(samples)}", "sample_id when more than one sample is present")
    return next(iter(samples), None)


def _count_paths(
    files: tuple[Path, ...], sample: str | None, suffixes: tuple[str, ...], directories: tuple[str, ...]
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in files
        if path.parent.name in directories and any(path.name == f"{sample}{suffix}" for suffix in suffixes)
    )


def _count_file(  # noqa: PLR0913, RUF100 - one resolver for every count role keeps selection explicit.
    root: DatasetRoot,
    files: tuple[Path, ...],
    sample: str | None,
    *,
    override: ArtifactFile | None,
    suffixes: tuple[str, ...],
    artifact: str,
    family: Family,
) -> ArtifactFile:
    context = ReaderErrorContext("stereoseq", artifact)

    if override is not None:
        return override
    if family == "flat":
        raise _error(root, f"flat {artifact} without explicit path", f"an explicit {artifact} source override")

    directories = _SQUARE_DIRECTORIES[family] if artifact == "square_counts_file" else _CELL_DIRECTORIES[family]
    candidates = _count_paths(files, sample, suffixes, directories)

    selected = require_unique_path(candidates, context=context, search_path=root)
    return require_metadata_file(root, selected, context=context)


def _calibration_sources(
    root: DatasetRoot, family: Family, files: tuple[Path, ...], sample: str | None
) -> tuple[SquareSource | CellSource, ...]:
    """Resolve optional exact-sample GEF companions from the existing inventory."""
    sources: list[SquareSource | CellSource] = []
    choices: tuple[tuple[tuple[str, ...], Literal["square", "unadjusted", "adjusted"]], ...] = (
        ((".tissue.gef", ".gef"), "square"),
        ((".cellbin.gef",), "unadjusted"),
        ((".adjusted.cellbin.gef",), "adjusted"),
    )
    context = ReaderErrorContext("stereoseq", "image calibration companion")
    for suffixes, kind in choices:
        directories = _SQUARE_DIRECTORIES[family] if kind == "square" else _CELL_DIRECTORIES[family]
        candidates = _count_paths(files, sample, suffixes, directories)
        if candidates:
            file = require_metadata_file(
                root, require_unique_path(candidates, context=context, search_path=root), context=context
            )
            sources.append(SquareSource(file, "all") if kind == "square" else CellSource(file, kind))
    return tuple(sources)


def _image_directories(family: Literal["compact_numbered", "numbered", "outs"], role: ImageRole) -> tuple[str, ...]:
    """Resolve role-specific locations in the documented producer trees."""
    if family == "compact_numbered":
        return ("03.register", "04.tissuecut")
    if family == "outs":
        return ("image",)

    directories = ("04.register", "05.tissuecut")
    if role == "distance_mask":
        # SAW 7 cellCorrect writes expanded masks beside cell GEFs:
        # A8 manual, PDF p.42; A9 manual, PDF p.45.
        # https://github.com/STOmics/SAW/blob/0808e44619f84b67d44c063b2fd24762f6633051/Documents/UserManual/Stereo-seq_Analysis_Workflow_User_Manual_A8.pdf
        return (*directories, "051.cellcut")
    return directories


def _images(
    root: DatasetRoot, family: Family, files: tuple[Path, ...], sample: str | None, roles: tuple[ImageRole, ...]
) -> tuple[ImageSource, ...]:
    selected: dict[str, ImageSource] = {}
    found_roles: set[ImageRole] = set()

    for path in files:
        parts = _image_parts(path)
        if parts is None or parts[1] not in roles:
            continue
        stem, role, distance = parts

        if family != "flat" and path.parent.name not in _image_directories(family, role):
            continue

        inferred, stain = _image_identity(stem, family, sample)
        if stem and inferred != sample:
            continue
        file = require_metadata_file(root, path, context=ReaderErrorContext("stereoseq", role))
        source: ImageSource
        if role == "distance_mask":
            if not stain:
                raise _error(path, "distance mask without stain", "a documented stain/sample distance-mask name")
            source = DistanceMask(file, stain, distance)
        elif stain and role != "expression":
            source = StainedImage(file, role, stain)
        else:
            source = GridImage(file, role)
        if source.name in selected:
            raise _error(path, f"duplicate image {source.name!r}", "one image per stain/role/distance")
        selected[source.name] = source
        found_roles.add(role)

    if missing := set(roles) - found_roles:
        raise _error(root, f"missing image roles {sorted(missing)}", "every selected image role")

    return tuple(selected[name] for name in sorted(selected))


def _validate_override_samples(
    root: DatasetRoot,
    overrides: tuple[ArtifactFile | None, ...],
    sample: str | None,
    requested: str | None,
) -> None:
    """Reconcile explicit source populations with the selected sample once."""
    for position, file in enumerate(overrides):
        if file is not None:
            identity = _count_sample(file, sample_id=requested if position == 0 else None)
            if identity is not None and identity != sample:
                raise _error(
                    root, "count override/sample disagreement", "selected count artifacts from the selected sample"
                )


def discover_layout(  # noqa: PLR0913, RUF100 - mirrors the public product selections.
    path: str | Path,
    *,
    sample_id: str | None,
    bins: tuple[int, ...] | Literal["all"],
    cell_bins: Literal["none", "unadjusted", "adjusted", "both"],
    square_counts_file: str | Path | None,
    cell_counts_file: str | Path | None,
    adjusted_cell_counts_file: str | Path | None,
    analysis_files: tuple[str | Path, ...],
    image_roles: tuple[ImageRole, ...],
) -> StereoseqLayout:
    """Resolve selected products once, allowing caller overrides outside the root."""
    context = ReaderErrorContext("stereoseq", "dataset layout")
    root = normalize_dataset_root(path, context=context)

    family, files = _inventory(root)
    analyses = tuple(_analysis_source(require_caller_file(root, file, context=context)) for file in analysis_files)
    overrides = tuple(
        None if file is None else require_caller_file(root, file, context=context)
        for file in (square_counts_file, cell_counts_file, adjusted_cell_counts_file)
    )

    selected_overrides = tuple(file for file in overrides if file is not None)
    sample = _select_sample(root, family, (*files, *selected_overrides), analyses, sample_id)
    _validate_override_samples(root, overrides, sample, sample_id)

    identities: set[tuple[str, str | int, Decimal | None]] = set()
    for source in analyses:
        if source.population is not None:
            key = (source.sample_id, source.population, source.resolution)
            if key in identities:
                raise _error(
                    source.file, "duplicate analysis identity", "distinct population/clustering-resolution products"
                )
            identities.add(key)

    if len({source.file for source in analyses}) != len(analyses):
        raise _error(root, "duplicate analysis artifact", "distinct selected analysis files")
    if any(source.sample_id != sample for source in analyses):
        raise _error(root, "analysis/sample disagreement", "selected analyses from the selected sample")

    # Resolve only requested count populations; missing unselected products are
    # irrelevant to this invocation.
    squares: tuple[SquareSource, ...] = ()
    if bins:
        file = _count_file(
            root,
            files,
            sample,
            override=overrides[0],
            suffixes=(".tissue.gef", ".gef"),
            artifact="square_counts_file",
            family=family,
        )
        squares = (SquareSource(file, bins),)

    cells: list[CellSource] = []
    cell_choices: tuple[tuple[Literal["unadjusted", "adjusted"], ArtifactFile | None, str, str], ...] = (
        ("unadjusted", overrides[1], ".cellbin.gef", "cell_counts_file"),
        ("adjusted", overrides[2], ".adjusted.cellbin.gef", "adjusted_cell_counts_file"),
    )
    for kind, override, suffix, artifact in cell_choices:
        if cell_bins in (kind, "both"):
            file = _count_file(
                root, files, sample, override=override, suffixes=(suffix,), artifact=artifact, family=family
            )
            cells.append(CellSource(file, kind))

    images = _images(root, family, files, sample, image_roles)
    calibration = (
        _calibration_sources(root, family, files, sample) if images and not (squares or cells or analyses) else ()
    )
    return StereoseqLayout(root, family, sample, squares, tuple(cells), analyses, images, calibration)
