"""Discover one Xenium bundle: family, selected sources, twins, images and aligned files.

Discovery reads ``experiment.xenium`` once and resolves only the artifacts that
the selected products need. Parquet files are preferred over their CSV twins;
an HDF5 count matrix over its MatrixMarket directory or archive. Image paths
come from the ``images`` block of ``experiment.xenium``, never from version
guesses, because Xenium Ranger keeps the input bundle's image files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    optional_directory,
    optional_file,
    require_file,
    require_metadata_file,
    sorted_paths,
)
from spatialdata_io.readers.xenium._metadata import Experiment, read_experiment

if TYPE_CHECKING:
    from pathlib import Path

type TabularKind = Literal["parquet", "csv", "csv_gz"]
type TableTarget = Literal["cell_labels", "cell_boundaries", "cell_circles"]

_ALIGNED_SUFFIXES: tuple[tuple[str, Literal["he", "if"]], ...] = (
    ("he_image.ome.tif", "he"),
    ("if_image.ome.tif", "if"),
)
_CSV_TWINS: tuple[tuple[str, TabularKind], ...] = ((".csv", "csv"), (".csv.gz", "csv_gz"))
_MTX_MEMBERS = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")


@dataclass(frozen=True, slots=True)
class Selection:
    """Validated products of one read; ``table_target`` is ``None`` for an unlinked or absent table."""

    table: bool
    table_target: TableTarget | None
    cell_labels: bool
    nucleus_labels: bool
    cell_polygons: bool
    cell_circles: bool
    nucleus_polygons: bool
    transcripts: bool
    morphology_focus: bool
    morphology_mip: bool
    aligned_images: bool
    analysis: bool


@dataclass(frozen=True, slots=True)
class TabularSource:
    """A parquet file or one CSV twin with an identical header."""

    file: ArtifactFile
    kind: TabularKind


@dataclass(frozen=True, slots=True)
class H5Counts:
    """A 10x feature-barcode HDF5 matrix; the only source of the protein scaling factor."""

    file: ArtifactFile


@dataclass(frozen=True, slots=True)
class MtxDirectory:
    """An extracted ``cell_feature_matrix/`` MatrixMarket directory."""

    directory: ArtifactDirectory


@dataclass(frozen=True, slots=True)
class MtxArchive:
    """A ``cell_feature_matrix.tar.gz`` MatrixMarket archive."""

    file: ArtifactFile


type CountSource = H5Counts | MtxDirectory | MtxArchive


@dataclass(frozen=True, slots=True)
class FocusImages:
    """The focus entry file and every OME-TIFF that must belong to its linked set."""

    entry: ArtifactFile
    files: tuple[ArtifactFile, ...]


@dataclass(frozen=True, slots=True)
class AlignedImage:
    """A post-Xenium H&E or IF image and its optional alignment matrix."""

    file: ArtifactFile
    kind: Literal["he", "if"]
    alignment: ArtifactFile | None

    @property
    def name(self) -> str:
        """Return the element name."""
        return f"{self.kind}_image"


@dataclass(frozen=True, slots=True)
class AnalysisDirectory:
    """An extracted ``analysis/`` directory."""

    directory: ArtifactDirectory


@dataclass(frozen=True, slots=True)
class AnalysisArchive:
    """An ``analysis.tar.gz`` archive."""

    file: ArtifactFile


type AnalysisSource = AnalysisDirectory | AnalysisArchive


@dataclass(frozen=True, slots=True)
class XeniumLayout:
    """Validated paths of one read; ``None`` marks an unselected or absent optional source."""

    root: DatasetRoot
    experiment: Experiment
    cells_store: ArtifactFile | None
    cells: TabularSource | None
    cell_boundaries: TabularSource | None
    nucleus_boundaries: TabularSource | None
    transcripts: TabularSource | None
    counts: CountSource | None
    focus: FocusImages | None
    mip: ArtifactFile | None
    aligned: tuple[AlignedImage, ...]
    analysis: AnalysisSource | None
    metrics: ArtifactFile | None
    gene_panel: ArtifactFile | None
    protein_panel: ArtifactFile | None


def _context(experiment: Experiment, artifact: str) -> ReaderErrorContext:
    version = experiment.software if experiment.version is None else str(experiment.version)
    return ReaderErrorContext("xenium", artifact, version)


def _tabular(root: DatasetRoot, stem: str, context: ReaderErrorContext) -> TabularSource:
    """Prefer ``<stem>.parquet``; otherwise require exactly one CSV twin."""
    parquet = optional_file(root / f"{stem}.parquet", context=context)
    if parquet is not None:
        return TabularSource(parquet, "parquet")
    twins = [
        TabularSource(file, kind)
        for suffix, kind in _CSV_TWINS
        if (file := optional_file(root / f"{stem}{suffix}", context=context)) is not None
    ]
    if len(twins) > 1:
        raise ReaderFormatError(
            context=context,
            path=root,
            found=f"both {stem}.csv and {stem}.csv.gz without {stem}.parquet",
            expected=f"{stem}.parquet or exactly one CSV twin",
        )
    if not twins:
        require_file(root / f"{stem}.parquet", context=context)
    return twins[0]


def _transcripts(root: DatasetRoot, context: ReaderErrorContext) -> TabularSource:
    """Require parquet or a plain CSV; one gzip stream cannot be split into lazy partitions."""
    source = _tabular(root, "transcripts", context)
    if source.kind == "csv_gz":
        raise ReaderFormatError(
            context=context,
            path=source.file,
            found="only a single-stream gzip transcript CSV",
            expected="transcripts.parquet or a decompressed transcripts.csv (for example gunzip -k transcripts.csv.gz)",
        )
    return source


def _counts(root: DatasetRoot, context: ReaderErrorContext) -> CountSource:
    h5 = optional_file(root / "cell_feature_matrix.h5", context=context)
    if h5 is not None:
        return H5Counts(h5)
    directory = optional_directory(root / "cell_feature_matrix", context=context)
    if directory is not None:
        for member in _MTX_MEMBERS:
            require_file(directory / member, context=context)
        return MtxDirectory(directory)
    archive = optional_file(root / "cell_feature_matrix.tar.gz", context=context)
    if archive is not None:
        return MtxArchive(archive)
    return H5Counts(require_file(root / "cell_feature_matrix.h5", context=context))


def _image_entry(root: DatasetRoot, experiment: Experiment, key: str, purpose: str) -> ArtifactFile:
    context = _context(experiment, f"{purpose} ({key})")
    entry = experiment.images.get(key)
    if entry is None:
        raise ReaderFormatError(
            context=context,
            path=root / "experiment.xenium",
            found=f"no images.{key} entry",
            expected=f"images.{key}; XOA 2.0+ bundles have no maximum-intensity projection"
            if key == "morphology_mip_filepath"
            else f"images.{key} naming the {purpose}",
        )
    return require_metadata_file(root, entry, context=context)


def _focus(root: DatasetRoot, experiment: Experiment) -> FocusImages:
    """Resolve the focus entry; a dedicated focus directory must hold only its linked set."""
    entry = _image_entry(root, experiment, "morphology_focus_filepath", "morphology focus image")
    if entry.parent == root:
        return FocusImages(entry, (entry,))
    context = _context(experiment, "morphology_focus")
    # macOS AppleDouble companions ("._name") are not images.
    candidates = (
        path for path in entry.parent.iterdir() if path.name.endswith(".ome.tif") and not path.name.startswith("._")
    )
    paths = sorted_paths(candidates, context=context, search_path=entry.parent)
    files = tuple(require_file(path, context=context) for path in paths)
    return FocusImages(entry, files)


def _aligned(root: DatasetRoot, experiment: Experiment) -> tuple[AlignedImage, ...]:
    """Find 10x-named aligned H&E/IF images at the root; their alignment CSVs are optional."""
    context = _context(experiment, "aligned images")
    entries = sorted_paths(root.iterdir(), context=context, search_path=root)
    images: list[AlignedImage] = []
    for suffix, kind in _ALIGNED_SUFFIXES:
        matches = [path for path in entries if path.name.endswith(suffix) and not path.name.startswith("._")]
        if len(matches) > 1:
            raise ReaderFormatError(
                context=context,
                path=root,
                found=f"{len(matches)} {kind.upper()} images {[path.name for path in matches]}",
                expected=f"at most one *{suffix}; read others with xenium_aligned_image",
            )
        if matches:
            image = require_file(matches[0], context=context)
            name = matches[0].name.removesuffix(".ome.tif") + "alignment.csv"
            alignment = optional_file(root / name, context=context)
            images.append(AlignedImage(image, kind, alignment))
    return tuple(images)


def _analysis(root: DatasetRoot, context: ReaderErrorContext) -> AnalysisSource:
    directory = optional_directory(root / "analysis", context=context)
    if directory is not None:
        return AnalysisDirectory(directory)
    return AnalysisArchive(require_file(root / "analysis.tar.gz", context=context))


def discover_layout(path: str | Path, selection: Selection) -> XeniumLayout:
    """Resolve the artifacts of one read without touching unselected products.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If the root or a required artifact is missing or of the wrong kind.
    ReaderFormatError
        If ``experiment.xenium`` is malformed or unsupported, a twin is
        ambiguous, transcripts exist only as gzip CSV, a requested image entry is
        absent, or aligned images are ambiguous.
    """
    root = normalize_dataset_root(path, context=ReaderErrorContext("xenium", "dataset root"))
    experiment = read_experiment(
        require_file(root / "experiment.xenium", context=ReaderErrorContext("xenium", "experiment.xenium"))
    )

    def context(artifact: str) -> ReaderErrorContext:
        return _context(experiment, artifact)

    # A cell_labels table target is validated to include the cell labels.
    labels = selection.cell_labels or selection.nucleus_labels
    uses_store = labels or selection.table or selection.cell_polygons or selection.nucleus_polygons
    cells_store = (
        require_file(root / "cells.zarr.zip", context=context("cells.zarr.zip (labels and their table link)"))
        if labels
        else optional_file(root / "cells.zarr.zip", context=context("cells.zarr.zip"))
        if uses_store
        else None
    )
    counts_needed = selection.table or selection.transcripts
    return XeniumLayout(
        root=root,
        experiment=experiment,
        cells_store=cells_store,
        cells=_tabular(root, "cells", context("cells table")) if selection.table or selection.cell_circles else None,
        cell_boundaries=_tabular(root, "cell_boundaries", context("cell boundaries"))
        if selection.cell_polygons
        else None,
        nucleus_boundaries=_tabular(root, "nucleus_boundaries", context("nucleus boundaries"))
        if selection.nucleus_polygons
        else None,
        transcripts=_transcripts(root, context("transcripts")) if selection.transcripts else None,
        counts=_counts(root, context("cell-feature matrix")) if counts_needed else None,
        focus=_focus(root, experiment) if selection.morphology_focus else None,
        mip=_image_entry(root, experiment, "morphology_mip_filepath", "maximum-intensity projection")
        if selection.morphology_mip
        else None,
        aligned=_aligned(root, experiment) if selection.aligned_images else (),
        analysis=_analysis(root, context("analysis")) if selection.analysis else None,
        metrics=optional_file(root / "metrics_summary.csv", context=context("metrics_summary.csv")),
        gene_panel=optional_file(root / "gene_panel.json", context=context("gene_panel.json")),
        protein_panel=optional_file(root / "protein_panel.json", context=context("protein_panel.json")),
    )
