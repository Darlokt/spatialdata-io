"""Discover one Xenium bundle: family, selected sources, twins, images and aligned files.

Discovery reads ``experiment.xenium`` once and resolves only the artifacts that
the selected products need. Parquet files are preferred over their CSV twins;
an HDF5 count matrix over its MatrixMarket directory or archive. Image paths
come from the ``images`` block of ``experiment.xenium``, never from version
guesses, because Xenium Ranger keeps the input bundle's image files.

Artifacts, with the alternatives discovery accepts::

    outs/
    ├── experiment.xenium                    # format version, pixel size, image paths
    ├── cells.parquet | cells.csv | cells.csv.gz
    ├── cell_boundaries.*, nucleus_boundaries.*   # same twins as cells
    ├── transcripts.parquet | transcripts.csv     # a gzip CSV alone is rejected
    ├── cell_feature_matrix.h5 | cell_feature_matrix/ | cell_feature_matrix.tar.gz
    ├── cells.zarr.zip                       # masks and cell-ID mappings
    ├── morphology_focus.ome.tif | morphology_focus/*.ome.tif
    ├── morphology_mip.ome.tif               # XOA 1.x only
    ├── *he_image.ome.tif, *if_image.ome.tif (+ *imagealignment.csv)
    ├── analysis/ | analysis.tar.gz
    └── metrics_summary.csv, gene_panel.json, protein_panel.json   # optional

A morphology image in its own directory must be an OME-linked set equal to the
``.ome.tif`` files beside it; the set is compared by file name, so symlinked
members keep their bundle names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    optional_metadata_directory,
    optional_metadata_file,
    require_metadata_file,
)
from spatialdata_io.readers.xenium._metadata import Experiment, read_experiment

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy

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
class MorphologyImage:
    """A morphology entry file and the file names its OME-linked set must have.

    ``name`` is the entry's file name as ``experiment.xenium`` declares it and
    ``members`` the names of every validated OME-TIFF in its directory, or only
    ``name`` for a file at the bundle root. OME-TIFF sets refer to their files
    by these names.
    """

    entry: ArtifactFile
    name: str
    members: frozenset[str]


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
    focus: MorphologyImage | None
    mip: MorphologyImage | None
    aligned: tuple[AlignedImage, ...]
    analysis: AnalysisSource | None
    metrics: ArtifactFile | None
    gene_panel: ArtifactFile | None
    protein_panel: ArtifactFile | None


def _tabular(root: DatasetRoot, stem: str, context: ReaderErrorContext) -> TabularSource:
    """Prefer ``<stem>.parquet``; otherwise require exactly one CSV twin."""
    parquet = optional_metadata_file(root, f"{stem}.parquet", context=context)
    if parquet is not None:
        return TabularSource(parquet, "parquet")
    twins = [
        TabularSource(file, kind)
        for suffix, kind in _CSV_TWINS
        if (file := optional_metadata_file(root, f"{stem}{suffix}", context=context)) is not None
    ]
    if len(twins) > 1:
        raise ReaderFormatError(
            context=context,
            path=root.path,
            found=f"both {stem}.csv and {stem}.csv.gz without {stem}.parquet",
            expected=f"{stem}.parquet or exactly one CSV twin",
        )
    if not twins:
        require_metadata_file(root, f"{stem}.parquet", context=context)
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
    h5 = optional_metadata_file(root, "cell_feature_matrix.h5", context=context)
    if h5 is not None:
        return H5Counts(h5)
    directory = optional_metadata_directory(root, "cell_feature_matrix", context=context)
    if directory is not None:
        for member in _MTX_MEMBERS:
            require_metadata_file(root, directory / member, context=context)
        return MtxDirectory(directory)
    archive = optional_metadata_file(root, "cell_feature_matrix.tar.gz", context=context)
    if archive is not None:
        return MtxArchive(archive)
    return H5Counts(require_metadata_file(root, "cell_feature_matrix.h5", context=context))


def _morphology(root: DatasetRoot, experiment: Experiment, key: str, purpose: str) -> MorphologyImage:
    """Resolve an ``images`` entry; a dedicated directory must hold only its linked set."""
    context = experiment.context(purpose)
    declared = experiment.images.get(key)
    if declared is None:
        raise ReaderFormatError(
            context=context,
            path=root.path / "experiment.xenium",
            found=f"no images.{key} entry",
            expected=f"images.{key}; XOA 2.0+ bundles have no maximum-intensity projection"
            if key == "morphology_mip_filepath"
            else f"images.{key} naming the {purpose}",
        )
    entry = require_metadata_file(root, declared, context=context)
    name = PurePosixPath(declared).name
    if PurePosixPath(declared).parent == PurePosixPath():
        return MorphologyImage(entry, name, frozenset({name}))

    paths = [path for path in list_directory(entry.parent, context=context) if path.name.endswith(".ome.tif")]
    for path in paths:
        require_metadata_file(root, path, context=context)
    return MorphologyImage(entry, name, frozenset(path.name for path in paths))


def _aligned(root: DatasetRoot, experiment: Experiment) -> tuple[AlignedImage, ...]:
    """Find 10x-named aligned H&E/IF images at the root; their alignment CSVs are optional."""
    context = experiment.context("aligned images")
    entries = list_directory(root.path, context=context)
    images: list[AlignedImage] = []
    for suffix, kind in _ALIGNED_SUFFIXES:
        matches = [path for path in entries if path.name.endswith(suffix)]
        if len(matches) > 1:
            raise ReaderFormatError(
                context=context,
                path=root.path,
                found=f"{len(matches)} {kind.upper()} images {[path.name for path in matches]}",
                expected=f"at most one *{suffix}; read others with xenium_aligned_image",
            )
        if matches:
            image = require_metadata_file(root, matches[0], context=context)
            name = matches[0].name.removesuffix(".ome.tif") + "alignment.csv"
            alignment = optional_metadata_file(root, name, context=context)
            images.append(AlignedImage(image, kind, alignment))
    return tuple(images)


def _analysis(root: DatasetRoot, context: ReaderErrorContext) -> AnalysisSource:
    directory = optional_metadata_directory(root, "analysis", context=context)
    if directory is not None:
        # The analysis reader opens every CSV below the directory.
        for file in list_directory(directory, context=context, recursive=True):
            if file.name.endswith(".csv"):
                require_metadata_file(root, file, context=context)
        return AnalysisDirectory(directory)
    return AnalysisArchive(require_metadata_file(root, "analysis.tar.gz", context=context))


def discover_layout(path: str | Path, selection: Selection, *, path_policy: PathPolicy = "strict") -> XeniumLayout:
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
    root = normalize_dataset_root(path, context=ReaderErrorContext("xenium", "dataset root"), path_policy=path_policy)
    experiment = read_experiment(
        require_metadata_file(root, "experiment.xenium", context=ReaderErrorContext("xenium", "experiment.xenium"))
    )

    context = experiment.context
    # A cell_labels table target is validated to include the cell labels.
    labels = selection.cell_labels or selection.nucleus_labels
    uses_store = labels or selection.table or selection.cell_polygons or selection.nucleus_polygons
    cells_store = (
        require_metadata_file(root, "cells.zarr.zip", context=context("cells.zarr.zip (labels and their table link)"))
        if labels
        else optional_metadata_file(root, "cells.zarr.zip", context=context("cells.zarr.zip"))
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
        focus=_morphology(root, experiment, "morphology_focus_filepath", "morphology focus image")
        if selection.morphology_focus
        else None,
        mip=_morphology(root, experiment, "morphology_mip_filepath", "maximum-intensity projection")
        if selection.morphology_mip
        else None,
        aligned=_aligned(root, experiment) if selection.aligned_images else (),
        analysis=_analysis(root, context("analysis")) if selection.analysis else None,
        metrics=optional_metadata_file(root, "metrics_summary.csv", context=context("metrics_summary.csv")),
        gene_panel=optional_metadata_file(root, "gene_panel.json", context=context("gene_panel.json")),
        protein_panel=optional_metadata_file(root, "protein_panel.json", context=context("protein_panel.json")),
    )
