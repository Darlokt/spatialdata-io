"""Discover one single-library Space Ranger output directory for classic Visium.

The count source is found first, so that the Space Ranger version read from an
HDF5 matrix names every later error. Space Ranger 2.0 and later write::

    outs/
    ├── filtered_feature_bc_matrix.h5       # preferred count source
    ├── filtered_feature_bc_matrix/         # MEX alternative, all gzip or all plain
    │   ├── matrix.mtx.gz
    │   ├── barcodes.tsv.gz
    │   └── features.tsv.gz
    ├── raw_feature_bc_matrix.h5            # only through counts_source
    └── spatial/
        ├── tissue_positions.csv            # headered
        ├── scalefactors_json.json
        ├── tissue_hires_image.png          # optional
        └── tissue_lowres_image.png         # optional

Space Ranger 1.x writes the same tree with a headerless
``spatial/tissue_positions_list.csv``; a ``tissue_positions_list.csv`` with a
header is also accepted. ``tissue_positions.csv`` takes precedence when both
exist. 10x sample downloads prefix the matrix with the dataset name and are
recognized when the standard names are absent::

    <prefix>_filtered_feature_bc_matrix.h5  # or <prefix>_filtered_feature_bc_matrix/
    spatial/                                # extracted <prefix>_spatial.tar.gz

The standard HDF5 matrix wins over the standard MEX directory, which wins over
one prefixed source; a prefixed HDF5 file beside a prefixed MEX directory is
ambiguous. A MEX directory must hold exactly one complete
``[<prefix>]matrix.mtx``/``barcodes.tsv``/``features.tsv`` triplet; a matching
uncompressed ``[<prefix>]genes.tsv`` would make the 10x MEX reader switch to
its legacy feature format and is rejected. Other files in ``spatial/`` (QC
images, CytAssist images, enrichment tables) are ignored. The full-resolution
tissue image is not part of the output and is read only from
``fullres_image_file``.

Overrides (``counts_source``, ``positions_file``, ``scale_factors_file``,
``fullres_image_file``) resolve relative paths against the dataset root and
accept absolute paths anywhere. ``counts_source`` is an HDF5 file when its
suffix is ``.h5`` or ``.hdf5`` and a MEX directory otherwise. Discovered
artifacts, including the three MEX member files, follow the root's path
policy.

The position format is classified from the first logical CSV record (quoted
newlines allowed): a record naming any position column is a header, any other
record must be a six-field data row.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    optional_metadata_directory,
    optional_metadata_file,
    optional_unique_path,
    require_caller_directory,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
)
from spatialdata_io.readers.visium._metadata import CountMetadata, read_count_metadata
from spatialdata_io.readers.visium._schemas import PositionFormat, classify_position_row

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import PathPolicy

FILTERED_H5 = "filtered_feature_bc_matrix.h5"
FILTERED_MTX = "filtered_feature_bc_matrix"
POSITIONS_V1 = "tissue_positions_list.csv"
POSITIONS_V2 = "tissue_positions.csv"
SCALE_FACTORS = "scalefactors_json.json"
HIRES_IMAGE = "tissue_hires_image.png"
LOWRES_IMAGE = "tissue_lowres_image.png"

_H5_SUFFIXES = frozenset({".h5", ".hdf5"})
_MTX_MEMBERS = ("matrix.mtx", "barcodes.tsv", "features.tsv")


@dataclass(frozen=True, slots=True)
class H5Counts:
    """A 10x HDF5 count matrix, its identity attributes and any sample-download prefix."""

    path: ArtifactFile
    filename_prefix: str | None
    metadata: CountMetadata


@dataclass(frozen=True, slots=True)
class MtxCounts:
    """One complete MEX triplet: its member prefix, compression and any sample-download prefix."""

    path: ArtifactDirectory
    prefix: str | None
    compressed: bool
    filename_prefix: str | None


type CountsLayout = H5Counts | MtxCounts


@dataclass(frozen=True, slots=True)
class VisiumLayout:
    """Validated artifacts and the position format of one Visium dataset."""

    counts: CountsLayout
    positions: ArtifactFile
    position_format: PositionFormat
    scale_factors: ArtifactFile
    hires_image: ArtifactFile | None
    lowres_image: ArtifactFile | None
    fullres_image: ArtifactFile | None

    @property
    def format_version(self) -> str | None:
        """Return the Space Ranger version recorded in an HDF5 count matrix, if recognized."""
        return self.counts.metadata.format_version if isinstance(self.counts, H5Counts) else None


def _context(artifact: str, version: str | None = None) -> ReaderErrorContext:
    """Identify a Visium layout artifact for contextual errors."""
    return ReaderErrorContext("visium", artifact, version)


def _inspect_mtx_directory(path: ArtifactDirectory, *, filename_prefix: str | None) -> MtxCounts:
    """Resolve one complete consistently compressed MEX triplet."""
    context = _context("count matrix")
    names = frozenset(entry.name for entry in list_directory(path, context=context))

    variants: list[tuple[str | None, bool]] = []
    for compressed in (True, False):
        extension = ".gz" if compressed else ""
        matrix, barcodes, features = (f"{name}{extension}" for name in _MTX_MEMBERS)
        prefixes = {name[: -len(matrix)] for name in names if name.endswith(matrix)}
        variants.extend(
            (prefix or None, compressed)
            for prefix in sorted(prefixes)
            if f"{prefix}{barcodes}" in names and f"{prefix}{features}" in names
        )

    if len(variants) != 1:
        found = "no complete MEX triplet" if not variants else f"multiple MEX triplets {variants!r}"
        raise ReaderFormatError(
            context=context,
            found=found,
            expected="exactly one matrix/barcodes/features triplet, consistently compressed or uncompressed",
            path=path,
        )

    prefix, compressed = variants[0]
    legacy_name = f"{prefix or ''}genes.tsv"
    legacy_path = path / legacy_name
    # Scanpy selects its legacy feature format from this file even beside
    # features.tsv; no vendor export is known to contain both.
    if legacy_name in names and legacy_path.is_file():
        raise ReaderFormatError(
            context=context,
            found=f"competing legacy feature selector {legacy_name!r}",
            expected=f"the selected modern {prefix or ''}features.tsv{'.gz' if compressed else ''} family only",
            path=legacy_path,
        )

    return MtxCounts(path=path, prefix=prefix, compressed=compressed, filename_prefix=filename_prefix)


def _require_mtx_members(root: DatasetRoot, counts: MtxCounts) -> MtxCounts:
    """Apply the root's path policy to the three discovered files that the MEX reader opens."""
    extension = ".gz" if counts.compressed else ""
    for name in _MTX_MEMBERS:
        require_metadata_file(
            root, counts.path / f"{counts.prefix or ''}{name}{extension}", context=_context("count matrix")
        )
    return counts


def _h5_counts(path: ArtifactFile, *, filename_prefix: str | None = None) -> H5Counts:
    """Pair an HDF5 count matrix with its identity attributes."""
    return H5Counts(path=path, filename_prefix=filename_prefix, metadata=read_count_metadata(path))


def _explicit_counts(root: DatasetRoot, source: str | Path, *, context: ReaderErrorContext) -> CountsLayout:
    """Read the caller's count source as HDF5 by suffix, otherwise as a MEX directory."""
    if Path(source).suffix.lower() in _H5_SUFFIXES:
        return _h5_counts(require_caller_file(root, source, context=context))
    return _inspect_mtx_directory(require_caller_directory(root, source, context=context), filename_prefix=None)


def _discover_counts(root: DatasetRoot, source: str | Path | None) -> CountsLayout:
    """Resolve the explicit or preferred count source without ambiguous fallback."""
    context = _context("count matrix")
    if source is not None:
        return _explicit_counts(root, source, context=context)

    standard_h5 = optional_metadata_file(root, FILTERED_H5, context=context)
    if standard_h5 is not None:
        return _h5_counts(standard_h5)

    standard_mtx = optional_metadata_directory(root, FILTERED_MTX, context=context)
    if standard_mtx is not None:
        return _require_mtx_members(root, _inspect_mtx_directory(standard_mtx, filename_prefix=None))

    entries = list_directory(root.path, context=context)
    h5_candidate = optional_unique_path(
        (entry for entry in entries if entry.name.endswith(f"_{FILTERED_H5}")),
        context=_context("legacy prefixed HDF5 count matrix"),
        search_path=root.path,
    )
    mtx_candidate = optional_unique_path(
        (entry for entry in entries if entry.name.endswith(f"_{FILTERED_MTX}")),
        context=_context("legacy prefixed MEX count matrix"),
        search_path=root.path,
    )
    if h5_candidate is not None and mtx_candidate is not None:
        raise ReaderFormatError(
            context=context,
            found=f"both {h5_candidate.name!r} and {mtx_candidate.name!r}",
            expected="one legacy prefixed count source",
            path=root.path,
        )

    if h5_candidate is not None:
        prefix = h5_candidate.name.removesuffix(f"_{FILTERED_H5}")
        return _h5_counts(require_metadata_file(root, h5_candidate, context=context), filename_prefix=prefix)

    if mtx_candidate is not None:
        prefix = mtx_candidate.name.removesuffix(f"_{FILTERED_MTX}")
        directory = require_metadata_directory(root, mtx_candidate, context=context)
        return _require_mtx_members(root, _inspect_mtx_directory(directory, filename_prefix=prefix))

    # Report the missing standard artifact with a precise FileNotFoundError.
    return _h5_counts(require_metadata_file(root, FILTERED_H5, context=context))


def _position_format(path: ArtifactFile, *, context: ReaderErrorContext) -> PositionFormat:
    """Read the first logical CSV record and classify its field contract."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream, strict=True), [])
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    except UnicodeDecodeError as error:
        raise ReaderFormatError(
            context=context, path=path, found="invalid UTF-8 header", expected="UTF-8 position CSV"
        ) from error
    except csv.Error as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="malformed first CSV row",
            expected="a valid header or six-field data row",
        ) from error

    return classify_position_row(columns, path=path, context=context)


def _discover_positions(
    root: DatasetRoot, override: str | Path | None, *, version: str | None
) -> tuple[ArtifactFile, PositionFormat]:
    """Resolve the position artifact and classify its named or positional layout."""
    context = _context("tissue positions", version)
    if override is not None:
        path = require_caller_file(root, override, context=context)
    else:
        current = optional_metadata_file(root, f"spatial/{POSITIONS_V2}", context=context)
        path = current or require_metadata_file(root, f"spatial/{POSITIONS_V1}", context=context)

    return path, _position_format(path, context=context)


def discover_layout(  # noqa: PLR0913, RUF100 - mirrors the public artifact selections.
    path: str | Path,
    *,
    counts_source: str | Path | None,
    fullres_image_file: str | Path | None,
    positions_file: str | Path | None,
    scale_factors_file: str | Path | None,
    path_policy: PathPolicy = "strict",
) -> VisiumLayout:
    """Discover exactly one supported single-library Visium layout."""
    root = normalize_dataset_root(path, context=_context("dataset root"), path_policy=path_policy)
    counts = _discover_counts(root, counts_source)
    version = counts.metadata.format_version if isinstance(counts, H5Counts) else None

    positions, position_format = _discover_positions(root, positions_file, version=version)
    scale_context = _context("scale factors", version)
    scale_factors = (
        require_caller_file(root, scale_factors_file, context=scale_context)
        if scale_factors_file is not None
        else require_metadata_file(root, f"spatial/{SCALE_FACTORS}", context=scale_context)
    )

    fullres = (
        None
        if fullres_image_file is None
        else require_caller_file(root, fullres_image_file, context=_context("full-resolution image", version))
    )
    spatial = optional_metadata_directory(root, "spatial", context=_context("spatial output", version))
    hires, lowres = (
        None if spatial is None else optional_metadata_file(root, spatial / name, context=_context(artifact, version))
        for name, artifact in ((HIRES_IMAGE, "hires image"), (LOWRES_IMAGE, "lowres image"))
    )

    return VisiumLayout(
        counts=counts,
        positions=positions,
        position_format=position_format,
        scale_factors=scale_factors,
        hires_image=hires,
        lowres_image=lowres,
        fullres_image=fullres,
    )
