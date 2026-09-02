"""Deterministic discovery of single-library Space Ranger output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    optional_directory,
    optional_file,
    optional_unique_path,
    require_caller_directory,
    require_caller_file,
    require_directory,
    require_file,
)

FILTERED_H5 = "filtered_feature_bc_matrix.h5"
FILTERED_MTX = "filtered_feature_bc_matrix"
POSITIONS_V1 = "tissue_positions_list.csv"
POSITIONS_V2 = "tissue_positions.csv"
SCALE_FACTORS = "scalefactors_json.json"
HIRES_IMAGE = "tissue_hires_image.png"
LOWRES_IMAGE = "tissue_lowres_image.png"
MAX_POSITION_HEADER_BYTES = 1_024

type PositionFormat = Literal["headered", "headerless"]


@dataclass(frozen=True, slots=True)
class H5Counts:
    """One 10x HDF5 count source and optional legacy filename prefix."""

    path: ArtifactFile
    filename_prefix: str | None


@dataclass(frozen=True, slots=True)
class MtxCounts:
    """One validated 10x MEX directory selection."""

    path: ArtifactDirectory
    prefix: str | None
    compressed: bool
    filename_prefix: str | None


type CountsLayout = H5Counts | MtxCounts


@dataclass(frozen=True, slots=True)
class VisiumLayout:
    """Closed immutable paths and format decisions for one Visium dataset."""

    root: DatasetRoot
    counts: CountsLayout
    positions: ArtifactFile
    position_format: PositionFormat
    scale_factors: ArtifactFile
    hires_image: ArtifactFile | None
    lowres_image: ArtifactFile | None
    fullres_image: ArtifactFile | None


def _context(artifact: str) -> ReaderErrorContext:
    return ReaderErrorContext(reader="visium", artifact=artifact)


def _inspect_mtx_directory(path: ArtifactDirectory, *, filename_prefix: str | None) -> MtxCounts:
    context = _context("count matrix")
    try:
        names = frozenset(entry.name for entry in path.iterdir())
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise

    variants: list[tuple[str | None, bool]] = []
    for compressed, suffixes in (
        (True, ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz")),
        (False, ("matrix.mtx", "barcodes.tsv", "features.tsv")),
    ):
        matrix_suffix, barcode_suffix, feature_suffix = suffixes
        prefixes = {name[: -len(matrix_suffix)] for name in names if name.endswith(matrix_suffix)}
        variants.extend(
            (prefix or None, compressed)
            for prefix in sorted(prefixes)
            if f"{prefix}{barcode_suffix}" in names and f"{prefix}{feature_suffix}" in names
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
    return MtxCounts(
        path=path,
        prefix=prefix,
        compressed=compressed,
        filename_prefix=filename_prefix,
    )


def _explicit_counts(root: DatasetRoot, source: str | Path) -> CountsLayout:
    source_path = Path(source).expanduser()
    if source_path.suffix.lower() in {".h5", ".hdf5"}:
        path = require_caller_file(root, source, context=_context("count matrix"))
        return H5Counts(path=path, filename_prefix=None)
    directory = require_caller_directory(root, source, context=_context("count matrix"))
    return _inspect_mtx_directory(directory, filename_prefix=None)


def _discover_counts(root: DatasetRoot, source: str | Path | None) -> CountsLayout:
    if source is not None:
        return _explicit_counts(root, source)

    standard_h5 = optional_file(root / FILTERED_H5, context=_context("count matrix"))
    if standard_h5 is not None:
        return H5Counts(path=standard_h5, filename_prefix=None)
    standard_mtx = optional_directory(root / FILTERED_MTX, context=_context("count matrix"))
    if standard_mtx is not None:
        return _inspect_mtx_directory(standard_mtx, filename_prefix=None)

    h5_candidate = optional_unique_path(
        root.glob(f"*_{FILTERED_H5}"),
        context=_context("legacy prefixed HDF5 count matrix"),
        search_path=root,
    )
    mtx_candidate = optional_unique_path(
        root.glob(f"*_{FILTERED_MTX}"),
        context=_context("legacy prefixed MEX count matrix"),
        search_path=root,
    )
    if h5_candidate is not None and mtx_candidate is not None:
        raise ReaderFormatError(
            context=_context("count matrix"),
            found=f"both {h5_candidate.name!r} and {mtx_candidate.name!r}",
            expected="one legacy prefixed count source",
            path=root,
        )
    if h5_candidate is not None:
        prefix = h5_candidate.name[: -len(f"_{FILTERED_H5}")]
        return H5Counts(
            path=require_file(h5_candidate, context=_context("count matrix")),
            filename_prefix=prefix,
        )
    if mtx_candidate is not None:
        prefix = mtx_candidate.name[: -len(f"_{FILTERED_MTX}")]
        directory = require_directory(mtx_candidate, context=_context("count matrix"))
        return _inspect_mtx_directory(directory, filename_prefix=prefix)
    # Preserve a precise path exception for the required default artifact.
    return H5Counts(
        path=require_file(root / FILTERED_H5, context=_context("count matrix")),
        filename_prefix=None,
    )


def _position_format(path: ArtifactFile) -> PositionFormat:
    context = _context("tissue positions")
    try:
        with path.open("rb") as stream:
            first_line = stream.readline(MAX_POSITION_HEADER_BYTES + 1)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    if len(first_line) > MAX_POSITION_HEADER_BYTES:
        raise ReaderFormatError(
            context=context,
            found=f"a first row longer than {MAX_POSITION_HEADER_BYTES} bytes",
            expected="a bounded six-column CSV header or data row",
            path=path,
        )
    return "headered" if first_line.removeprefix(b"\xef\xbb\xbf").startswith(b"barcode,") else "headerless"


def _discover_positions(root: DatasetRoot, override: str | Path | None) -> tuple[ArtifactFile, PositionFormat]:
    if override is not None:
        path = require_caller_file(root, override, context=_context("tissue positions"))
    else:
        current = optional_file(root / "spatial" / POSITIONS_V2, context=_context("tissue positions"))
        path = current or require_file(root / "spatial" / POSITIONS_V1, context=_context("tissue positions"))
    return path, _position_format(path)


def discover_layout(
    path: str | Path,
    *,
    counts_source: str | Path | None,
    fullres_image_file: str | Path | None,
    positions_file: str | Path | None,
    scale_factors_file: str | Path | None,
) -> VisiumLayout:
    """Discover exactly one supported single-library Visium layout."""
    root = normalize_dataset_root(path, context=_context("dataset root"))
    spatial = optional_directory(root / "spatial", context=_context("spatial output"))
    positions, position_format = _discover_positions(root, positions_file)
    scale_factors = (
        require_caller_file(root, scale_factors_file, context=_context("scale factors"))
        if scale_factors_file is not None
        else require_file(root / "spatial" / SCALE_FACTORS, context=_context("scale factors"))
    )
    fullres = (
        require_caller_file(root, fullres_image_file, context=_context("full-resolution image"))
        if fullres_image_file is not None
        else None
    )
    hires = None if spatial is None else optional_file(spatial / HIRES_IMAGE, context=_context("hires image"))
    lowres = None if spatial is None else optional_file(spatial / LOWRES_IMAGE, context=_context("lowres image"))
    return VisiumLayout(
        root=root,
        counts=_discover_counts(root, counts_source),
        positions=positions,
        position_format=position_format,
        scale_factors=scale_factors,
        hires_image=hires,
        lowres_image=lowres,
        fullres_image=fullres,
    )
