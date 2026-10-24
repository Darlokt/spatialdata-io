"""Assemble one CosMx flat-file export into calibrated rasters, points and a table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.cosmx._layout import discover_layout
from spatialdata_io.readers.cosmx._rasters import make_raster_options, read_image, read_labels
from spatialdata_io.readers.cosmx._tables import build_table, read_counts, read_fov_positions, read_metadata
from spatialdata_io.readers.cosmx._transcripts import read_transcripts
from spatialdata_io.readers.cosmx._transforms import solve_placements

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from dask.dataframe import DataFrame
    from xarray import DataArray, DataTree


def cosmx(
    path: str | Path,
    *,
    dataset_id: str | None = None,
    include_transcripts: bool = True,
    scale_factors: Sequence[int] | None = None,
    chunks: int | tuple[int, int, int] | None = None,
) -> SpatialData:
    """Read a CosMx Spatial Molecular Imager flat-file export.

    Supported exports contain, with ``.csv`` or ``.csv.gz`` flat files::

        export/
        ├── <dataset_id>_exprMat_file.csv
        ├── <dataset_id>_metadata_file.csv
        ├── <dataset_id>_fov_positions_file.csv      # optional
        ├── <dataset_id>_tx_file.csv                 # with include_transcripts
        ├── CellComposite/<prefix>_F<fov>.jpg        # optional; .jpeg, .png or .tif
        └── CellLabels/<prefix>_F<fov>.tif           # optional; .tiff

    The 2021 flat-file release and AtoMx flat-file exports are recognized by
    their columns, not by a version label. Polygon, overlay, compartment and
    morphology files are not read.

    Parameters
    ----------
    path
        Export directory.
    dataset_id
        File prefix. None infers it from the single ``*_exprMat_file`` file.
    include_transcripts
        Return the lazy transcript points.
    scale_factors
        Optional integer factors greater than one for generated image and
        label pyramid levels. None keeps the full-resolution single scale.
    chunks
        Positive integer or three positive CYX sizes; labels use the final two.
        None keeps native source chunks. Rechunking remains lazy.

    Returns
    -------
    spatialdata.SpatialData
        ``tables['table']`` follows metadata row order. ``X`` is an eager
        canonical CSR int64 count array whose features are every count-matrix
        column except ``fov`` and ``cell_ID``, in source order, including
        negative and system controls. The per-FOV background rows with
        ``cell_ID`` 0 are removed; the remaining count and metadata populations
        must be identical. Observation names are the vendor ``cell`` IDs when
        present and row positions otherwise. ``fov`` and ``cell_ID`` are int64;
        ``Area``, ``AspectRatio``, ``Width``, ``Height`` and ``Mean.*``/``Max.*``
        intensities are nullable float64; other metadata stay text.
        ``obsm['spatial']`` and ``obsm['global']`` hold local and global cell
        centers in pixels. With labels, ``region`` links each cell to
        ``<fov>_labels`` with instance key ``cell_ID``; without a
        ``CellLabels`` directory the table is unlinked.

        ``images['<fov>_image']`` is a lazy ``(c, y, x)`` array with source
        pixels unchanged (8-bit color composites use channels ``r, g, b``), and
        ``labels['<fov>_labels']`` is lazy integer ``(y, x)`` on the same grid.
        ``points['<dataset_id>_transcripts']`` holds one lazy point per
        transcript at its global ``x``/``y`` pixel position, with ``target``
        categories equal to the count features and ``fov``, ``cell_ID``,
        ``x_local``, ``y_local`` and, when present, ``z_raw``, ``CellComp`` and
        ``cell`` annotations. ``cell_ID`` 0 marks unassigned transcripts.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        Required artifacts are missing or have the wrong filesystem kind.
    ValueError
        Layouts, schemas, identifiers, populations, placements, values or
        options violate the supported contract.
    TypeError
        Public options have invalid types.

    Notes
    -----
    Coordinate system ``global`` is the export's global pixel frame; each
    ``<fov>`` system is that FOV's local pixel frame. Local coordinates are
    pixel indices with a reversed vertical axis, so the raster pixel at row
    ``r`` and column ``c`` of an ``H``-row FOV has local coordinates
    ``(c, H - r)``. This convention was measured on the 2021 release and an
    AtoMx export. Each FOV's local-to-global map is ``global = ±local +
    offset`` per axis, derived from every metadata cell and accepted only when
    one offset satisfies all printed rounding bounds; pixel FOV positions,
    when present, must agree. Rasters are supported for exports whose axes
    are not reversed between frames; the raster convention of exports with a
    reversed global axis has not been evidenced, so such rasters fail.

    Transcript partitions are newline-delimited blocks of plain CSV, or the
    independently compressed, record-aligned members of a ``.csv.gz`` file.
    A single-stream gzip transcript file fails; decompress it or recompress it
    in blocks. Unknown targets and malformed values fail when their partition
    is computed. Encoded JPEG and PNG images decode their whole frame for any
    selection. Construction executes no raster or transcript tasks.
    """
    if not isinstance(include_transcripts, bool):
        message = "include_transcripts must be a boolean."
        raise TypeError(message)
    options = make_raster_options(scale_factors, chunks)
    layout = discover_layout(path, dataset_id=dataset_id, include_transcripts=include_transcripts)

    metadata = read_metadata(layout.metadata)
    counts = read_counts(layout.counts)
    positions = {} if layout.fov_positions is None else read_fov_positions(layout.fov_positions)
    raster_fovs = frozenset(item.fov for item in (*layout.images, *layout.labels))
    placements = solve_placements(
        metadata.fovs, metadata.samples, required=raster_fovs, positions=positions, path=layout.metadata
    )
    if raster_fovs and any((p.sign_x, p.sign_y) != (1, 1) for p in placements.values()):
        raise ReaderFormatError(
            context=ReaderErrorContext("cosmx", "FOV rasters"),
            path=layout.root,
            found="rasters in an export whose global axes are reversed relative to local axes",
            expected="rasters only in exports with the evidenced local and global orientation",
        )

    images: dict[str, DataArray | DataTree] = {}
    grids: dict[int, tuple[int, int]] = {}
    for item in layout.images:
        images[f"{item.fov}_image"], grids[item.fov] = read_image(
            item.path, fov=item.fov, placement=placements[item.fov], options=options
        )
    labels: dict[str, DataArray | DataTree] = {}
    for item in layout.labels:
        labels[f"{item.fov}_labels"], _ = read_labels(
            item.path, fov=item.fov, grid=grids.get(item.fov), placement=placements[item.fov], options=options
        )

    # A CellLabels directory is the table target; every FOV with cells needs its labels.
    linked = bool(layout.labels)
    missing = sorted(set(metadata.fovs.tolist()) - {item.fov for item in layout.labels})
    if linked and missing:
        raise ReaderFormatError(
            context=ReaderErrorContext("cosmx", "CellLabels"),
            path=layout.root,
            found=f"no labels for FOVs {missing}",
            expected="labels for every FOV with cells",
        )
    table = build_table(counts, metadata, path=layout.metadata, linked=linked)

    points: dict[str, DataFrame] = {}
    if layout.transcripts is not None:
        points[f"{layout.dataset_id}_transcripts"] = read_transcripts(
            layout.transcripts, features=counts.features, placements=placements
        )

    result = SpatialData(images=images, labels=labels, points=points, tables={"table": table})
    set_reader_provenance(result.attrs, reader="cosmx")
    return result
