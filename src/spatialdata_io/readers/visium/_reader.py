"""Assemble one single-library classic Visium dataset from Space Ranger output.

One call validates every option, discovers the layout (reading the HDF5
identity attributes), reads the scale factors, builds the lazy images, reads
the positions and the sparse counts, builds the spot circles and links the
table to them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.names import sanitize_element_name, validate_element_name
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.visium._geometry import make_circles, make_frames
from spatialdata_io.readers.visium._images import image_kind, make_image_options, read_images
from spatialdata_io.readers.visium._layout import H5Counts, discover_layout
from spatialdata_io.readers.visium._metadata import read_scale_factors
from spatialdata_io.readers.visium._tables import link_table, prepare_table, read_positions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers.visium._layout import CountsLayout

# The circles share SpatialData's case-insensitive namespace with the table.
_RESERVED_NAMES = ("table",)


def _derived_dataset_id(counts: CountsLayout) -> str:
    """Name the dataset after the HDF5 library ID or the sample-download filename prefix."""
    if isinstance(counts, H5Counts) and counts.metadata.library_id is not None:
        value, source = counts.metadata.library_id, "visium: the HDF5 library ID"
    elif counts.filename_prefix is not None:
        value, source = counts.filename_prefix, "visium: the count filename prefix"
    else:
        message = "dataset_id is required when the count source has no HDF5 library ID or filename prefix."
        raise ValueError(message)
    return sanitize_element_name(value, source=source, parameter="dataset_id", reserved=_RESERVED_NAMES)


def visium(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    dataset_id: str | None = None,
    counts_source: str | Path | None = None,
    fullres_image_file: str | Path | None = None,
    positions_file: str | Path | None = None,
    scale_factors_file: str | Path | None = None,
    make_var_names_unique: bool = True,
    data_axes: str | None = None,
    tiff_series: int | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read one single-library Space Ranger output directory of classic Visium.

    Space Ranger 1.x and 2.0 or later layouts are supported (``outs/`` of
    ``spaceranger count``)::

        outs/
        ├── filtered_feature_bc_matrix.h5   # or filtered_feature_bc_matrix/ (MEX)
        └── spatial/
            ├── tissue_positions.csv        # 2.0+, headered
            │   or tissue_positions_list.csv  # 1.x, headerless
            ├── scalefactors_json.json
            ├── tissue_hires_image.png      # optional
            └── tissue_lowres_image.png     # optional

    The HDF5 matrix is preferred over the MEX directory; a 10x sample download
    with a ``<prefix>_filtered_feature_bc_matrix.h5`` (or MEX directory) is
    used when neither standard name exists. A MEX directory holds one
    ``matrix.mtx``/``barcodes.tsv``/``features.tsv`` triplet, all gzip or all
    plain. Visium HD output is read by :func:`spatialdata_io.visium_hd`.

    Positions have one row per capture spot: the barcode, ``in_tissue``
    (``0`` or ``1``), ``array_row`` and ``array_col`` (nonnegative integers),
    and the spot centre ``pxl_row_in_fullres`` (``y``) and
    ``pxl_col_in_fullres`` (``x``) in full-resolution image pixels. A header
    names these fields in any order and may add columns; a headerless file has
    exactly these six. ``scalefactors_json.json`` must give finite positive
    ``tissue_hires_scalef``, ``tissue_lowres_scalef`` and
    ``spot_diameter_fullres``.

    Parameters
    ----------
    path
        Space Ranger ``outs`` directory.
    dataset_id
        Name of the circles element and prefix of the image and coordinate
        system names. Must be a valid
        SpatialData element name: nonempty, only letters, digits, ``_``,
        ``-`` and ``.``, not ``.`` or ``..``, not starting with ``__``, and
        not ``table`` in any letter case; it is checked before any I/O.
        Defaults to the HDF5 ``library_ids`` entry, else the sample-download
        filename prefix, repaired with one warning if it breaks these rules:
        other characters become ``_``, a leading run of underscores becomes
        one, an empty name, ``.`` or ``..`` becomes ``unnamed``, and a
        reserved name gains a trailing ``_``. Required for a MEX directory
        without a prefix.
    counts_source
        HDF5 file (suffix ``.h5`` or ``.hdf5``) or MEX directory to read
        instead of the discovered filtered matrix, for example
        ``"raw_feature_bc_matrix.h5"``. Relative paths resolve against
        ``path``; absolute paths may point elsewhere.
    fullres_image_file
        Full-resolution tissue image given to Space Ranger: TIFF, OME-TIFF,
        BigTIFF (``.tif``, ``.tiff``, ``.btf``), PNG or JPEG (``.jpg``,
        ``.jpeg``), resolved like ``counts_source``. The suffix is checked
        before any I/O. The positions must refer to this image's pixels.
    positions_file, scale_factors_file
        Position CSV or scale-factor JSON to read instead of the discovered
        one, resolved like ``counts_source``.
    make_var_names_unique
        Make duplicate feature names unique by appending ``-1``, ``-2``, ...
    data_axes
        Source axes of the full-resolution TIFF as one string of distinct
        letters including ``X`` and ``Y``, case-insensitive, for example
        ``"CYX"`` for a stack without axis metadata; ``S`` (samples) counts as
        a channel axis. Defaults to the axes the file declares. Requires a TIFF
        ``fullres_image_file``.
    tiff_series
        Zero-based series of the full-resolution TIFF. Required when the file
        has several series. Requires a TIFF ``fullres_image_file``.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        full-resolution image; empty for a single scale. Requires
        ``fullres_image_file``; the hires and lowres PNGs stay single-scale.
        Unrelated to ``scale_factors_file``.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes, applied to
        every image; by default one channel per chunk and square 1024-pixel
        tiles.
    path_policy
        How symlinks inside the dataset directory are treated. ``"strict"``
        requires every discovered artifact and relative file option to resolve
        inside the directory. ``"relaxed"`` also accepts symlinks to targets
        stored elsewhere, as in git-annex or DVC datasets. Relative paths may
        not leave the directory with ``..`` in either mode, and absolute file
        options may point anywhere. File options keep the name the caller
        gave, and relaxed reads keep each artifact's own name rather than that
        of its link target.

    Returns
    -------
    SpatialData
        One linked table, the spot circles and up to three images:

        ``tables["table"]``
            One row per count-matrix barcode in matrix order, including spots
            outside the tissue when raw counts are read; features keep the
            matrix order and every feature type (gene expression, antibody
            capture, ...).

            - ``X``: canonical finite CSR sparse array of counts, float32 when
              every value is exact in float32, else float64. Duplicate sparse
              entries are summed.
            - ``var``: ``gene_ids``, ``feature_types`` and, when present,
              ``genome``; ``var_names`` are the feature names.
            - ``obs``: int64 ``in_tissue``, ``array_row`` and ``array_col``;
              ``spot_id`` (the barcode, string) and categorical ``region``
              (``dataset_id``) link the rows to the circles.
            - ``obsm["spatial_pixels"]``: float64 spot centres ``x, y`` in
              full-resolution pixels, declared in
              ``uns["spatialdata_io_positions"]`` with frame
              ``dataset_id + "_pixels"``; there is no physical
              ``obsm["spatial"]``.
            - ``uns["spatial"][dataset_id]``: ``scalefactors`` (the complete
              JSON object) and ``metadata`` (HDF5 ``chemistry_description``
              and ``software_version`` when present), as Scanpy stores them,
              without images.

        ``shapes[dataset_id]``
            One circle per table row, indexed by barcode (``spot_id``), centred
            on the spot with radius ``spot_diameter_fullres / 2``. This is the
            display size Space Ranger draws, not a measured capture area.
        ``images[dataset_id + "_full_image"]`` (only with ``fullres_image_file``)
            Lazy CYX image of the selected TIFF series, or of the PNG or JPEG
            file, with the source dtype and channel order; a pyramid only when
            ``scale_factors`` asks for one.
        ``images[dataset_id + "_hires_image"]``, ``images[dataset_id + "_lowres_image"]``
            The Space Ranger PNGs that are present, lazy and single-scale.

        Image channels are named ``r, g, b`` (``r, g, b, a`` with declared
        alpha) only when the file declares RGB: TIFF
        ``PhotometricInterpretation`` RGB or JPEG-compressed YCbCr, PNG colour
        types 2, 3 and 6, and three-component JPEG. Otherwise the OME channel
        names, or IDs, name the channels, and an OME channel with several
        samples names them ``<name or ID>:sample:<index>``; other images use
        positions. A four-channel fluorescence image is therefore never
        labelled RGB(A).

        Coordinate systems: Space Ranger estimates the spot diameter in
        pixels from the fiducial registration and a nominal spot size, which
        does not calibrate the image, so no element uses ``global``
        (micrometres).

        - ``dataset_id + "_pixels"``, full-resolution pixels: the circles and
          the full-resolution image by identity, each PNG by
          ``Scale(1 / tissue_*_scalef)``;
        - ``dataset_id + "_hires_pixels"`` and
          ``dataset_id + "_lowres_pixels"``, PNG pixels: that PNG by identity
          and the circles by ``Scale(tissue_*_scalef)``.

        Earlier releases named these frames ``dataset_id``,
        ``dataset_id + "_downscaled_hires"`` and
        ``dataset_id + "_downscaled_lowres"``.

        ``attrs`` record the reader, the package version and the Space Ranger
        version of an HDF5 matrix (``2.1.0`` for ``spaceranger-2.1.0``). Image
        pixels are not read until computed: TIFF computations decode only the
        tiles or strips they overlap, PNG and JPEG decode the whole frame.
        Files are closed on return and no source file is modified.

    Warns
    -----
    UserWarning
        Once if the HDF5 library ID or filename prefix had to be repaired to
        name the dataset, naming both names.

    Raises
    ------
    TypeError
        If an option has the wrong type, or expression storage is unsupported.
    ValueError
        If an option has an invalid value, an image option is given without a
        suitable ``fullres_image_file``, ``dataset_id`` is needed but missing,
        the axes of an image cannot be mapped to CYX, or a count is not finite
        or not exactly representable.
    IndexError
        If ``tiff_series`` is out of range.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a path is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact, path and
        detected Space Ranger version if an artifact is ambiguous or
        malformed, for example several count sources, an incomplete MEX
        triplet, invalid position fields or scale factors, a count barcode
        without a position row, several HDF5 library IDs, or a multi-series
        TIFF without ``tiff_series``.
    """
    if dataset_id is not None:
        validate_element_name(dataset_id, parameter="dataset_id", reserved=_RESERVED_NAMES)
    fullres_kind = None if fullres_image_file is None else image_kind(fullres_image_file)
    if not isinstance(make_var_names_unique, bool):
        message = "make_var_names_unique must be a boolean."
        raise TypeError(message)
    image_options = make_image_options(
        fullres_kind, data_axes=data_axes, tiff_series=tiff_series, scale_factors=scale_factors, chunks=chunks
    )

    layout = discover_layout(
        path,
        counts_source=counts_source,
        fullres_image_file=fullres_image_file,
        positions_file=positions_file,
        scale_factors_file=scale_factors_file,
        path_policy=path_policy,
    )
    if dataset_id is None:
        dataset_id = _derived_dataset_id(layout.counts)

    vendor_scales = read_scale_factors(layout.scale_factors, version=layout.format_version)
    frames = make_frames(dataset_id, vendor_scales)
    images = read_images(layout, dataset_id=dataset_id, frames=frames, options=image_options)

    positions = read_positions(layout)
    prepared = prepare_table(
        layout,
        positions,
        dataset_id=dataset_id,
        pixel_frame=frames.fullres,
        scale_factors=vendor_scales,
        make_var_names_unique=make_var_names_unique,
    )
    # Space Ranger reports a spot diameter; ShapesModel stores the radius.
    circles = make_circles(prepared.positions, radius=vendor_scales.spot_diameter / 2.0, frames=frames)
    table = link_table(prepared.adata, dataset_id=dataset_id, shape_index=circles.index, layout=layout)

    result = SpatialData(images=images, shapes={dataset_id: circles}, tables={"table": table})
    set_reader_provenance(result.attrs, reader="visium", format_version=layout.format_version)
    return result
