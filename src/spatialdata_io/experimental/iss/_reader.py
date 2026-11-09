"""Public ISS reader: link a caller-selected raw image, labels and ``.h5ad`` table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.experimental.iss._images import read_rasters, validate_data_axes
from spatialdata_io.experimental.iss._layout import discover_layout, validate_artifact_names
from spatialdata_io.experimental.iss._schemas import DEFAULT_INSTANCE_KEY, REGION_KEY
from spatialdata_io.experimental.iss._tables import read_table
from spatialdata_io.readers._utils.errors import ReaderErrorContext, warn_user
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.names import validate_element_name
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.table import SparseDuplicateSummary


def iss(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    raw_image_file: str | Path,
    labels_file: str | Path,
    table_file: str | Path,
    instance_key: str | None = None,
    dataset_id: str = "region",
    data_axes: str | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read an ISS raw image, its segmentation labels and a linked ``.h5ad`` table.

    ISS pipelines have no fixed directory tree or format version, so the caller
    names each of the three artifacts::

        dataset/
        ├── raw.tiff     # raw_image_file: TIFF, OME-TIFF, BigTIFF or BTF image
        ├── labels.tiff  # labels_file: TIFF-family integer segmentation
        └── cells.h5ad   # table_file: AnnData expression and cell metadata

    Each TIFF must hold exactly one series, read at full resolution. The image
    and labels must be registered on the same pixel grid; equal YX extents are
    checked, alignment itself is the caller's assertion. Table rows link to
    label values through an integer instance column.

    Parameters
    ----------
    path
        Dataset directory.
    raw_image_file, labels_file, table_file
        The three artifacts, with suffix ``.tif``, ``.tiff`` or ``.btf`` for the
        rasters and ``.h5ad`` for the table. Relative paths resolve against
        ``path``; absolute paths may point elsewhere.
    instance_key
        Observation column holding the label value of each row: non-null
        positive integers. By default the observation names are used, and
        must then consist of ASCII digits only.
    dataset_id
        Prefix of the raster element names. It must be a valid SpatialData
        element name: nonempty, only letters, digits, ``_``, ``-`` and ``.``,
        not ``.`` or ``..``, and not starting with ``__``.
    data_axes
        Axes of the raw image as one string of distinct letters including X
        and Y, case-insensitive, for example ``"CYX"`` for a stack without
        axis metadata or ``"YXC"`` to read interleaved RGB samples (``YXS``)
        as channels. Defaults to the axes the file declares. Labels always use
        their declared axes.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        image and labels; empty for a single scale. Pyramids require labels of
        at most 32 bits, because downsampling rounds larger IDs.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; labels use
        the ``y, x`` sizes. By default one channel per chunk and square
        1024-pixel tiles.
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
        One image, one labels element and one linked table:

        ``images["<dataset_id>_raw_image"]``
            Lazy CYX image with the source dtype and channel order. Singleton
            nonspatial axes are dropped; a missing C axis is added. Channel
            names are ``r, g, b`` (``r, g, b, a`` with a declared alpha sample)
            for samples the TIFF declares RGB (``PhotometricInterpretation``
            RGB or JPEG-compressed YCbCr); otherwise OME channel names when all
            are nonempty and unique, else OME channel IDs, with
            ``<name or ID>:sample:<index>`` for an OME channel expanded from
            samples; otherwise, including channels that ``data_axes`` reads
            from another declared axis such as Z, positions. A three- or
            four-channel image is never guessed to be RGB.
        ``labels["<dataset_id>_labels_image"]``
            Lazy YX labels with the source integer dtype. Zero is background.
        ``tables["table"]``
            The ``.h5ad`` table in source row and feature order.

            - ``X`` and the optional ``layers["counts"]``: canonical CSR sparse
              arrays with the source values and dtype. Duplicate sparse entries
              are summed; an integer sum that overflows the source dtype
              promotes to float32 or float64 when exact, and raises otherwise.
            - ``obs["region"]``: the labels element name, categorical.
            - ``obs[instance_key]`` (kept, widened to 16 bits when narrower) or
              ``obs["instance_id"]`` (uint64)
              holds the label values. IDs must fit the labels dtype. Membership
              in the labels is not checked, so labels are never computed.
            - ``var_names``: duplicates are made unique (``GeneB``, ``GeneB-1``).
            - An imported ``obsm["spatial"]`` moves unchanged to
              ``obsm["spatial_source"]``; no axes or units are inferred.
            - Other fields, layers, ``raw`` and ``uns`` entries are kept with
              their dtypes and missingness. Source fields named ``region`` or
              like the instance column must equal the generated values.
              Other ``obs`` and ``var`` column names and top-level ``uns``
              keys that a SpatialData table does not accept (``my note``) are
              sanitized with a warning (``my_note``; a name that then
              collides, ignoring letter case, with another one, ``region``,
              the instance column or the reserved ``uns`` keys
              ``spatialdata_io_positions`` and ``spatialdata_attrs`` gets a
              ``_1``, ``_2``, ... suffix). ``instance_key`` keeps its name;
              ``uns`` values and nested keys stay as stored.

        Coordinate systems: image and labels share one coordinate system.

        - ``global`` in micrometres, with a ``Scale`` of the pixel size, when
          the raw image declares complete OME ``PhysicalSizeX``/``Y`` in µm,
          nm or mm. Label OME sizes must agree where present.
        - ``image_pixels`` (identity) otherwise, or when ``data_axes`` moves X
          or Y. ``global`` is then absent, because it always means
          micrometres.

        The source origin is kept; stage placement is not inferred. The table
        has no coordinates.

        ``attrs`` record the reader and package version. Pixels are not read
        until computed, and a partial computation decodes only the TIFF tiles
        or strips it overlaps. All files are closed on return; caller-owned
        objects and source files are not modified.

    Warns
    -----
    UserWarning
        Once per call when duplicate sparse entries were summed, naming the
        affected expression slots and any dtype change; once each if ``obs``
        or ``var`` column names or top-level ``uns`` keys were sanitized.

    Raises
    ------
    TypeError
        If an option has the wrong type, or the expression storage is
        unsupported.
    ValueError
        If an option has an invalid value, for example a wrong file suffix, or
        ``data_axes`` does not fit the raw image (with a note naming it).
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a path is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if an
        artifact is malformed: several TIFF series, unsupported declared axes,
        unequal extents, non-integer labels, conflicting calibration, a
        missing or duplicate table field, invalid or out-of-range IDs, or
        non-finite expression.
    """
    validate_artifact_names(raw_image_file, labels_file, table_file)
    _validate_options(instance_key, dataset_id)
    raw_axes = validate_data_axes(data_axes)
    raster = make_raster_options(scale_factors, chunks)

    layout = discover_layout(
        path, raw_image_file=raw_image_file, labels_file=labels_file, table_file=table_file, path_policy=path_policy
    )
    rasters = read_rasters(layout, data_axes=raw_axes, raster=raster)
    region = f"{dataset_id}_labels_image"
    table = read_table(layout.table, region=region, instance_key=instance_key, label_dtype=rasters.label_dtype)

    duplicate_sums: list[SparseDuplicateSummary] = []
    table = parse_linked_table(
        table,
        TableLinkage((region,), REGION_KEY, instance_key or DEFAULT_INSTANCE_KEY),
        context=ReaderErrorContext("iss", "linked table"),
        path=layout.table,
        duplicate_sums=duplicate_sums,
        expression_layers=("counts",) if "counts" in table.layers else (),
    )

    result = SpatialData(
        images={f"{dataset_id}_raw_image": rasters.image}, labels={region: rasters.labels}, tables={"table": table}
    )
    set_reader_provenance(result.attrs, reader="iss")
    if duplicate_sums:
        warn_user(f"iss {layout.table.name}: " + "; ".join(map(str, duplicate_sums)))
    return result


def _validate_options(instance_key: str | None, dataset_id: str) -> None:
    """Validate the table column and element-name options before any I/O.

    ``dataset_id`` follows SpatialData's element-name rule. The derived names
    end in ``_raw_image`` and ``_labels_image``, so they cannot collide with
    ``table``.
    """
    if instance_key is not None:
        if not isinstance(instance_key, str):
            message = "instance_key must be a string or None."
            raise TypeError(message)
        if not instance_key or instance_key == REGION_KEY:
            message = f"instance_key must be a nonempty observation column other than {REGION_KEY!r}."
            raise ValueError(message)

    validate_element_name(dataset_id, parameter="dataset_id")
