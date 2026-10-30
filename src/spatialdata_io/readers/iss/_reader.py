"""Assemble the ISS raw image, segmentation, table linkage, and provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.iss._images import _read_rasters, _validate_data_axes
from spatialdata_io.readers.iss._layout import _discover_layout
from spatialdata_io.readers.iss._schemas import DEFAULT_INSTANCE_KEY, REGION_KEY, _validate_output_column
from spatialdata_io.readers.iss._tables import _read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def iss(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    raw_image_file: str | Path,
    labels_file: str | Path,
    table_file: str | Path,
    instance_key: str | None = None,
    dataset_id: str = "region",
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    data_axes: str | None = None,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> SpatialData:
    """Read an unversioned ISS image, segmentation, and linked h5ad table.

    The caller selects three artifacts. There is no inferred pipeline version
    or fixed vendor directory tree::

        raw.tiff  # TIFF-family scientific image
        labels.tiff  # TIFF-family integer segmentation
        cells.h5ad  # expression and observation metadata

    Each TIFF must contain exactly one series. Its full-resolution level zero
    is used. The caller must select images and labels registered on the same
    base pixel grid; equal YX extents are required but do not prove alignment.
    Complete raw-image OME XY spacing calibrates this pair in micrometres.

    Parameters
    ----------
    path
        Dataset directory.
    raw_image_file, labels_file, table_file
        Required TIFF, OME-TIFF, BigTIFF/BTF and h5ad paths. Relative paths
        resolve within the dataset root. Absolute paths may be outside it.
    instance_key
        Integer observation column containing strictly positive label IDs.
        Otherwise decimal observation names become ``instance_id``. Zero is
        reserved for label background. IDs must fit the labels' integer dtype.
        Exact pixel membership is not checked by computing lazy labels.
    dataset_id
        Nonempty identifier used to name the two raster elements. Contains only
        alphanumeric characters, underscores, dots or hyphens, and cannot start
        with two underscores.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images and labels; empty for a single scale. Generated label pyramids
        require <=32-bit integer labels: upstream downsampling rounds 64-bit
        IDs. Exact base int64/uint64 labels are supported at a single scale.
    data_axes
        Optional source raw-image axes, such as ``CYX`` for an unlabeled
        stack. Otherwise TIFF axes are used. Scientific C channels never
        become RGB based on their number. Non-singleton sample axes require
        an explicit interpretation as C. Singleton nonspatial axes may be
        removed, while singleton X/Y dimensions are retained. Labels require
        YX and only singleton extra axes.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images; labels use the ``y`` and ``x`` sizes. By default one channel per
        chunk and square 1024-pixel tiles.

    Returns
    -------
    SpatialData
        ``images['<dataset_id>_raw_image']`` is lazy CYX with source dtype,
        channel order, and unique available OME channel names or IDs. Channels
        without OME metadata use positions. ``labels['<dataset_id>_labels_image']`` is lazy YX with
        source integer dtype and instance values. Both use ``global`` in micrometres
        when the base raw image declares complete OME XY sizes in µm, nm or mm;
        labels inherit missing calibration and conflicting declarations fail.
        Otherwise both retain shared source pixels. Explicit reinterpretation
        of spatial axes retains pixels. The source origin is unchanged; stage
        placement is not inferred. Construction executes zero
        pixel tasks. Computing a selection decodes only the TIFF tiles or strips
        of the output chunks it intersects. TIFF resources close after inspection and inside each lazy task.
        ``tables['table']`` is owned, eager, finite canonical CSR sparse-array
        expression with source dtype and row/feature order. Duplicate feature
        names become unique. The optional ``counts`` layer is normalized in the
        same way. Other layers, additional H5AD annotation fields and their
        dtypes/missingness, and existing spatial metadata are retained. Imported
        H5AD embeddings keep their source values and units; label linkage does
        not establish their calibration. Annotation
        columns may be reordered; duplicate column names are rejected. Existing
        reserved output columns must be semantically equal to generated values.
        Final linkage uses categorical ``region`` and the selected integer
        instance column. No shapes or points are returned. Caller-owned state
        is not mutated. Provenance records the reader and package version.

    Raises
    ------
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If an artifact is missing or has the wrong filesystem kind.
    ValueError
        If raster axes, integer IDs, table linkage, shared extents, options,
        TIFF metadata, expression values or generated pyramids violate the
        supported contract. Malformed layouts and schemas raise the
        ``ReaderFormatError`` subclass.
    TypeError
        If option or expression types are unsupported.
    """
    # Validate public configuration before opening any artifact.
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id.startswith("__")
        or not all(character.isalnum() or character in "_-." for character in dataset_id)
    ):
        message = (
            "dataset_id must contain only alphanumeric characters, underscores, dots or hyphens "
            "and cannot start with '__'."
        )
        raise ValueError(message)
    if instance_key is not None and (
        not isinstance(instance_key, str) or not instance_key or instance_key == REGION_KEY
    ):
        message = "instance_key must be a nonempty observation name distinct from 'region'."
        raise ValueError(message)

    raw_axes = _validate_data_axes(data_axes)
    raster = make_raster_options(scale_factors, chunks)

    layout = _discover_layout(path, raw_image_file=raw_image_file, labels_file=labels_file, table_file=table_file)
    rasters = _read_rasters(layout, data_axes=raw_axes, raster=raster)
    table = _read_table(layout.table, instance_key=instance_key, label_dtype=rasters.label_dtype)

    # _read_table validates eager frames; AnnData also types backed Dataset2D.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)

    region = f"{dataset_id}_labels_image"
    regions = pd.Series(pd.Categorical([region] * table.n_obs), index=table.obs_names)
    _validate_output_column(obs, REGION_KEY, regions, path=layout.table)
    obs[REGION_KEY] = regions

    # Annotation storage is finalized after reader mutations and before linkage.
    context = ReaderErrorContext("iss", "linked counts")
    prepare_anndata_dataframe(obs, context=context)
    prepare_anndata_dataframe(var, context=context)

    try:
        table = parse_linked_table(
            table,
            TableLinkage((region,), REGION_KEY, instance_key or DEFAULT_INSTANCE_KEY),
            context=context,
            expression_layers=("counts",) if "counts" in table.layers else (),
        )
    except (ValueError, TypeError) as error:
        add_reader_context(error, context=context, path=layout.table)
        raise

    result = SpatialData(
        images={f"{dataset_id}_raw_image": rasters.image}, labels={region: rasters.labels}, tables={"table": table}
    )
    set_reader_provenance(result.attrs, reader="iss")
    return result
