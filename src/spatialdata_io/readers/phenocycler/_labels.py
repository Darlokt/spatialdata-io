"""Register an explicitly associated instance-label TIFF to an Akoya regional image.

Each mask represents one integer YX plane. Supplying its path declares that it
segments the selected image. On the same pixel grid the shapes must agree.
A cropped, rotated or resampled mask instead needs a measured affine mapping
its native XY pixel coordinates into the image's pixel coordinates. The image
then supplies the transformation into the output coordinate system.

Full-resolution label values remain lazy and zero denotes background. Neither
registration nor CODEX table linkage requires decoding the mask to enumerate
cell IDs. The original files must remain accessible during later computations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from spatialdata.models import Labels2DModel
from spatialdata.transformations import Affine, get_transformation
from xarray import DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    normalize_raster_axes,
    ome_pixel_sizes_um,
    read_tiff,
    require_matching_pixel_sizes,
    select_tiff_series,
)
from spatialdata_io.readers._utils.paths import require_caller_file
from spatialdata_io.readers.phenocycler._transforms import validate_xy_affine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers._utils.paths import ArtifactFile, DatasetRoot

_XY = ("x", "y")


@dataclass(frozen=True, slots=True)
class MaskSource:
    """Validated label TIFF path and optional registration into its image pixels."""

    path: ArtifactFile
    to_image: Affine | None


def resolve_mask_files(
    root: DatasetRoot, labels: str | Path | Mapping[str, str | Path] | None, *, context: ReaderErrorContext
) -> dict[str | None, ArtifactFile]:
    """Resolve caller paths against the dataset root before discovering image candidates."""
    if labels is None:
        return {}
    if isinstance(labels, (str, Path)):
        return {None: require_caller_file(root, labels, context=context)}
    if not isinstance(labels, Mapping):
        message = "labels must be a file path or a mapping from region names to paths."
        raise TypeError(message)
    if not labels or any(not isinstance(region, str) or not region.strip() for region in labels):
        message = "labels requires at least one nonempty region name."
        raise ValueError(message)
    return {region: require_caller_file(root, path, context=context) for region, path in labels.items()}


def bind_masks(
    files: Mapping[str | None, ArtifactFile],
    labels_to_image: Affine | Mapping[str, Affine] | None,
    *,
    image_regions: Sequence[str],
    region_count: int,
) -> dict[str, MaskSource]:
    """Associate masks and optional XY registrations with the selected primary images."""
    if None in files:
        if region_count != 1 or len(image_regions) != 1:
            message = (
                "A single labels path requires one selected region with an image. For several regions use a mapping."
            )
            raise ValueError(message)
        named = {image_regions[0]: files[None]}
    else:
        named = {key: file for key, file in files.items() if key is not None}
    if unknown := named.keys() - image_regions:
        message = f"labels references regions without selected images: {sorted(unknown)}."
        raise ValueError(message)
    if labels_to_image is None:
        registrations: dict[str, Affine] = {}
    elif isinstance(labels_to_image, Affine):
        if len(named) != 1:
            message = "A single labels_to_image affine requires exactly one mask."
            raise ValueError(message)
        registrations = {next(iter(named)): labels_to_image}
    elif isinstance(labels_to_image, Mapping):
        if not labels_to_image or any(key not in named for key in labels_to_image):
            message = "labels_to_image keys must identify supplied masks."
            raise ValueError(message)
        registrations = dict(labels_to_image)
    else:
        message = "labels_to_image must be an XY Affine or a mapping from mask regions to affines."
        raise TypeError(message)
    for affine in registrations.values():
        validate_xy_affine(affine, option="labels_to_image")
    return {region: MaskSource(path, registrations.get(region)) for region, path in named.items()}


def read_mask(
    source: MaskSource, image: DataArray | DataTree, *, options: RasterOptions, context: ReaderErrorContext
) -> DataArray | DataTree:
    """Construct the full-resolution YX label grid and any requested nearest-neighbour pyramid."""
    metadata = inspect_tiff(source.path, context=context, include_pages=True)
    series = select_tiff_series(metadata, None, context=context)
    raster = read_tiff(metadata, series=series, level=0, context=context)
    try:
        pixels = normalize_raster_axes(raster.data, raster.axes, target_axes=("y", "x"))
    except ValueError as error:
        raise ReaderFormatError(
            context=context, path=source.path, found=str(error), expected="one YX mask plane"
        ) from error
    orientations = {page.orientation for page in metadata.pages or () if page.orientation is not None}
    if pixels.dtype.kind not in "iu" or orientations - {1}:
        raise ReaderFormatError(
            context=context,
            path=source.path,
            found=f"dtype {pixels.dtype}, declared TIFF orientations {sorted(orientations)}",
            expected="a top-left oriented integer instance mask",
        )
    base = image["scale0"] if isinstance(image, DataTree) else image
    if source.to_image is None and pixels.shape != (base.sizes["y"], base.sizes["x"]):
        raise ReaderFormatError(
            context=context,
            path=source.path,
            found=f"mask shape {pixels.shape} for image YX {(base.sizes['y'], base.sizes['x'])}",
            expected="equal extents on a caller-associated grid, or an explicit labels_to_image registration",
        )
    # SpatialData returns a mapping for get_all=True; its annotation combines both overloads.
    inherited = cast("dict[str, BaseTransformation]", get_transformation(image, get_all=True))
    transformations: dict[str, BaseTransformation] = {}
    declared = ome_pixel_sizes_um(metadata, series=series, axes=_XY, context=context)
    for frame, transform in inherited.items():
        composed = transform
        if source.to_image is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = transform.to_affine_matrix(_XY, _XY) @ source.to_image.matrix
            composed = Affine(matrix, input_axes=_XY, output_axes=_XY)
            validate_xy_affine(composed, option="composed mask registration")
        # Both Akoya readers reserve global and *_um for calibrated micrometre frames.
        if frame == "global" or frame.endswith("_um"):
            linear = composed.to_affine_matrix(_XY, _XY)[:2, :2]
            require_matching_pixel_sizes(
                declared,
                np.hypot(linear[0], linear[1]).tolist(),
                axes="xy",
                context=context,
                path=source.path,
                expected="mask OME physical sizes consistent with its registration",
            )
        transformations[frame] = composed
    options.validate_label_pyramid(pixels, element=f"mask {source.path.name!r}")
    return Labels2DModel.parse(
        pixels,
        dims=("y", "x"),
        transformations=transformations,
        chunks=options.label_chunks(),
        scale_factors=options.model_scale_factors,
    )
