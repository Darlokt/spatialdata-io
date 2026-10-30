"""Convert between SpatialData and the legacy spatial AnnData representation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from spatialdata import (
    SpatialData,
    get_centroids,
    get_extent,
    join_spatialelement_table,
    to_circles,
)
from spatialdata._core.operations._utils import transform_to_data_extent
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel, get_table_keys
from spatialdata.transformations import Identity, Scale

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata.transformations import BaseTransformation
    from xarray import DataArray, DataTree

_DOWNSCALED_HIRES_LENGTH = 2000
_DOWNSCALED_LOWRES_LENGTH = 600
_DEFAULT_SPOT_DIAMETER = 10


def to_legacy_anndata(
    sdata: SpatialData,
    coordinate_system: str | None = None,
    table_name: str | None = None,
    *,
    include_images: bool = False,
) -> AnnData:
    """Convert a SpatialData object to a (legacy) spatial AnnData object.

    This is useful for using packages expecting spatial information in AnnData, for example Scanpy and older versions
    of Squidpy. Using this format for any new package is not recommended.

    This function by default ignores images (recommended); setting the `include_images` parameter to `True` will
    include a downscaled version of the images in the output AnnData object.

    Parameters
    ----------
    sdata
        SpatialData object
    coordinate_system
        The coordinate system to consider. The AnnData object will be populated with the data transformed to this
        coordinate system.
    table_name
        The name of the table in the SpatialData object to consider. If None and the SpatialData object has only one
        table, that table will be used.
    include_images
        Keyword-only option. If True, includes downscaled versions of the images in the output AnnData object.
        It is recommended to handle the full resolution images separately and keep them in the SpatialData object.

    Returns
    -------
    The legacy spatial AnnData object.

    Raises
    ------
    ValueError
        If coordinate-system or table selection is missing or ambiguous, or the
        selected table does not annotate exactly one nonempty Shapes or Labels element.

    Notes
    -----
    Edge cases and limitations:

        - The Table can only annotate Shapes or Labels.
        - Labels will be approximated to circles by using the centroids of each label and an average approximated
          radius.
        - The Table cannot annotate more than one Shapes element. If multiple Shapes elements are present, please merge
          them into a single Shapes element and make the table annotate the new Shapes element.
        - Table rows not annotating geometries in the coordinate system will be dropped. Similarly, Shapes rows (in the
          case of Labels, the circle rows approximating the Labels) not annotated by Table rows will be dropped.

    How resolutions, coordinates, indices and Table rows will be affected:

        - When `include_images` is `True`, the Images will be scaled (see later) and will not maintain the original
          resolution.
        - The Shapes centroids will likely change to match the alignment in the specified coordinate system, and to the
          images bounding boxes when `include_images` is `True` (see below).
        - The Table and Shapes rows will have the same indices as before the conversion, which is useful for plugging
          results back to the SpatialData object, but as mentioned above, some rows may be dropped.

    When `include_images` is `True`, the generated AnnData object will contain low-resolution images and their origin
    will be reset to the pixel (0, 0). In particular, the ImageContainer class used by Squidpy is not used. Eventual
    transformations are applied before conversion. Precisely, this is how the Images are prepared for the AnnData
    object:

        - For each Image aligned to the specified coordinate system, the coordinate transformation to the coordinate
          system is applied; all the Images are rendered to a common target bounding box, which is the bounding box of
          the union of the Images aligned to the target coordinate system.
        - The origins of the new images will match the pixel (0, 0) of the target bounding box.
        - Each Image is downscaled (or upscaled) during rendering to fit a 2000x2000 pixels image (downscaled_hires) and
          a 600x600 pixels image (downscaled_lowres).
        - Practically the above implies that if the Images have very dissimilar locations, most the rendered Images will
          have empty spaces; in such cases it is recommended to drop or crop some images before the conversion.

    Matching of spatial coordinates and pixel coordinates:

        - When `include_images` is `True`, the coordinates in `obsm['spatial']` will match the downscaled (generated)
          image "hires" (i.e. coordinate (x, y) will match with the pixel (y, x) in the image). Please note that the
          "hires" naming is used for legacy reasons, but the image is actually downscaled and fairly low resolution.
        - The above matching is not perfect due to this bug: https://github.com/scverse/spatialdata/issues/165, which
          will eventually be addressed.

    Imperfect downscaling:

        - Due to the bug mentioned above, https://github.com/scverse/spatialdata/issues/165, when the downscaling factor
          is large, the error between the original and the downscale image can be significant. This is another reason to
          not recommend using include_images=True. Note, however, that the error can be large only between the original
          and the downscaled images, not between the downscaled images and the new spatial coordinates. Thus the use
          may be acceptable for some use cases.

    Final consideration:

        - Due to the legacy nature of the AnnData object generated, this function may not handle all the edge cases,
          please report any issue you may find.
    """
    sdata, coordinate_system, table_name = _select_table(sdata, coordinate_system, table_name)
    geometry_sdata, region_name = _linked_circles(sdata, table_name)
    if include_images:
        hires, lowres = _render_images(sdata, geometry_sdata, coordinate_system)
        adata = hires.tables[table_name].copy()
        _add_images(adata, hires, lowres, region_name)
        geometry_sdata = hires
    else:
        adata = geometry_sdata.tables[table_name].copy()
    adata.obsm["spatial"] = (
        get_centroids(geometry_sdata.shapes[region_name], coordinate_system=coordinate_system).compute().to_numpy()
    )
    return adata


def _select_table(
    sdata: SpatialData, coordinate_system: str | None, table_name: str | None
) -> tuple[SpatialData, str, str]:
    coordinate_systems = sdata.coordinate_systems
    if coordinate_system is None:
        if len(coordinate_systems) != 1:
            msg = "The SpatialData object must have exactly one coordinate system. Please specify one."
            raise ValueError(msg)
        coordinate_system = coordinate_systems[0]
    elif coordinate_system not in coordinate_systems:
        msg = f"The SpatialData object does not have the coordinate system {coordinate_system}."
        raise ValueError(msg)
    sdata = sdata.filter_by_coordinate_system(coordinate_system)
    if table_name is None:
        if len(sdata.tables) != 1:
            msg = (
                "When the table_name is not specified, the SpatialData object must have exactly one table. "
                "The specified "
                f"sdata object, after being filtered for the coordinate system {coordinate_system}, has "
                f"{len(sdata.tables)} tables."
            )
            raise ValueError(msg)
        table_name = next(iter(sdata.tables))
    elif table_name not in sdata.tables:
        msg = f"The table {table_name} is not present in the SpatialData object."
        raise ValueError(msg)
    return sdata, coordinate_system, table_name


def _linked_circles(sdata: SpatialData, table_name: str) -> tuple[SpatialData, str]:
    table = sdata.tables[table_name]
    region, _, _ = get_table_keys(table)
    regions = region if isinstance(region, list) else [region]
    if len(regions) != 1:
        msg = f"The table needs to annotate exactly one element. Found {len(regions)}."
        raise ValueError(msg)
    region_name = regions[0]
    if region_name not in sdata.shapes and region_name not in sdata.labels:
        msg = "The table needs to annotate a Shapes or Labels element, not Points."
        raise ValueError(msg)
    # Labels and polygonal shapes use the same legacy circle approximation.
    shapes = to_circles(sdata[region_name])
    circles_sdata = SpatialData(tables={table_name: table}, shapes={region_name: shapes})
    joined_elements, new_table = join_spatialelement_table(
        sdata=circles_sdata, spatial_element_names=region_name, table_name=table_name, how="inner", match_rows="left"
    )
    if len(new_table) == 0:
        msg = (
            "The table does not annotate any geometry in the Shapes element. This could also be caused by a mismatch "
            "between the type of the indices of the geometries and the type of the INSTANCE_KEY column of the table."
        )
        raise ValueError(msg)
    return SpatialData(tables={table_name: new_table}, shapes={region_name: joined_elements[region_name]}), region_name


def _render_images(
    sdata: SpatialData, geometry_sdata: SpatialData, coordinate_system: str
) -> tuple[SpatialData, SpatialData]:
    sdata_images = sdata.subset(element_names=list(sdata.images))
    for image_name, image in sdata_images.images.items():
        geometry_sdata[image_name] = image
    bounding_box = get_extent(sdata_images, coordinate_system=coordinate_system)
    parameters = {"x": "target_width", "y": "target_height", "z": "target_depth"}
    longest_side = max(bounding_box, key=bounding_box.__getitem__)
    hires = transform_to_data_extent(
        geometry_sdata,
        coordinate_system=coordinate_system,
        maintain_positioning=False,
        **{parameters[longest_side]: _DOWNSCALED_HIRES_LENGTH},
    )
    lowres = transform_to_data_extent(
        geometry_sdata,
        coordinate_system=coordinate_system,
        maintain_positioning=False,
        **{parameters[longest_side]: _DOWNSCALED_LOWRES_LENGTH},
    )
    return hires, lowres


def _add_images(adata: AnnData, hires: SpatialData, lowres: SpatialData, region_name: str) -> None:
    for image_name in hires.images:
        if "spatial" not in adata.uns:
            spatial_metadata: dict[str, object] = {}
            adata.uns["spatial"] = spatial_metadata
        # transform_to_data_extent produces single-scale images; SpatialData.images
        # retains a DataArray | DataTree annotation for its general container contract.
        hires_image = cast("DataArray", hires.images[image_name])
        lowres_image = cast("DataArray", lowres.images[image_name])
        adata.uns["spatial"][image_name] = {
            "images": {
                "hires": hires_image.data.compute().transpose([1, 2, 0]),
                "lowres": lowres_image.data.compute().transpose([1, 2, 0]),
            }
        }
        with suppress(KeyError):
            adata.uns["spatial"][image_name]["scalefactors"] = {
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": _DOWNSCALED_LOWRES_LENGTH / _DOWNSCALED_HIRES_LENGTH,
                "spot_diameter_fullres": hires.shapes[region_name]["radius"].iloc[0] * 2,
            }


def from_legacy_anndata(adata: AnnData, *, rgb: bool | None = None) -> SpatialData:
    """Convert (legacy) spatial AnnData object to SpatialData object.

    This is useful for parsing a (legacy) spatial AnnData object, for example the ones produced by Scanpy and older
    version of Squidpy.

    Parameters
    ----------
    adata
        (legacy) spatial AnnData object
    rgb
        Keyword-only argument passed to `spatialdata.models.Image2DModel.parse()`. If `None`,
        3-(4) channels images will be interpreted as RGB(A).

    Returns
    -------
    The SpatialData object

    Raises
    ------
    TypeError
        If spatial coordinates are present and observation metadata is not a pandas DataFrame.

    Notes
    -----
    The SpatialData object will have one hires and one lowres image for each dataset in the AnnData object.
    """
    images, transformations, diameters = _legacy_images(adata, rgb=rgb)
    diameter = _spot_diameter(diameters)
    shapes = {}
    new_table = adata.copy()
    if "spatial" in adata.obsm:
        shapes["locations"] = ShapesModel.parse(
            adata.obsm["spatial"], geometry=0, radius=diameter / 2, transformations=transformations
        )
        new_table.uns.pop(TableModel.ATTRS_KEY, None)
        if not isinstance(new_table.obs, pd.DataFrame):
            msg = "Legacy spatial conversion requires pandas DataFrame observation metadata."
            raise TypeError(msg)
        new_table.obs["region"] = "locations"
        new_table.obs["region"] = new_table.obs["region"].astype("category")
        new_table.obs["instance_id"] = shapes["locations"].index.to_numpy()
        new_table = TableModel.parse(new_table, region="locations", region_key="region", instance_key="instance_id")
    return SpatialData(tables={"table": new_table}, images=images, shapes=shapes)


def _legacy_images(
    adata: AnnData, *, rgb: bool | None
) -> tuple[dict[str, DataArray | DataTree], dict[str, BaseTransformation], list[float]]:
    images: dict[str, DataArray | DataTree] = {}
    transformations: dict[str, BaseTransformation] = {}
    diameters: list[float] = []
    for dataset_id, dataset in adata.uns.get("spatial", {}).items():
        keys = set(dataset.keys())
        scalefactors_raw = dataset.get("scalefactors", {}) if isinstance(dataset, Mapping) else {}
        scalefactors = scalefactors_raw if isinstance(scalefactors_raw, Mapping) else {}
        if "spot_diameter_fullres" in scalefactors:
            diameters.append(scalefactors["spot_diameter_fullres"])
        image_data = dataset.get("images", {}) if "images" in keys and isinstance(dataset, Mapping) else {}
        if not isinstance(image_data, Mapping):
            continue
        for image_key, image_value in image_data.items():
            if image_key not in ("hires", "lowres"):
                logger.warning(
                    "Found non-standard image key '%s' in dataset '%s' - attempting to parse.", image_key, dataset_id
                )
            scalefactor = _scale_factor(scalefactors, dataset_id, image_key)
            image_name = f"{dataset_id}_{image_key}"
            images[image_name] = Image2DModel.parse(
                image_value, dims=("y", "x", "c"), transformations={image_name: Identity()}, rgb=rgb
            )
            transformations[image_name] = Scale([scalefactor, scalefactor], axes=("x", "y"))
    return images, transformations, diameters


def _scale_factor(scalefactors: Mapping[str, float | None], dataset_id: str, image_key: str) -> float:
    # A direct image-name entry takes precedence, including an explicit None.
    source = image_key if image_key in scalefactors else f"tissue_{image_key}_scalef"
    scalefactor = scalefactors.get(source)
    if scalefactor is None:
        logger.warning(
            "Scale factor missing for image '%s' in dataset '%s'; defaulting to %s", image_key, dataset_id, 1.0
        )
        return 1.0
    logger.debug("Using scalefactor '%s' for image '%s' in dataset '%s'", source, image_key, dataset_id)
    return scalefactor


def _spot_diameter(diameters: list[float]) -> float:
    if not diameters:
        logger.warning("spot_diameter_fullres is not present. Using %s as default value.", _DEFAULT_SPOT_DIAMETER)
        return _DEFAULT_SPOT_DIAMETER
    values = np.array(diameters)
    if not np.allclose(values, values[0]):
        logger.warning("spot_diameter_fullres is not constant across datasets. Using the average value.")
        return values.mean()
    return values[0]
