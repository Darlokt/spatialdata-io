from __future__ import annotations

import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from dask_image.imread import imread
from geopandas import GeoDataFrame, GeoSeries
from scipy.sparse import coo_matrix
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from tqdm import tqdm

from spatialdata_io._constants._constants import StereoseqKeys as SK
from spatialdata_io._docs import inject_docs
from spatialdata_io.readers._utils.provenance import set_reader_provenance

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["stereoseq"]


@inject_docs(xx=SK)
def stereoseq(
    path: str | Path,
    dataset_id: str | None = None,
    read_square_bin: bool = True,
    optional_tif: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *Stereo-seq* formatted dataset.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    dataset_id
        Dataset identifier. If not given will be determined automatically.
    read_square_bin
        If True, will read the square bin ``{xx.GEF_FILE!r}`` file and build corresponding points element.
    optional_tif
        If True, will read ``{xx.TISSUE_TIF!r}`` files.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`

    Notes
    -----
    The cell segmentation, which encodes the background as 0 and the cells as 1, is parsed as an image (i.e. (c, y, x))
    object and not as labels object (i.e. (y, x)). If you want to visualize this binary image with napari you will
    have to adjust the color limit to be able to see the cells.
    """
    image_models_kwargs = dict(image_models_kwargs)
    image_models_kwargs.setdefault("chunks", (1, 4096, 4096))
    image_models_kwargs.setdefault("scale_factors", [2, 2, 2, 2])
    path = Path(path)

    if dataset_id is None:
        dataset_id_path = path / SK.COUNT_DIRECTORY
        gef_files = [filename for filename in os.listdir(dataset_id_path) if filename.endswith(SK.GEF_FILE)]

        if gef_files:
            first_gef_file = gef_files[0]
            square_bin = str(first_gef_file.split(".")[0]) + SK.GEF_FILE
        else:
            raise ValueError(f"No {SK.GEF_FILE!r} files found in {dataset_id_path!r}")
    else:
        square_bin = dataset_id + SK.GEF_FILE

    image_patterns = [
        re.compile(r".*" + re.escape(SK.MASK_TIF)),
        re.compile(r".*" + re.escape(SK.REGIST_TIF)),
    ]
    if optional_tif:
        image_patterns.append(
            re.compile(r".*" + re.escape(SK.TISSUE_TIF)),
        )

    gef_patterns = [
        re.compile(r".*" + re.escape(SK.RAW_GEF)),
        re.compile(r".*" + re.escape(SK.CELLBIN_GEF)),
        re.compile(r".*" + re.escape(SK.TISSUECUT_GEF)),
        re.compile(r".*" + re.escape(square_bin)),
    ]

    image_filenames = [i for i in os.listdir(path / SK.REGISTER) if any(pattern.match(i) for pattern in image_patterns)]
    cell_mask_file = [x for x in image_filenames if (f"{SK.MASK_TIF}" in x)]
    image_filenames = [x for x in image_filenames if x not in cell_mask_file]

    cellbin_gef_filename = [
        i for i in os.listdir(path / SK.CELLCUT) if any(pattern.match(i) for pattern in gef_patterns)
    ]
    squarebin_gef_filename = [
        i for i in os.listdir(path / SK.TISSUECUT) if any(pattern.match(i) for pattern in gef_patterns)
    ]

    table_pattern = re.compile(r".*" + re.escape(SK.CELL_CLUSTER_H5AD))
    table_filename = [i for i in os.listdir(path / SK.CELLCLUSTER) if table_pattern.match(i)]

    # create table using .h5ad and cellbin.gef
    adata = ad.read_h5ad(path / SK.CELLCLUSTER / table_filename[0])
    path_cellbin = path / SK.CELLCUT / cellbin_gef_filename[0]
    with h5py.File(path_cellbin, "r") as cellbin_gef:
        cell_bin = cellbin_gef[SK.CELL_BIN]

        # add cell info to obs
        obs = pd.DataFrame(cell_bin[SK.CELL_DATASET][:])
        obs.columns = [
            SK.CELL_ID,
            SK.COORD_X,
            SK.COORD_Y,
            SK.OFFSET,
            SK.GENE_COUNT,
            SK.EXP_COUNT,
            SK.DNBCOUNT,
            SK.CELL_AREA,
            SK.CELL_TYPE_ID,
            SK.CLUSTER_ID,
        ]
        obs[SK.CELL_EXON] = cell_bin[SK.CELL_EXON][:]

        # add centroids to obsm
        obsm_spatial = obs[[SK.COORD_X, SK.COORD_Y]].to_numpy()
        obs = obs.drop([SK.COORD_X, SK.COORD_Y], axis=1)
        adata.obsm[SK.SPATIAL_KEY] = obsm_spatial

        # add gene info to var
        var = pd.DataFrame(
            cell_bin[SK.FEATURE_KEY][:],
            columns=[
                SK.GENE_NAME,
                SK.OFFSET,
                SK.CELL_COUNT,
                SK.EXP_COUNT,
                SK.MAX_MID_COUNT,
            ],
        )
        var[SK.GENE_NAME] = var[SK.GENE_NAME].str.decode("utf-8")
        var[SK.GENE_EXON] = cell_bin[SK.GENE_EXON][:]

        # add all leftover columns in cellbin which don't fit .obs or .var to uns
        cell_exp = cell_bin[SK.CELL_EXP]
        gene_exp = cell_bin[SK.GENE_EXP]
        cellbin_uns = {
            SK.GENE_ID: cell_exp[SK.GENE_ID][:],
            SK.CELL_EXP: cell_exp[SK.COUNT][:],
            SK.GENE_EXP_EXON: cell_bin[SK.CELL_EXP_EXON][:],
            SK.CELL_ID: gene_exp[SK.CELL_ID][:],
            SK.GENE_EXP: gene_exp[SK.COUNT][:],
            SK.GENE_EXP_EXON: cell_bin[SK.GENE_EXP_EXON][:],
        }
        adata.uns["cellBin_cell_gene_exon_exp"] = pd.DataFrame(cellbin_uns)
        adata.uns["cellBin_blockIndex"] = cell_bin[SK.BLOCK_INDEX][:]
        adata.uns["cellBin_blockSize"] = cell_bin[SK.BLOCK_SIZE][:]
        adata.uns["cellBin_cellTypeList"] = cell_bin[SK.CELL_TYPE_LIST][:]
        # add cellbin attrs to uns
        adata.uns["cellBin_attrs"] = dict(cellbin_gef.attrs)
        cell_border = cell_bin[SK.CELL_BORDER][:]

    # merge columns of obs and var to adata.obs and adata.var
    if not isinstance(adata.obs, pd.DataFrame) or not isinstance(adata.var, pd.DataFrame):
        raise TypeError("Stereo-seq annotation data must expose pandas DataFrame 'obs' and 'var' tables.")
    obs.index = adata.obs.index
    adata.obs = pd.merge(adata.obs, obs, left_index=True, right_index=True)
    var.index = adata.var.index
    adata.var = pd.merge(adata.var, var, left_index=True, right_index=True)

    # add region and instance_id to obs for the TableModel
    adata.obs[SK.REGION_KEY] = f"{SK.REGION}_circles"
    adata.obs[SK.REGION_KEY] = adata.obs[SK.REGION_KEY].astype("category")
    adata.obs[SK.INSTANCE_KEY] = adata.obs.index

    # let's correct the dtype for some columns
    for column_name in [SK.CELL_TYPE_ID, SK.CLUSTER_ID]:
        adata.obs[column_name] = adata.obs[column_name].astype("category")

    images = {
        Path(name).stem: Image2DModel.parse(
            imread(path / SK.REGISTER / name, **imread_kwargs), dims=("c", "y", "x"), **image_models_kwargs
        )
        for name in image_filenames
    }

    cells_table = TableModel.parse(
        adata,
        region=f"{SK.REGION.value}_circles",
        region_key=SK.REGION_KEY.value,
        instance_key=SK.INSTANCE_KEY.value,
    )
    tables = {f"{SK.REGION}_table": cells_table}

    radii = np.sqrt(adata.obs[SK.CELL_AREA].to_numpy() / np.pi)
    shapes = {
        f"{SK.REGION}_circles": ShapesModel.parse(
            adata.obsm[SK.SPATIAL_KEY], geometry=0, radius=radii, index=adata.obs[SK.INSTANCE_KEY]
        )
    }
    shapes[f"{SK.REGION}_circles"].index.name = None
    points = {}

    if read_square_bin:
        # create points model using SquareBin.gef
        path_squarebin = path / SK.TISSUECUT / squarebin_gef_filename[0]
        with h5py.File(path_squarebin, "r") as squarebin_gef:
            gene_expression = squarebin_gef[SK.GENE_EXP]
            for i in gene_expression.keys():
                bin_attrs = dict(gene_expression[i][SK.EXPRESSION].attrs)
                # this is the center to center distance between bins and could be used to calculate the radius of the
                # circles (or side of the square) to represent the bins as circles (squares)
                bin_attrs[SK.RESOLUTION].item()
                # get gene info
                arr = gene_expression[i][SK.FEATURE_KEY][:]
                df_gene = pd.DataFrame(arr, columns=[SK.FEATURE_KEY, SK.OFFSET, SK.COUNT])
                df_gene[SK.FEATURE_KEY] = df_gene[SK.FEATURE_KEY].str.decode("utf-8")

                # create df for points model
                arr = gene_expression[i][SK.EXPRESSION][:]
                df_points = pd.DataFrame(arr, columns=[SK.COORD_X, SK.COORD_Y, SK.COUNT])
                # the exon information is skipped for the moment, but it could be added analogously to what is done for the
                # 'count' column; in such a case it should be added in a separate table (one per bin)
                # df_points[SK.EXON] = squarebin_gef[SK.GENE_EXP][i][SK.EXON][:]

                # check that the column 'offset' is redundant with information in 'count'
                expected_offsets = np.insert(np.cumsum(df_gene[SK.COUNT]), 0, 0)[:-1]
                if not np.array_equal(df_gene[SK.OFFSET], expected_offsets):
                    raise ValueError(
                        f"Invalid Stereo-seq square-bin GEF {path_squarebin}: feature offsets do not match counts."
                    )
                # unroll gene names by count such that there exists a mapping between coordinate counts and gene names
                df_points[SK.FEATURE_KEY] = [
                    name
                    for _, (name, cell_count) in df_gene[[SK.FEATURE_KEY, SK.COUNT]].iterrows()
                    for _ in range(cell_count)
                ]
                df_points[SK.FEATURE_KEY] = df_points[SK.FEATURE_KEY].astype("category")
                # this is unique for a given bin; also the "wholeExp" information (not parsed here) may use more bins than
                # the ones used for the gene expression, so ids constructed from there are different from the ones
                # constructed here from "geneExp" (in fact max_y would likely be different, leading to a different set of
                # bin ids)
                # df_points["bin_id"] = df_points[SK.COORD_Y] + df_points[SK.COORD_X] * (max_y + 1)
                points_coords = df_points[[SK.COORD_X, SK.COORD_Y]].copy()
                points_coords.drop_duplicates(inplace=True)
                points_coords.reset_index(inplace=True, drop=True)
                points_coords["bin_id"] = points_coords.index

                name_points_element = f"{i}_genes"
                name_table_element = f"{i}_table"
                index_to_bin_id = pd.merge(
                    df_points[[SK.COORD_X, SK.COORD_Y]],
                    points_coords,
                    on=[SK.COORD_X, SK.COORD_Y],
                    how="left",
                    validate="many_to_one",
                )

                obs = pd.DataFrame({SK.INSTANCE_KEY: points_coords.index, SK.REGION_KEY: name_points_element})
                obs[SK.REGION_KEY] = obs[SK.REGION_KEY].astype("category")

                expression = coo_matrix(
                    (
                        df_points[SK.COUNT].to_numpy(),
                        (
                            index_to_bin_id.loc[df_points.index]["bin_id"].to_numpy(),
                            df_points[SK.FEATURE_KEY].cat.codes.to_numpy(),
                        ),
                    ),
                    shape=(len(points_coords), len(df_points[SK.FEATURE_KEY].cat.categories)),
                ).tocsr()

                points_coords.drop(columns=["bin_id"], inplace=True)
                # it would be more natural to use shapes, but for performance reasons we use points
                points_element = PointsModel.parse(
                    points_coords,
                    coordinates={"x": SK.COORD_X, "y": SK.COORD_Y},
                )

                # add more gene info to var
                df_gene = df_gene.set_index(SK.FEATURE_KEY)
                df_gene.index.name = None
                df_gene = df_gene.loc[df_points[SK.FEATURE_KEY].cat.categories, :]
                adata = ad.AnnData(expression, obs=obs, var=df_gene)

                table = TableModel.parse(
                    adata,
                    region=name_points_element,
                    region_key=SK.REGION_KEY.value,
                    instance_key=SK.INSTANCE_KEY.value,
                )

                tables[name_table_element] = table
                points[name_points_element] = points_element

    # add cell border shapes element
    cell_coordinates = [x for i in range(1, 33) for x in ("x_" + str(i), "y_" + str(i))]
    n_cells, n_coords, xy = cell_border.shape
    new_arr = cell_border.reshape(n_cells, n_coords * xy)
    df_coords = pd.DataFrame(new_arr, columns=cell_coordinates, index=cells_table.obs.index)

    x_original = shapes[f"{SK.REGION}_circles"].geometry.centroid.x
    y_original = shapes[f"{SK.REGION}_circles"].geometry.centroid.y

    x_coords = df_coords.filter(regex="x_")
    y_coords = df_coords.filter(regex="y_")
    polygons = []
    for (x_index, x_row), (y_index, y_row) in tqdm(
        zip(x_coords.iterrows(), y_coords.iterrows(), strict=False), desc="creating polygons", total=len(df_coords)
    ):
        if x_index != y_index:
            raise ValueError(
                f"Invalid Stereo-seq cell borders in {path_cellbin}: x/y coordinate rows are not aligned "
                f"({x_index!r} != {y_index!r})."
            )
        # the polygonal cells coordinates are stored as offsets from the centroids, so let's add the centroids
        x = x_row[x_row != int(SK.PADDING_VALUE)]
        y = y_row[y_row != int(SK.PADDING_VALUE)]
        x = x + x_original[x_index]
        y = y + y_original[y_index]
        if len(x) != len(y):
            raise ValueError(
                f"Invalid Stereo-seq cell border for {x_index!r} in {path_cellbin}: "
                f"found {len(x)} x coordinates and {len(y)} y coordinates."
            )
        xy_pairs = np.vstack((x, y)).T
        polygon = Polygon(xy_pairs)
        polygons.append(polygon)
    gs = GeoSeries(polygons, index=df_coords.index)
    polygons_gdf = GeoDataFrame(geometry=gs)
    polygons_gdf = ShapesModel.parse(polygons_gdf)
    shapes[f"{SK.REGION}_polygons"] = polygons_gdf

    for cell_mask_name in cell_mask_file:
        masks = imread(path / SK.REGISTER / cell_mask_name, **imread_kwargs)
        masks = Image2DModel.parse(
            masks,
            dims=("c", "y", "x"),
            **image_models_kwargs,
        )
        images[Path(cell_mask_name).stem] = masks

    sdata = SpatialData(images=images, tables=tables, shapes=shapes, points=points)
    set_reader_provenance(sdata.attrs, reader="stereoseq")
    return sdata
