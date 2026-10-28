from enum import unique

from spatialdata_io._constants._enum import ModeEnum


@unique
class VisiumHDKeys(ModeEnum):
    """Keys for *10X Genomics Visium hd* formatted dataset."""

    # directories
    SPATIAL = "spatial"
    DEFAULT_BIN = "square_008um"
    BIN_PREFIX = "square_"
    MICROSCOPE_IMAGE = "microscope_image"
    BINNED_OUTPUTS = "binned_outputs"
    SEGMENTATION_OUTPUTS = "segmented_outputs"

    # counts and locations files
    FILTERED_COUNTS_FILE = "filtered_feature_bc_matrix.h5"
    RAW_COUNTS_FILE = "raw_feature_bc_matrix.h5"
    TISSUE_POSITIONS_FILE = "tissue_positions.parquet"
    BARCODE_MAPPINGS_FILE = "barcode_mappings.parquet"
    FILTERED_CELL_COUNTS_FILE = "filtered_feature_cell_matrix.h5"
    CELL_SEGMENTATION_GEOJSON_PATH = "cell_segmentations.geojson"
    NUCLEUS_SEGMENTATION_GEOJSON_PATH = "nucleus_segmentations.geojson"

    # images
    IMAGE_HIRES_FILE = "tissue_hires_image.png"
    IMAGE_LOWRES_FILE = "tissue_lowres_image.png"
    IMAGE_CYTASSIST = "cytassist_image.tiff"

    # scalefactors
    SCALEFACTORS_FILE = "scalefactors_json.json"

    # scalefactors keys
    SCALEFACTORS_HIRES = "tissue_hires_scalef"
    SCALEFACTORS_LOWRES = "tissue_lowres_scalef"
    SCALEFACTORS_SPOT_DIAMETER_FULLRES = "spot_diameter_fullres"
    SCALEFACTORS_BIN_SIZE_UM = "bin_size_um"
    SCALEFACTORS_MICRONS_PER_PIXEL = "microns_per_pixel"

    # locations keys
    LOCATIONS_X = "pxl_col_in_fullres"
    LOCATIONS_Y = "pxl_row_in_fullres"
    BARCODE = "barcode"
    IN_TISSUE = "in_tissue"
    ARRAY_ROW = "array_row"
    ARRAY_COL = "array_col"

    # table keys
    REGION_KEY = "region"
    INSTANCE_KEY = "location_id"

    # feature slice file (it contains transformation matrices in the metadata)
    FEATURE_SLICE_FILE = "feature_slice.h5"

    # METADATA_KEYS
    METADATA_JSON = "metadata_json"
    HD_LAYOUT_JSON = "hd_layout_json"
    TRANSFORM = "transform"
    TRANSFORM_MATRICES = "transform_matrices"
    CYTASSIST_COLROW_TO_SPOT_COLROW = ("cytassist_colrow_to_spot_colrow",)
    SPOT_COLROW_TO_CYTASSIST_COLROW = ("spot_colrow_to_cytassist_colrow",)
    MICROSCOPE_COLROW_TO_SPOT_COLROW = ("microscope_colrow_to_spot_colrow",)
    SPOT_COLROW_TO_MICROSCOPE_COLROW = ("spot_colrow_to_microscope_colrow",)
    FILE_FORMAT = "file_format"

    # Cell Segmentation keys
    CELL_SEG_KEY_HD = "cell_segmentations"
    NUCLEUS_SEG_KEY_HD = "nucleus_segmentations"
