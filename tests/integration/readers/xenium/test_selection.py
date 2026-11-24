"""Register the three coordinates exports of one Xenium Explorer annotation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Affine, BaseTransformation, get_transformation

from spatialdata_io import xenium_aligned_image, xenium_explorer_selection

if TYPE_CHECKING:
    from pathlib import Path

# Exported from a rectangle drawn in 10x's public Human FFPE Kidney demo.
# The demo URL also supplies the H&E image-to-morphology matrix, and its
# experiment.xenium gives 0.2125 µm per morphology pixel.
# https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma
CSV = (
    "#Selection names: Selection 1\n"
    "#Areas (µm^2): 65449.7\n"
    "Selection,X,Y,Class,Color\n"
    "Selection 1,3808.328166878173,2159.4504676395945,Unclassified,#c2c2c2\n"
    "Selection 1,4077.998217639594,2159.4504676395945,Unclassified,#c2c2c2\n"
    "Selection 1,4077.998217639594,2402.1535133248735,Unclassified,#c2c2c2\n"
    "Selection 1,3808.328166878173,2402.1535133248735,Unclassified,#c2c2c2\n"
    "Selection 1,3808.328166878173,2159.4504676395945,Unclassified,#c2c2c2\n"
)
MORPHOLOGY = """{"type": "FeatureCollection", "features": [{
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "plane": {"z": 0},
        "coordinates": [[
            [17921.544314720813, 10162.11984771574],
            [19190.579847715737, 10162.11984771574],
            [19190.579847715737, 11304.25182741117],
            [17921.544314720813, 11304.25182741117],
            [17921.544314720813, 10162.11984771574]
        ]]
    },
    "properties": {
        "classification": {"color": [194, 194, 194], "name": "Fixture class"},
        "color": [194, 194, 194],
        "name": "Selection 1",
        "objectType": "annotation"
    }
}]}"""
HE = """{"type": "FeatureCollection", "features": [{
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [14594.709476513792, 23086.95611219732],
            [14597.01951179868, 24071.002464247285],
            [13711.37779495371, 24073.081496003684],
            [13709.067759668822, 23089.03514395372],
            [14594.709476513792, 23086.95611219732]
        ]]
    },
    "properties": {
        "color": [194, 194, 194],
        "name": "Selection 1",
        "objectType": "annotation"
    }
}]}"""
PIXEL_SIZE = 0.2125
ALIGNMENT = (
    "0.0030273239131817036,1.2896024024947854,-11895.632666864709\n"
    "-1.2896024024947854,0.0030273239131817036,28913.600558020305\n"
    "0.0,0.0,1.0\n"
)


def test_matching_explorer_exports_survive_round_trip(tmp_path: Path) -> None:
    csv_file = tmp_path / "annotation.csv"
    csv_file.write_text(CSV, encoding="utf-8")
    csv = xenium_explorer_selection(csv_file)

    # Only image metadata is needed to obtain the transform. These tiny TIFFs
    # stand in for the original public images, using the demo's registrations.
    pixels = np.zeros((16, 16), dtype=np.uint16)
    tiff = tmp_path / "source.ome.tif"
    tifffile.imwrite(tiff, pixels, ome=True, metadata={"axes": "YX"})
    alignment_file = tmp_path / "he_imagealignment.csv"
    alignment_file.write_text(ALIGNMENT)
    morphology_image = xenium_aligned_image(tiff, pixel_size=PIXEL_SIZE, already_registered=True)
    he_image = xenium_aligned_image(tiff, alignment_file, pixel_size=PIXEL_SIZE)

    path = tmp_path / "morphology.geojson"
    path.write_text(MORPHOLOGY)
    morphology_to_global = get_transformation(morphology_image)
    assert isinstance(morphology_to_global, BaseTransformation)
    morphology = xenium_explorer_selection(path, image_to_global=morphology_to_global)
    path = tmp_path / "he.geojson"
    path.write_text(HE)
    he_to_global = get_transformation(he_image)
    assert isinstance(he_to_global, BaseTransformation)
    he = xenium_explorer_selection(path, image_to_global=he_to_global)

    output = tmp_path / "annotation.zarr"
    SpatialData(shapes={"morphology": morphology, "he": he}).write(output)
    reloaded = read_zarr(output)

    assert reloaded.shapes["morphology"]["annotation_class"].tolist() == ["Fixture class"]
    assert reloaded.shapes["morphology"]["annotation_class_color"].tolist() == ["#c2c2c2"]
    assert reloaded.shapes["morphology"]["z_plane"].tolist() == [0]
    target = shapely.get_coordinates(csv.geometry)
    for name, original in (("morphology", morphology), ("he", he)):
        shapes = reloaded.shapes[name]
        assert shapes.index.tolist() == ["Selection 1"]
        assert shapes["color"].tolist() == ["#c2c2c2"]
        coords = shapely.get_coordinates(shapes.geometry)
        np.testing.assert_array_equal(coords, shapely.get_coordinates(original.geometry))
        affine = get_transformation(shapes, to_coordinate_system="global")
        assert isinstance(affine, Affine)
        matrix = affine.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_allclose(coords @ matrix[:2, :2].T + matrix[:2, 2], target, rtol=0, atol=1e-9)
