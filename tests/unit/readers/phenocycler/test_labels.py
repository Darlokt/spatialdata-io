"""Check regional mask selection, TIFF grids and coordinate registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import tifffile
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine, Scale, get_transformation

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import normalize_dataset_root, require_file
from spatialdata_io.readers.phenocycler._labels import MaskSource, bind_masks, read_mask, resolve_mask_files

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata.transformations import BaseTransformation

_CONTEXT = ReaderErrorContext("phenocycler", "segmentation mask")
_XY = ("x", "y")


class TestResolveMaskFiles:
    def test_relative_region_paths(self, tmp_path: Path) -> None:
        root = normalize_dataset_root(tmp_path, path_policy="strict", context=_CONTEXT)
        path = tmp_path / "cells.tif"
        path.touch()
        assert resolve_mask_files(root, {"A": "cells.tif"}, context=_CONTEXT) == {"A": path}
        assert resolve_mask_files(root, path, context=_CONTEXT) == {None: path}

    @pytest.mark.parametrize("labels", [[], {}, {"": "mask"}])
    def test_invalid_associations(self, tmp_path: Path, labels: object) -> None:
        root = normalize_dataset_root(tmp_path, path_policy="strict", context=_CONTEXT)
        with pytest.raises((ValueError, TypeError), match="labels"):
            resolve_mask_files(root, cast("dict[str, str]", labels), context=_CONTEXT)


class TestBindMasks:
    def test_registrations_only_for_supplied_selected_images(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.tif"
        path.touch()
        mask = require_file(path, context=_CONTEXT)
        affine = Affine(np.diag([2, 2, 1]), _XY, _XY)
        masks = bind_masks({"A": mask, "B": mask}, {"B": affine}, image_regions=["A", "B"], region_count=2)
        assert masks == {"A": MaskSource(mask, None), "B": MaskSource(mask, affine)}
        with pytest.raises(ValueError, match="regions without selected images"):
            bind_masks({"C": mask}, None, image_regions=["A", "B"], region_count=2)
        with pytest.raises(ValueError, match="single labels path"):
            bind_masks({None: mask}, None, image_regions=["B"], region_count=2)
        with pytest.raises(ValueError, match="keys must identify"):
            bind_masks({"B": mask}, {"A": affine}, image_regions=["B"], region_count=1)
        with pytest.raises(ValueError, match="exactly one mask"):
            bind_masks({}, affine, image_regions=["A"], region_count=1)
        with pytest.raises(TypeError, match="XY Affine"):
            bind_masks({"A": mask}, cast("Affine", []), image_regions=["A"], region_count=1)


class TestReadMask:
    @pytest.mark.parametrize(
        ("shape", "dtype", "orientation", "message"),
        [
            ((10, 8), "float32", 1, "integer instance"),
            ((10, 8), "uint16", 4, "top-left"),
            ((2, 10, 8), "uint16", 1, "one YX mask plane"),
            ((5, 4), "uint16", 1, "equal extents"),
        ],
    )
    def test_rejects_unsupported_grids(
        self, tmp_path: Path, shape: tuple[int, ...], dtype: str, orientation: int, message: str
    ) -> None:
        image = Image2DModel.parse(np.zeros((1, 10, 8)), dims=("c", "y", "x"))
        path = tmp_path / "cells.tif"
        tifffile.imwrite(
            path, np.zeros(shape, dtype=dtype), photometric="minisblack", extratags=[(274, "H", 1, orientation, False)]
        )
        with pytest.raises(ReaderFormatError, match=message):
            read_mask(
                MaskSource(require_file(path, context=_CONTEXT), None),
                image,
                options=make_raster_options(),
                context=_CONTEXT,
            )

    def test_measured_crop_and_ome_spacing(self, tmp_path: Path) -> None:
        image = Image2DModel.parse(
            np.zeros((1, 10, 8)), dims=("c", "y", "x"), transformations={"global": Scale([0.5, 0.25], axes=_XY)}
        )
        path = tmp_path / "crop.ome.tif"
        tifffile.imwrite(
            path,
            np.full((2, 3), 17, dtype=np.uint32),
            metadata={"axes": "YX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 0.5},
        )
        mapping = Affine(np.array([[2, 0, 1], [0, 2, 4], [0, 0, 1]]), _XY, _XY)
        mask = read_mask(
            MaskSource(require_file(path, context=_CONTEXT), mapping),
            image,
            options=make_raster_options(),
            context=_CONTEXT,
        )
        transform = cast("BaseTransformation", get_transformation(mask, to_coordinate_system="global"))
        np.testing.assert_allclose(transform.to_affine_matrix(_XY, _XY), [[1, 0, 0.5], [0, 0.5, 1], [0, 0, 1]])
        with pytest.raises(ReaderFormatError, match="mask OME physical sizes"):
            read_mask(
                MaskSource(require_file(path, context=_CONTEXT), Affine(np.eye(3), _XY, _XY)),
                image,
                options=make_raster_options(),
                context=_CONTEXT,
            )
