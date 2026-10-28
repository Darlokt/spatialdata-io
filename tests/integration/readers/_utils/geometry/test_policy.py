"""Demonstrate one invalid-geometry definition on two existing reader consumers.

MERSCOPE and Stereo-seq currently reject invalid source polygons. These tests show
that the shared classification flags exactly the polygons each reader rejects, on
each reader's own source representation, without changing either reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import Polygon
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import invalid_geometry_mask, raise_invalid_geometries
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._geometry import read_boundaries
from spatialdata_io.readers.stereoseq._geometry import make_polygons

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

VALID = [(0, 0), (4, 0), (4, 3), (0, 3)]
BOWTIE = [(0, 0), (4, 3), (4, 0), (0, 3)]
PINCHED = [(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)]


class TestMerscopeConsumer:
    @pytest.mark.parametrize(("rings", "invalid"), [((VALID, VALID), ()), ((VALID, BOWTIE, PINCHED), (1, 2))])
    def test_shared_mask_matches_reader_rejection(
        self, tmp_path: Path, rings: tuple[list[tuple[int, int]], ...], invalid: tuple[int, ...]
    ) -> None:
        ids = [f"cell-{index}" for index in range(len(rings))]
        geometries = shapely.from_wkb(shapely.to_wkb([Polygon(ring) for ring in rings]))
        path = ArtifactFile(tmp_path / "cell_boundaries.parquet")
        pq.write_table(
            pa.table({"EntityID": ids, "ZIndex": [0] * len(ids), "Geometry": shapely.to_wkb(geometries)}), path
        )
        identity = Affine(np.eye(3), input_axes=("x", "y"), output_axes=("x", "y"))

        mask = invalid_geometry_mask(geometries)

        assert np.flatnonzero(mask).tolist() == list(invalid)
        if invalid:
            with pytest.raises(ReaderFormatError, match="invalid"):
                read_boundaries(path, transform=identity)
            with pytest.raises(ReaderFormatError, match=r"2 invalid geometries.*cell-1.*cell-2"):
                raise_invalid_geometries(mask, ids, context=ReaderErrorContext("merscope", "cell boundaries"))
        else:
            assert len(read_boundaries(path, transform=identity)) == len(ids)


def _borders(rings: tuple[list[tuple[int, int]], ...]) -> NDArray[np.int16]:
    """Encode centroid-relative SAW border offsets with paired SHRT_MAX padding."""
    borders = np.full((len(rings), 32, 2), 32767, dtype=np.int16)
    for index, ring in enumerate(rings):
        borders[index, : len(ring)] = ring
    return borders


class TestStereoseqConsumer:
    @pytest.mark.parametrize(("rings", "invalid"), [((VALID, VALID), ()), ((VALID, BOWTIE), (1,)), ((PINCHED,), (0,))])
    def test_shared_mask_matches_reader_rejection(
        self, tmp_path: Path, rings: tuple[list[tuple[int, int]], ...], invalid: tuple[int, ...]
    ) -> None:
        centers = np.array([[100, 200]] * len(rings), dtype=np.int64)
        index = pd.Index(range(55, 55 + len(rings)))
        geometries = np.array([Polygon(np.asarray(ring) + centers[0]) for ring in rings], dtype=object)

        mask = invalid_geometry_mask(geometries)

        assert np.flatnonzero(mask).tolist() == list(invalid)
        if invalid:
            with pytest.raises(ReaderFormatError, match=f"invalid polygon for cell {index[invalid[0]]!r}"):
                make_polygons(centers, _borders(rings), index, path=tmp_path / "cells.gef")
        else:
            shapes = make_polygons(centers, _borders(rings), index, path=tmp_path / "cells.gef")
            assert shapely.equals_exact(shapes.geometry.to_numpy(), geometries, tolerance=0).all()
