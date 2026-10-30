"""Boundary polygons, mask-label agreement, the invalid-geometry policy and circles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from spatialdata.transformations import Identity, Translation, get_transformation

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import _geometry
from spatialdata_io.readers.xenium._cells import read_cells_store
from spatialdata_io.readers.xenium._geometry import (
    apply_policy,
    make_cell_polygons,
    make_circles,
    make_nucleus_polygons,
    read_polygons,
)
from spatialdata_io.readers.xenium._layout import TabularSource
from spatialdata_io.readers.xenium._tables import read_cells
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_mock import MockerFixture

    from spatialdata_io.readers.xenium._cells import CellStore

type Change = Callable[[list[object]], list[object]]

# Each change of one column breaks a structural rule that no invalid-geometry
# policy may repair. Every polygon of the fixture has five vertex rows.
STRUCTURAL_DEFECTS: dict[str, tuple[str, Change, str]] = {
    "nonfinite": ("vertex_x", lambda values: [np.nan, *values[1:]], "nonfinite vertices"),
    "open": ("vertex_x", lambda values: [-1.0, *values[1:]], "open rings"),
    # Labels 1, 1, 2, 1, 1, 2, ...: polygon 1 resumes after polygon 2 began.
    "interleaved": ("label_id", lambda values: [*values[:2], values[5], *values[3:]], "interleaved polygon rows"),
    "several-cells": ("cell_id", lambda values: [values[0], "zzzzzzzz-1", *values[2:]], "spanning several cells"),
    "zero-label": ("label_id", lambda values: [0] * 5 + values[5:], "uint32 labels from 1"),
}


def _store(bundle: fx.XeniumFixture) -> CellStore:
    return read_cells_store(
        ArtifactFile(bundle.root / "cells.zarr.zip"),
        bundle.profile.family,
        pixel_size=fx.PIXEL_SIZE,
        context=ReaderErrorContext("xenium", "cells.zarr.zip"),
    )


def _source(bundle: fx.XeniumFixture, kind: str, suffix: str = "parquet") -> TabularSource:
    file_kind = "parquet" if suffix == "parquet" else "csv_gz"
    return TabularSource(ArtifactFile(bundle.root / f"{kind}_boundaries.{suffix}"), file_kind)


def _rewrite(path: Path, column: str, values: list[object]) -> None:
    table = pq.read_table(path)
    position = table.column_names.index(column)
    pq.write_table(table.set_column(position, column, pa.array(values, table.schema.field(column).type)), path)


class TestReadPolygons:
    """Contiguous vertex rows form exact polygons whose labels agree with the cell store."""

    @pytest.mark.parametrize("profile", [fx.XOA_1_0, fx.XOA_1_3, fx.XOA_1_9, fx.XOA_2_0, fx.XOA_4_0])
    def test_cells_match_their_rectangles_and_labels(self, tmp_path: Path, profile: fx.Profile) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile)

        polygons = read_polygons(_source(bundle, "cell"), profile.family, "cell", _store(bundle))

        assert polygons.cell_ids.tolist() == list(bundle.cell_ids)
        np.testing.assert_array_equal(polygons.labels, bundle.cell_labels)
        assert not polygons.degenerate.any()
        for geometry, rect in zip(polygons.geometries, fx.CELL_RECTS, strict=True):
            # The vendor files store float32 vertices, which the reader keeps exactly.
            ring = np.asarray(rect.ring(), dtype=np.float32).astype(np.float64)
            assert shapely.equals_exact(geometry, shapely.Polygon(ring), tolerance=0)

    def test_nuclei_of_polygon_sets_follow_label_id(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)

        polygons = read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", _store(bundle))

        # Cell 2 has two nuclei, cell 3 none.
        assert polygons.cell_ids.tolist() == [fx.HEX_IDS[0], fx.HEX_IDS[1], fx.HEX_IDS[2], fx.HEX_IDS[2]]
        np.testing.assert_array_equal(polygons.labels, [1, 2, 3, 4])

    @pytest.mark.parametrize("profile", [fx.XOA_1_0, fx.XOA_1_9])
    def test_nuclei_of_one_to_one_families_carry_cell_labels(self, tmp_path: Path, profile: fx.Profile) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile)

        polygons = read_polygons(_source(bundle, "nucleus"), profile.family, "nucleus", _store(bundle))

        np.testing.assert_array_equal(polygons.labels, [1, 2, 3, 4])

    def test_csv_twins_equal_parquet(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0, fx.Options(csv_twins=True))

        parquet = read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", None)
        csv = read_polygons(_source(bundle, "nucleus", "csv.gz"), "polygon_sets", "nucleus", None)

        assert parquet.cell_ids.equals(csv.cell_ids)
        np.testing.assert_array_equal(parquet.labels, csv.labels)
        assert shapely.equals_exact(parquet.geometries, csv.geometries, tolerance=0).all()

    def test_labels_without_the_store(self, tmp_path: Path) -> None:
        integer = fx.write_xenium_bundle(tmp_path / "integer", fx.XOA_1_0)
        hexadecimal = fx.write_xenium_bundle(tmp_path / "hex", fx.XOA_1_9)

        np.testing.assert_array_equal(
            read_polygons(_source(integer, "nucleus"), "integer_ids", "nucleus", None).labels, [1, 2, 3, 4]
        )
        assert read_polygons(_source(hexadecimal, "nucleus"), "hex_ids", "nucleus", None).labels is None

    def test_early_xoa_2_0_nuclei_without_label_id(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0, fx.Options(nucleus_label_id=False))

        with pytest.raises(ReaderFormatError, match="without label_id"):
            read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", _store(bundle))

    def test_store_agreement(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        path = bundle.root / "nucleus_boundaries.parquet"
        labels = pq.read_table(path)["label_id"].to_pylist()

        # Swapping the labels of nuclei owned by different cells contradicts cell_index.
        _rewrite(path, "label_id", [{1: 2, 2: 1}.get(label, label) for label in labels])
        with pytest.raises(ReaderFormatError, match="owned by other cells"):
            read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", _store(bundle))

        _rewrite(path, "label_id", [9 if label == 4 else {1: 2, 2: 1}.get(label, label) for label in labels])
        with pytest.raises(ReaderFormatError, match="owned by other cells"):
            read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", _store(bundle))

    def test_unknown_cells_and_missing_nucleus_set(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        path = bundle.root / "cell_boundaries.parquet"
        ids = pq.read_table(path)["cell_id"].to_pylist()
        _rewrite(path, "cell_id", ["zzzzzzzz-1" if value == ids[0] else value for value in ids])

        with pytest.raises(ReaderFormatError, match=r"absent from cells\.zarr\.zip"):
            read_polygons(_source(bundle, "cell"), "polygon_sets", "cell", _store(bundle))

        imported = fx.write_xenium_bundle(tmp_path / "import", fx.XOA_3_0, fx.Options(import_kind="transcripts"))
        nuclei = fx.write_xenium_bundle(tmp_path / "nuclei", fx.XOA_3_0)
        with pytest.raises(ReaderFormatError, match="no nucleus polygon set"):
            read_polygons(_source(nuclei, "nucleus"), "polygon_sets", "nucleus", _store(imported))

    @pytest.mark.parametrize(
        ("column", "change", "match"), list(STRUCTURAL_DEFECTS.values()), ids=list(STRUCTURAL_DEFECTS)
    )
    def test_structural_errors_raise_in_every_policy(
        self, tmp_path: Path, column: str, change: Change, match: str
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(cells_store=False))
        path = bundle.root / "cell_boundaries.parquet"
        _rewrite(path, column, change(pq.read_table(path)[column].to_pylist()))

        with pytest.raises(ReaderFormatError, match=match):
            read_polygons(_source(bundle, "cell"), "polygon_sets", "cell", None)


class TestApplyPolicy:
    """The invalid-geometry policy keeps, drops or rejects whole entities."""

    def test_keep_drop_raise_for_invalid_polygons(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(bowtie_cells=(1,)))
        polygons = read_polygons(_source(bundle, "cell"), "polygon_sets", "cell", _store(bundle))

        np.testing.assert_array_equal(apply_policy(polygons, "keep", "cell"), [True] * 4)
        np.testing.assert_array_equal(apply_policy(polygons, "drop", "cell"), [True, False, True, True])
        with pytest.raises(ReaderFormatError, match=r"1 invalid geometries .*\['ffkpbaba-1'\]"):
            apply_policy(polygons, "raise", "cell")

    def test_keep_does_not_classify_topology_or_materialize_cell_identifiers(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(bowtie_cells=(1,)))
        polygons = read_polygons(_source(bundle, "cell"), "polygon_sets", "cell", _store(bundle))
        topology = mocker.spy(_geometry, "invalid_geometry_mask")
        identifiers = mocker.spy(pd.Index, "tolist")
        np.testing.assert_array_equal(apply_policy(polygons, "keep", "cell"), [True] * 4)
        topology.assert_not_called()
        identifiers.assert_not_called()

    def test_degenerate_rings(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(short_nuclei=(0,)))
        polygons = read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", _store(bundle))

        assert polygons.degenerate.tolist() == [True, False, False, False]
        assert polygons.geometries[0] is None
        with pytest.raises(
            ReaderFormatError, match=r"'drop' removes them.*found 1 rings with fewer than four coordinates, e.g. \[1\]"
        ):
            apply_policy(polygons, "keep", "nucleus")
        np.testing.assert_array_equal(apply_policy(polygons, "drop", "nucleus"), [False, True, True, True])
        with pytest.raises(ReaderFormatError, match=r"1 invalid geometries"):
            apply_policy(polygons, "raise", "nucleus")


class TestElements:
    """Shapes elements carry the documented index, columns and micrometre registration."""

    def test_cell_polygons_index_and_registration(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        polygons = read_polygons(_source(bundle, "cell"), "polygon_sets", "cell", _store(bundle))

        shapes = make_cell_polygons(polygons, np.array([True, False, True, True]), fx.PIXEL_SIZE)

        assert shapes.index.name == "cell_id"
        assert shapes.index.tolist() == [fx.HEX_IDS[0], fx.HEX_IDS[2], fx.HEX_IDS[3]]
        transformation = get_transformation(shapes, to_coordinate_system="global")
        assert isinstance(transformation, Translation)
        np.testing.assert_array_equal(transformation.translation, [fx.PIXEL_SIZE / 2, fx.PIXEL_SIZE / 2])
        assert transformation.axes == ("x", "y")
        # Translated vertices bound the mask pixels' centres in the corner frame.
        bounds = shapely.bounds(shapes.geometry.iloc[0])
        rect = fx.CELL_RECTS[0]
        np.testing.assert_allclose(
            bounds + fx.PIXEL_SIZE / 2,
            np.array([rect.left + 0.5, rect.top + 0.5, rect.right + 0.5, rect.bottom + 0.5]) * fx.PIXEL_SIZE,
        )

    def test_nucleus_polygons_index_by_label(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        polygons = read_polygons(_source(bundle, "nucleus"), "polygon_sets", "nucleus", _store(bundle))

        shapes = make_nucleus_polygons(polygons, np.ones(4, dtype=bool), fx.PIXEL_SIZE)

        assert shapes.index.name == "label_id"
        assert shapes.index.tolist() == [1, 2, 3, 4]
        assert shapes["cell_id"].tolist() == [fx.HEX_IDS[0], fx.HEX_IDS[1], fx.HEX_IDS[2], fx.HEX_IDS[2]]
        assert not shapes.geometry.isna().any()

    def test_hex_nuclei_need_the_store_and_empty_elements_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_9)
        polygons = read_polygons(_source(bundle, "nucleus"), "hex_ids", "nucleus", None)

        with pytest.raises(ReaderFormatError, match="seg_mask_value labels"):
            make_nucleus_polygons(polygons, np.ones(4, dtype=bool), fx.PIXEL_SIZE)
        with pytest.raises(ReaderFormatError, match="include_cell_boundaries=False; found no polygons to return"):
            make_cell_polygons(polygons, np.zeros(4, dtype=bool), fx.PIXEL_SIZE)

    def test_circles_have_the_exact_mask_area(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        cells = read_cells(TabularSource(ArtifactFile(bundle.root / "cells.parquet"), "parquet"), "polygon_sets")

        circles = make_circles(cells, np.array([True, True, False, True]))

        assert circles.index.tolist() == [fx.HEX_IDS[0], fx.HEX_IDS[1], fx.HEX_IDS[3]]
        pixels = np.array([fx.CELL_RECTS[i].pixels for i in (0, 1, 3)])
        np.testing.assert_allclose(np.pi * circles["radius"].to_numpy() ** 2, pixels * fx.PIXEL_SIZE**2, rtol=1e-14)
        assert isinstance(get_transformation(circles, to_coordinate_system="global"), Identity)
        np.testing.assert_array_equal(circles.geometry.x, pd.Series(cells.frame["x_centroid"]).to_numpy()[[0, 1, 3]])
