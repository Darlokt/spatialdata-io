"""Public option validation, grid reconciliation and one final linkage boundary."""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import pytest
from numpy.testing import assert_array_equal
from spatialdata.transformations import get_transformation

import spatialdata_io
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.stereoseq import stereoseq
from spatialdata_io.readers.stereoseq._reader import _bin_selection, _sequence, _validate_options
from tests.integration.readers.stereoseq._fixtures import write_analysis, write_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TestBinSelection:
    """Normalize trusted internal sizes once at public ingress."""

    @pytest.mark.parametrize("value", [True, 2, "1", [True], [1.5], ["all", 1]])
    def test_invalid_type(self, value: Sequence[int] | Literal["all"]) -> None:
        with pytest.raises(TypeError):
            _bin_selection(value)

    @pytest.mark.parametrize("value", [[0], [-1], [1, 1]])
    def test_invalid_domain(self, value: Sequence[int]) -> None:
        with pytest.raises(ValueError, match="distinct positive"):
            _bin_selection(value)

    def test_sorted_owned_tuple(self) -> None:
        value = [50, 1, 20]

        assert _bin_selection(value) == (1, 20, 50)
        assert value == [50, 1, 20]
        assert _bin_selection("all") == "all"
        assert _bin_selection(()) == ()


class TestSequence:
    """Reject scalar collection options instead of interpreting their characters."""

    @pytest.mark.parametrize("value", ["file.h5ad", b"file.h5ad", None, 5], ids=["str", "bytes", "none", "int"])
    def test_rejects_nonsequence(self, value: Sequence[str]) -> None:
        with pytest.raises(TypeError):
            _sequence(value, "analysis_files")


class TestValidateOptions:
    """Validate option domains before any layout I/O."""

    @pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b", "a\0b", False])
    def test_sample(self, value: str) -> None:
        with pytest.raises(ValueError, match="sample_id"):
            _validate_options(sample_id=value, cell_bins="none", cell_geometry="centroids", roles=(), analyses=())

    @pytest.mark.parametrize("value", ["invalid", True])
    def test_invalid_cell_selection(self, value: Literal["none", "unadjusted", "adjusted", "both"]) -> None:
        with pytest.raises(ValueError, match="cell_bins"):
            _validate_options(sample_id=None, cell_bins=value, cell_geometry="centroids", roles=(), analyses=())

    @pytest.mark.parametrize("value", ["centroids "])
    def test_invalid_geometry(self, value: Literal["centroids", "polygons"]) -> None:
        with pytest.raises(ValueError, match="cell_geometry"):
            _validate_options(sample_id=None, cell_bins="none", cell_geometry=value, roles=(), analyses=())

    @pytest.mark.parametrize("roles", [("mask", "mask"), ("labels",)])
    def test_invalid_roles(self, roles: tuple[Literal["mask"], ...]) -> None:
        with pytest.raises(ValueError, match="image_roles"):
            _validate_options(sample_id=None, cell_bins="none", cell_geometry="centroids", roles=roles, analyses=())

    @pytest.mark.parametrize("value", [1])
    def test_invalid_analysis_path(self, value: str | Path) -> None:
        with pytest.raises(TypeError, match="filesystem paths"):
            _validate_options(sample_id=None, cell_bins="none", cell_geometry="centroids", roles=(), analyses=(value,))


class TestStereoseq:
    """Enforce transaction boundaries not owned by individual product parsers."""

    def test_invalid_polygon_policy_precedes_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid_geometries"):
            stereoseq(tmp_path / "missing", invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_no_products_fails_before_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Select products"):
            stereoseq(tmp_path / "missing")

    def test_invalid_raster_option_fails_before_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scale_factors"):
            stereoseq(tmp_path / "missing", image_roles=("registered",), scale_factors=(1,))

    def test_raster_option_defaults(self) -> None:
        parameters = inspect.signature(stereoseq).parameters

        assert list(parameters)[-2:] == ["scale_factors", "chunks"]
        assert parameters["scale_factors"].default == DEFAULT_SCALE_FACTORS
        assert parameters["chunks"].default == DEFAULT_CHUNKS

    @pytest.mark.parametrize("selection", ["square", "cells", "adjusted"])
    def test_override_does_not_select_product(self, tmp_path: Path, selection: str) -> None:
        with pytest.raises(ValueError, match="overrides require"):
            stereoseq(
                tmp_path / "missing",
                image_roles=("registered",),
                square_counts_file="file.gef" if selection == "square" else None,
                cell_counts_file="file.gef" if selection == "cells" else None,
                adjusted_cell_counts_file="file.gef" if selection == "adjusted" else None,
            )

    @pytest.mark.parametrize("field", ["origin", "resolution", "sample"])
    def test_rejects_conflicting_count_frames(self, tmp_path: Path, field: str) -> None:
        fixture = write_dataset(tmp_path, products=("square", "cells"))
        with h5py.File(fixture.cells, "r+") as handle:
            if field == "origin":
                handle.attrs["offsetX"] = 100
            elif field == "resolution":
                handle.attrs["resolution"] = 600
            else:
                handle.attrs["sn"] = "other_sample"

        with pytest.raises(ReaderFormatError, match="coordinate frame"):
            stereoseq(tmp_path, bin_sizes=(1,), cell_bins="unadjusted")

    def test_common_nonzero_origin_is_not_added_to_one_element(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("square", "cells", "registered"))
        with h5py.File(fixture.cells, "r+") as handle:
            handle.attrs["offsetX"] = 100
            handle.attrs["offsetY"] = 200
        with h5py.File(fixture.square, "r+") as handle:
            handle["geneExp/bin1/expression"].attrs["minX"] = 100
            handle["geneExp/bin1/expression"].attrs["minY"] = 200
        result = stereoseq(tmp_path, bin_sizes=(1,), cell_bins="unadjusted", image_roles=("registered",))
        assert_array_equal(result.tables["cells_table"].obsm["spatial_grid"], [[10, 20], [30, 40]])
        assert_array_equal(result.tables["cells_table"].obsm["spatial"], [[5, 10], [15, 20]])
        assert_array_equal(result.tables["bin1_table"].obsm["spatial_grid"], [[5, 6], [1, 2]])
        assert_array_equal(result.tables["bin1_table"].obsm["spatial"], [[2.5, 3], [0.5, 1]])

        assert result.tables["cells_table"].uns["stereoseq"]["origin"] == [100, 200]
        for element in (*result.images.values(), *result.points.values()):
            transform = get_transformation(element, to_coordinate_system="global")
            assert not isinstance(transform, dict)
            assert_array_equal(
                transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")) @ [10, 20, 1], [5, 10, 1]
            )

    def test_duplicate_normalized_analysis_names_fail(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("analysis",))
        duplicate = fixture.analysis.with_name("sample.cellbin_0.50.h5ad")
        duplicate.write_bytes(fixture.analysis.read_bytes())

        with pytest.raises(ReaderFormatError, match="duplicate"):
            stereoseq(tmp_path, analysis_files=(fixture.analysis, duplicate))

    def test_analysis_only_requires_consistent_grid_pitch(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("analysis",))
        other = fixture.analysis.with_name("sample.cellbin_1.h5ad")
        analysis = ad.read_h5ad(fixture.analysis)
        analysis.uns["resolution"] = 600
        analysis.write_h5ad(other)

        with pytest.raises(ReaderFormatError, match="common grid pitch"):
            stereoseq(tmp_path, analysis_files=(fixture.analysis, other))

    def test_historical_analysis_metadata_cannot_overwrite_a_product(self, tmp_path: Path) -> None:
        files = tuple(tmp_path / directory / "sample.spatial.cluster.h5ad" for directory in ("first", "second"))
        for path in files:
            write_analysis(path)
            table = ad.read_h5ad(path)
            table.uns["bin_type"] = "bins"
            table.uns["bin_size"] = 50
            table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="duplicate analysis name"):
            stereoseq(tmp_path, analysis_files=files)

    def test_package_and_root_export(self) -> None:
        package = importlib.import_module("spatialdata_io.readers.stereoseq")

        assert spatialdata_io.stereoseq is stereoseq
        assert package.__all__ == ["stereoseq"]
