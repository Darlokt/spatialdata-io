"""Public option validation, grid reconciliation and one final linkage boundary."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array
from spatialdata.transformations import Identity, Scale, get_transformation

import spatialdata_io
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq import stereoseq
from spatialdata_io.readers.stereoseq._reader import (
    _CELL_BINS,
    IMAGE_PIXELS,
    _bin_selection,
    _choice,
    _format_version,
    _image_roles,
    _path,
    _sample_id,
    _sequence,
    _square_products,
    _transformations,
)
from spatialdata_io.readers.stereoseq._square import SquareCounts
from tests.integration.readers.stereoseq._fixtures import write_analysis, write_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers.stereoseq._layout import ImageRole


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
        with pytest.raises(TypeError, match="analysis_files must be a sequence"):
            _sequence(value, "analysis_files")


class TestChoice:
    """Literal options: wrong types raise TypeError, wrong values ValueError."""

    def test_valid(self) -> None:
        assert _choice("both", "cell_bins", _CELL_BINS) == "both"

    @pytest.mark.parametrize("value", [True, 1, None, b"both"])
    def test_wrong_type(self, value: str) -> None:
        with pytest.raises(TypeError, match=r"cell_bins must be one of \['none', 'unadjusted', 'adjusted', 'both'\]"):
            _choice(value, "cell_bins", _CELL_BINS)

    @pytest.mark.parametrize("value", ["invalid", "Both", "both "])
    def test_wrong_value(self, value: str) -> None:
        with pytest.raises(ValueError, match=f"found {value!r}"):
            _choice(value, "cell_bins", _CELL_BINS)


class TestSampleId:
    """The sample selector is a filename component, never an element name."""

    @pytest.mark.parametrize("value", [None, "sample", "sample.with.dots", "sample 1"])
    def test_valid(self, value: str | None) -> None:
        assert _sample_id(value) == value

    @pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b", "a\0b"])
    def test_invalid_value(self, value: str) -> None:
        with pytest.raises(ValueError, match="sample_id must be a nonempty sample filename component"):
            _sample_id(value)

    @pytest.mark.parametrize("value", [False, 1, b"sample"])
    def test_wrong_type(self, value: str) -> None:
        with pytest.raises(TypeError, match="sample_id must be a string"):
            _sample_id(value)


class TestPath:
    """File options accept strings and paths only."""

    @pytest.mark.parametrize("value", ["a.gef", Path("a.gef")])
    def test_valid(self, value: str | Path) -> None:
        assert _path(value, "cell_counts_file") is value

    @pytest.mark.parametrize("value", [1, b"a.gef", None])
    def test_wrong_type(self, value: str) -> None:
        with pytest.raises(TypeError, match="cell_counts_file must contain filesystem paths"):
            _path(value, "cell_counts_file")


class TestImageRoles:
    """Roles are distinct members of the supported vocabulary."""

    def test_valid(self) -> None:
        assert _image_roles(["mask", "registered"]) == ("mask", "registered")

    def test_duplicate(self) -> None:
        with pytest.raises(ValueError, match="image_roles must be distinct"):
            _image_roles(("mask", "mask"))

    def test_unknown(self) -> None:
        with pytest.raises(ValueError, match="image_roles must be one of"):
            _image_roles(("labels",))  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    @pytest.mark.parametrize("value", ["mask", (1,)])
    def test_wrong_type(self, value: Sequence[ImageRole]) -> None:
        with pytest.raises(TypeError, match="image_roles"):
            _image_roles(value)


class TestTransformations:
    """``global`` is always micrometres; without a pitch only a pixel frame exists."""

    def test_pitch_scales_to_micrometres(self) -> None:
        transformations = _transformations(500)
        assert list(transformations) == ["global"]
        scale = transformations["global"]
        assert isinstance(scale, Scale)
        assert_array_equal(scale.scale, [0.5, 0.5])
        assert scale.axes == ("x", "y")

    def test_without_pitch_uses_image_pixels(self) -> None:
        transformations = _transformations(None)
        assert list(transformations) == [IMAGE_PIXELS] == ["image_pixels"]
        assert isinstance(transformations[IMAGE_PIXELS], Identity)


class TestSquareProducts:
    """Unsigned row identifiers leave signed grid coordinates unchanged."""

    def test_identifiers_and_signed_coordinates(self, tmp_path: Path) -> None:
        coordinates = np.array([[-(2**31), -1], [0, 0], [2**32 - 1, 2**31]], dtype=np.int64)
        table = ad.AnnData(
            csr_array(np.eye(3, dtype=np.float32)),
            obs=pd.DataFrame(index=["0", "1", "2"]),
        )
        table.obsm["spatial"] = coordinates
        square = SquareCounts(
            ArtifactFile(tmp_path / "square.gef"), table, coordinates, 1, (0, 0), 500, "sample", "gef-4"
        )

        points, linked = _square_products(square, {"global": Identity()})

        frame = points.compute()
        assert frame.index.dtype == linked.obs["instance_id"].dtype == np.uint64
        assert_array_equal(frame.index, [0, 1, 2])
        assert_array_equal(linked.obs["instance_id"], [0, 1, 2])
        assert linked.obs_names.tolist() == ["0", "1", "2"]
        assert frame["x"].dtype == frame["y"].dtype == np.int64
        assert_array_equal(frame[["x", "y"]], coordinates)
        assert_array_equal(linked.obsm["spatial_grid"], coordinates)
        assert_array_equal(linked.obsm["spatial"], coordinates * 0.5)


class TestFormatVersion:
    """Record a GEF schema version only when every read GEF reports the same one."""

    def test_no_gef_records_none(self) -> None:
        assert _format_version({}) is None

    def test_agreement_records_the_common_version(self) -> None:
        files = {ArtifactFile(Path("a.gef")): "gef-4", ArtifactFile(Path("b.gef")): "gef-4"}
        assert _format_version(files) == "gef-4"

    @pytest.mark.parametrize("versions", [(None,), (None, "gef-4"), ("gef-4", None)])
    def test_unknown_version_records_none_without_warning(self, versions: tuple[str | None, ...]) -> None:
        files = {ArtifactFile(Path(f"{index}.gef")): version for index, version in enumerate(versions)}
        assert _format_version(files) is None

    def test_disagreement_warns_once_and_records_none(self) -> None:
        files = {
            ArtifactFile(Path("dir/a.raw.gef")): "gef-2",
            ArtifactFile(Path("dir/a.cellbin.gef")): "gef-4",
            ArtifactFile(Path("dir/a.adjusted.cellbin.gef")): None,
        }
        message = (
            r"^stereoseq: the read GEFs have different schema versions \(a\.raw\.gef: gef-2, "
            r"a\.cellbin\.gef: gef-4, a\.adjusted\.cellbin\.gef: no version\); no format version is recorded\.$"
        )
        with pytest.warns(UserWarning, match=message) as caught:
            assert _format_version(files) is None
        assert len(caught) == 1
        assert caught[0].filename == __file__


class TestStereoseq:
    """Enforce transaction boundaries not owned by individual product parsers."""

    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    def test_records_common_gef_version(self, tmp_path: Path, profile: Literal["name_only", "id_and_name"]) -> None:
        write_dataset(tmp_path, profile=profile, products=("square", "cells"))

        result = stereoseq(tmp_path, bin_sizes=(1, 50), cell_bins="unadjusted")
        assert result.attrs["spatialdata_io_format_version"] == ("gef-2" if profile == "name_only" else "gef-4")

    def test_disagreeing_gef_versions_warn_and_record_none(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("square", "cells"))
        with h5py.File(fixture.cells, "r+") as handle:
            handle.attrs["version"] = np.uint32(4)

        with pytest.warns(UserWarning, match=r"sample\.gef: gef-2, sample\.cellbin\.gef: gef-4") as caught:
            result = stereoseq(tmp_path, bin_sizes=(1,), cell_bins="unadjusted")
        assert [warning.filename for warning in caught] == [__file__]
        assert "spatialdata_io_format_version" not in result.attrs
        assert set(result.tables) == {"bin1_table", "cells_table"}

    def test_absent_gef_version_records_none(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("square", "cells"))
        with h5py.File(fixture.cells, "r+") as handle:
            del handle.attrs["version"]

        result = stereoseq(tmp_path, bin_sizes=(1,), cell_bins="unadjusted")
        assert "spatialdata_io_format_version" not in result.attrs

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

        assert list(parameters)[-3:] == ["scale_factors", "chunks", "path_policy"]
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

        with pytest.raises(ReaderFormatError, match="coordinate frame', detected version 'gef-2'"):
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
