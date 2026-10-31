from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata.transformations import BaseTransformation, get_transformation
from xarray import DataArray

from spatialdata_io.converters.legacy_anndata._images import read_images, select_library

if TYPE_CHECKING:
    import numpy.typing as npt


def _table() -> AnnData:
    table = AnnData(shape=(3, 1))
    table.uns["spatial"] = {
        "sample": {
            "images": {"hires": np.arange(6, dtype=np.uint16).reshape(2, 3)},
            "scalefactors": {"tissue_hires_scalef": 0.5},
        }
    }
    return table


class TestSelectLibrary:
    def test_infers_single_library_and_validates_image_selection(self) -> None:
        table = _table()
        library = select_library(table, library_id=None, library_key=None)
        assert library.name == "sample"
        assert library.rows is None
        assert [(image.key, image.scale) for image in library.images] == [("hires", 0.5)]

    def test_no_library_metadata_or_images_is_valid(self) -> None:
        table = AnnData(shape=(0, 1))
        library = select_library(table, library_id=None, library_key=None)
        assert library.name is None
        assert library.rows is None
        assert library.images == ()
        dataset: dict[str, object] = {}
        table.uns["spatial"] = {"sample": dataset}
        assert select_library(table, library_id=None, library_key=None).images == ()

    @pytest.mark.parametrize("option", ["library_id", "library_key"])
    @pytest.mark.parametrize("value", ["", 1, False, []])
    def test_rejects_invalid_option_types(self, option: str, value: object) -> None:
        # Deliberately violate the static option contract to test external validation.
        invalid = cast("str", value)
        with pytest.raises(TypeError, match=f"{option} must be"):
            select_library(
                _table(),
                library_id=invalid if option == "library_id" else None,
                library_key=invalid if option == "library_key" else None,
            )

    @pytest.mark.parametrize("location", ["spatial", "sample", "images", "scalefactors"])
    @pytest.mark.parametrize("value", [None, [], {1: {}}, {"": {}}])
    def test_rejects_malformed_consumed_mappings(self, location: str, value: object) -> None:
        table = _table()
        if location == "spatial":
            table.uns["spatial"] = value
        elif location == "sample":
            table.uns["spatial"]["sample"] = value
        else:
            table.uns["spatial"]["sample"][location] = value
        with pytest.raises(TypeError, match="mapping with nonempty string keys"):
            select_library(table, library_id=None, library_key=None)

    @pytest.mark.parametrize("scale", [1, np.float32(0.25), np.int64(2)])
    def test_direct_image_scale_takes_precedence(self, scale: float) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["scalefactors"] = {"hires": scale, "tissue_hires_scalef": "unused"}
        library = select_library(table, library_id=None, library_key=None)
        assert library.images[0].scale == float(scale)

    @pytest.mark.parametrize("scale", [None, True, False, "0.5", [], 1j, np.bool_(1)])
    def test_rejects_nonreal_scale_types(self, scale: object) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["scalefactors"]["hires"] = scale
        with pytest.raises(TypeError, match=r"sample.*hires.*scale factor"):
            select_library(table, library_id=None, library_key=None)

    def test_requires_consumed_image_scale(self) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["scalefactors"] = {}
        with pytest.raises(TypeError, match=r"sample.*hires.*scale factor"):
            select_library(table, library_id=None, library_key=None)

    @pytest.mark.parametrize("scale", [0, -1, np.inf, -np.inf, np.nan, np.nextafter(0.0, 1.0), 10**1000])
    def test_rejects_nonpositive_nonfinite_or_unrepresentable_scale(self, scale: float) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["scalefactors"]["tissue_hires_scalef"] = scale
        with pytest.raises(ValueError, match=r"sample.*hires.*invalid scale factor"):
            select_library(table, library_id=None, library_key=None)

    def test_selects_multilibrary_rows_in_source_order(self) -> None:
        table = _table()
        table.uns["spatial"]["other"] = {"images": "unconsumed malformed state"}
        table.obs["library"] = pd.Categorical(["sample", "other", "sample"])
        library = select_library(table, library_id="sample", library_key="library")
        assert library.name == "sample"
        np.testing.assert_array_equal(library.rows, [0, 2])
        assert library.images[0].key == "hires"
        np.testing.assert_array_equal(table.obs["library"], ["sample", "other", "sample"])

    @pytest.mark.parametrize("library_id", [None, "sample"])
    def test_multilibrary_requires_explicit_id_and_association(self, library_id: str | None) -> None:
        table = _table()
        table.uns["spatial"]["other"] = {}
        with pytest.raises(ValueError, match=r"[Mm]ultiple legacy libraries"):
            select_library(table, library_id=library_id, library_key=None)

    def test_multilibrary_association_still_requires_selected_id(self) -> None:
        table = _table()
        table.uns["spatial"]["other"] = {}
        table.obs["library"] = ["sample", "other", "sample"]
        with pytest.raises(ValueError, match="Multiple legacy libraries"):
            select_library(table, library_id=None, library_key="library")

    def test_honors_single_library_association(self) -> None:
        table = _table()
        table.obs["library"] = ["sample"] * 3
        library = select_library(table, library_id=None, library_key="library")
        np.testing.assert_array_equal(library.rows, [0, 1, 2])

    def test_observation_associations_can_supply_library_without_images(self) -> None:
        table = AnnData(shape=(3, 1))
        table.obs["library"] = ["b", "a", "b"]
        library = select_library(table, library_id="b", library_key="library")
        assert library.name == "b"
        np.testing.assert_array_equal(library.rows, [0, 2])
        assert library.images == ()

    @pytest.mark.parametrize(
        "memberships",
        [["sample", "missing", "sample"], ["sample", None, "sample"], ["sample", "", "sample"], [1, 2, 3]],
    )
    def test_rejects_invalid_or_unknown_associations(self, memberships: list[str | int | None]) -> None:
        table = _table()
        table.obs["library"] = memberships
        with pytest.raises(ValueError, match=r"library.*(non-null|unknown)"):
            select_library(table, library_id=None, library_key="library")

    def test_rejects_unknown_requested_library(self) -> None:
        with pytest.raises(ValueError, match="Unknown legacy library 'missing'"):
            select_library(_table(), library_id="missing", library_key=None)

    def test_rejects_missing_association_column(self) -> None:
        with pytest.raises(ValueError, match="column 'library' is missing"):
            select_library(_table(), library_id=None, library_key="library")

    def test_rejects_ambiguous_duplicate_association_columns(self) -> None:
        table = _table()
        table.obs = pd.DataFrame({"first": ["sample"] * 3, "second": ["sample"] * 3}, index=table.obs_names)
        table.obs.columns = ["library", "library"]
        with pytest.raises(ValueError, match=r"library.*(unique|ambiguous|duplicate|one column)"):
            select_library(table, library_id=None, library_key="library")

    def test_rejects_empty_explicit_library_selection(self) -> None:
        table = _table()
        table.uns["spatial"]["other"] = {}
        table.obs["library"] = ["other"] * 3
        with pytest.raises(ValueError, match=r"sample.*no associated observations"):
            select_library(table, library_id="sample", library_key="library")

    def test_rejects_supplied_association_for_zero_observations(self) -> None:
        table = _table()[:0].copy()
        table.obs["library"] = pd.Series(index=table.obs_names, dtype="str")
        with pytest.raises(ValueError, match="no associated observations"):
            select_library(table, library_id=None, library_key="library")

    @pytest.mark.parametrize("pixels", [[[1, 2]], da.zeros((2, 3)), "pixels"])
    def test_rejects_noneager_numpy_images(self, pixels: object) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["images"]["hires"] = pixels
        with pytest.raises(TypeError, match=r"sample.*hires.*eager NumPy"):
            select_library(table, library_id=None, library_key=None)

    @pytest.mark.parametrize("shape", [(), (2,), (2, 3, 4, 5), (0, 3), (2, 0, 3), (2, 3, 0)])
    def test_rejects_invalid_image_dimensions(self, shape: tuple[int, ...]) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["images"]["hires"] = np.zeros(shape)
        with pytest.raises(ValueError, match="nonempty YX or YXC real numerical pixels"):
            select_library(table, library_id=None, library_key=None)

    @pytest.mark.parametrize("dtype", [object, "U1", "S1", np.complex64, "datetime64[ns]"])
    def test_rejects_nonreal_image_dtype(self, dtype: npt.DTypeLike) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["images"]["hires"] = np.zeros((2, 3), dtype=dtype)
        with pytest.raises(ValueError, match="real numerical pixels"):
            select_library(table, library_id=None, library_key=None)


class TestReadImages:
    def test_no_images_returns_empty_mapping(self) -> None:
        table = AnnData(shape=(0, 1))
        library = select_library(table, library_id=None, library_key=None)
        assert read_images(table, library, frame="pixels", rgb=None) == {}

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64, np.bool_])
    def test_preserves_asymmetric_grayscale_values_and_native_resolution(self, dtype: npt.DTypeLike) -> None:
        table = _table()
        source = np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)
        table.uns["spatial"]["sample"]["images"]["hires"] = source
        library = select_library(table, library_id=None, library_key=None)
        images = read_images(table, library, frame="pixel_frame", rgb=None)
        assert set(images) == {"sample_hires"}
        image = images["sample_hires"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == (1, 2, 3)
        assert image.dtype == source.dtype
        np.testing.assert_array_equal(image.values, source[None, :, :])
        np.testing.assert_array_equal(source, np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype))

    @pytest.mark.parametrize(
        ("channels", "rgb", "channel_names"),
        [
            (3, None, ["r", "g", "b"]),
            (4, True, ["r", "g", "b", "a"]),
            (3, False, [0, 1, 2]),
            (5, None, [0, 1, 2, 3, 4]),
        ],
    )
    def test_preserves_rgb_rgba_and_arbitrary_channels(
        self, channels: int, *, rgb: bool | None, channel_names: list[str] | list[int]
    ) -> None:
        table = _table()
        source = np.arange(6 * channels, dtype=np.uint16).reshape(2, 3, channels)
        table.uns["spatial"]["sample"]["images"]["hires"] = source
        library = select_library(table, library_id=None, library_key=None)
        image = read_images(table, library, frame="pixels", rgb=rgb)["sample_hires"]
        assert isinstance(image, DataArray)
        assert image.shape == (channels, 2, 3)
        assert image.dtype == source.dtype
        np.testing.assert_array_equal(image.values, source.transpose(2, 0, 1))
        np.testing.assert_array_equal(image.coords["c"], channel_names)

    def test_image_centres_map_to_unchanged_full_resolution_pixel_indices(self) -> None:
        table = _table()
        table.uns["spatial"]["sample"]["images"]["lowres"] = np.array([[9, 7]], dtype=np.uint8)
        table.uns["spatial"]["sample"]["scalefactors"]["tissue_lowres_scalef"] = 0.25
        library = select_library(table, library_id=None, library_key=None)
        images = read_images(table, library, frame="fullres_pixels", rgb=False)
        assert set(images) == {"sample_hires", "sample_lowres"}
        # Analytic landmarks: index (2, 1) in hires is (4, 2) fullres;
        # index (1, 0) in lowres is (4, 0) fullres. Intrinsic centres add 0.5.
        landmarks = [("sample_hires", [2.5, 1.5, 1], [4, 2, 1]), ("sample_lowres", [1.5, 0.5, 1], [4, 0, 1])]
        for name, intrinsic, expected in landmarks:
            image = images[name]
            transform = get_transformation(image, "fullres_pixels")
            assert isinstance(transform, BaseTransformation)
            matrix = transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
            np.testing.assert_array_equal(matrix @ intrinsic, expected)
            np.testing.assert_array_equal(matrix @ [0.5, 0.5, 1], [0, 0, 1])

    def test_rgb_requires_three_or_four_channels_and_reports_source(self) -> None:
        table = _table()
        library = select_library(table, library_id=None, library_key=None)
        with pytest.raises(ValueError, match="rgb is True") as captured:
            read_images(table, library, frame="pixels", rgb=True)
        assert any("sample" in note and "hires" in note for note in captured.value.__notes__)
