"""Exercise deterministic DBiT layouts, ordering, geometry, and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import shapely
import tifffile
from PIL import Image
from scipy import sparse
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io import __version__
from spatialdata_io.readers.dbit import dbit

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture


_SOURCE_NAMES = ["CCCCAAAATTGGCCAA", "GGGGTTTTAACCGGTT"]
_ORDERED_NAMES = _SOURCE_NAMES[::-1]


def _write_layout(path: Path, *, sparse_source: bool = True) -> None:
    expression = np.asarray([[3, 4], [1, 2]], dtype=np.int32)
    table = ad.AnnData(
        X=sparse.csr_matrix(expression) if sparse_source else expression,
        obs=pd.DataFrame(
            {"quality": np.asarray([0.2, 0.7], dtype=np.float32), "annotation": ["second", "first"]},
            index=_SOURCE_NAMES,
        ),
        var=pd.DataFrame({"feature_id": ["id-b", "id-a"]}, index=["gene-b", "gene-a"]),
        uns={"experiment": {"assay": "DBiT", "replicate": 4}},
        layers={"counts": sparse.csr_matrix(expression * 2), "quality": expression / 10},
    )
    table.obsm["embedding"] = np.asarray([[20, 21], [10, 11]], dtype=np.float64)
    table.varm["weights"] = np.asarray([[0.5], [0.25]], dtype=np.float32)
    table.write_h5ad(path / "sample.h5ad")

    (path / "barcode_list.txt").write_text(
        "A2\tTTGGCCAA\nB2\tCCCCAAAA\nA1\tAACCGGTT\nB1\tGGGGTTTT\n",
        encoding="utf-8",
    )


def _write_image(path: Path) -> NDArray[np.uint8]:
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    if path.suffix == ".png":
        Image.fromarray(pixels).save(path)
    else:
        tifffile.imwrite(path, pixels, photometric="rgb")
    return pixels


class TestDbit:
    """Check the public reader against small format-faithful inputs."""

    @pytest.mark.parametrize("sparse_source", [True, False])
    def test_synthetic_layout_preserves_public_contract(self, tmp_path: Path, *, sparse_source: bool) -> None:
        _write_layout(tmp_path, sparse_source=sparse_source)

        result = dbit(tmp_path, dataset_id="sample", include_border=True, border_scale=1.0)

        assert list(result.shapes) == ["sample"]
        assert list(result.tables) == ["table"]
        assert not result.images
        assert not result.labels
        assert not result.points
        assert result.attrs == {
            "spatialdata_io_reader": "dbit",
            "spatialdata_io_software_version": __version__,
        }

        shapes = result.shapes["sample"]
        assert shapes.index.tolist() == [0, 1]
        assert shapes.geometry.geom_type.tolist() == ["Polygon", "Polygon"]
        np.testing.assert_allclose(shapes.total_bounds, [0.125, 0.125, 1.875, 1.875])
        assert get_transformation(shapes, get_all=True) == {"global": Identity()}

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        expression = table.X.toarray()
        np.testing.assert_array_equal(expression, np.asarray([[1, 2], [3, 4]], dtype=np.int32))
        assert expression.dtype == np.int32

        assert table.obs_names.tolist() == _ORDERED_NAMES
        assert table.var_names.tolist() == ["gene-b", "gene-a"]
        assert table.var["feature_id"].tolist() == ["id-b", "id-a"]
        assert table.obs["array_A"].tolist() == [1, 2]
        assert table.obs["array_B"].tolist() == [1, 2]
        assert table.obs["pixel_id"].tolist() == [0, 1]
        assert table.obs["sample"].tolist() == ["sample", "sample"]
        assert table.obs["region"].tolist() == ["sample", "sample"]
        assert isinstance(table.obs["region"].dtype, pd.CategoricalDtype)

        np.testing.assert_allclose(table.obs["quality"], [0.7, 0.2])
        assert table.obs["quality"].dtype == np.float32
        assert table.obs["annotation"].tolist() == ["first", "second"]
        np.testing.assert_array_equal(table.obsm["spatial"], [[1, 1], [2, 2]])
        np.testing.assert_array_equal(table.obsm["embedding"], [[10, 11], [20, 21]])

        weights = table.varm["weights"]
        assert isinstance(weights, np.ndarray)
        np.testing.assert_allclose(weights, [[0.5], [0.25]])

        counts = table.layers["counts"]
        assert isinstance(counts, sparse.csr_array)
        np.testing.assert_array_equal(counts.toarray(), [[2, 4], [6, 8]])

        quality = table.layers["quality"]
        assert isinstance(quality, np.ndarray)
        np.testing.assert_allclose(quality, [[0.1, 0.2], [0.3, 0.4]])

        assert table.uns["experiment"] == {"assay": "DBiT", "replicate": 4}
        assert get_table_keys(table) == ("sample", "region", "pixel_id")
        assert set(table.obs["pixel_id"]) == set(shapes.index)
        assert not table.is_view

    @pytest.mark.parametrize(
        ("border", "border_scale", "inset"), [(True, 1.0, 0.125), (True, 2.0, 0.25), (False, 2.0, 0)]
    )
    def test_border_geometry(self, tmp_path: Path, *, border: bool, border_scale: float, inset: float) -> None:
        _write_layout(tmp_path)

        result = dbit(tmp_path, include_border=border, border_scale=border_scale)

        first = result.shapes["sample"].geometry.iloc[0]
        assert isinstance(first, shapely.Polygon)
        np.testing.assert_allclose(
            first.exterior.coords,
            [[inset, inset], [inset, 1 - inset], [1 - inset, 1 - inset], [1 - inset, inset], [inset, inset]],
        )
        np.testing.assert_allclose(result.shapes["sample"].total_bounds, [inset, inset, 2 - inset, 2 - inset])

    @pytest.mark.parametrize("image_name", ["tissue_lowres_image.png", "histology.tif"])
    @pytest.mark.parametrize("border", [True, False])
    def test_rgb_image_values_axes_scale_and_transforms(self, tmp_path: Path, image_name: str, *, border: bool) -> None:
        _write_layout(tmp_path)
        image_path = tmp_path / image_name
        pixels = _write_image(image_path)

        result = dbit(tmp_path, image_file=str(image_path), include_border=border, border_scale=2.0)

        assert list(result.images) == ["sample_image"]
        image = result.images["sample_image"]
        assert image.dims == ("c", "y", "x")
        assert image.shape == (3, 8, 10)
        assert image.dtype == np.uint8
        assert isinstance(image.data, da.Array)
        np.testing.assert_array_equal(image.c, [0, 1, 2])
        np.testing.assert_array_equal(image.data.compute(), pixels.transpose(2, 0, 1))

        assert get_transformation(image, get_all=True) == {"global": Identity()}
        assert get_transformation(result.shapes["sample"], get_all=True) == {"global": Identity()}

        inset = 1.25 if border else 0
        np.testing.assert_allclose(result.shapes["sample"].total_bounds, [inset, inset, 10 - inset, 10 - inset])
        np.testing.assert_array_equal(result.tables["table"].obsm["spatial"], [[1, 1], [2, 2]])

    def test_image_auto_discovery(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        _write_image(tmp_path / "tissue_lowres_image.png")

        assert list(dbit(tmp_path).images) == ["sample_image"]

    def test_absolute_overrides_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        source = tmp_path / "source"
        source.mkdir()
        _write_layout(source)
        _write_image(source / "histology.tif")

        result = dbit(
            root,
            table_file=str(source / "sample.h5ad"),
            barcode_positions_file=str(source / "barcode_list.txt"),
            image_file=str(source / "histology.tif"),
            dataset_id="override",
        )

        assert list(result.shapes) == ["override"]
        assert list(result.images) == ["override_image"]
        assert get_table_keys(result.tables["table"]) == ("override", "region", "pixel_id")

    def test_relative_override_is_resolved_from_dataset_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _write_layout(source)
        monkeypatch.chdir(tmp_path)

        result = dbit(source, table_file="sample.h5ad", barcode_positions_file="barcode_list.txt")

        assert list(result.shapes) == ["sample"]

    def test_missing_explicit_optional_image_raises(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)

        with pytest.raises(FileNotFoundError, match=r"absent\.tif"):
            dbit(tmp_path, image_file="absent.tif")

    def test_missing_required_table(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            dbit(tmp_path)

    @pytest.mark.parametrize("sparse_source", [True, False])
    def test_expression_normalization_does_not_densify(
        self, tmp_path: Path, mocker: MockerFixture, *, sparse_source: bool
    ) -> None:
        _write_layout(tmp_path, sparse_source=sparse_source)

        def reject_densification(*_args: object, **_kwargs: object) -> None:
            pytest.fail("DBiT expression data was densified")

        for sparse_type in (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array):
            mocker.patch.object(sparse_type, "toarray", side_effect=reject_densification)

        result = dbit(tmp_path)

        matrix = result.tables["table"].X
        assert isinstance(matrix, sparse.csr_array)
        assert matrix.has_canonical_format
        np.testing.assert_array_equal(matrix.data, [1, 2, 3, 4])
        np.testing.assert_array_equal(matrix.indices, [0, 1, 0, 1])
        np.testing.assert_array_equal(matrix.indptr, [0, 2, 4])

    @pytest.mark.parametrize(("lower_map", "lower_names"), [(True, False), (False, True), (True, True)])
    def test_case_insensitive_barcode_lookup_preserves_observation_identifiers(
        self, tmp_path: Path, *, lower_map: bool, lower_names: bool
    ) -> None:
        _write_layout(tmp_path)
        if lower_map:
            barcode_file = tmp_path / "barcode_list.txt"
            rows = barcode_file.read_text(encoding="utf-8").splitlines()
            barcode_file.write_text(
                "".join(
                    f"{position}\t{sequence.lower()}\n" for position, sequence in (row.split("\t") for row in rows)
                ),
                encoding="utf-8",
            )

        source = ad.read_h5ad(tmp_path / "sample.h5ad")
        source.obs_names = [f"{name.lower() if lower_names else name}-sample" for name in _SOURCE_NAMES]
        source.write_h5ad(tmp_path / "sample.h5ad")

        result = dbit(tmp_path)

        expected_names = [f"{name.lower() if lower_names else name}-sample" for name in _ORDERED_NAMES]
        assert result.tables["table"].obs_names.tolist() == expected_names
        np.testing.assert_array_equal(result.tables["table"].obsm["spatial"], [[1, 1], [2, 2]])

    def test_case_insensitive_h5ad_suffix_and_multidot_identifier(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        (tmp_path / "sample.h5ad").rename(tmp_path / "assay.rep1.H5AD")

        result = dbit(tmp_path)

        assert list(result.shapes) == ["assay.rep1"]
        assert get_table_keys(result.tables["table"]) == ("assay.rep1", "region", "pixel_id")

    def test_reader_does_not_mutate_source_h5ad(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        original_bytes = (tmp_path / "sample.h5ad").read_bytes()

        result = dbit(tmp_path)
        result.tables["table"].obs["quality"] = 0

        assert (tmp_path / "sample.h5ad").read_bytes() == original_bytes

        source = ad.read_h5ad(tmp_path / "sample.h5ad")
        assert source.obs_names.tolist() == _SOURCE_NAMES
        assert "region" not in source.obs
        assert "spatialdata_attrs" not in source.uns
        np.testing.assert_allclose(source.obs["quality"], [0.2, 0.7])
