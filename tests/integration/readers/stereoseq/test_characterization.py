"""Exact public reader contracts independent of the characterized legacy defects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, csr_matrix, sparray, spmatrix
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale, get_transformation
from xarray import DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.stereoseq import stereoseq
from tests.integration.readers.stereoseq._fixtures import Family, Profile, write_analysis, write_dataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture

    from tests.integration.readers.stereoseq._fixtures import Product

pytestmark = [pytest.mark.integration, pytest.mark.stereoseq]


def _counts(table: ad.AnnData, *, layer: str | None = None) -> csr_array:
    value = table.X if layer is None else table.layers[layer]
    assert isinstance(value, csr_array)
    return value


class TestStereoseq:
    """Check supported compositions against explicit source values and identities."""

    def test_image_only_ambiguous_calibration_companion_fails(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("square", "registered"))
        fixture.square.with_name("sample.gef").write_bytes(fixture.square.read_bytes())
        with pytest.raises(ReaderFormatError, match="image calibration companion"):
            stereoseq(fixture.root, image_roles=("registered",))

    def test_polygon_drop_retains_uint64_count_and_exon_extremes(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("cells",), invalid_cell_ids=(200,))
        maximum = 2**64 - 1
        with h5py.File(fixture.cells, "r+") as handle:
            group = handle["cellBin"]
            del group["cellExp"]
            group["cellExp"] = np.array([(0, 11), (1, maximum)], dtype=[("geneID", "u4"), ("count", "u8")])
            del group["cellExpExon"]
            group["cellExpExon"] = np.array([5, maximum - 1], dtype=np.uint64)

        result = stereoseq(fixture.root, cell_bins="unadjusted", cell_geometry="polygons", invalid_geometries="drop")

        table = result.tables["cells_table"]
        assert_array_equal(_counts(table).data, np.array([maximum], dtype=np.uint64))
        assert_array_equal(_counts(table, layer="exon").data, np.array([maximum - 1], dtype=np.uint64))
        assert table.obs_names.tolist() == ["100"]

    @pytest.mark.parametrize("family", ["compact_numbered", "numbered", "outs", "flat"])
    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    def test_polygon_drop_preserves_independent_populations_and_sparse_layers(
        self, tmp_path: Path, family: Family, profile: Profile
    ) -> None:
        fixture = write_dataset(
            tmp_path,
            family=family,
            profile=profile,
            products=("cells", "adjusted_cells"),
            invalid_cell_ids=(200,),
            invalid_adjusted_cell_ids=(100,),
        )

        result = stereoseq(
            fixture.root,
            cell_bins="both",
            cell_geometry="polygons",
            invalid_geometries="drop",
            cell_counts_file=fixture.cells if family == "flat" else None,
            adjusted_cell_counts_file=fixture.adjusted_cells if family == "flat" else None,
        )

        for name, identifier, count, exon in (("cells", 100, 22, 6), ("adjusted_cells", 200, 11, 5)):
            table = result.tables[f"{name}_table"]
            shapes = result.shapes[f"{name}_polygons"]
            assert table.obs_names.tolist() == [str(identifier)]
            assert shapes.index.tolist() == [identifier]
            assert_array_equal(_counts(table).data, [count])
            assert_array_equal(_counts(table, layer="exon").data, [exon])
            assert _counts(table).dtype == np.uint64
            assert table.obs["instance_id"].tolist() == [identifier]

    def test_geometry_drop_preserves_analysis_of_original_cell_population(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=("cells", "analysis"), invalid_cell_ids=(200,))

        result = stereoseq(
            fixture.root,
            cell_bins="unadjusted",
            cell_geometry="polygons",
            invalid_geometries="drop",
            analysis_files=(fixture.analysis,),
        )

        assert result.tables["cells_table"].obs_names.tolist() == ["100"]
        assert result.tables["cells_analysis_r0_5_table"].obs["id"].tolist() == [200, 100]
        assert len(result.points["cells_analysis_r0_5_points"].compute()) == 2

    @pytest.mark.parametrize(
        "products", [("cells", "registered"), ("analysis", "registered"), ("square", "registered")]
    )
    def test_physical_pitch_maps_every_element_without_changing_native_coordinates(
        self, tmp_path: Path, products: tuple[Product, ...]
    ) -> None:
        fixture = write_dataset(tmp_path, family="outs", products=products)
        result = stereoseq(
            fixture.root,
            cell_bins="unadjusted" if "cells" in products else "none",
            bin_sizes=(50,) if "square" in products else (),
            analysis_files=(fixture.analysis,) if "analysis" in products else (),
            image_roles=("registered",),
        )
        for elements in (result.images, result.labels, result.points, result.shapes):
            for element in elements.values():
                transform = get_transformation(element, to_coordinate_system="global")
                assert not isinstance(transform, dict)
                assert_array_equal(
                    transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")) @ [10, 20, 1], [5, 10, 1]
                )
        if "cells" in products:
            assert_array_equal(result.tables["cells_table"].obsm["spatial_grid"], [[10, 20], [30, 40]])
        if "square" in products:
            assert_array_equal(result.tables["bin50_table"].obsm["spatial_grid"], [[5, 6], [1, 2]])

    @pytest.mark.parametrize("family", ["compact_numbered", "numbered", "outs"])
    @pytest.mark.parametrize("companion", ["square", "cells"])
    def test_image_only_calibration_reads_only_associated_gef_metadata(
        self, tmp_path: Path, mocker: MockerFixture, family: Family, companion: Literal["square", "cells"]
    ) -> None:
        fixture = write_dataset(tmp_path, family=family, products=(companion, "registered"))
        read = mocker.spy(h5py.Dataset, "__getitem__")
        result = stereoseq(fixture.root, image_roles=("registered",))

        read.assert_not_called()
        image = next(iter(result.images.values()))
        transform = get_transformation(image, to_coordinate_system="global")
        assert not isinstance(transform, dict)
        assert_array_equal(
            transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.diag([0.5, 0.5, 1])
        )

    def test_saw7_cell_correction_distance_mask(self, tmp_path: Path) -> None:
        # A8/A9 place this cellCorrect output in 051.cellcut, not 04.register.
        path = tmp_path / "051.cellcut" / "HE_sample_mask_edm_dis_10.tif"
        path.parent.mkdir()
        pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
        tifffile.imwrite(path, pixels, metadata={"axes": "YX"})

        result = stereoseq(tmp_path, image_roles=("distance_mask",))

        assert set(result.labels) == {"HE_distance_mask_10"}
        assert result.labels["HE_distance_mask_10"].dims == ("y", "x")
        assert_array_equal(result.labels["HE_distance_mask_10"].data.compute(), pixels)
        assert result.images == result.tables == result.points == result.shapes == {}

    @pytest.mark.parametrize("family", ["compact_numbered", "numbered", "outs", "flat"])
    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    def test_raw_selected_products(self, tmp_path: Path, family: Family, profile: Profile) -> None:
        fixture = write_dataset(
            tmp_path, family=family, profile=profile, products=("square", "cells", "adjusted_cells")
        )

        result = stereoseq(
            fixture.root,
            bin_sizes=(50,),
            cell_bins="both",
            square_counts_file=fixture.square if family == "flat" else None,
            cell_counts_file=fixture.cells if family == "flat" else None,
            adjusted_cell_counts_file=fixture.adjusted_cells if family == "flat" else None,
        )
        assert set(result.images) == set()
        assert set(result.shapes) == set()
        assert set(result.tables) == {"bin50_table", "cells_table", "adjusted_cells_table"}
        assert set(result.points) == {"bin50_genes", "cells_centroids", "adjusted_cells_centroids"}
        features = ["z", "a"] if profile == "name_only" else ["g_z", "g_a"]
        square = result.tables["bin50_table"]
        assert square.var_names.tolist() == features
        assert _counts(square).dtype == np.uint64
        assert_array_equal(_counts(square).toarray(), [[6, 0], [0, 3]])
        assert_array_equal(_counts(square, layer="exon").toarray(), [[3, 0], [0, 1]])
        assert_array_equal(result.points["bin50_genes"].compute()[["x", "y"]], [[5, 6], [1, 2]])
        for name, offset in (("cells", 0), ("adjusted_cells", 1)):
            table = result.tables[f"{name}_table"]
            assert table.var_names.tolist() == features
            assert table.obs_names.tolist() == ["200", "100"]
            assert _counts(table).dtype == np.uint64
            assert_array_equal(_counts(table).toarray(), [[11, 0], [0, 22]])
            assert_array_equal(_counts(table, layer="exon").toarray(), [[5, 0], [0, 6]])
            assert_array_equal(
                result.points[f"{name}_centroids"].compute()[["x", "y"]], [[10 + offset, 20], [30 + offset, 40]]
            )
        for table in result.tables.values():
            assert _counts(table).has_canonical_format
            assert isinstance(table.X, sparray)
            assert not isinstance(table.X, spmatrix)

    def test_overflow_has_independent_integer_oracle(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, overflow=True, products=("square",))

        result = stereoseq(fixture.root, bin_sizes=(1,))

        # These two independently written records represent the same coordinate/feature.
        assert int(_counts(result.tables["bin1_table"])[0, 0]) == sum([200, 100]) == 300

    @pytest.mark.parametrize("reordered", [False, True])
    def test_analysis_preserves_order_and_checks_identity(self, tmp_path: Path, *, reordered: bool) -> None:
        fixture = write_dataset(tmp_path, family="outs", profile="id_and_name", products=("cells", "analysis"))
        write_analysis(fixture.analysis, profile="id_and_name", reordered=reordered)
        source = ad.read_h5ad(fixture.analysis)

        result = stereoseq(fixture.root, cell_bins="unadjusted", analysis_files=(fixture.analysis,))
        table = result.tables["cells_analysis_r0_5_table"]
        assert table.obs_names.tolist() == source.obs_names.tolist()
        assert table.var_names.tolist() == source.var_names.tolist()
        assert isinstance(source.X, csr_array | csr_matrix)
        assert_array_equal(_counts(table).toarray(), source.X.toarray())
        assert_array_equal(table.obsm["spatial_grid"], source.obsm["spatial"])
        assert_array_equal(table.obsm["X_pca"], source.obsm["X_pca"])
        assert_array_equal(table.obs["id"], source.obs["id"])
        points = result.points["cells_analysis_r0_5_points"].compute()
        assert_array_equal(points[["x", "y"]], source.obsm["spatial"])
        assert set(table.obs["instance_id"]) == set(points.index)
        assert isinstance(table.X, sparray)

    def test_unrelated_analysis_identity_is_rejected(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", profile="id_and_name", products=("cells", "analysis"))
        write_analysis(fixture.analysis, profile="id_and_name", mismatched=True)

        with pytest.raises(ReaderFormatError):
            stereoseq(fixture.root, cell_bins="unadjusted", analysis_files=(fixture.analysis,))

    def test_analysis_only_needs_no_gef(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", profile="id_and_name", products=("analysis",))

        result = stereoseq(fixture.root, analysis_files=(fixture.analysis,))
        assert set(result.tables) == {"cells_analysis_r0_5_table"}
        assert set(result.points) == {"cells_analysis_r0_5_points"}

    def test_filtered_analysis_retains_its_independent_target(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", profile="id_and_name", products=("cells", "analysis"))
        subset = ad.read_h5ad(fixture.analysis)[[1], [1]].copy()
        subset.write_h5ad(fixture.analysis)

        result = stereoseq(fixture.root, cell_bins="unadjusted", analysis_files=(fixture.analysis,))
        analysis = result.tables["cells_analysis_r0_5_table"]
        assert analysis.shape == (1, 1)
        assert_array_equal(_counts(analysis).toarray(), [[5]])
        assert analysis.var_names.tolist() == ["g_a"]
        assert analysis.obs["id"].tolist() == [100]
        assert result.tables["cells_table"].shape == (2, 2)
        assert set(analysis.obs["instance_id"]) == set(result.points["cells_analysis_r0_5_points"].compute().index)

    def test_adjusted_analysis_is_a_separate_population(self, tmp_path: Path) -> None:
        fixture = write_dataset(
            tmp_path, family="outs", profile="id_and_name", products=("adjusted_cells", "adjusted_analysis")
        )

        result = stereoseq(fixture.root, cell_bins="adjusted", analysis_files=(fixture.adjusted_analysis,))
        assert set(result.tables) == {"adjusted_cells_table", "adjusted_cells_analysis_r0_5_table"}
        analysis = result.tables["adjusted_cells_analysis_r0_5_table"]
        assert_array_equal(analysis.obsm["spatial_grid"], [[11, 20], [31, 40]])
        assert_array_equal(_counts(analysis).toarray(), [[1.25, 2.5], [3.75, 5]])

    @pytest.mark.parametrize("historical", [False, True])
    def test_square_analysis_uses_explicit_native_bin_identity(self, tmp_path: Path, *, historical: bool) -> None:
        family: Family = "numbered" if historical else "outs"
        fixture = write_dataset(tmp_path, family=family, profile="id_and_name", products=("square",))
        relative = (
            "06.spatialcluster/sample.spatial.cluster.h5ad" if historical else "outs/analysis/sample.bin50_0.5.h5ad"
        )
        path = tmp_path / relative
        write_analysis(path, profile="id_and_name")
        analysis = ad.read_h5ad(path)
        positions = np.array([[5, 6], [1, 2]], dtype=np.uint32)
        observations = analysis.obs
        assert isinstance(observations, pd.DataFrame)
        analysis.obs = observations.drop(columns=["id"])
        analysis.obs["x"] = positions[:, 0]
        analysis.obs["y"] = positions[:, 1]
        analysis.obs_names = [str((int(x) << 32) | int(y)) for x, y in positions]
        analysis.obsm["spatial"] = positions
        analysis.uns["bin_type"] = "bins"
        analysis.uns["bin_size"] = 50
        analysis.write_h5ad(path)

        result = stereoseq(fixture.root, bin_sizes=(50,), analysis_files=(path,))
        name = "bin50_analysis_table" if historical else "bin50_analysis_r0_5_table"
        assert set(result.tables) == {"bin50_table", name}
        assert_array_equal(_counts(result.tables[name]).toarray(), [[1.25, 2.5], [3.75, 5]])

    def test_centroid_mode_does_not_read_unselected_borders(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("cells",))
        with h5py.File(fixture.cells, "r+") as handle:
            del handle["cellBin/cellBorder"]

        result = stereoseq(fixture.root, cell_bins="unadjusted")
        assert result.tables["cells_table"].n_obs == 2
        with pytest.raises(ReaderFormatError):
            stereoseq(fixture.root, cell_bins="unadjusted", cell_geometry="polygons")

    def test_polygons_and_linkage(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("cells",))

        result = stereoseq(fixture.root, cell_bins="unadjusted", cell_geometry="polygons")
        shape = result.shapes["cells_polygons"]
        assert shape.geometry.iloc[0].bounds == (9, 19, 11, 21)
        assert shape.geometry.iloc[1].bounds == (29, 39, 31, 41)
        assert get_transformation(shape) == Scale([0.5, 0.5], axes=("x", "y"))
        assert set(result.tables["cells_table"].obs["instance_id"]) == set(shape.index)

    def test_selected_empty_population_fails_linkage(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("square",), bins=(1,))
        with h5py.File(fixture.square, "r+") as handle:
            group = handle["geneExp/bin1"]
            features = group["gene"][:]
            features["offset"] = 0
            features["count"] = 0
            group["gene"][:] = features
            records = group["expression"]
            attributes = dict(records.attrs)
            empty = records[:0]
            del group["expression"]
            dataset = group.create_dataset("expression", data=empty)
            for name, value in attributes.items():
                dataset.attrs[name] = value
            del group["exon"]
            group["exon"] = np.empty(0, dtype=np.uint8)

        with pytest.raises(ReaderFormatError):
            stereoseq(fixture.root, bin_sizes=(1,))

    def test_requested_missing_bin_fails(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("square",), bins=(1,))

        with pytest.raises(ReaderFormatError, match="50"):
            stereoseq(fixture.root, bin_sizes=(50,))

    def test_all_bins_is_explicit(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, products=("square",))

        result = stereoseq(fixture.root, bin_sizes="all")
        assert set(result.tables) == {"bin1_table", "bin50_table"}

    def test_sparse_round_trip(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path / "source", family="outs", profile="id_and_name", products=("square", "cells"))

        result = stereoseq(fixture.root, bin_sizes=(1,), cell_bins="unadjusted")
        destination = tmp_path / "result.zarr"
        result.write(destination)
        restored = SpatialData.read(destination)
        for name in result.tables:
            restored_matrix = restored.tables[name].X
            assert isinstance(restored_matrix, csr_array | csr_matrix)
            assert_array_equal(restored_matrix.toarray(), _counts(result.tables[name]).toarray())
            assert restored.tables[name].obs_names.tolist() == result.tables[name].obs_names.tolist()
            assert restored.tables[name].var_names.tolist() == result.tables[name].var_names.tolist()
            assert_array_equal(restored.tables[name].obs["instance_id"], result.tables[name].obs["instance_id"])

    @pytest.mark.parametrize("family", ["compact_numbered", "numbered", "outs"])
    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    def test_roles_and_lazy_pixels(self, tmp_path: Path, family: Family, profile: Profile) -> None:
        fixture = write_dataset(
            tmp_path,
            family=family,
            profile=profile,
            products=("registered", "mask", "tissue_cut", "expression", "expression_tissue_cut"),
        )

        executed: list[object] = []

        def record(key: object, *_: object) -> None:
            executed.append(key)

        with Callback(posttask=record):
            result = stereoseq(fixture.root, image_roles=("registered", "mask", "tissue_cut", "expression"))
        assert executed == []
        stain = "" if family == "compact_numbered" else "HE_"
        assert set(result.images) == {f"{stain}registered", "expression"}
        assert set(result.labels) == {f"{stain}mask", f"{stain}tissue_cut", "expression_tissue_cut"}
        assert not result.tables
        assert not result.points
        assert not result.shapes
        source = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
        binary = (source % 3 == 0).astype(np.uint8)
        for image in result.images.values():
            assert image.dims == ("c", "y", "x")
            assert image.shape == (1, 16, 24)
            assert isinstance(get_transformation(image), Identity)
            with Callback(posttask=record):
                computed = image.data[:, 2:4, 5:8].compute()
            assert_array_equal(computed, source[None, 2:4, 5:8])
        for name, labels in result.labels.items():
            assert labels.dims == ("y", "x")
            assert labels.shape == (16, 24)
            assert isinstance(get_transformation(labels), Identity)
            expected = source if name == f"{stain}tissue_cut" else binary
            assert labels.dtype == expected.dtype
            with Callback(posttask=record):
                computed = labels.data[2:4, 5:8].compute()
            assert_array_equal(computed, expected[2:4, 5:8])
        assert executed

    def test_scale_factors_apply_to_images_and_masks(self, tmp_path: Path) -> None:
        fixture = write_dataset(tmp_path, family="outs", profile="name_only", products=("registered", "mask"))

        result = stereoseq(fixture.root, image_roles=("registered", "mask"), scale_factors=(2,), chunks=(1, 8, 8))

        assert set(result.images) == {"HE_registered"}
        assert set(result.labels) == {"HE_mask"}
        source = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
        image = result.images["HE_registered"]
        assert isinstance(image, DataTree)
        assert list(image.children) == ["scale0", "scale1"]
        assert image["scale0"]["image"].data.chunksize == (1, 8, 8)
        assert_array_equal(image["scale0"]["image"].data.compute(), source[None])
        assert image["scale1"]["image"].shape == (1, 8, 12)

        mask = result.labels["HE_mask"]
        assert isinstance(mask, DataTree)
        assert list(mask.children) == ["scale0", "scale1"]
        assert mask["scale0"]["image"].data.chunksize == (8, 8)
        assert_array_equal(mask["scale0"]["image"].data.compute(), (source % 3 == 0).astype(np.uint8))
        downsampled = mask["scale1"]["image"].data.compute()
        assert downsampled.shape == (8, 12)
        assert downsampled.dtype == np.uint8
        assert set(np.unique(downsampled).tolist()) == {0, 1}

        destination = tmp_path / "result.zarr"
        result.write(destination)
        restored = SpatialData.read(destination).labels["HE_mask"]
        assert isinstance(restored, DataTree)
        assert list(restored.children) == ["scale0", "scale1"]
        assert_array_equal(restored["scale1"]["image"].data.compute(), downsampled)


class TestWriteDataset:
    """Freeze source records separately from every reader implementation."""

    @pytest.mark.parametrize(
        ("family", "profile"),
        [
            pytest.param("compact_numbered", "name_only", id="compact-structural-export"),
            pytest.param("numbered", "name_only", id="shared-saw7-numbered-contract"),
            pytest.param("outs", "id_and_name", id="shared-saw8-outs-contract"),
            pytest.param("flat", "id_and_name", id="explicit-flat-export"),
        ],
    )
    def test_source_profiles(self, tmp_path: Path, family: Family, profile: Profile) -> None:
        fixture = write_dataset(
            tmp_path,
            family=family,
            profile=profile,
            products=(
                "square",
                "cells",
                "adjusted_cells",
                "analysis",
                "adjusted_analysis",
                "registered",
                "mask",
                "tissue_cut",
                "expression",
                "expression_tissue_cut",
            ),
        )
        with h5py.File(fixture.square, "r") as handle:
            assert sorted(handle["geneExp"]) == ["bin1", "bin50"]
            # GEF schema version is meaningful; no synthetic SAW release is claimed.
            assert set(handle.attrs) == {"version", "offsetX", "offsetY"}
            expression = handle["geneExp/bin1/expression"][:]
            assert sum(int(value) for value in expression["count"][:2]) == 6
            assert_array_equal(expression["x"], [5, 5, 1])
            assert_array_equal(expression["y"], [6, 6, 2])
        with h5py.File(fixture.cells, "r") as handle:
            assert_array_equal(handle["cellBin/cell"]["id"], [200, 100])
            assert_array_equal(handle["cellBin/cellExp"]["count"], [11, 22])
            assert_array_equal(handle["cellBin/cellExpExon"][:], [5, 6])
        assert fixture.adjusted_cells.is_file()
        assert fixture.analysis.is_file()
        assert fixture.adjusted_analysis.is_file()
        assert len(list(fixture.image_directory.glob("*.tif"))) == 5
