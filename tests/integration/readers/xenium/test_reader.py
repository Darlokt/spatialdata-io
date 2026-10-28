"""Public Xenium workflows per family: exact elements, frames, linkage, policies and round trips."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest
import shapely
from click.testing import CliRunner
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Affine, BaseTransformation, Identity, Scale, Translation, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io import __version__
from spatialdata_io._cli.commands.xenium import xenium_command
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium import xenium
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame as DaskDataFrame
    from geopandas import GeoDataFrame
    from numpy.typing import NDArray
    from spatialdata import SpatialData

PROFILES = [fx.XOA_1_0, fx.XOA_1_3, fx.XOA_1_9, fx.XOA_2_0, fx.XOA_3_0, fx.XOA_4_0, fx.FORMAT_6_1]
PROFILE_IDS = ["xoa-1.0", "xoa-1.3", "xoa-1.9", "xoa-2.0", "xoa-3.0", "xoa-4.0", "format-6.1"]
DEFAULT_TYPES = ("Gene Expression", "Protein Expression")
HALF = fx.PIXEL_SIZE / 2
STORE_COLUMNS = ("label_id", "nucleus_centroid_x", "nucleus_centroid_y")


def _x(table: AnnData) -> sparse.csr_array:
    """Narrow AnnData's broad expression union to the reader's sparse contract."""
    assert isinstance(table.X, sparse.csr_array)
    return table.X


def _frame(frame: object) -> pd.DataFrame:
    """Narrow AnnData's annotation union; readers return pandas annotations."""
    assert isinstance(frame, pd.DataFrame)
    return frame


def _array(element: DataArray | DataTree) -> DataArray:
    assert isinstance(element, DataArray)
    return element


def _transform(element: DataArray | DataTree | GeoDataFrame | DaskDataFrame) -> BaseTransformation:
    transformation = get_transformation(element, to_coordinate_system="global")
    assert isinstance(transformation, BaseTransformation)
    return transformation


def _read_without_store(root: Path, *, nuclei: bool) -> SpatialData:
    """Read the products a bundle without cells.zarr.zip still supports."""
    return xenium(
        root,
        include_cell_labels=False,
        include_nucleus_labels=False,
        include_morphology_focus=False,
        include_nucleus_boundaries=nuclei,
        table_target="cell_boundaries",
    )


def _expected_x(bundle: fx.XeniumFixture, types: tuple[str, ...] = DEFAULT_TYPES) -> NDArray[np.float32]:
    """Count-matrix columns of the selected types; protein divided once in float32."""
    columns = [index for index, kind in enumerate(bundle.feature_types) if kind in types]
    values = bundle.counts[:, columns].astype(np.float32)
    protein = np.array([bundle.feature_types[index] == "Protein Expression" for index in columns], dtype=bool)
    values[:, protein] /= np.float32(fx.PROTEIN_FACTOR)
    return values


def _q20_counts(result: SpatialData, table: AnnData) -> NDArray[np.int64]:
    """Count assigned q20 transcripts per table cell and table feature."""
    points = result.points["transcripts"].compute()
    assigned = points[(points["qv"] >= 20) & points["cell_id"].isin(table.obs["cell_id"])]
    rows = pd.Index(table.obs["cell_id"]).get_indexer(assigned["cell_id"])
    columns = table.var_names.get_indexer(assigned["feature_name"].astype("str"))
    counts = np.zeros(table.shape, dtype=np.int64)
    np.add.at(counts, (rows[columns >= 0], columns[columns >= 0]), 1)
    return counts


class TestXenium:
    """Every family returns exact, explicitly linked micrometre elements."""

    @pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
    def test_family_contract(self, tmp_path: Path, profile: fx.Profile) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile)
        tasks: list[object] = []

        with Callback(posttask=lambda *args: tasks.append(args)):
            result = xenium(bundle.root)

        assert tasks == []
        assert sorted(result.images) == ["morphology_focus"]
        assert sorted(result.labels) == ["cell_labels", "nucleus_labels"]
        assert sorted(result.shapes) == ["cell_boundaries", "nucleus_boundaries"]
        assert sorted(result.points) == ["transcripts"]
        assert sorted(result.tables) == ["table"]

        table = result.tables["table"]
        x = _x(table)
        assert x.dtype == np.float32
        assert x.has_canonical_format
        np.testing.assert_array_equal(x.toarray(), _expected_x(bundle))
        assert table.obs_names.tolist() == list(bundle.barcodes)
        assert table.var["feature_types"].tolist() == [kind for kind in bundle.feature_types if kind in DEFAULT_TYPES]
        assert get_table_keys(table) == ("cell_labels", "region", "label_id")
        np.testing.assert_array_equal(table.obs["label_id"].to_numpy(), np.asarray(bundle.cell_labels, np.uint32))
        assert table.obs["label_id"].dtype == np.uint32
        np.testing.assert_array_equal(table.obsm["spatial"], _frame(table.obs)[["x_centroid", "y_centroid"]].to_numpy())

        cell_labels = _array(result.labels["cell_labels"])
        nucleus_labels = _array(result.labels["nucleus_labels"])
        np.testing.assert_array_equal(cell_labels.data.compute(), bundle.cell_mask)
        np.testing.assert_array_equal(nucleus_labels.data.compute(), bundle.nucleus_mask)
        image = _array(result.images["morphology_focus"])
        assert image.coords["c"].to_numpy().tolist() == list(profile.channels)
        np.testing.assert_array_equal(image.data.compute(), bundle.focus)
        for raster in (cell_labels, nucleus_labels, image):
            scale = _transform(raster)
            assert isinstance(scale, Scale)
            np.testing.assert_array_equal(scale.scale, [fx.PIXEL_SIZE, fx.PIXEL_SIZE])

        cells = result.shapes["cell_boundaries"]
        nuclei = result.shapes["nucleus_boundaries"]
        assert cells.index.name == "cell_id"
        assert cells.index.tolist() == list(bundle.cell_ids)
        assert nuclei.index.name == "label_id"
        assert nuclei.index.tolist() == list(bundle.nucleus_labels)
        assert nuclei["cell_id"].tolist() == [bundle.cell_ids[owner] for owner, _ in bundle.nuclei]
        for shapes in (cells, nuclei):
            translation = _transform(shapes)
            assert isinstance(translation, Translation)
            np.testing.assert_array_equal(translation.translation, [HALF, HALF])
        assert isinstance(_transform(result.points["transcripts"]), Identity)

        assert result.attrs["spatialdata_io_reader"] == "xenium"
        assert result.attrs["spatialdata_io_software_version"] == __version__
        version = (
            None if profile.version is None else ".".join(str(part) for part in profile.version if part is not None)
        )
        assert result.attrs.get("spatialdata_io_format_version") == version
        experiment = result.attrs["xenium"]["experiment"]
        assert experiment["pixel_size"] == fx.PIXEL_SIZE
        assert "images" not in experiment
        assert "xenium_explorer_files" not in experiment

    @pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
    def test_registered_vertices_fall_on_their_mask_pixels(self, tmp_path: Path, profile: fx.Profile) -> None:
        # Vertices trace boundary-pixel centres in index coordinates; the half-pixel
        # translation moves them onto those pixel centres in the corner frame.
        result = xenium(fx.write_xenium_bundle(tmp_path, profile).root)
        mask = _array(result.labels["cell_labels"]).data.compute()
        labels = _frame(result.tables["table"].obs)["label_id"]
        shapes = result.shapes["cell_boundaries"]
        translation = _transform(shapes)
        assert isinstance(translation, Translation)

        for label, geometry in zip(labels, shapes.geometry, strict=True):
            registered = shapely.get_coordinates(geometry) + translation.translation
            pixels = np.floor(registered / fx.PIXEL_SIZE).astype(np.int64)
            assert (mask[pixels[:, 1], pixels[:, 0]] == label).all()

    @pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
    def test_assigned_transcripts_fall_in_their_cell_labels(self, tmp_path: Path, profile: fx.Profile) -> None:
        result = xenium(fx.write_xenium_bundle(tmp_path, profile).root)
        points = result.points["transcripts"].compute()
        mask = _array(result.labels["cell_labels"]).data.compute()
        labels = _frame(result.tables["table"].obs).set_index("cell_id")["label_id"]

        assigned = points[points["cell_id"].isin(labels.index)]
        pixels = np.floor(assigned[["x", "y"]].to_numpy() / fx.PIXEL_SIZE).astype(np.int64)
        assert len(assigned) == 24
        np.testing.assert_array_equal(mask[pixels[:, 1], pixels[:, 0]], labels[assigned["cell_id"]].to_numpy())

    @pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
    def test_q20_transcripts_reproduce_the_counts(self, tmp_path: Path, profile: fx.Profile) -> None:
        result = xenium(fx.write_xenium_bundle(tmp_path, profile).root, feature_types=("Gene Expression",))
        table = result.tables["table"]

        np.testing.assert_array_equal(_q20_counts(result, table), _x(table).toarray())

    def test_protein_metadata(self, tmp_path: Path) -> None:
        result = xenium(fx.write_xenium_bundle(tmp_path, fx.XOA_4_0).root)
        table = result.tables["table"]
        var = _frame(table.var)
        protein = var[var["feature_types"] == "Protein Expression"]

        np.testing.assert_array_equal(
            _x(table[:, protein.index]).toarray(), np.asarray(fx.PROTEIN_STORED, np.float32) / np.float32(10)
        )
        assert protein["long_name"].tolist() == ["PD-1 protein", "VISTA protein"]
        assert protein["codeword_index"].tolist() == [16, 17]
        assert result.attrs["xenium"]["protein"] == {"scaling_factor": 10, "metric": "MEAN_PER_CELL_STAIN"}
        assert result.attrs["xenium"]["protein_panel"]["panel_type"] == "PANEL_TYPE_PROTEIN"

    @pytest.mark.parametrize(
        "feature_types", ["all", ("Negative Control Probe", "Gene Expression")], ids=["all", "explicit"]
    )
    def test_feature_types(self, tmp_path: Path, feature_types: tuple[str, ...] | Literal["all"]) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        selected = tuple(dict.fromkeys(bundle.feature_types)) if feature_types == "all" else feature_types

        table = xenium(bundle.root, feature_types=feature_types).tables["table"]

        # Columns keep the source order whatever order the selection names.
        np.testing.assert_array_equal(_x(table).toarray(), _expected_x(bundle, selected))
        assert table.var_names.tolist() == [
            name for name, kind in zip(bundle.features, bundle.feature_types, strict=True) if kind in selected
        ]

    def test_unknown_feature_type_raises(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_9)

        with pytest.raises(ReaderFormatError, match="Protein Expression"):
            xenium(bundle.root, feature_types=("Protein Expression",))

    @pytest.mark.parametrize(
        ("target", "geometry"),
        [("cell_boundaries", "polygons"), ("cell_circles", "both")],
        ids=["polygons", "circles"],
    )
    def test_shape_targets(
        self,
        tmp_path: Path,
        target: Literal["cell_boundaries", "cell_circles"],
        geometry: Literal["polygons", "both"],
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        keys = (target, "region", "cell_id")

        result = xenium(bundle.root, table_target=target, cell_geometry=geometry)

        table = result.tables["table"]
        assert get_table_keys(table) == keys
        assert table.obs["cell_id"].tolist() == list(bundle.cell_ids)
        assert table.obs["region"].tolist() == [keys[0]] * 4

    def test_circles_have_cell_area_and_centroids(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)

        result = xenium(bundle.root, cell_geometry="circles", table_target="cell_circles")

        assert sorted(result.shapes) == ["cell_circles", "nucleus_boundaries"]
        circles = result.shapes["cell_circles"]
        obs = result.tables["table"].obs
        np.testing.assert_allclose(np.pi * circles["radius"].to_numpy() ** 2, obs["cell_area"], rtol=1e-15)
        np.testing.assert_array_equal(shapely.get_coordinates(circles.geometry), obs[["x_centroid", "y_centroid"]])
        assert isinstance(_transform(circles), Identity)

    def test_unlinked_and_absent_tables(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)

        unlinked = xenium(bundle.root, table_target=None).tables["table"]
        without = xenium(bundle.root, include_table=False)

        assert "spatialdata_attrs" not in unlinked.uns
        assert "region" not in unlinked.obs
        assert dict(without.tables) == {}

    def test_drop_removes_whole_entities(self, tmp_path: Path) -> None:
        # Cell 2 has a self-intersecting polygon and owns nuclei 3 and 4.
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(bowtie_cells=(2,), bowtie_nuclei=(0,)))
        kept_cells = [cell for index, cell in enumerate(fx.HEX_IDS) if index != 2]

        result = xenium(bundle.root, invalid_geometries="drop", cell_geometry="both", table_target="cell_boundaries")

        assert result.shapes["cell_boundaries"].index.tolist() == kept_cells
        assert result.shapes["cell_circles"].index.tolist() == kept_cells
        assert result.shapes["nucleus_boundaries"].index.tolist() == [2]
        assert result.tables["table"].obs_names.tolist() == kept_cells
        np.testing.assert_array_equal(_x(result.tables["table"]).toarray(), _expected_x(bundle)[[0, 1, 3]])
        np.testing.assert_array_equal(_array(result.labels["cell_labels"]).data.compute(), bundle.cell_mask)
        assert len(result.points["transcripts"]) == len(bundle.transcripts.x)

    def test_keep_and_raise_invalid_polygons(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(bowtie_cells=(2,)))

        kept = xenium(bundle.root).shapes["cell_boundaries"]
        assert not kept.geometry.iloc[2].is_valid
        with pytest.raises(ReaderFormatError, match=r"1 invalid geometries .*\['iaaaaaaf-1'\]"):
            xenium(bundle.root, invalid_geometries="raise")

    def test_degenerate_import_rings_need_drop(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(short_nuclei=(0,)))

        with pytest.raises(ReaderFormatError, match="fewer than four coordinates"):
            xenium(bundle.root)
        dropped = xenium(bundle.root, invalid_geometries="drop")
        assert dropped.shapes["nucleus_boundaries"].index.tolist() == [2, 3, 4]
        assert dropped.tables["table"].n_obs == 4

    def test_transcript_assignment_import(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(import_kind="transcripts"))

        with pytest.raises(FileNotFoundError, match="nucleus_boundaries"):
            xenium(bundle.root)
        result = xenium(
            bundle.root,
            include_cell_labels=False,
            include_nucleus_labels=False,
            include_nucleus_boundaries=False,
            table_target="cell_boundaries",
        )
        assert sorted(result.shapes) == ["cell_boundaries"]
        assert dict(result.labels) == {}
        obs = result.tables["table"].obs
        assert obs["segmentation_method"].tolist() == ["Imported Cell Segmentation"] * 4
        # The store still defines each cell's label as its polygon index + 1.
        assert obs["label_id"].tolist() == [1, 2, 3, 4]

    def test_import_without_masks_cannot_return_labels(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(import_kind="transcripts"))

        with pytest.raises(ReaderFormatError, match="no usable cell mask"):
            xenium(bundle.root, include_nucleus_labels=False, include_nucleus_boundaries=False)

    def test_cells_only_import(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(import_kind="cells_only"))

        with pytest.raises(ReaderFormatError, match="include_nucleus_boundaries=False; found no polygons"):
            xenium(bundle.root)
        result = xenium(bundle.root, include_nucleus_boundaries=False)
        assert sorted(result.shapes) == ["cell_boundaries"]
        assert not _array(result.labels["nucleus_labels"]).data.compute().any()
        assert result.tables["table"].obs["nucleus_count"].tolist() == [0, 0, 0, 0]

    @pytest.mark.parametrize("profile", [fx.XOA_1_0, fx.XOA_1_9, fx.XOA_3_0], ids=["integer", "hex", "polygon-sets"])
    def test_stripped_deposit_equals_the_full_bundle(self, tmp_path: Path, profile: fx.Profile) -> None:
        stripped = fx.write_xenium_bundle(
            tmp_path / "stripped",
            profile,
            fx.Options(tables="csv_gz", counts=("mtx_archive",), transcripts="csv", cells_store=False, images=False),
        )
        full = fx.write_xenium_bundle(tmp_path / "full", profile)
        # Hexadecimal nuclei carry their labels only in cells.zarr.zip.
        nuclei = profile.family != "hex_ids"

        with pytest.raises(FileNotFoundError, match=r"cells\.zarr\.zip"):
            xenium(stripped.root)
        result = _read_without_store(stripped.root, nuclei=nuclei)
        reference = _read_without_store(full.root, nuclei=nuclei)

        np.testing.assert_array_equal(_x(result.tables["table"]).toarray(), _x(reference.tables["table"]).toarray())
        obs, reference_obs = _frame(result.tables["table"].obs), _frame(reference.tables["table"].obs)
        assert set(reference_obs.columns) - set(obs.columns) == set(STORE_COLUMNS)
        pd.testing.assert_frame_equal(obs, reference_obs[obs.columns])
        for name in result.shapes:
            assert result.shapes[name].geometry.geom_equals_exact(reference.shapes[name].geometry, tolerance=0).all()
        pd.testing.assert_frame_equal(
            result.points["transcripts"].compute().reset_index(drop=True),
            reference.points["transcripts"].compute(),
        )

    def test_hex_nuclei_without_the_store_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_9, fx.Options(cells_store=False))

        with pytest.raises(ReaderFormatError, match=r"seg_mask_value labels from cells\.zarr\.zip"):
            xenium(bundle.root, include_cell_labels=False, include_nucleus_labels=False, table_target=None)

    def test_aligned_images(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_2_0, fx.Options(aligned=("he", "if")))

        result = xenium(bundle.root)

        assert sorted(result.images) == ["he_image", "if_image", "morphology_focus"]
        assert isinstance(_transform(result.images["he_image"]), Affine)
        assert isinstance(_transform(result.images["if_image"]), Scale)
        assert _array(result.images["if_image"]).coords["c"].to_numpy().tolist() == ["CD20", "DAPI"]
        assert sorted(xenium(bundle.root, include_aligned_images=False).images) == ["morphology_focus"]

    @pytest.mark.parametrize("kind", ["directory", "archive"])
    def test_analysis(self, tmp_path: Path, kind: Literal["directory", "archive"]) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(analysis=kind))

        table = xenium(bundle.root, include_analysis=True).tables["table"]

        assert table.obs["gene_expression_graphclust"].astype("object").tolist()[:3] == ["1", "2", "1"]
        assert pd.isna(table.obs["gene_expression_graphclust"].iloc[3])
        assert table.obsm["pca_gene_expression_3_components"].shape == (4, 3)
        assert table.obsm["umap_gene_expression_2_components"].shape == (4, 2)
        assert get_table_keys(table) == ("cell_labels", "region", "label_id")

    def test_analysis_names_must_not_shadow_annotations(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(analysis="directory"))
        clustering = bundle.root / "analysis" / "clustering"
        (clustering / "gene_expression_graphclust").rename(clustering / "cell_area")

        with pytest.raises(ReaderFormatError, match=r"clusterings \['cell_area'\] named like cell annotations"):
            xenium(bundle.root, include_analysis=True)

    def test_mip_of_xoa_1(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_9)

        mip = _array(xenium(bundle.root, include_morphology_mip=True).images["morphology_mip"])

        np.testing.assert_array_equal(mip.data.compute(), bundle.focus + np.uint16(500))

    @pytest.mark.parametrize(("profile", "ranger"), [(fx.XOA_3_0, "relabel"), (fx.XOA_1_9, "rename")])
    def test_ranger_outputs_keep_the_bundle_format_version(
        self, tmp_path: Path, profile: fx.Profile, ranger: Literal["relabel", "rename"]
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile, fx.Options(ranger=ranger))

        result = xenium(bundle.root)

        assert profile.version is not None
        assert result.attrs["spatialdata_io_format_version"] == ".".join(map(str, profile.version))
        ranger_attrs = result.attrs["xenium"]["experiment"]["xenium_ranger"]
        assert "command_line" not in ranger_attrs
        assert ranger_attrs["version"] == "xenium-3.1.0.5"

    def test_pyramids_and_chunks(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)

        result = xenium(bundle.root, scale_factors=(2,), chunks=(1, 8, 16))

        image = result.images["morphology_focus"]
        labels = result.labels["cell_labels"]
        assert isinstance(image, DataTree)
        assert isinstance(labels, DataTree)
        assert image["scale0"]["image"].data.chunksize == (1, 8, 16)
        assert labels["scale0"]["image"].data.chunksize == (8, 16)
        assert labels["scale1"]["image"].shape == (12, 16)
        np.testing.assert_array_equal(labels["scale0"]["image"].data.compute(), bundle.cell_mask)

    def test_zarr_round_trip_through_the_cli(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path / "input", fx.XOA_4_0)
        output = tmp_path / "output.zarr"

        invoked = CliRunner().invoke(xenium_command, ["-i", str(bundle.root), "-o", str(output)])

        assert invoked.exit_code == 0, invoked.output
        restored = read_zarr(output)
        table = restored.tables["table"]
        # Storage round trips return AnnData's own sparse type; compare exact values only.
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        np.testing.assert_array_equal(table.X.toarray(), _expected_x(bundle))
        assert get_table_keys(table) == ("cell_labels", "region", "label_id")
        assert len(restored.points["transcripts"]) == len(bundle.transcripts.x)
        assert restored.shapes["nucleus_boundaries"].index.tolist() == list(bundle.nucleus_labels)
        assert restored.attrs["xenium"]["protein"] == {"scaling_factor": 10, "metric": "MEAN_PER_CELL_STAIN"}
        translation = _transform(restored.shapes["cell_boundaries"])
        assert isinstance(translation, Translation)
        np.testing.assert_array_equal(translation.translation, [HALF, HALF])
