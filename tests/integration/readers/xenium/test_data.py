"""Downloaded 10x Xenium bundles from XOA 1.0.2 to 4.0, checked against their raw artifacts.

Expected values come from the vendor files themselves, read without the reader:
HDF5 counts through h5py, cells, boundaries and transcripts through Arrow, image
windows through tifffile and alignment matrices as text.
"""

from __future__ import annotations

import io
import shutil
import tarfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import shapely
import tifffile
import zarr
from click.testing import CliRunner
from scipy import sparse
from spatialdata import read_zarr
from spatialdata.models import get_table_keys
from spatialdata.transformations import Affine, BaseTransformation, Scale, Translation, get_transformation
from xarray import DataArray

from spatialdata_io._cli.commands.xenium import xenium_command
from spatialdata_io.readers.xenium import xenium

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from anndata import AnnData
    from dask.dataframe import DataFrame as DaskDataFrame
    from geopandas import GeoDataFrame
    from numpy.typing import NDArray
    from spatialdata import SpatialData
    from xarray import DataTree

PIXEL_SIZE = 0.2125
FOUR = ("DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "alphaSMA/Vimentin")
PROTEIN_KIT = (
    *FOUR,
    "Blue background",
    "Green background",
    "Yellow background",
    "Red background",
    "PD-1",
    "VISTA",
    "PD-L1",
    "LAG-3",
)
DEFAULT_TYPES = ("Gene Expression", "Protein Expression")
TWINS = ("cells.csv.gz", "cell_boundaries.csv.gz", "nucleus_boundaries.csv.gz", "cell_feature_matrix.tar.gz")
METADATA = ("experiment.xenium", "metrics_summary.csv", "gene_panel.json")
STORE_COLUMNS = {"label_id", "nucleus_centroid_x", "nucleus_centroid_y"}
POLYGON_SET_COLUMNS = {"nucleus_count", "segmentation_method"}
# cell_area is mask pixels times a pixel area: XOA 1.0-1.3 use the float64 square
# of the pixel size, XOA 1.9 onwards the float32 product (measured on these bundles).
FLOAT64_AREA = PIXEL_SIZE**2
FLOAT32_AREA = float(np.float32(PIXEL_SIZE) * np.float32(PIXEL_SIZE))
WINDOW_MARGIN = 8


@dataclass(frozen=True, slots=True)
class Dataset:
    """One registered bundle and the facts its raw files state."""

    key: str
    version: str | None
    family: Literal["integer_ids", "hex_ids", "polygon_sets"]
    channels: tuple[str, ...]
    twins: bool = True
    pixel_area: float = FLOAT32_AREA


DATASETS = (
    Dataset("xenium_mouse_brain_subset", None, "integer_ids", ("DAPI",), pixel_area=FLOAT64_AREA),
    Dataset("xenium_brain_addon", "2.0", "hex_ids", ("DAPI",), twins=False, pixel_area=FLOAT64_AREA),
    Dataset("xenium_heart", "3.0.0", "hex_ids", ("DAPI",)),
    Dataset("xenium_breast", "4.0.0", "polygon_sets", FOUR),
    Dataset("xenium_lung", "4.0.0", "polygon_sets", ("DAPI",)),
    Dataset("xenium_prime_mouse_ileum", "5.0.0", "polygon_sets", ("DAPI",)),
    Dataset("xenium_prime_multicell_mouse_ileum", "5.0.0", "polygon_sets", FOUR),
    Dataset("xenium_ovary", "6.0.0", "polygon_sets", ("DAPI",)),
    Dataset("xenium_multicell_ovary", "6.0.0", "polygon_sets", FOUR),
    Dataset("xenium_protein_kidney", "6.0.0", "polygon_sets", PROTEIN_KIT),
)
WITH_TWINS = tuple(dataset for dataset in DATASETS if dataset.twins)


def _ids(datasets: tuple[Dataset, ...]) -> list[str]:
    return [dataset.key for dataset in datasets]


def _x(table: AnnData) -> sparse.csr_array:
    assert isinstance(table.X, sparse.csr_array)
    return table.X


def _obs(table: AnnData) -> pd.DataFrame:
    assert isinstance(table.obs, pd.DataFrame)
    return table.obs


def _same(actual: sparse.csr_array, expected: sparse.csr_array) -> bool:
    """Exact sparse equality, independent of stored zeros and index order."""
    return actual.shape == expected.shape and (actual - expected).count_nonzero() == 0


def _array(element: object) -> DataArray:
    assert isinstance(element, DataArray)
    return element


def _transform(element: DataArray | DataTree | GeoDataFrame | DaskDataFrame) -> BaseTransformation:
    transformation = get_transformation(element, to_coordinate_system="global")
    assert isinstance(transformation, BaseTransformation)
    return transformation


def _read_without_store(root: Path, *, nuclei: bool) -> SpatialData:
    """Read the gene table and polygons; MatrixMarket twins have no protein scaling factor."""
    return xenium(
        root,
        feature_types=("Gene Expression",),
        include_cell_labels=False,
        include_nucleus_labels=False,
        include_nucleus_boundaries=nuclei,
        include_transcripts=False,
        include_morphology_focus=False,
        table_target="cell_boundaries",
    )


def _text(values: list[object]) -> list[object]:
    return [value.decode() if isinstance(value, bytes) else value for value in values]


@dataclass(frozen=True, slots=True)
class RawMatrix:
    """The selected HDF5 matrix as 10x stores it."""

    matrix: sparse.csr_array
    features: list[object]
    barcodes: list[object]


def _raw_matrix(root: Path, types: tuple[str, ...] = DEFAULT_TYPES) -> RawMatrix:
    """Read the CSC feature-barcode matrix and divide protein once in float32 by the root factor."""
    with h5py.File(root / "cell_feature_matrix.h5", "r") as handle:
        group = handle["matrix"]
        assert isinstance(group, h5py.Group)
        features, cells = (int(value) for value in group["shape"][()])
        data, indices, indptr = (np.asarray(group[name][()]) for name in ("data", "indices", "indptr"))
        csc = sparse.csc_array((data, indices, indptr), shape=(features, cells))
        kinds = np.asarray(_text(group["features/feature_type"][()].tolist()))
        names = _text(group["features/name"][()].tolist())
        barcodes = _text(group["barcodes"][()].tolist())
        factor = handle.attrs.get("protein_scaling_factor")

    selected = np.isin(kinds, types)
    matrix = sparse.csr_array(csc.T)[:, np.flatnonzero(selected)].astype(np.float32)
    protein = kinds[selected] == "Protein Expression"
    if protein.any():
        assert factor is not None
        entries = protein[matrix.indices]
        matrix.data[entries] /= np.float32(factor)
    return RawMatrix(matrix, [name for name, keep in zip(names, selected, strict=True) if keep], barcodes)


def _window_bounds(vertices: NDArray[np.float64], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Return a mask window around a polygon with a margin, clipped to the raster."""
    low = np.floor(vertices.min(axis=0) / PIXEL_SIZE).astype(np.int64) - WINDOW_MARGIN
    high = np.ceil(vertices.max(axis=0) / PIXEL_SIZE).astype(np.int64) + WINDOW_MARGIN + 1
    left, top = (int(value) for value in np.maximum(low, 0))
    right, bottom = (int(value) for value in np.minimum(high, (shape[1], shape[0])))
    return left, top, right, bottom


def _window(path: Path, *, channel: int | None, rows: slice, columns: slice) -> NDArray[np.generic]:
    """Decode one window of the full-resolution OME series with tifffile."""
    with tifffile.TiffFile(path) as tiff:
        store = tiff.series[0].aszarr(level=0)
        array = zarr.open_array(store=store, mode="r")
        selection = (rows, columns) if channel is None else (channel, rows, columns)
        return np.asarray(array[selection])


class TestXeniumData:
    """Every registered bundle reproduces its raw vendor artifacts exactly."""

    @pytest.mark.parametrize("dataset", DATASETS, ids=_ids(DATASETS))
    def test_table_reproduces_the_raw_matrix(
        self, dataset: Dataset, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset.key)
        raw = _raw_matrix(root)

        result = xenium(root, include_transcripts=False, include_morphology_focus=False)

        table = result.tables["table"]
        x = _x(table)
        assert x.dtype == np.float32
        assert x.has_canonical_format
        assert _same(x, raw.matrix)
        assert table.var_names.tolist() == raw.features
        assert table.obs_names.tolist() == raw.barcodes
        cells = pq.read_table(root / "cells.parquet", columns=["cell_id"])["cell_id"].to_pylist()
        assert _obs(table)["cell_id"].tolist() == _text(cells)
        assert get_table_keys(table) == ("cell_labels", "region", "label_id")
        assert _obs(table)["label_id"].is_unique
        assert result.attrs.get("spatialdata_io_format_version") == dataset.version

    @pytest.mark.parametrize("dataset", DATASETS, ids=_ids(DATASETS))
    def test_q20_transcripts_reproduce_the_gene_counts(
        self, dataset: Dataset, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset.key)
        result = xenium(
            root,
            feature_types=("Gene Expression",),
            include_cell_labels=False,
            include_nucleus_labels=False,
            include_nucleus_boundaries=False,
            include_morphology_focus=False,
            table_target="cell_boundaries",
        )
        table = result.tables["table"]
        points = result.points["transcripts"][["cell_id", "feature_name", "qv"]].compute()

        # 10x counts assigned transcripts with qv >= 20 of the matrix features.
        rows = pd.Index(_obs(table)["cell_id"]).get_indexer(points["cell_id"])
        columns = table.var_names.get_indexer(points["feature_name"].astype("str"))
        kept = (points["qv"].to_numpy() >= 20) & (rows >= 0) & (columns >= 0)
        ones = np.ones(int(kept.sum()), dtype=np.int64)
        counts = sparse.csr_array((ones, (rows[kept], columns[kept])), shape=table.shape)
        counts.sum_duplicates()
        assert _same(counts, _x(table).astype(np.int64))

    @pytest.mark.parametrize("dataset", DATASETS, ids=_ids(DATASETS))
    def test_labels_polygons_and_rows_identify_the_same_cells(
        self, dataset: Dataset, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset.key)
        result = xenium(
            root, include_transcripts=False, include_morphology_focus=False, include_nucleus_boundaries=False
        )
        labels = _array(result.labels["cell_labels"]).data
        obs = _obs(result.tables["table"])
        shapes = result.shapes["cell_boundaries"]
        translation = _transform(shapes)
        assert isinstance(translation, Translation)
        assert shapes.index.tolist() == obs["cell_id"].tolist()

        # Candidate registrations of the vertices: the reader's half pixel and its neighbours.
        offsets = {"registered": translation.translation, "none": np.zeros(2), "opposite": -translation.translation}
        hits = dict.fromkeys(offsets, 0)
        vertices_seen = 0
        for row in range(min(25, len(obs))):
            vertices = shapely.get_coordinates(shapes.geometry.iloc[row])
            left, top, right, bottom = _window_bounds(vertices, labels.shape)
            window = labels[top:bottom, left:right].compute()
            label = obs["label_id"].iloc[row]
            # The window holds the whole cell: its label never touches an inner border.
            inner = [
                side
                for side, inside in (
                    (window[0], top > 0),
                    (window[-1], bottom < labels.shape[0]),
                    (window[:, 0], left > 0),
                    (window[:, -1], right < labels.shape[1]),
                )
                if inside
            ]
            assert all(label not in side for side in inner)
            np.testing.assert_allclose(
                int((window == label).sum()) * dataset.pixel_area, obs["cell_area"].iloc[row], rtol=1e-15
            )
            for name, offset in offsets.items():
                pixels = np.floor((vertices + offset) / PIXEL_SIZE).astype(np.int64)
                hits[name] += int((window[pixels[:, 1] - top, pixels[:, 0] - left] == label).sum())
            vertices_seen += len(vertices)

        # 10x simplifies polygons, so a few vertices may cut a neighbour; the
        # registration still puts at least 99 % on the cell's own pixels.
        assert hits["registered"] >= 0.99 * vertices_seen
        assert hits["registered"] > max(hits["none"], hits["opposite"])

    @pytest.mark.parametrize("dataset", DATASETS, ids=_ids(DATASETS))
    def test_morphology_focus_reproduces_the_ome_files(
        self, dataset: Dataset, require_test_dataset: Callable[[str], Path]
    ) -> None:
        root = require_test_dataset(dataset.key)

        image = _array(
            xenium(
                root,
                include_table=False,
                include_cell_labels=False,
                include_nucleus_labels=False,
                include_cell_boundaries=False,
                include_nucleus_boundaries=False,
                include_transcripts=False,
            ).images["morphology_focus"]
        )

        assert image.coords["c"].to_numpy().tolist() == list(dataset.channels)
        assert image.dtype == np.uint16
        scale = _transform(image)
        assert isinstance(scale, Scale)
        np.testing.assert_array_equal(scale.scale, [PIXEL_SIZE, PIXEL_SIZE])
        files = sorted((root / "morphology_focus").glob("*.ome.tif")) or [root / "morphology_focus.ome.tif"]
        rows = slice(image.sizes["y"] // 2, image.sizes["y"] // 2 + 48)
        columns = slice(image.sizes["x"] // 2, image.sizes["x"] // 2 + 64)
        for channel in range(len(dataset.channels)):
            # Multi-file sets store every channel in each file's CYX series.
            source = files[0] if len(files) == 1 else files[channel]
            index = None if len(dataset.channels) == 1 else channel
            expected = _window(source, channel=index, rows=rows, columns=columns)
            np.testing.assert_array_equal(image.data[channel, rows, columns].compute(), expected)

    @pytest.mark.parametrize("dataset", WITH_TWINS, ids=_ids(WITH_TWINS))
    def test_manufacturer_twins_equal_the_full_bundle(
        self, dataset: Dataset, require_test_dataset: Callable[[str], Path], tmp_path: Path
    ) -> None:
        root = require_test_dataset(dataset.key)
        for name in (*METADATA, *TWINS):
            shutil.copyfile(root / name, tmp_path / name)
        # Hexadecimal nuclei carry their labels only in cells.zarr.zip.
        nuclei = dataset.family != "hex_ids"

        stripped = _read_without_store(tmp_path, nuclei=nuclei)
        full = _read_without_store(root, nuclei=nuclei)

        table, reference = stripped.tables["table"], full.tables["table"]
        assert _same(_x(table), _x(reference))
        obs, reference_obs = _obs(table), _obs(reference)
        # Only columns the cells table lacks come from cells.zarr.zip.
        supplied = STORE_COLUMNS | (POLYGON_SET_COLUMNS if dataset.family == "polygon_sets" else set())
        stored = set(pq.read_schema(root / "cells.parquet").names)
        assert set(reference_obs.columns) - set(obs.columns) == supplied - stored
        pd.testing.assert_frame_equal(obs, reference_obs[obs.columns])
        assert sorted(stripped.shapes) == sorted(full.shapes)
        for name in stripped.shapes:
            assert stripped.shapes[name].index.equals(full.shapes[name].index)
            assert stripped.shapes[name].geometry.geom_equals_exact(full.shapes[name].geometry, tolerance=0).all()

    def test_protein_values_and_metadata(self, require_test_dataset: Callable[[str], Path]) -> None:
        root = require_test_dataset("xenium_protein_kidney")
        raw = _raw_matrix(root, ("Protein Expression",))
        with h5py.File(root / "cell_feature_matrix.h5", "r") as handle:
            factor = int(handle.attrs["protein_scaling_factor"])

        result = xenium(root, feature_types=("Protein Expression",), include_transcripts=False)

        table = result.tables["table"]
        assert _same(_x(table), raw.matrix)
        assert table.var_names.tolist() == ["PD-1", "VISTA", "PD-L1", "LAG-3"]
        assert result.attrs["xenium"]["protein"] == {"scaling_factor": factor, "metric": "MEAN_PER_CELL_STAIN"}

    def test_aligned_he_image(self, require_test_dataset: Callable[[str], Path]) -> None:
        root = require_test_dataset("xenium_heart")
        matrix = np.loadtxt(root / "he_imagealignment.csv", delimiter=",")

        image = _array(
            xenium(
                root,
                include_table=False,
                include_cell_labels=False,
                include_nucleus_labels=False,
                include_cell_boundaries=False,
                include_nucleus_boundaries=False,
                include_transcripts=False,
                include_morphology_focus=False,
            ).images["he_image"]
        )

        assert image.dims == ("c", "y", "x")
        assert image.coords["c"].to_numpy().tolist() == ["r", "g", "b"]
        affine = _transform(image)
        assert isinstance(affine, Affine)
        linear = affine.to_affine_matrix(("x", "y"), ("x", "y"))
        np.testing.assert_array_equal(linear, np.diag([PIXEL_SIZE, PIXEL_SIZE, 1.0]) @ matrix)
        # The composed scale reproduces the OME pixel size the H&E file declares.
        with tifffile.TiffFile(root / "he_image.ome.tif") as tiff:
            ome = tiff.ome_metadata
        assert ome is not None
        declared = float(ome.split('PhysicalSizeX="')[1].split('"')[0])
        np.testing.assert_allclose(np.sqrt(abs(np.linalg.det(linear[:2, :2]))), declared, rtol=1e-3)
        rows, columns = slice(5000, 5016), slice(12000, 12032)
        expected = _window(root / "he_image.ome.tif", channel=None, rows=rows, columns=columns)
        np.testing.assert_array_equal(image.data[:, rows, columns].compute(), np.moveaxis(expected, -1, 0))

    def test_analysis_matches_the_archive(self, require_test_dataset: Callable[[str], Path]) -> None:
        root = require_test_dataset("xenium_ovary")
        with tarfile.open(root / "analysis.tar.gz") as archive:
            member = archive.extractfile("analysis/clustering/gene_expression_graphclust/clusters.csv")
            assert member is not None
            clusters = pd.read_csv(io.BytesIO(member.read()), dtype={"Barcode": "str", "Cluster": "int64"})

        table = xenium(root, include_analysis=True, include_transcripts=False).tables["table"]

        observed = _obs(table)["gene_expression_graphclust"]
        expected = clusters.set_index("Barcode")["Cluster"].astype("str")
        assert observed.loc[expected.index].astype("str").tolist() == expected.tolist()
        assert observed.cat.categories.tolist() == [str(value) for value in sorted(clusters["Cluster"].unique())]
        assert table.obsm["pca_gene_expression_10_components"].shape == (table.n_obs, 10)

    def test_cli_round_trip(self, require_test_dataset: Callable[[str], Path], tmp_path: Path) -> None:
        root = require_test_dataset("xenium_protein_kidney")
        output = tmp_path / "kidney.zarr"

        invoked = CliRunner().invoke(xenium_command, ["-i", str(root), "-o", str(output)])

        assert invoked.exit_code == 0, invoked.output
        restored = read_zarr(output)
        table = restored.tables["table"]
        assert isinstance(table.X, sparse.csr_array | sparse.csr_matrix)
        assert _same(sparse.csr_array(table.X), _raw_matrix(root).matrix)
        assert get_table_keys(table) == ("cell_labels", "region", "label_id")
        assert _array(restored.images["morphology_focus"]).coords["c"].to_numpy().tolist() == list(PROTEIN_KIT)
