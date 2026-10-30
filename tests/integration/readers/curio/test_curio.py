"""Curio five-artifact export workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.tests.helpers import assert_equal
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import get_table_keys
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io import __version__, curio
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path


class TestCurio:
    """Verify scientific values, metadata, geometry, and serialization."""

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_inherited_slots_and_raw_survive_persistence(self, tmp_path: Path, storage: str) -> None:
        """Keep inherited values aligned, including raw's independent feature vocabulary."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        source = ad.read_h5ad(path)
        rows, features = source.shape
        source.obs["nullable_annotation"] = pd.array([1] + [None] * (rows - 1), dtype="Int16")
        source.layers["counts"] = sparse.csc_matrix(np.arange(rows * features).reshape(rows, features))
        source.layers["dense_metadata"] = np.arange(rows * features, dtype=np.float32).reshape(rows, features)
        source.layers["sparse_metadata"] = sparse.csc_matrix(source.layers["dense_metadata"])
        source.obsm["inherited"] = np.arange(rows * 2).reshape(rows, 2)
        source.varm["inherited"] = np.arange(features * 2).reshape(features, 2)
        source.obsp["inherited"] = sparse.csr_matrix(np.arange(rows * rows).reshape(rows, rows))
        source.varp["inherited"] = sparse.csr_matrix(np.arange(features * features).reshape(features, features))
        source.uns["inherited"] = {"nested": {"meaning": "unchanged", "number": 3}}
        raw = ad.AnnData(
            sparse.csr_matrix(np.arange(rows * 4, dtype=np.int32).reshape(rows, 4)),
            obs=pd.DataFrame(index=source.obs_names),
            var=pd.DataFrame({"annotation": [1.0, np.nan, 3.0, 4.0]}, index=["raw-z", "raw-a", "raw-b", "raw-c"]),
        )
        source.raw = raw
        source.write_h5ad(path)
        original = ad.read_h5ad(path)

        result = curio(tmp_path)
        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert isinstance(table.layers["counts"], sparse.csc_matrix)
        assert isinstance(table.layers["sparse_metadata"], sparse.csc_matrix)
        destination = tmp_path / f"inherited.{storage}"
        if storage == "h5ad":
            table.write_h5ad(destination)
            restored = ad.read_h5ad(destination)
        else:
            result.write(destination)
            restored = SpatialData.read(destination).tables["table"]

        order = [0, 1, 2]
        expected = original[order].copy()
        for actual in (table, restored):
            assert actual.obs_names.tolist() == expected.obs_names.tolist()
            assert isinstance(actual.obs, pd.DataFrame)
            assert isinstance(expected.obs, pd.DataFrame)
            pd.testing.assert_series_equal(actual.obs["nullable_annotation"], expected.obs["nullable_annotation"])
            for name in ("counts", "dense_metadata", "sparse_metadata"):
                assert_equal(actual.layers[name], expected.layers[name], exact=True)
            for mapping in ("obsm", "varm", "obsp", "varp"):
                assert_equal(getattr(actual, mapping)["inherited"], getattr(expected, mapping)["inherited"], exact=True)
            assert_equal(actual.raw, expected.raw, exact=True)
            assert actual.raw is not None
            assert actual.raw.obs_names.tolist() == expected.obs_names.tolist()
            assert actual.uns["inherited"] == expected.uns["inherited"]
            assert get_table_keys(actual) == get_table_keys(table)
            np.testing.assert_array_equal(actual.obsm["spatial"], table.obsm["spatial"])
        unchanged = ad.read_h5ad(path)
        assert unchanged.obs_names.equals(original.obs_names)
        assert "region" not in unchanged.obs

    def test_preserved_annotations_and_dynamic_enrichment(self, tmp_path: Path) -> None:
        """Preserve imported annotations while aligning extensible enrichment by identifier."""
        write_curio(tmp_path)
        source = tmp_path / "sample_anndata.h5ad"
        original = ad.read_h5ad(source)
        original.obs["batch"] = pd.Categorical(["b", "a", "b"], categories=["a", "b"], ordered=True)
        original.obs["retained"] = np.asarray([True, False, True])
        original.var["feature_number"] = np.asarray([2**53 + 1, 2, 3], dtype=np.int64)
        original.write_h5ad(source)
        (tmp_path / "sample_variable_features_clusters.txt").write_text(
            "note\tavg_log2FC\tgene\tcluster\nextra\t2.5\tgene-c\t1\n", encoding="utf-8"
        )
        (tmp_path / "sample_variable_features_spatial_moransi.txt").write_text(
            "gene\tcustom_statistic\tselected\ngene-c\t0.8\tTRUE\ngene-b\t0.4\tFALSE\n", encoding="utf-8"
        )

        table = curio(tmp_path).tables["table"]

        assert isinstance(table.obs, pd.DataFrame)
        assert isinstance(table.var, pd.DataFrame)
        assert isinstance(original.obs, pd.DataFrame)
        assert isinstance(original.var, pd.DataFrame)
        pd.testing.assert_series_equal(table.obs["batch"], original.obs["batch"], check_exact=True)
        pd.testing.assert_series_equal(table.obs["retained"], original.obs["retained"], check_exact=True)
        pd.testing.assert_series_equal(table.var["feature_number"], original.var["feature_number"], check_exact=True)
        np.testing.assert_array_equal(table.var["custom_statistic"], [0.4, np.nan, 0.8])
        assert table.var["selected"].loc[["gene-b", "gene-c"]].tolist() == [False, True]
        assert pd.isna(table.var["selected"].iloc[1])
        defining = table.uns["Top_cluster_defining_features"]
        assert defining.columns.tolist() == ["note", "avg_log2FC", "gene", "cluster"]
        assert defining.iloc[0].tolist() == ["extra", 2.5, "gene-c", 1]

    def test_positional_metric_fields_allow_arbitrary_names_and_trailing_metadata(self, tmp_path: Path) -> None:
        """Retain positional metric semantics with arbitrary names and trailing category metadata."""
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").write_text(
            "Measurement,Result,Note,Category\nReads,100,extra,QC\nBeads,3,extra,QC\n", encoding="utf-8"
        )

        table = curio(tmp_path).tables["table"]

        assert table.uns["QC"] == {"Reads": 100, "Beads": 3}

    def test_complete_export(self, tmp_path: Path) -> None:
        """Verify exact count buffers, linked geometry, metadata, and source ownership."""
        write_curio(tmp_path)

        result = curio(tmp_path)

        assert list(result.tables) == ["table"]
        assert list(result.shapes) == ["cells"]
        assert not result.images
        assert not result.labels
        assert not result.points

        table = result.tables["table"]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert table.X.has_canonical_format
        assert table.X.dtype == np.dtype(np.int32)
        np.testing.assert_array_equal(table.X.data, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(table.X.indices, [0, 2, 1, 0, 2])
        np.testing.assert_array_equal(table.X.indptr, [0, 2, 3, 5])

        assert table.obs_names.tolist() == ["bead-b", "bead-a", "bead-c"]
        assert table.var_names.tolist() == ["gene-b", "gene-a", "gene-c"]
        assert table.obs["quality"].tolist() == [3, 2, 1]
        assert table.obs["cluster"].tolist() == [1, 0, 1]
        assert isinstance(table.obs["cluster"].dtype, pd.CategoricalDtype)
        assert table.obs["region"].tolist() == ["cells"] * 3
        assert table.obs["instance_id"].tolist() == table.obs_names.tolist()
        assert get_table_keys(table) == ("cells", "region", "instance_id")

        assert table.var["symbol"].tolist() == ["B", "A", "C"]
        np.testing.assert_array_equal(table.var["MoransI_observed"], [0.4, np.nan, 0.8])
        np.testing.assert_array_equal(table.var["MoransI_p.value"], [0.05, np.nan, 0.01])
        assert table.uns["Sequencing"] == {"Reads, total": 100}
        assert table.uns["Mapping"] == {"Beads": 3}
        pd.testing.assert_frame_equal(
            table.uns["Top_cluster_defining_features"],
            pd.DataFrame(
                {
                    "gene": pd.Series(["gene-c"], dtype=pd.StringDtype(storage="pyarrow")),
                    "cluster": [1],
                    "avg_log2FC": [2.5],
                }
            ),
        )
        assert list(table.obsm) == ["spatial"]
        np.testing.assert_array_equal(table.layers["other"], np.ones((3, 3), dtype=np.float32))

        shapes = result.shapes["cells"]
        assert shapes.index.tolist() == table.obs_names.tolist()
        assert shapes.geometry.geom_type.tolist() == ["Point"] * 3
        np.testing.assert_array_equal(shapes.geometry.x, [30, 10, 20])
        np.testing.assert_array_equal(shapes.geometry.y, [4, 8, 2])
        np.testing.assert_array_equal(shapes["radius"], [10, 10, 10])
        np.testing.assert_array_equal(table.obsm["spatial"], np.column_stack((shapes.geometry.x, shapes.geometry.y)))
        assert isinstance(get_transformation(shapes, to_coordinate_system="global"), Identity)

        assert result.attrs == {
            "spatialdata_io_reader": "curio",
            "spatialdata_io_software_version": __version__,
        }

        # The source is reader-owned after loading. No mutations are persisted.
        source = ad.read_h5ad(tmp_path / "sample_anndata.h5ad")
        assert "X_spatial" in source.obsm
        assert "cluster" not in source.obs
        assert list(source.var.columns) == ["symbol"]

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Persist the complete reader result while preserving sparse values and shape identifiers."""
        write_curio(tmp_path)

        result = curio(tmp_path)
        destination = tmp_path / "converted.zarr"
        result.write(destination)

        loaded = SpatialData.read(destination)

        # SpatialData delegates deserialization to AnnData, which returns a CSR matrix.
        assert isinstance(result.tables["table"].X, sparse.csr_array)
        assert isinstance(loaded.tables["table"].X, sparse.csr_matrix)
        np.testing.assert_array_equal(loaded.tables["table"].X.data, result.tables["table"].X.data)
        assert loaded.shapes["cells"].index.tolist() == result.shapes["cells"].index.tolist()

    @pytest.mark.parametrize("storage", ["h5ad", "zarr"])
    def test_nullable_statistics_roundtrip(self, tmp_path: Path, storage: str) -> None:
        """Persist exact large integers, missing statistics, and nullable uns annotations."""
        write_curio(tmp_path)
        (tmp_path / "sample_variable_features_clusters.txt").write_text(
            "gene\tcluster\toptional\ngene-c\t1\t0.25\ngene-b\t0\t\ngene-a\t1\t0.75\n", encoding="utf-8"
        )
        (tmp_path / "sample_variable_features_spatial_moransi.txt").write_text(
            "gene\texact_integer\tselected\tprecision\ngene-c\t2\tTRUE\t0.10000000000000002\n"
            "gene-b\t9007199254740993\tFALSE\t0.25\n",
            encoding="utf-8",
        )
        table = curio(tmp_path).tables["table"]
        destination = tmp_path / f"table.{storage}"

        if storage == "h5ad":
            table.write_h5ad(destination)
            loaded = ad.read_h5ad(destination)
        else:
            table.write_zarr(destination)
            loaded = ad.read_zarr(destination)

        assert table.var["exact_integer"].dtype == pd.Int64Dtype()
        assert table.var["selected"].dtype == pd.BooleanDtype()
        assert loaded.var["exact_integer"].iloc[0] == 2**53 + 1
        assert loaded.var["exact_integer"].isna().tolist() == [False, True, False]
        assert loaded.var["selected"].isna().tolist() == [False, True, False]
        assert loaded.var["precision"].iloc[2] == float("0.10000000000000002")
        assert loaded.obs["instance_id"].tolist() == table.obs_names.tolist()
        np.testing.assert_array_equal(loaded.uns["Top_cluster_defining_features"]["optional"], [0.25, np.nan, 0.75])

    def test_lexical_nan_cluster_stays_a_category(self, tmp_path: Path) -> None:
        """Keep a literal NaN label distinct from a missing cluster assignment."""
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text("bead-b\tNaN\nbead-a\t1\nbead-c\t2\n", encoding="utf-8")

        table = curio(tmp_path).tables["table"]

        assert table.obs["cluster"].tolist() == ["NaN", "1", "2"]
        assert not table.obs["cluster"].isna().any()

    def test_reordered_clusters_align_without_reordering_counts(self, tmp_path: Path) -> None:
        """Align shuffled cluster rows while preserving the count observation order."""
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text("bead-c\t1\nbead-b\t1\nbead-a\t0\n", encoding="utf-8")

        table = curio(tmp_path).tables["table"]

        assert table.obs_names.tolist() == ["bead-b", "bead-a", "bead-c"]
        assert table.obs["cluster"].tolist() == [1, 0, 1]

    @pytest.mark.parametrize("identifiers", [["002", "001", "003"], ["NA", "NULL", "nan"]])
    def test_string_identifiers_preserved(self, tmp_path: Path, identifiers: list[str]) -> None:
        """Retain leading zeros and literal missing-value words in source identifiers."""
        write_curio(tmp_path)
        source = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(source)
        table.obs_names = identifiers
        table.var_names = identifiers
        table.write_h5ad(source)
        (tmp_path / "sample_cluster_assignment.txt").write_text(
            "".join(f"{identifier}\t{cluster}\n" for identifier, cluster in zip(identifiers, [1, 0, 1], strict=True)),
            encoding="utf-8",
        )
        (tmp_path / "sample_variable_features_spatial_moransi.txt").write_text(
            f"\tI\n{identifiers[2]}\t0.8\n{identifiers[0]}\t0.4\n", encoding="utf-8"
        )

        result = curio(tmp_path)

        assert result.tables["table"].obs_names.tolist() == identifiers
        assert result.tables["table"].var_names.tolist() == identifiers
        np.testing.assert_array_equal(result.tables["table"].var["I"], [0.4, np.nan, 0.8])
