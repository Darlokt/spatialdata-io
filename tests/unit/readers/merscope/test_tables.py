"""MERSCOPE bounded sparse counts, blank storage, exact alignment and final linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Polygon
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope import _tables as merscope_tables
from spatialdata_io.readers.merscope._geometry import CellBoundaries
from spatialdata_io.readers.merscope._layout import Names, TableFiles
from spatialdata_io.readers.merscope._schemas import _FLOAT_DTYPE, STRING_DTYPE
from spatialdata_io.readers.merscope._tables import finalize_table, read_feature_names, read_table

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _sources(
    path: Path,
    counts: str = "cell,B,Blank-1,A,BlanketGene,blank-2,Blank-0\nb,0,2,4,7,10,1\na,7,5,0,8,11,0\n",
    metadata: str = "EntityID,center_x,annotation,center_y,volume\na,2,0001,3,4\nb,7,NA,5,6\n",
) -> TableFiles:
    """Write exact lexical sources, independently ordered by default."""
    count_path = path / "counts.csv"
    metadata_path = path / "metadata.csv"
    count_path.write_text(counts, encoding="utf-8")
    metadata_path.write_text(metadata, encoding="utf-8")
    return TableFiles(ArtifactFile(count_path), ArtifactFile(metadata_path))


class TestReadFeatureNames:
    """Vocabulary discovery reads only the header without decoding count rows."""

    def test_header_only(self, tmp_path: Path) -> None:
        counts = _sources(tmp_path, "cell,B,Blank-1,A\nnot,a,valid,record,at,all\n").counts
        assert read_feature_names(counts) == ("B", "Blank-1", "A")

    @pytest.mark.parametrize("header", ["cell,A,A\n", '"unterminated\n'])
    def test_bad_header(self, tmp_path: Path, header: str) -> None:
        counts = _sources(tmp_path, header).counts
        with pytest.raises(ReaderFormatError) as caught:
            read_feature_names(counts)
        assert caught.value.context == ReaderErrorContext("merscope", "count matrix")


class TestReadTable:
    """Keep count order, sparsity, annotations and a complete feature vocabulary."""

    def test_sparse_order_blank_names_and_metadata(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.merscope._tables._BATCH_MEMORY_BYTES", 1)
        table, vocabulary = read_table(_sources(tmp_path))
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert table.X.has_canonical_format
        assert table.obs_names.tolist() == ["b", "a"]
        assert table.var_names.tolist() == ["B", "A", "BlanketGene", "blank-2", "Blank-0"]
        assert vocabulary == ("B", "Blank-1", "A", "BlanketGene", "blank-2", "Blank-0")
        np.testing.assert_array_equal(table.X.data, [4, 7, 10, 1, 7, 8, 11])
        blanks = table.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert blanks.shape == (2, 1)
        np.testing.assert_array_equal(blanks.data, [2, 5])
        assert table.uns["blank_names"] == ["Blank-1"]
        assert table.obs["annotation"].tolist() == ["NA", "0001"]
        assert table.obs["annotation"].dtype == STRING_DTYPE
        assert table.obs["volume"].dtype == _FLOAT_DTYPE
        np.testing.assert_array_equal(table.obsm["spatial"], [[7, 5], [2, 3]])
        assert "spatialdata_attrs" not in table.uns

    def test_exact_unsigned_counts_across_batches(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.merscope._tables._BATCH_MEMORY_BYTES", 1)
        large = 2**63
        table, _ = read_table(_sources(tmp_path, f"cell,A\na,1\nb,{large}\n"))
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.float32)
        assert table.X.data.tolist() == [1, large]

    def test_row_batches_are_bounded_only_by_bytes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """20,001 single-feature rows fit one 8 MiB block; 1,000-row blocks give the identical table."""
        rows = 20_001
        counts = "cell,A,Blank-1\n" + "".join(f"c{row},{row % 3},{row % 5 == 0:d}\n" for row in range(rows))
        metadata = "EntityID,center_x,center_y\n" + "".join(f"c{row},{row},1\n" for row in range(rows))
        sources = _sources(tmp_path, counts, metadata)
        blocks = mocker.spy(merscope_tables, "_count_block")

        table, _ = read_table(sources)

        assert [call.args[0].num_rows for call in blocks.call_args_list] == [rows]
        # 16 bytes per feature and row; two features.
        mocker.patch.object(merscope_tables, "_BATCH_MEMORY_BYTES", 16 * 2 * 1000)
        blocks.reset_mock()
        small, _ = read_table(sources)
        assert len(blocks.call_args_list) == 21
        for expected, actual in ((table.X, small.X), (table.obsm["blank"], small.obsm["blank"])):
            assert isinstance(expected, sparse.csr_array)
            assert isinstance(actual, sparse.csr_array)
            assert expected.nnz == actual.nnz
            assert expected.dtype == actual.dtype
            np.testing.assert_array_equal(expected.data, actual.data)
            np.testing.assert_array_equal(expected.indices, actual.indices)
            np.testing.assert_array_equal(expected.indptr, actual.indptr)

    @pytest.mark.parametrize("batch_bytes", [1, 8 * 1024 * 1024])
    def test_exact_integer_boundaries_and_spellings(
        self, tmp_path: Path, mocker: MockerFixture, batch_bytes: int
    ) -> None:
        mocker.patch("spatialdata_io.readers.merscope._tables._BATCH_MEMORY_BYTES", batch_bytes)
        table, _ = read_table(
            _sources(
                tmp_path,
                f"cell,A,Blank-1,B\na,{2**24 + 1},2.0,3e0\nb,{2**63},{2**64 - 2048},0\n",
            )
        )
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(table.X.data, np.array([2**24 + 1, 3, 2**63], dtype=np.float64))
        blanks = table.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert blanks.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(blanks.data, np.array([2, 2**64 - 2048], dtype=np.float64))

    @pytest.mark.parametrize("feature", ["A", "Blank-1"])
    def test_rejects_inexact_count_or_control(self, tmp_path: Path, feature: str) -> None:
        with pytest.raises(ReaderFormatError, match="not exactly representable") as caught:
            read_table(_sources(tmp_path, f"cell,{feature}\na,1\nb,{2**64 - 1}\n"))
        assert caught.value.path == tmp_path / "counts.csv"
        assert caught.value.context == ReaderErrorContext("merscope", "count matrix")

    @pytest.mark.parametrize(
        "counts",
        [
            "cell,A\n",
            "cell,A\na,1\na,2\n",
            "cell,A\na,1.0000000000000001\n",
            "cell,A\na,-1\n",
            "cell,A\na,1e-400\n",
            "cell,A\na,18446744073709551616\n",
            "cell,A\na,\n",
            "cell,A\na,1\x00\n",
            "cell,A\na,1,2\n",
            "cell,A,B\na,1\n",
        ],
    )
    def test_bad_counts(self, tmp_path: Path, counts: str) -> None:
        with pytest.raises(ReaderFormatError):
            read_table(_sources(tmp_path, counts))

    @pytest.mark.parametrize(
        "metadata",
        [
            "EntityID,center_x,center_y\na,1,2\n",
            "EntityID,center_x,center_y\na,1,2\na,3,4\n",
            "EntityID,center_x,center_y\na,1\n",
            "EntityID,center_x,center_y\na,1,2,extra\n",
            "EntityID,center_x,center_y\na,1,2\nb,nan,3\n",
        ],
    )
    def test_bad_metadata(self, tmp_path: Path, metadata: str) -> None:
        with pytest.raises(ReaderFormatError):
            read_table(_sources(tmp_path, metadata=metadata))

    @pytest.mark.parametrize("features", ["A", "Blank-1"])
    def test_empty_expression_or_blank_width(self, tmp_path: Path, features: str) -> None:
        table, vocabulary = read_table(_sources(tmp_path, f"cell,{features}\na,1\nb,0\n"))
        assert vocabulary == (features,)
        assert isinstance(table.X, sparse.csr_array)
        blanks = table.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert table.X.shape == (2, 0 if features == "Blank-1" else 1)
        assert blanks.shape == (2, 1 if features == "Blank-1" else 0)

    def test_quoted_records_and_integral_exponents(self, tmp_path: Path) -> None:
        table, _ = read_table(
            _sources(
                tmp_path,
                'cell,"Gene,A"\na,1e0\nb,2.0\n',
                'EntityID,center_x,center_y,extra\na,.10000000000000002,2,"a multiline\nannotation"\nb,3,4,NA\n',
            )
        )
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.data.tolist() == [1, 2]
        assert table.obs["extra"].tolist() == ["a multiline\nannotation", "NA"]
        assert table.obsm["spatial"][0, 0] == 0.10000000000000002


def _boundaries(ids: tuple[str, ...]) -> CellBoundaries:
    """Create complete normalized boundaries with a separately ordered table population."""
    shapes = gpd.GeoDataFrame(
        {"geometry_representation": ["union_footprint"] * len(ids)},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * len(ids),
        index=pd.Index(ids, dtype=STRING_DTYPE, name="EntityID"),
    )
    return CellBoundaries(shapes, shapes.index)


class TestFinalizeTable:
    """Apply compatible annotation storage and exact independent target membership once."""

    @pytest.mark.parametrize("wide_feature", ["A", "Blank-1"])
    @pytest.mark.parametrize("retain_wide", [False, True])
    def test_geometry_drop_selects_dtype_from_retained_count_family(
        self, tmp_path: Path, wide_feature: str, *, retain_wide: bool
    ) -> None:
        ordinary, wide = 3, 2**24 + 1
        a, b = (wide, ordinary) if retain_wide else (ordinary, wide)
        text = (
            f"cell,A,Blank-1\na,{a if wide_feature == 'A' else 1},{a if wide_feature == 'Blank-1' else 2}\n"
            f"b,{b if wide_feature == 'A' else 1},{b if wide_feature == 'Blank-1' else 2}\n"
        )
        table, _ = read_table(_sources(tmp_path, text))
        boundaries = CellBoundaries(_boundaries(("a",)).shapes, pd.Index(["a", "b"]))
        result = finalize_table(table, names=Names(region="region", slide="slide"), boundaries=boundaries)
        assert result.obs_names.tolist() == ["a"]
        assert isinstance(result.X, sparse.csr_array)
        blanks = result.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert result.X.dtype == blanks.dtype == np.dtype(np.float64 if retain_wide else np.float32)
        np.testing.assert_array_equal(result.X.data, [a if wide_feature == "A" else 1])
        np.testing.assert_array_equal(blanks.data, [a if wide_feature == "Blank-1" else 2])
        assert table.n_obs == 2

    @pytest.mark.parametrize("linked", [False, True])
    def test_linkage_and_blank_roundtrip(self, tmp_path: Path, *, linked: bool) -> None:
        table, _ = read_table(_sources(tmp_path))
        native = table.obsm["spatial"]
        expression = table.X
        result = finalize_table(
            table,
            names=Names(region="region", slide="slide"),
            boundaries=_boundaries(("a", "b")) if linked else None,
        )
        assert result is table
        assert result.obsm["spatial"] is native
        assert result.X is expression
        assert result.uns["spatialdata_io_positions"]["coordinates"]["spatial"]["frame"] == "global"
        assert result.obs["EntityID"].tolist() == ["b", "a"]
        assert result.obs["region"].tolist() == ["region", "region"]
        if linked:
            assert get_table_keys(result) == ("slide_region_polygons", "cells_region", "EntityID")
        else:
            assert "spatialdata_attrs" not in result.uns
            assert "cells_region" not in result.obs
        destination = tmp_path / "table.h5ad"
        result.write_h5ad(destination)
        restored = ad.read_h5ad(destination)
        assert restored.obs["EntityID"].tolist() == ["b", "a"]
        assert restored.uns["blank_names"].tolist() == ["Blank-1"]
        restored_blanks = sparse.csr_array(restored.obsm["blank"])
        np.testing.assert_array_equal(restored_blanks.data, [2, 5])

    def test_wrong_target_population(self, tmp_path: Path) -> None:
        table, _ = read_table(_sources(tmp_path))
        with pytest.raises(ReaderFormatError):
            finalize_table(
                table,
                names=Names(region="region", slide="slide"),
                boundaries=_boundaries(("a", "different")),
            )

    def test_conflicting_source_annotations_fail(self, tmp_path: Path) -> None:
        table, _ = read_table(_sources(tmp_path))
        table.obs["region"] = pd.Series("conflicting source value", index=table.obs_names, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="conflicting source"):
            finalize_table(
                table,
                names=Names(region="region", slide="slide"),
                boundaries=_boundaries(("a", "b")),
            )

    def test_matching_annotations_are_preserved_as_declared_categories(self, tmp_path: Path) -> None:
        table, _ = read_table(_sources(tmp_path))
        expected = {
            "region": "region",
            "slide": "slide",
            "dataset_id": "slide_region",
            "cells_region": "slide_region_polygons",
        }
        for column, value in expected.items():
            table.obs[column] = pd.Series(value, index=table.obs_names, dtype=STRING_DTYPE)
        native = table.obsm["spatial"]
        expression = table.X
        result = finalize_table(
            table,
            names=Names(region="region", slide="slide"),
            boundaries=_boundaries(("a", "b")),
        )
        assert result.obsm["spatial"] is native
        assert result.X is expression
        for column, value in expected.items():
            assert result.obs[column].tolist() == [value, value]
            assert result.obs[column].dtype == pd.CategoricalDtype([value])

    def test_unlinked_source_target_annotation_fails(self, tmp_path: Path) -> None:
        table, _ = read_table(_sources(tmp_path))
        table.obs["cells_region"] = pd.Series("cells", index=table.obs_names, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="unlinked table"):
            finalize_table(
                table,
                names=Names(region="region", slide="slide"),
                boundaries=None,
            )
