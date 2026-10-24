"""MERSCOPE bounded sparse counts, blank storage, exact alignment and final linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._schemas import FLOAT_DTYPE, STRING_DTYPE
from spatialdata_io.readers.merscope._tables import finalize_table, read_feature_names, read_table

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _sources(
    path: Path,
    counts: str = "cell,B,Blank-1,A,BlanketGene,blank-2,Blank-0\nb,0,2,4,7,10,1\na,7,5,0,8,11,0\n",
    metadata: str = "EntityID,center_x,annotation,center_y,volume\na,2,0001,3,4\nb,7,NA,5,6\n",
) -> tuple[ArtifactFile, ArtifactFile]:
    """Write exact lexical sources, independently ordered by default."""
    count_path = path / "counts.csv"
    metadata_path = path / "metadata.csv"
    count_path.write_text(counts, encoding="utf-8")
    metadata_path.write_text(metadata, encoding="utf-8")
    return ArtifactFile(count_path), ArtifactFile(metadata_path)


class TestReadFeatureNames:
    """Vocabulary discovery reads only a bounded header without decoding count rows."""

    def test_header_only(self, tmp_path: Path) -> None:
        counts, _ = _sources(tmp_path, "cell,B,Blank-1,A\nnot,a,valid,record,at,all\n")
        assert read_feature_names(counts) == ("B", "Blank-1", "A")

    @pytest.mark.parametrize("header", ["cell,A,A\n", '"unterminated\n'])
    def test_bad_header(self, tmp_path: Path, header: str) -> None:
        counts, _ = _sources(tmp_path, header)
        with pytest.raises(ReaderFormatError):
            read_feature_names(counts)

    def test_header_bound(self, tmp_path: Path, mocker: MockerFixture) -> None:
        counts, _ = _sources(tmp_path)
        mocker.patch("spatialdata_io.readers.merscope._tables._MAX_HEADER_CHARACTERS", 2)
        with pytest.raises(ReaderFormatError, match="oversized"):
            read_feature_names(counts)


class TestReadTable:
    """Keep count order, sparsity, annotations and a complete feature vocabulary."""

    def test_sparse_order_blank_names_and_metadata(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.merscope._tables._MAX_BATCH_ROWS", 1)
        table, vocabulary = read_table(*_sources(tmp_path))
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
        assert table.obs["volume"].dtype == FLOAT_DTYPE
        np.testing.assert_array_equal(table.obsm["spatial"], [[7, 5], [2, 3]])
        assert "spatialdata_attrs" not in table.uns

    def test_exact_unsigned_counts_across_batches(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.merscope._tables._MAX_BATCH_ROWS", 1)
        large = 2**63 + 1
        table, _ = read_table(*_sources(tmp_path, f"cell,A\na,1\nb,{large}\n"))
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.uint64)
        assert table.X.data.tolist() == [1, large]

    @pytest.mark.parametrize("rows_per_batch", [1, 10_000])
    def test_exact_integer_boundaries_and_spellings(
        self, tmp_path: Path, mocker: MockerFixture, rows_per_batch: int
    ) -> None:
        mocker.patch("spatialdata_io.readers.merscope._tables._MAX_BATCH_ROWS", rows_per_batch)
        table, _ = read_table(
            *_sources(
                tmp_path,
                f"cell,A,Blank-1,B\na,{2**63 - 1},2.0,3e0\nb,{2**63},{2**64 - 1},0\n",
            )
        )
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(table.X.data, np.array([2**63 - 1, 3, 2**63], dtype=np.uint64))
        blanks = table.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert blanks.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(blanks.data, np.array([2, 2**64 - 1], dtype=np.uint64))

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
            read_table(*_sources(tmp_path, counts))

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
            read_table(*_sources(tmp_path, metadata=metadata))

    @pytest.mark.parametrize("features", ["A", "Blank-1"])
    def test_empty_expression_or_blank_width(self, tmp_path: Path, features: str) -> None:
        table, vocabulary = read_table(*_sources(tmp_path, f"cell,{features}\na,1\nb,0\n"))
        assert vocabulary == (features,)
        assert isinstance(table.X, sparse.csr_array)
        blanks = table.obsm["blank"]
        assert isinstance(blanks, sparse.csr_array)
        assert table.X.shape == (2, 0 if features == "Blank-1" else 1)
        assert blanks.shape == (2, 1 if features == "Blank-1" else 0)

    def test_quoted_records_and_integral_exponents(self, tmp_path: Path) -> None:
        table, _ = read_table(
            *_sources(
                tmp_path,
                'cell,"Gene,A"\na,1e0\nb,2.0\n',
                'EntityID,center_x,center_y,extra\na,.10000000000000002,2,"a multiline\nannotation"\nb,3,4,NA\n',
            )
        )
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.data.tolist() == [1, 2]
        assert table.obs["extra"].tolist() == ["a multiline\nannotation", "NA"]
        assert table.obsm["spatial"][0, 0] == 0.10000000000000002


class TestFinalizeTable:
    """Apply compatible annotation storage and exact independent target membership once."""

    @pytest.mark.parametrize("target", [None, "slide_region_polygons"])
    def test_linkage_and_blank_roundtrip(self, tmp_path: Path, target: str | None) -> None:
        table, _ = read_table(*_sources(tmp_path))
        result = finalize_table(
            table,
            region_name="region",
            slide_name="slide",
            dataset_id="slide_region",
            target=target,
            expected_instances=pd.Index(["a", "b"]) if target else None,
        )
        assert result is table
        assert result.obs["EntityID"].tolist() == ["b", "a"]
        assert result.obs["region"].tolist() == ["region", "region"]
        if target:
            assert get_table_keys(result) == (target, "cells_region", "EntityID")
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
        table, _ = read_table(*_sources(tmp_path))
        with pytest.raises(ReaderFormatError):
            finalize_table(
                table,
                region_name="region",
                slide_name="slide",
                dataset_id="slide_region",
                target="cells",
                expected_instances=pd.Index(["a", "different"]),
            )

    def test_conflicting_source_annotations_fail(self, tmp_path: Path) -> None:
        table, _ = read_table(*_sources(tmp_path))
        table.obs["region"] = pd.Series("conflicting source value", index=table.obs_names, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="conflicting source"):
            finalize_table(
                table,
                region_name="region",
                slide_name="slide",
                dataset_id="slide_region",
                target="cells",
                expected_instances=pd.Index(["a", "b"]),
            )

    def test_matching_annotations_are_preserved_as_declared_categories(self, tmp_path: Path) -> None:
        table, _ = read_table(*_sources(tmp_path))
        expected = {"region": "region", "slide": "slide", "dataset_id": "slide_region", "cells_region": "cells"}
        for column, value in expected.items():
            table.obs[column] = pd.Series(value, index=table.obs_names, dtype=STRING_DTYPE)
        result = finalize_table(
            table,
            region_name="region",
            slide_name="slide",
            dataset_id="slide_region",
            target="cells",
            expected_instances=pd.Index(["a", "b"]),
        )
        for column, value in expected.items():
            assert result.obs[column].tolist() == [value, value]
            assert result.obs[column].dtype == pd.CategoricalDtype([value])

    def test_unlinked_source_target_annotation_fails(self, tmp_path: Path) -> None:
        table, _ = read_table(*_sources(tmp_path))
        table.obs["cells_region"] = pd.Series("cells", index=table.obs_names, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="unlinked table"):
            finalize_table(
                table,
                region_name="region",
                slide_name="slide",
                dataset_id="slide_region",
                target=None,
                expected_instances=None,
            )
