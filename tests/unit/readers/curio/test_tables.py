"""Curio owned H5AD and identifier-aligned enrichment."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.curio._layout import discover_layout
from spatialdata_io.readers.curio._tables import _add_metrics, _join_moran, _read_named_rows, _read_rows, read_table
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestReadTable:
    """Validate owned H5AD and eager enrichment schemas."""

    def test_malformed_sparse_storage_retains_count_context(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        with h5py.File(path, "r+") as source:
            source["X/indices"][0] = 3

        with pytest.raises(ValueError, match="active indices") as raised:
            curio(tmp_path)

        assert "while normalizing adata.X" in raised.value.__notes__
        assert any("curio" in note and str(path) in note for note in raised.value.__notes__)

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(("dtype", "values"), [("int8", [100, 100]), ("uint8", [150, 150])])
    def test_integer_duplicate_overflow(self, tmp_path: Path, storage: str, dtype: str, values: list[int]) -> None:
        """Promote imported integer sums losslessly and report the normalization."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        buffers = (np.array(values, dtype=dtype), np.array([0, 0]), np.array([0, 2, 2, 2]))
        table.X = (
            sparse.csr_array(buffers, shape=(3, 3)) if storage == "csr" else sparse.csc_array(buffers, shape=(3, 3))
        )
        table.write_h5ad(path)

        with pytest.warns(UserWarning, match="combined.*X.*promoted") as reported:
            result = curio(tmp_path).tables["table"].X
        assert len(reported) == 1
        assert isinstance(result, sparse.csr_array)
        assert int(result[0, 0]) == sum(values)
        assert result.dtype == np.dtype(np.float32)
        assert path.name in str(reported[0].message)
        unchanged = ad.read_h5ad(path).X
        assert isinstance(unchanged, (sparse.csr_matrix, sparse.csc_matrix))
        np.testing.assert_array_equal(unchanged.data, values)

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(
        ("dtype", "values", "expected"), [("int16", [100, 100], 200), ("int8", [127, 127, -127], 127)]
    )
    def test_safe_integer_duplicates(
        self, tmp_path: Path, storage: str, dtype: str, values: list[int], expected: int
    ) -> None:
        """Accept exact completed sums, preserving dtype, sparsity and source storage."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        buffers = (
            np.array(values, dtype=dtype),
            np.zeros(len(values), dtype=np.int32),
            np.array([0, len(values), len(values), len(values)]),
        )
        table.X = (
            sparse.csr_array(buffers, shape=(3, 3)) if storage == "csr" else sparse.csc_array(buffers, shape=(3, 3))
        )
        table.write_h5ad(path)

        with pytest.warns(UserWarning, match="combined") as reported:
            result = curio(tmp_path).tables["table"].X
        assert len(reported) == 1
        assert reported[0].filename == __file__

        assert isinstance(result, sparse.csr_array)
        assert result.dtype == np.dtype(dtype)
        assert result.has_canonical_format
        assert result[0, 0] == expected
        unchanged = ad.read_h5ad(path).X
        assert isinstance(unchanged, (sparse.csr_matrix, sparse.csc_matrix))
        np.testing.assert_array_equal(unchanged.data, values)

    @pytest.mark.parametrize(
        "rows",
        ["bead-b\t1\t2\n", "bead-b\t\n", "bead-b\t1\nbead-b\t0\nbead-c\t1\n", "missing\t1\nbead-a\t0\nbead-c\t1\n"],
    )
    def test_invalid_clusters(self, tmp_path: Path, rows: str) -> None:
        """Reject malformed assignments and incomplete or duplicated bead populations."""
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="cluster"):
            read_table(discover_layout(tmp_path))

    @pytest.mark.parametrize("kind", ["empty", "duplicate_obs", "duplicate_var"])
    def test_invalid_h5ad_identifiers(self, tmp_path: Path, kind: str) -> None:
        """Require a nonempty observation population and unambiguous count identifiers."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        if kind == "empty":
            table = table[:0].copy()
        elif kind == "duplicate_obs":
            table.obs_names = ["b", "b", "c"]
        else:
            table.var_names = ["b", "b", "c"]
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="identifiers"):
            read_table(discover_layout(tmp_path))

    def test_missing_expression(self, tmp_path: Path) -> None:
        """Reject an annotation-only H5AD at the count-bearing reader boundary."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        table.X = None
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="missing expression") as raised:
            read_table(discover_layout(tmp_path))

        assert raised.value.path == path
        assert raised.value.context.reader == "curio"

    def test_h5ad_failure_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Retain storage failure identity and add reader artifact context."""
        write_curio(tmp_path)
        error = OSError("injected read failure")
        mocker.patch("spatialdata_io.readers.curio._tables.ad.read_h5ad", side_effect=error)

        with pytest.raises(OSError, match="injected") as raised:
            read_table(discover_layout(tmp_path))

        assert any("curio" in note and "counts" in note for note in raised.value.__notes__)

    @pytest.mark.parametrize("axis", ["obs", "var"])
    @pytest.mark.parametrize("malformed", ["empty_id", "null_id", "duplicate_columns"])
    def test_malformed_h5ad_annotation_boundary(
        self, tmp_path: Path, mocker: MockerFixture, axis: str, malformed: str
    ) -> None:
        """Reject missing identifiers and duplicate annotation names returned by H5AD loading."""
        write_curio(tmp_path)
        layout = discover_layout(tmp_path)
        table = ad.read_h5ad(layout.counts)
        frame = table.obs if axis == "obs" else table.var
        assert isinstance(frame, pd.DataFrame)
        if malformed == "duplicate_columns":
            frame["second"] = frame.iloc[:, 0]
            frame.columns = ["duplicate", "duplicate"]
        else:
            frame.index = pd.Index(["" if malformed == "empty_id" else None, "second", "third"])
        mocker.patch("spatialdata_io.readers.curio._tables.ad.read_h5ad", return_value=table)

        with pytest.raises(ReaderFormatError, match=r"identifiers|column names"):
            read_table(layout)

    def test_sparse_expression_does_not_densify(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Keep count expression sparse throughout the complete reader transaction."""
        write_curio(tmp_path)
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("densification"))
        mocker.patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("densification"))

        assert isinstance(curio(tmp_path).tables["table"].X, sparse.csr_array)

    def test_nonfinite_expression(self, tmp_path: Path) -> None:
        """Reject nonfinite expression before returning a linked scientific table."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        table.X = sparse.csr_array([[np.inf, 0, 0], [0, 1, 0], [0, 0, 2]])
        table.write_h5ad(path)

        with pytest.raises(ValueError, match="finite"):
            curio(tmp_path)

    @pytest.mark.parametrize("header", ["gene\tcluster\tcluster", "gene\t"])
    def test_ambiguous_defining_feature_header(self, tmp_path: Path, header: str) -> None:
        """Reject duplicate or unnamed defining-feature fields before pandas repairs them."""
        write_curio(tmp_path)
        (tmp_path / "sample_variable_features_clusters.txt").write_text(f"{header}\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="header"):
            curio(tmp_path)

    def test_invalid_quoted_defining_header_context(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_clusters.txt"
        path.write_text('"gene\tcluster\n', encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed CSV header") as raised:
            read_table(discover_layout(tmp_path))

        assert raised.value.path == path

    def test_nul_cannot_truncate_a_bead_identifier(self, tmp_path: Path) -> None:
        """Prevent parser truncation from aligning a different bead identifier."""
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text(
            "bead-b\x00suffix\t1\nbead-a\t0\nbead-c\t1\n", encoding="utf-8"
        )

        with pytest.raises(ReaderFormatError, match="NUL"):
            curio(tmp_path)

    def test_nul_cluster_label_is_rejected(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text(
            "bead-b\t1\x00suffix\nbead-a\t0\nbead-c\t1\n", encoding="utf-8"
        )

        with pytest.raises(ReaderFormatError, match="NUL"):
            read_table(discover_layout(tmp_path))

    @pytest.mark.parametrize("row", ['"bead-b"suffix\t1\n', 'bead-b\t"unterminated\n', "bead-b\t1\t2\n"])
    def test_malformed_cluster_rows_are_rejected(self, tmp_path: Path, row: str) -> None:
        """Reject malformed quoting and extra fields before lexical IDs can change meaning."""
        write_curio(tmp_path)
        path = tmp_path / "sample_cluster_assignment.txt"
        path.write_text(row, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed rows") as raised:
            read_table(discover_layout(tmp_path))

        assert raised.value.path == path
        assert raised.value.context.artifact == "cluster assignments"

    @pytest.mark.parametrize(
        ("suffix", "artifact"),
        [
            ("cluster_assignment.txt", "cluster assignments"),
            ("Metrics.csv", "metrics"),
            ("variable_features_clusters.txt", "cluster features"),
            ("variable_features_spatial_moransi.txt", "Moran features"),
        ],
    )
    def test_invalid_utf8_rows_name_the_artifact(self, tmp_path: Path, suffix: str, artifact: str) -> None:
        """Report undecodable records as a format error with a stable artifact name."""
        write_curio(tmp_path)
        path = tmp_path / f"sample_{suffix}"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(lines[0] + b"\xff" + b"".join(lines[1:]))

        with pytest.raises(ReaderFormatError, match="not valid UTF-8") as raised:
            read_table(discover_layout(tmp_path))

        assert raised.value.context.artifact == artifact
        assert raised.value.path == path

    def test_uns_keys_are_sanitized_with_one_warning(self, tmp_path: Path) -> None:
        """Rename invalid and reader-colliding top-level keys only, keeping values and order."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        source = ad.read_h5ad(path)
        source.uns["my key"] = {"inner key": "kept"}
        source.uns["valid"] = "v"
        source.uns["Spatialdata_io_positions"] = np.asarray([1, 2])
        source.uns["top_cluster_defining_features"] = "t"
        source.write_h5ad(path)

        stored = list(ad.read_h5ad(path).uns)

        with pytest.warns(UserWarning, match="uns keys") as caught:
            table = read_table(discover_layout(tmp_path))

        # HDF5 stores group members by name, so the stored order is alphabetical.
        assert stored == ["Spatialdata_io_positions", "my key", "top_cluster_defining_features", "valid"]
        assert [str(warning.message) for warning in caught] == [
            (
                "curio: the H5AD uns has names that a SpatialData table does not accept as uns keys; using "
                "'Spatialdata_io_positions' -> 'Spatialdata_io_positions_1', 'my key' -> 'my_key', "
                "'top_cluster_defining_features' -> 'top_cluster_defining_features_1'. Rename them in the source to "
                "choose the names explicitly."
            )
        ]
        assert list(table.uns) == [
            "Spatialdata_io_positions_1",
            "my_key",
            "top_cluster_defining_features_1",
            "valid",
            "Sequencing",
            "Mapping",
            "Top_cluster_defining_features",
        ]
        assert table.uns["my_key"] == {"inner key": "kept"}
        assert table.uns["valid"] == "v"
        np.testing.assert_array_equal(table.uns["Spatialdata_io_positions_1"], [1, 2])
        assert table.uns["top_cluster_defining_features_1"] == "t"

    def test_valid_uns_keys_are_untouched_without_warning(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        source = ad.read_h5ad(path)
        source.uns["qc"] = {"inner key": 1}
        source.write_h5ad(path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            table = read_table(discover_layout(tmp_path))

        assert list(table.uns)[:1] == ["qc"]
        assert table.uns["qc"] == {"inner key": 1}

    def test_metric_categories_are_reserved_against_sanitized_uns_keys(self, tmp_path: Path) -> None:
        """Categories yield to the already sanitized H5AD keys, which keep their repaired names."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        source = ad.read_h5ad(path)
        source.uns["run metrics"] = "h5ad"
        source.write_h5ad(path)
        (tmp_path / "sample_Metrics.csv").write_text("Metric,Value,Category\nReads,100,Run Metrics\n", encoding="utf-8")

        with pytest.warns(UserWarning, match="uns keys") as caught:
            table = read_table(discover_layout(tmp_path))

        assert [str(warning.message) for warning in caught] == [
            (
                "curio: the H5AD uns has names that a SpatialData table does not accept as uns keys; using "
                "'run metrics' -> 'run_metrics'. Rename them in the source to choose the names explicitly."
            ),
            (
                "curio: the Category column of the metrics file has names that a SpatialData table does not accept as "
                "uns keys; using 'Run Metrics' -> 'Run_Metrics_1'. Rename them in the source to choose the names "
                "explicitly."
            ),
        ]
        assert table.uns["run_metrics"] == "h5ad"
        assert table.uns["Run_Metrics_1"] == {"Reads": "100"}

    def test_exact_reserved_uns_key_is_rejected(self, tmp_path: Path) -> None:
        """An exact reader-owned key is not renamed: it still fails as an occupied destination."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        source = ad.read_h5ad(path)
        source.uns["Top_cluster_defining_features"] = "t"
        source.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="reserved output annotations"):
            read_table(discover_layout(tmp_path))


class TestReadNamedRows:
    """Keep source headers, exact physical row widths, and explicit file ownership."""

    @pytest.mark.parametrize("delimiter", [",", "\t"])
    @pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
    def test_header_only_has_explicit_lexical_columns(self, tmp_path: Path, delimiter: str, ending: str) -> None:
        """Retain named string columns even without a trailing header newline."""
        path = tmp_path / "features.txt"
        path.write_text(f"gene{delimiter}annotation{ending}", encoding="utf-8")

        frame = _read_named_rows(ArtifactFile(path), "features", delimiter=delimiter)

        pd.testing.assert_frame_equal(
            frame,
            pd.DataFrame(
                {column: pd.Series([], dtype=pd.ArrowDtype(pa.string())) for column in ["gene", "annotation"]}
            ),
            check_exact=True,
        )

    @pytest.mark.parametrize("delimiter", [",", "\t"])
    def test_quoted_lexemes_empty_fields_and_blank_records(self, tmp_path: Path, delimiter: str) -> None:
        """Distinguish empty text from missing fields while preserving valid CSV quoting."""
        path = tmp_path / "features.txt"
        path.write_text(
            f'\ufeffgene{delimiter}annotation\r\n"00{delimiter}1"{delimiter}"line 1\nline ""2"""\r\n'
            f"\r\n \r\nNA{delimiter}\r\nNULL{delimiter}NaN\r\n",
            encoding="utf-8",
        )

        frame = _read_named_rows(ArtifactFile(path), "features", delimiter=delimiter)

        pd.testing.assert_frame_equal(
            frame,
            pd.DataFrame(
                {"gene": [f"00{delimiter}1", "NA", "NULL"], "annotation": ['line 1\nline "2"', "", "NaN"]},
                dtype=pd.ArrowDtype(pa.string()),
            ),
            check_exact=True,
        )

    @pytest.mark.parametrize("row", ['gene\t"unterminated\n', '"gene"suffix\tvalue\n', '"gene" \tvalue\n'])
    def test_malformed_quoting_is_rejected(self, tmp_path: Path, row: str) -> None:
        """Reject unclosed quoted fields and non-delimiter text after a closing quote."""
        path = tmp_path / "features.tsv"
        path.write_text(f"gene\tannotation\n{row}", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed rows"):
            _read_named_rows(ArtifactFile(path), "features", delimiter="\t")

    def test_nul_is_rejected_without_truncation(self, tmp_path: Path) -> None:
        """Reject embedded NUL at ingestion instead of truncating a lexical annotation."""
        path = tmp_path / "features.tsv"
        path.write_text("gene\tannotation\n001\tbefore\x00after\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="NUL"):
            _read_named_rows(ArtifactFile(path), "features", delimiter="\t")

    @pytest.mark.parametrize("content", ["gene\tI\ngene\n"])
    def test_malformed_artifact(self, tmp_path: Path, content: str) -> None:
        """Reject physically incomplete enrichment rows."""
        path = tmp_path / "features.tsv"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=r"header|missing fields"):
            _read_named_rows(ArtifactFile(path), "features", delimiter="\t")

    def test_closes_after_header_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Close the owned enrichment stream when header validation fails."""
        path = tmp_path / "features.tsv"
        path.write_text("gene\tgene\n", encoding="utf-8")
        opened = mocker.spy(Path, "open")

        with pytest.raises(ReaderFormatError, match="header"):
            _read_named_rows(ArtifactFile(path), "features", delimiter="\t")

        assert opened.spy_return_list
        assert all(stream.closed for stream in opened.spy_return_list)

    def test_closes_after_row_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Close the owned stream when strict parsing rejects a malformed quoted value."""
        path = tmp_path / "features.tsv"
        path.write_text('gene\tannotation\n001\t"unterminated\n', encoding="utf-8")
        opened = mocker.spy(Path, "open")

        with pytest.raises(ReaderFormatError, match="malformed rows"):
            _read_named_rows(ArtifactFile(path), "features", delimiter="\t")

        assert opened.spy_return_list
        assert all(stream.closed for stream in opened.spy_return_list)


class TestAddMetrics:
    """Validate metric names and category dictionaries."""

    def test_ignored_fields_still_reject_nul(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").write_text(
            "Metric,Value,Category,Ignored\nReads,100,QC,before\x00after\n", encoding="utf-8"
        )

        with pytest.raises(ReaderFormatError, match="NUL"):
            _add_metrics(ad.AnnData(), discover_layout(tmp_path).metrics)

    @pytest.mark.parametrize(
        "rows",
        [
            "Name,Value\nReads,2\n",
            "Metric,Value,Category\nReads,1,QC\nReads,2,QC\n",
            "Metric,Value,Category\n,1,QC\n",
            "Metric,Value,Category\nReads,1,\n",
            "Metric,Value,Category\nReads,100,QC,extra\n",
            "Metric,Value,Category,Category\nReads,100,QC,extra\n",
        ],
    )
    def test_invalid_metrics(self, tmp_path: Path, rows: str) -> None:
        """Reject ambiguous categories, duplicate metric keys, and malformed metric rows."""
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=r"metric|columns"):
            _add_metrics(ad.AnnData(), discover_layout(tmp_path).metrics)

    @pytest.mark.parametrize("category", ["spatialdata_io_positions", "spatialdata_attrs"])
    def test_reader_output_categories_are_reserved(self, tmp_path: Path, category: str) -> None:
        """Reject categories that would collide with uns keys written later by the reader."""
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").write_text(f"Metric,Value,Category\nReads,1,{category}\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="conflicting metric categories") as raised:
            curio(tmp_path)

        assert raised.value.context.artifact == "metrics"

    def test_invalid_quoted_header_context(self, tmp_path: Path) -> None:
        """Preserve malformed-header parser identity and artifact context."""
        write_curio(tmp_path)
        path = tmp_path / "sample_Metrics.csv"
        path.write_text('"Metric,Value,Category\n', encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed CSV header") as raised:
            _add_metrics(ad.AnnData(), discover_layout(tmp_path).metrics)

        assert raised.value.path == path


class TestJoinMoran:
    """Join by explicit feature ID and retain count-column order."""

    @pytest.mark.parametrize(
        "rows",
        [
            "gene\ngene-b\n",
            "gene\tI\ngene-b\t0.5\ngene-b\t0.4\n",
            "gene\tI\nmissing\t0.5\n",
            "gene\tI\n\t0.5\n",
            "gene\tsymbol\ngene-b\t1\n",
            "gene\tI\ngene-b\tinf\n",
            "gene\tI\ngene-b\tNA\n",
            "gene\tI\ngene-b\t\n",
            "gene\tI\tI\ngene-b\t0.1\t0.2\n",
            "gene\tI\ngene-b\t0.1\t0.2\n",
            "gene\t\ngene-b\t0.1\n",
        ],
    )
    def test_malformed_moran(self, tmp_path: Path, rows: str) -> None:
        """Reject ambiguous identifiers and invalid supplied feature statistics."""
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError):
            _join_moran(pd.DataFrame({"symbol": ["B"]}, index=["gene-b"]), discover_layout(tmp_path).moran_features)

    def test_header_only_moran(self, tmp_path: Path) -> None:
        """Represent unlisted features as missing statistics without changing feature order."""
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text("gene\tI\n", encoding="utf-8")

        table = curio(tmp_path).tables["table"]

        assert table.var["I"].isna().all()
        assert table.var_names.tolist() == ["gene-b", "gene-a", "gene-c"]

    @pytest.mark.parametrize(("true", "false"), [("TRUE", "FALSE"), ("True", "False"), ("true", "false")])
    def test_boolean_statistics(self, tmp_path: Path, true: str, false: str) -> None:
        """Accept source boolean statistics as scientific booleans."""
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text(f"gene\tvariable\ngene-b\t{true}\ngene-a\t{false}\ngene-c\t{true}\n", encoding="utf-8")

        assert curio(tmp_path).tables["table"].var["variable"].tolist() == [True, False, True]

    def test_invalid_quoted_header_context(self, tmp_path: Path) -> None:
        """Preserve malformed-header parser identity and artifact context."""
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text('"gene\tI\n', encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed CSV header") as raised:
            _join_moran(pd.DataFrame(index=["gene-b"]), discover_layout(tmp_path).moran_features)

        assert raised.value.path == path


class TestReadRows:
    """Read every cell as an exact string with one field per column."""

    def test_headerless_rows_use_given_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "clusters.txt"
        path.write_text("\ufeffbead-b\t0007\nbead-a\tNA\n", encoding="utf-8")

        frame = _read_rows(ArtifactFile(path), "clusters", delimiter="\t", columns=["bead_id", "cluster"], header=False)

        pd.testing.assert_frame_equal(
            frame,
            pd.DataFrame(
                {"bead_id": ["bead-b", "bead-a"], "cluster": ["0007", "NA"]}, dtype=pd.ArrowDtype(pa.string())
            ),
            check_exact=True,
        )

    def test_empty_headerless_file(self, tmp_path: Path) -> None:
        path = tmp_path / "clusters.txt"
        path.write_bytes(b"")

        frame = _read_rows(ArtifactFile(path), "clusters", delimiter="\t", columns=["bead_id", "cluster"], header=False)

        assert frame.columns.tolist() == ["bead_id", "cluster"]
        assert len(frame) == 0
        assert frame.dtypes.tolist() == [pd.ArrowDtype(pa.string())] * 2

    def test_short_row_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "clusters.txt"
        path.write_text("bead-b\t1\nbead-a\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="missing fields") as raised:
            _read_rows(ArtifactFile(path), "clusters", delimiter="\t", columns=["bead_id", "cluster"], header=False)

        assert raised.value.context.artifact == "clusters"

    def test_read_failure_names_the_artifact(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Keep the filesystem exception and add reader and artifact context."""
        path = tmp_path / "clusters.txt"
        path.write_text("bead-b\t1\n", encoding="utf-8")
        mocker.patch.object(Path, "open", side_effect=PermissionError("denied"))

        with pytest.raises(PermissionError, match="denied") as raised:
            _read_rows(ArtifactFile(path), "clusters", delimiter="\t", columns=["bead_id", "cluster"], header=False)

        assert any("curio" in note and "clusters" in note and str(path) in note for note in raised.value.__notes__)
