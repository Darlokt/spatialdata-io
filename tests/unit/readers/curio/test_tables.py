"""Curio owned H5AD and identifier-aligned enrichment."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.curio._layout import _discover_layout
from spatialdata_io.readers.curio._tables import _enrich_features, _enrich_metrics, _read_named_table, _read_table
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadTable:
    """Validate owned H5AD and eager enrichment schemas."""

    @pytest.mark.parametrize(
        "rows",
        ["bead-b\t1\t2\n", "bead-b\t\n", "bead-b\t1\nbead-b\t0\nbead-c\t1\n", "missing\t1\nbead-a\t0\nbead-c\t1\n"],
    )
    def test_invalid_clusters(self, tmp_path: Path, rows: str) -> None:
        """Reject malformed assignments and incomplete or duplicated bead populations."""
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="cluster"):
            _read_table(_discover_layout(tmp_path))

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
            _read_table(_discover_layout(tmp_path))

    def test_missing_expression(self, tmp_path: Path) -> None:
        """Reject an annotation-only H5AD at the count-bearing reader boundary."""
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        table.X = None
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="missing expression") as raised:
            _read_table(_discover_layout(tmp_path))

        assert raised.value.path == path
        assert raised.value.context.reader == "curio"

    def test_h5ad_failure_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Retain storage failure identity and add reader artifact context."""
        write_curio(tmp_path)
        error = OSError("injected read failure")
        mocker.patch("spatialdata_io.readers.curio._tables.ad.read_h5ad", side_effect=error)

        with pytest.raises(OSError, match="injected") as raised:
            _read_table(_discover_layout(tmp_path))

        assert any("curio" in note and "counts" in note for note in raised.value.__notes__)

    @pytest.mark.parametrize("axis", ["obs", "var"])
    @pytest.mark.parametrize("malformed", ["empty_id", "null_id", "duplicate_columns"])
    def test_malformed_h5ad_annotation_boundary(
        self, tmp_path: Path, mocker: MockerFixture, axis: str, malformed: str
    ) -> None:
        """Reject missing identifiers and duplicate annotation names returned by H5AD loading."""
        write_curio(tmp_path)
        layout = _discover_layout(tmp_path)
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
            _read_table(layout)

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

    def test_nul_cannot_truncate_a_bead_identifier(self, tmp_path: Path) -> None:
        """Prevent parser truncation from aligning a different bead identifier."""
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text(
            "bead-b\x00suffix\t1\nbead-a\t0\nbead-c\t1\n", encoding="utf-8"
        )

        with pytest.raises(ReaderFormatError, match="bead IDs"):
            curio(tmp_path)

    @pytest.mark.parametrize("row", ['"bead-b"suffix\t1\n', 'bead-b\t"unterminated\n'])
    def test_malformed_cluster_quoting_is_rejected(self, tmp_path: Path, row: str) -> None:
        """Reject malformed quoted assignments before lexical IDs can change meaning."""
        write_curio(tmp_path)
        path = tmp_path / "sample_cluster_assignment.txt"
        path.write_text(row, encoding="utf-8")

        with pytest.raises(pd.errors.ParserError) as raised:
            _read_table(_discover_layout(tmp_path))

        assert any(str(path) in note and "cluster assignments" in note for note in raised.value.__notes__)


class TestReadNamedTable:
    """Keep source headers, exact physical row widths, and explicit file ownership."""

    @pytest.mark.parametrize("delimiter", [",", "\t"])
    @pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
    def test_header_only_has_explicit_lexical_columns(self, tmp_path: Path, delimiter: str, ending: str) -> None:
        """Retain named string columns even without a trailing header newline."""
        path = tmp_path / "features.txt"
        path.write_text(f"gene{delimiter}annotation{ending}", encoding="utf-8")

        frame = _read_named_table(ArtifactFile(path), delimiter=delimiter)

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

        frame = _read_named_table(ArtifactFile(path), delimiter=delimiter)

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

        with pytest.raises(ReaderFormatError, match="malformed enrichment rows"):
            _read_named_table(ArtifactFile(path), delimiter="\t")

    def test_nul_is_preserved_for_semantic_rejection(self, tmp_path: Path) -> None:
        """Preserve embedded NUL instead of truncating a lexical feature annotation."""
        path = tmp_path / "features.tsv"
        path.write_text("gene\tannotation\n001\tbefore\x00after\n", encoding="utf-8")

        frame = _read_named_table(ArtifactFile(path), delimiter="\t")

        assert frame.iloc[0].tolist() == ["001", "before\x00after"]

    @pytest.mark.parametrize("content", ["gene\tI\ngene\n"])
    def test_malformed_artifact(self, tmp_path: Path, content: str) -> None:
        """Reject physically incomplete enrichment rows."""
        path = tmp_path / "features.tsv"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=r"header|row width"):
            _read_named_table(ArtifactFile(path), delimiter="\t")

    def test_closes_after_header_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Close the owned enrichment stream when header validation fails."""
        path = tmp_path / "features.tsv"
        path.write_text("gene\tgene\n", encoding="utf-8")
        stream = path.open(encoding="utf-8")
        mocker.patch("pathlib.Path.open", return_value=stream)

        with pytest.raises(ReaderFormatError, match="header"):
            _read_named_table(ArtifactFile(path), delimiter="\t")

        assert stream.closed

    def test_closes_after_row_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Close the owned stream when strict parsing rejects a malformed quoted value."""
        path = tmp_path / "features.tsv"
        path.write_text('gene\tannotation\n001\t"unterminated\n', encoding="utf-8")
        stream = path.open(encoding="utf-8")
        mocker.patch("pathlib.Path.open", return_value=stream)

        with pytest.raises(ReaderFormatError, match="malformed enrichment rows"):
            _read_named_table(ArtifactFile(path), delimiter="\t")

        assert stream.closed


class TestEnrichMetrics:
    """Validate metric names and category dictionaries."""

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
            _enrich_metrics(ad.AnnData(), _discover_layout(tmp_path).metrics)

    def test_invalid_quoted_header_context(self, tmp_path: Path) -> None:
        """Preserve malformed-header parser identity and artifact context."""
        write_curio(tmp_path)
        path = tmp_path / "sample_Metrics.csv"
        path.write_text('"Metric,Value,Category\n', encoding="utf-8")

        with pytest.raises(csv.Error) as raised:
            _enrich_metrics(ad.AnnData(), _discover_layout(tmp_path).metrics)

        assert any(str(path) in note for note in raised.value.__notes__)


class TestEnrichFeatures:
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
            _enrich_features(
                pd.DataFrame({"symbol": ["B"]}, index=["gene-b"]), _discover_layout(tmp_path).moran_features
            )

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

        with pytest.raises(csv.Error) as raised:
            _enrich_features(pd.DataFrame(index=["gene-b"]), _discover_layout(tmp_path).moran_features)

        assert any(str(path) in note for note in raised.value.__notes__)
