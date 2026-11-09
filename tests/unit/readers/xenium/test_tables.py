"""Counts, protein scaling, cell annotations, identity checks and final linkage."""

from __future__ import annotations

import io
import tarfile
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.counts import read_10x_h5
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.xenium._analysis import Analysis
from spatialdata_io.readers.xenium._cells import read_cells_store
from spatialdata_io.readers.xenium._layout import H5Counts, MtxArchive, MtxDirectory, TabularSource
from spatialdata_io.readers.xenium._metadata import ProteinPanel, ProteinTarget
from spatialdata_io.readers.xenium._schemas import cell_types
from spatialdata_io.readers.xenium._tables import (
    Cells,
    build_table,
    link_table,
    read_cells,
    read_counts,
    read_feature_index,
    read_header,
)
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad
    from pytest_mock import MockerFixture

    from spatialdata_io.readers.xenium._cells import CellStore
    from spatialdata_io.readers.xenium._layout import TableTarget
    from spatialdata_io.readers.xenium._tables import FeatureSelection
    from spatialdata_io.readers.xenium._version import Family

COUNTS = ReaderErrorContext("xenium", "cell-feature matrix", "6.0.0")
CELLS = ReaderErrorContext("xenium", "cells table", "6.0.0")
TABLE = ReaderErrorContext("xenium", "table", "6.0.0")
PANEL = ProteinPanel(
    {"name": "panel"},
    {"TXP000019": ProteinTarget("PD-1 protein", "n/a", 16), "TXP000023": ProteinTarget("VISTA protein", "c2", 17)},
)


def _h5(bundle: fx.XeniumFixture) -> H5Counts:
    return H5Counts(ArtifactFile(bundle.root / "cell_feature_matrix.h5"))


def _forbid_densification(mocker: MockerFixture) -> None:
    mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("densified"))
    mocker.patch.object(sparse.csr_array, "todense", side_effect=AssertionError("densified"))


class TestReadCounts:
    """Counts keep source order, select feature types and scale protein exactly."""

    def test_count_precision_error_identifies_source(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        source = _h5(bundle)
        with h5py.File(source.file, "r+") as handle:
            values = handle["matrix/data"][:].astype(np.int64)
            values[0] = 2**53 + 1
            del handle["matrix/data"]
            handle["matrix/data"] = values
        with pytest.raises(ReaderFormatError, match="not exactly representable") as caught:
            read_counts(source, "all", None, context=COUNTS)
        assert caught.value.path == source.file

    @pytest.mark.parametrize("feature_types", [None, "all", ("Protein Expression", "Gene Expression")])
    @pytest.mark.parametrize("panel", [None, PANEL], ids=["without-panel", "with-panel"])
    def test_preserves_selected_feature_annotations(
        self, tmp_path: Path, feature_types: FeatureSelection, panel: ProteinPanel | None
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        source = read_10x_h5(_h5(bundle).file, context=ReaderErrorContext("xenium", "counts"))
        source_var = cast("pd.DataFrame", source.var)
        selected = (
            np.ones(len(bundle.features), dtype=bool)
            if feature_types == "all"
            else np.isin(bundle.feature_types, ("Gene Expression", "Protein Expression"))
        )

        counts = read_counts(_h5(bundle), feature_types, panel, context=COUNTS)
        var = cast("pd.DataFrame", counts.table.var)

        pd.testing.assert_frame_equal(var[source_var.columns], source_var.loc[selected])
        assert var["probe_annotation"].tolist() == [f"panel-{i}" for i in np.flatnonzero(selected)]
        np.testing.assert_array_equal(var["is_target"], np.flatnonzero(selected) % 2 == 0)
        expected = bundle.counts[:, selected].astype(np.float32)
        proteins = np.asarray(bundle.feature_types)[selected] == "Protein Expression"
        expected[:, proteins] /= np.float32(fx.PROTEIN_FACTOR)
        np.testing.assert_array_equal(_dense(counts.table.X), expected)
        assert list(var.columns[: len(source_var.columns)]) == list(source_var.columns)

    def test_preserves_missing_annotations_from_the_count_boundary(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        source = read_10x_h5(_h5(bundle).file, context=ReaderErrorContext("xenium", "counts"))
        # The reader preserves the owned decoder result, including missing values;
        # Scanpy currently stringifies nonboolean additional HDF5 feature tags.
        source.var["score"] = np.asarray([0.5, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        expected = cast("pd.DataFrame", source.var).iloc[:3]
        mocker.patch("spatialdata_io.readers.xenium._tables.read_10x_h5", return_value=source)

        table = read_counts(_h5(bundle), None, None, context=COUNTS).table

        pd.testing.assert_frame_equal(cast("pd.DataFrame", table.var), expected)

    def test_unusable_feature_annotation_names_are_renamed(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        source = read_10x_h5(_h5(bundle).file, context=ReaderErrorContext("xenium", "counts"))
        source.var["probe set"] = "v1"
        source.var["Long_Name"] = "x"
        mocker.patch("spatialdata_io.readers.xenium._tables.read_10x_h5", return_value=source)

        with pytest.warns(UserWarning, match=r"using 'probe set' -> 'probe_set', 'Long_Name' -> 'Long_Name_1'"):
            var = cast("pd.DataFrame", read_counts(_h5(bundle), None, PANEL, context=COUNTS).table.var)

        assert list(var.columns[-5:]) == ["probe_set", "Long_Name_1", "long_name", "clone", "codeword_index"]
        assert var["long_name"].tolist()[3:] == ["PD-1 protein", "VISTA protein"]

    @pytest.mark.parametrize("column", ["long_name", "clone", "codeword_index"])
    def test_panel_annotations_never_overwrite_source_columns(self, tmp_path: Path, column: str) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        values = np.asarray([f"source-{i}" for i in range(len(bundle.features))], dtype="S16")
        with h5py.File(_h5(bundle).file, "r+") as handle:
            handle["matrix/features"].create_dataset(column, data=values)

        with pytest.raises(ReaderFormatError, match=f"reserved protein annotation columns.*{column}"):
            read_counts(_h5(bundle), None, PANEL, context=COUNTS)

        # These names are source annotations when no panel enrichment takes place.
        np.testing.assert_array_equal(
            read_counts(_h5(bundle), "all", None, context=COUNTS).table.var[column], values.astype(np.str_)
        )
        genes = read_counts(_h5(bundle), ("Gene Expression",), PANEL, context=COUNTS).table
        np.testing.assert_array_equal(genes.var[column], values[:3].astype(np.str_))

    def test_default_selects_gene_expression_and_protein(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        _forbid_densification(mocker)

        counts = read_counts(_h5(bundle), None, PANEL, context=COUNTS)
        mocker.stopall()

        table = counts.table
        selected = [
            i for i, kind in enumerate(bundle.feature_types) if kind in ("Gene Expression", "Protein Expression")
        ]
        assert table.var_names.tolist() == [bundle.features[i] for i in selected]
        assert isinstance(table.X, sparse.csr_array)
        assert not isinstance(table.X, sparse.spmatrix)
        assert table.X.dtype == np.float32
        expected = bundle.counts[:, selected].astype(np.float32)
        protein = np.asarray([bundle.feature_types[i] == "Protein Expression" for i in selected])
        # One float32 division of the exact stored integers.
        expected[:, protein] = bundle.counts[:, selected][:, protein].astype(np.float32) / np.float32(10)
        np.testing.assert_array_equal(_dense(table.X), expected)
        assert counts.protein == {"scaling_factor": 10, "metric": "MEAN_PER_CELL_STAIN"}
        assert counts.features.tolist() == list(bundle.features)
        var = cast("pd.DataFrame", table.var)
        assert var["feature_types"].tolist() == [bundle.feature_types[i] for i in selected]
        assert var.loc["VISTA", "long_name"] == "VISTA protein"
        assert var.loc["VISTA", "codeword_index"] == 17
        assert pd.isna(var.loc["GeneA", "long_name"])

    def test_all_and_explicit_selections(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)

        everything = read_counts(_h5(bundle), "all", None, context=COUNTS).table
        controls = read_counts(_h5(bundle), ("Negative Control Probe", "Genomic Control"), None, context=COUNTS).table

        assert everything.var_names.tolist() == list(bundle.features)
        np.testing.assert_array_equal(_dense(everything.X), bundle.counts.astype(np.float32))
        assert controls.var_names.tolist() == ["NegControlProbe_00001", "Intergenic_Region_1"]
        with pytest.raises(ReaderFormatError, match=r"feature types \['Protein Expression'\] absent"):
            read_counts(_h5(bundle), ("Protein Expression",), None, context=COUNTS)

    def test_matrix_market_twins_equal_the_h5(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("h5", "mtx", "mtx_archive")))
        reference = read_counts(_h5(bundle), "all", None, context=COUNTS).table

        for source in (
            MtxDirectory(ArtifactDirectory(tmp_path / "cell_feature_matrix")),
            MtxArchive(ArtifactFile(tmp_path / "cell_feature_matrix.tar.gz")),
        ):
            table = read_counts(source, "all", None, context=COUNTS).table
            np.testing.assert_array_equal(_dense(table.X), _dense(reference.X))
            assert table.obs_names.equals(reference.obs_names)
            assert table.var_names.equals(reference.var_names)
            common = ["gene_ids", "feature_types"]
            pd.testing.assert_frame_equal(
                cast("pd.DataFrame", table.var)[common], cast("pd.DataFrame", reference.var)[common]
            )
            assert "genome" not in table.var

    def test_protein_needs_the_h5_factor_and_metric(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0, fx.Options(counts=("h5", "mtx")))
        mtx = MtxDirectory(ArtifactDirectory(tmp_path / "cell_feature_matrix"))

        with pytest.raises(ReaderFormatError, match="MatrixMarket matrix without the scaling factor"):
            read_counts(mtx, None, None, context=COUNTS)
        assert read_counts(mtx, ("Gene Expression",), None, context=COUNTS).protein is None

        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            handle.attrs["protein_metric"] = "MEDIAN"
        with pytest.raises(ReaderFormatError, match="MEAN_PER_CELL_STAIN"):
            read_counts(_h5(bundle), None, None, context=COUNTS)
        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            handle.attrs["protein_metric"] = b"MEAN_PER_CELL_STAIN"
            handle.attrs["protein_scaling_factor"] = 0
        with pytest.raises(ReaderFormatError, match="positive integer"):
            read_counts(_h5(bundle), None, None, context=COUNTS)
        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            del handle.attrs["protein_scaling_factor"]
        with pytest.raises(ReaderFormatError, match="positive integer"):
            read_counts(_h5(bundle), None, None, context=COUNTS)

    def test_protein_panel_must_cover_every_protein(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        partial = ProteinPanel({}, {"TXP000019": ProteinTarget("a", "b", 1)})

        with pytest.raises(ReaderFormatError, match=r"no targets for \['TXP000023'\]"):
            read_counts(_h5(bundle), None, partial, context=COUNTS)

    @pytest.mark.filterwarnings("ignore:Variable names are not unique")
    def test_repeated_features_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            names = handle["matrix/features/name"][...]
            names[1] = names[0]
            handle["matrix/features/name"][...] = names

        with pytest.raises(ReaderFormatError, match="repeated feature names or barcodes"):
            read_counts(_h5(bundle), None, None, context=COUNTS)

    @pytest.mark.parametrize("member", ["../escape.tsv.gz", "cell_feature_matrix/link.tsv.gz", None])
    def test_unsafe_or_incomplete_archives_raise(self, tmp_path: Path, member: str | None) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("mtx_archive",)))
        archive = tmp_path / "cell_feature_matrix.tar.gz"
        with tarfile.open(archive, "r:gz") as source:
            members = [(info, source.extractfile(info)) for info in source if info.isfile()]
            payloads = [(info, None if handle is None else handle.read()) for info, handle in members]
        with tarfile.open(archive, "w:gz") as target:
            for info, payload in payloads:
                if member is None and info.name.endswith("matrix.mtx.gz"):
                    continue
                target.addfile(info, io.BytesIO(payload or b""))
            if member is not None:
                extra = tarfile.TarInfo(member)
                extra.size = 3
                target.addfile(extra, io.BytesIO(b"bad"))

        source_spec = MtxArchive(ArtifactFile(archive))
        if member is None:
            with pytest.raises(ReaderFormatError, match="cell_feature_matrix/"):
                read_counts(source_spec, None, None, context=COUNTS)
        else:
            # Unrelated members are never extracted.
            read_counts(source_spec, None, None, context=COUNTS)
            assert not (tmp_path / "escape.tsv.gz").exists()


class TestReadFeatureIndex:
    """Every count source yields the same features and barcodes."""

    @pytest.mark.parametrize("member", ["features.tsv.gz", "barcodes.tsv.gz", "matrix.mtx.gz"])
    def test_repeated_required_archive_members_raise(self, tmp_path: Path, member: str) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("mtx_archive",)))
        path = tmp_path / "cell_feature_matrix.tar.gz"
        with tarfile.open(path, "r:gz") as archive:
            payloads: list[tuple[tarfile.TarInfo, bytes]] = []
            for info in archive:
                if info.isfile():
                    handle = archive.extractfile(info)
                    assert handle is not None
                    with handle:
                        payloads.append((info, handle.read()))
        with tarfile.open(path, "w:gz") as archive:
            for info, payload in payloads:
                archive.addfile(info, io.BytesIO(payload))
                if info.name.endswith(member):
                    archive.addfile(info, io.BytesIO(payload))
        source = MtxArchive(ArtifactFile(path))
        with pytest.raises(ReaderFormatError, match="exactly one"):
            read_feature_index(source, context=COUNTS)
        with pytest.raises(ReaderFormatError, match="exactly one"):
            read_counts(source, None, None, context=COUNTS)

    def test_metadata_only_archive_reads_never_extract_matrix(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("mtx_archive",)))
        extract = mocker.spy(tarfile.TarFile, "extractall")
        index = read_feature_index(MtxArchive(ArtifactFile(tmp_path / "cell_feature_matrix.tar.gz")), context=COUNTS)
        assert index.features.tolist() == list(bundle.features)
        assert index.barcodes.tolist() == list(bundle.barcodes)
        members = extract.call_args.kwargs["members"]
        assert {info.name for info in members} == {
            "cell_feature_matrix/features.tsv.gz",
            "cell_feature_matrix/barcodes.tsv.gz",
        }

    def test_all_sources_agree(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("h5", "mtx", "mtx_archive")))

        for source in (
            _h5(bundle),
            MtxDirectory(ArtifactDirectory(tmp_path / "cell_feature_matrix")),
            MtxArchive(ArtifactFile(tmp_path / "cell_feature_matrix.tar.gz")),
        ):
            index = read_feature_index(source, context=COUNTS)
            assert index.features.tolist() == list(bundle.features)
            assert index.barcodes.tolist() == list(bundle.barcodes)


class TestReadCells:
    """Cell tables from parquet and CSV twins are typed, finite and identical."""

    @pytest.mark.parametrize("profile", [fx.XOA_1_0, fx.XOA_1_3, fx.XOA_2_0, fx.XOA_4_0])
    def test_parquet_and_csv_twins_agree(self, tmp_path: Path, profile: fx.Profile) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile, fx.Options(csv_twins=True))

        parquet = read_cells(
            TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types(profile.family)),
            context=CELLS,
        )
        csv = read_cells(
            TabularSource(ArtifactFile(tmp_path / "cells.csv.gz"), "csv_gz", cell_types(profile.family)), context=CELLS
        )

        assert parquet.ids.tolist() == list(bundle.cell_ids)
        assert parquet.ids.dtype == pd.ArrowDtype(pa.uint64() if profile.family == "integer_ids" else pa.string())
        pd.testing.assert_frame_equal(parquet.frame, csv.frame)
        assert parquet.frame["nucleus_area"].dtype == pd.ArrowDtype(pa.float64())

    @pytest.mark.parametrize("kind", ["csv", "parquet"])
    def test_unsigned_ids_and_counts_preserve_full_width(self, tmp_path: Path, kind: Literal["csv", "parquet"]) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_1_0)
        table = pq.read_table(tmp_path / "cells.parquet")
        ids = [1, 2**63 - 1, 2**63 + 1, 2**64 - 1]
        counts = [0, 2**63 - 1, 2**63 + 1, 2**64 - 1]
        for name, values in (("cell_id", ids), ("total_counts", counts)):
            table = table.set_column(table.column_names.index(name), name, pa.array(values, pa.uint64()))
        path = tmp_path / f"cells.{kind}"
        if kind == "parquet":
            pq.write_table(table, path)
        else:
            pacsv.write_csv(table, path)
        cells = read_cells(TabularSource(ArtifactFile(path), kind, cell_types("integer_ids")), context=CELLS)
        assert cells.ids.tolist() == ids
        assert cells.frame["total_counts"].tolist() == counts
        assert cells.ids.dtype == pd.ArrowDtype(pa.uint64())
        assert cells.frame["total_counts"].dtype == pd.ArrowDtype(pa.uint64())

    @pytest.mark.parametrize("value", ["-1", str(2**64)])
    def test_csv_counts_reject_unsigned_domain_violations(self, tmp_path: Path, value: str) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv"))
        path = bundle.root / "cells.csv"
        lines = path.read_text().splitlines()
        first = lines[1].split(",")
        first[lines[0].replace('"', "").split(",").index("total_counts")] = value
        path.write_text("\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n")
        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cells(TabularSource(ArtifactFile(path), "csv", cell_types("polygon_sets")), context=CELLS)

    def test_null_nucleus_area_becomes_nan(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        table = pq.read_table(tmp_path / "cells.parquet")
        area = pa.array([None if np.isnan(v) else v for v in table["nucleus_area"].to_numpy()], pa.float64())
        pq.write_table(
            table.set_column(table.column_names.index("nucleus_area"), "nucleus_area", area), tmp_path / "cells.parquet"
        )
        assert pq.read_table(tmp_path / "cells.parquet")["nucleus_area"].null_count == 1

        cells = read_cells(
            TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types("polygon_sets")),
            context=CELLS,
        )

        values = cells.frame["nucleus_area"].to_numpy(dtype=np.float64, na_value=np.nan)
        assert np.isnan(values[3])
        assert not cells.frame["nucleus_area"].isna().to_numpy()[:3].any()

    @pytest.mark.parametrize(
        ("column", "value", "match"),
        [
            ("cell_area", np.inf, "nonfinite cell_area"),
            ("x_centroid", np.nan, "nonfinite x_centroid"),
            ("nucleus_area", np.inf, "nonfinite nucleus_area"),
            ("total_counts", -1, "total_counts as uint64"),
            ("cell_area", -1.0, "negative cell_area"),
            ("cell_id", "aaaaaaab-1", "one row per cell"),
            ("cell_id", None, "complete"),
            ("transcript_counts", None, "missing required values"),
        ],
    )
    def test_rejects_invalid_cells(self, tmp_path: Path, column: str, value: object, match: str) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        table = pq.read_table(tmp_path / "cells.parquet")
        values = table[column].to_pylist()
        values[1] = value
        position = table.column_names.index(column)
        pq.write_table(
            table.set_column(position, column, pa.array(values, table.schema.field(column).type)),
            tmp_path / "cells.parquet",
        )

        with pytest.raises(ReaderFormatError, match=match):
            read_cells(
                TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types("polygon_sets")),
                context=CELLS,
            )

    def test_csv_header_contracts(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,cell_id\n")
        with pytest.raises(ReaderFormatError, match="repeated column names"):
            read_cells(TabularSource(ArtifactFile(path), "csv", cell_types("hex_ids")), context=CELLS)

        path.write_text('"cell_id\n')
        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_header(TabularSource(ArtifactFile(path), "csv", cell_types("hex_ids")), context=CELLS)

    def test_csv_missing_values_and_arrow_grammar(self, tmp_path: Path) -> None:
        """An empty typed field is missing, NaN text is NaN, other numbers follow Arrow's grammar."""
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv"))
        path = bundle.root / "cells.csv"
        lines = path.read_text().splitlines()
        header = lines[0].replace('"', "").split(",")
        first = lines[1].split(",")
        first[header.index("nucleus_area")] = "NaN"
        first[header.index("unassigned_codeword_counts")] = ""
        first[header.index("total_counts")] = " 0x10 "
        path.write_text("\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n")

        cells = read_cells(TabularSource(ArtifactFile(path), "csv", cell_types("polygon_sets")), context=CELLS)

        assert np.isnan(cells.frame["nucleus_area"].to_numpy(dtype=np.float64, na_value=0.0)[0])
        assert cells.frame["unassigned_codeword_counts"].isna().tolist() == [True, False, False, False]
        assert cells.frame["unassigned_codeword_counts"].dtype == pd.ArrowDtype(pa.uint64())
        assert cells.frame["total_counts"].tolist()[0] == 16

    @pytest.mark.parametrize(("value", "found"), [("NA", "invalid value 'NA'"), ("2.0", "invalid value '2.0'")])
    def test_csv_counts_are_integers(self, tmp_path: Path, value: str, found: str) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv"))
        path = bundle.root / "cells.csv"
        lines = path.read_text().splitlines()
        first = lines[1].split(",")
        first[lines[0].replace('"', "").split(",").index("total_counts")] = value
        path.write_text("\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n")

        with pytest.raises(ReaderFormatError) as caught:
            read_cells(TabularSource(ArtifactFile(path), "csv", cell_types("polygon_sets")), context=CELLS)
        assert found in caught.value.found
        assert caught.value.path == path

    def test_csv_record_width(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv"))
        path = bundle.root / "cells.csv"
        path.write_text(path.read_text() + "extra,1\n")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cells(TabularSource(ArtifactFile(path), "csv", cell_types("polygon_sets")), context=CELLS)


def _assemble(
    tmp_path: Path, profile: fx.Profile, options: fx.Options | None = None
) -> tuple[fx.XeniumFixture, ad.AnnData]:
    bundle = fx.write_xenium_bundle(tmp_path, profile, options)
    cells = read_cells(
        TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types(profile.family)), context=CELLS
    )
    store = read_cells_store(
        ArtifactFile(tmp_path / "cells.zarr.zip"),
        profile.family,
        pixel_size=fx.PIXEL_SIZE,
        context=ReaderErrorContext("xenium", "cells.zarr.zip"),
    )
    counts = read_counts(_h5(bundle), None, None, context=COUNTS)
    return bundle, build_table(cells, counts, store, None, path=tmp_path, context=CELLS)


def _store(root: Path, family: Family) -> CellStore:
    context = ReaderErrorContext("xenium", "cells.zarr.zip")
    return read_cells_store(ArtifactFile(root / "cells.zarr.zip"), family, pixel_size=fx.PIXEL_SIZE, context=context)


def _dense(matrix: object) -> np.ndarray:
    return sparse.csr_array(cast("sparse.csr_array", matrix)).toarray()


class TestBuildTable:
    """The expression table joins counts, cells and store columns by exact identity."""

    def test_store_columns_for_xoa_2_0(self, tmp_path: Path) -> None:
        bundle, table = _assemble(tmp_path, fx.XOA_2_0)

        obs = cast("pd.DataFrame", table.obs)
        assert table.obs_names.tolist() == list(fx.HEX_IDS)
        np.testing.assert_array_equal(obs["label_id"], [1, 2, 3, 4])
        assert obs["label_id"].dtype == np.uint32
        # XOA 2.0 parquet lacks nucleus_count and segmentation_method: the store supplies them.
        np.testing.assert_array_equal(obs["nucleus_count"], [1, 1, 2, 0])
        assert obs["nucleus_count"].dtype == np.uint64
        assert obs["segmentation_method"].tolist() == [fx.SEGMENTATION_METHODS[m] for m in fx.CELL_METHODS]
        assert np.isnan(obs["nucleus_centroid_x"].to_numpy()[3])
        np.testing.assert_array_equal(
            table.obsm["spatial"], obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
        )
        assert table.obsm["spatial"].dtype == np.float64
        description = table.uns["spatialdata_io_positions"]["coordinates"]["spatial"]
        assert description["frame"] == "global"
        assert description["position"] == "vendor cell centroid"
        assert description["units"] == "micrometer"
        assert bundle.cell_ids == tuple(obs["cell_id"])

    def test_identity_mismatches_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        cells = read_cells(
            TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types("polygon_sets")),
            context=CELLS,
        )
        counts = read_counts(_h5(bundle), None, None, context=COUNTS)
        store = read_cells_store(
            ArtifactFile(tmp_path / "cells.zarr.zip"),
            "polygon_sets",
            pixel_size=fx.PIXEL_SIZE,
            context=ReaderErrorContext("x", "y"),
        )

        reordered = Cells(cells.frame.iloc[::-1].reset_index(drop=True), cells.ids[::-1])
        with pytest.raises(ReaderFormatError, match="differing from the count barcodes"):
            build_table(reordered, counts, None, None, path=tmp_path, context=CELLS)

        shuffled_store = replace(store, cell_ids=store.cell_ids[::-1])
        with pytest.raises(ReaderFormatError, match=r"differing from cells\.zarr\.zip"):
            build_table(cells, counts, shuffled_store, None, path=tmp_path, context=CELLS)

        truncated_store = replace(store, cell_ids=store.cell_ids[:-1])
        with pytest.raises(ReaderFormatError, match=r"differing from cells\.zarr\.zip"):
            build_table(cells, counts, truncated_store, None, path=tmp_path, context=CELLS)

        frame = cells.frame.assign(label_id=1)
        with pytest.raises(ReaderFormatError, match=r"reserved columns \['label_id'\]"):
            build_table(Cells(frame, cells.ids), counts, store, None, path=tmp_path, context=CELLS)

        frame = cells.frame.assign(segmentation_method="other")
        with pytest.raises(ReaderFormatError, match="segmentation_method differing"):
            build_table(Cells(frame, cells.ids), counts, store, None, path=tmp_path, context=CELLS)

    def test_unusable_column_names_are_renamed(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        cells = read_cells(
            TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types("polygon_sets")),
            context=CELLS,
        )
        counts = read_counts(_h5(bundle), None, None, context=COUNTS)
        store = _store(tmp_path, "polygon_sets")
        # A case variant listed before a reader-relied-on column must not displace it.
        frame = pd.concat(
            [
                pd.DataFrame({"Cell_Area": 1.0, "my note": "x", "Label_ID": 2, "_index": 3}, index=cells.frame.index),
                cells.frame,
            ],
            axis=1,
        )

        with pytest.warns(UserWarning, match=r"xenium: the cells table header has column names") as caught:
            table = build_table(Cells(frame, cells.ids), counts, store, None, path=tmp_path, context=CELLS)

        assert str(caught[0].message) == (
            "xenium: the cells table header has column names that a SpatialData table does not accept; using "
            "'Cell_Area' -> 'Cell_Area_1', 'my note' -> 'my_note', 'Label_ID' -> 'Label_ID_1', '_index' -> 'index'. "
            "Rename the source columns to choose the names explicitly."
        )
        obs = cast("pd.DataFrame", table.obs)
        assert list(obs.columns[:4]) == ["Cell_Area_1", "my_note", "Label_ID_1", "index"]
        np.testing.assert_array_equal(obs["cell_area"], cells.frame["cell_area"])
        np.testing.assert_array_equal(obs["label_id"], [1, 2, 3, 4])
        assert list(cells.frame.columns) == list(frame.columns[4:])

    def test_analysis_annotates_the_table(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        cells = read_cells(
            TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet", cell_types("polygon_sets")),
            context=CELLS,
        )
        counts = read_counts(_h5(bundle), None, None, context=COUNTS)
        clusters = pd.Categorical.from_codes([0, 1, 0, -1], categories=pd.Index(["1", "2"], dtype="str"))
        projection = np.arange(8, dtype=np.float64).reshape(4, 2)
        analysis = Analysis({"graphclust": clusters, "Region": clusters, "_index": clusters}, {"pca_m": projection})

        with pytest.warns(UserWarning, match=r"'Region' -> 'Region_1', '_index' -> 'index'"):
            table = build_table(cells, counts, None, analysis, path=tmp_path, context=CELLS)

        obs = cast("pd.DataFrame", table.obs)
        assert list(obs.columns[-3:]) == ["graphclust", "Region_1", "index"]
        assert obs["graphclust"].cat.codes.tolist() == [0, 1, 0, -1]
        np.testing.assert_array_equal(table.obsm["pca_m"], projection)

        shadowing = Analysis({"cell_area": clusters}, {})
        with pytest.raises(
            ReaderFormatError, match=r"clusterings \['cell_area'\] named like cell annotations"
        ) as error:
            build_table(cells, counts, None, shadowing, path=tmp_path, context=CELLS)
        assert error.value.context == ReaderErrorContext("xenium", "analysis", "6.0.0")

    def test_integer_ids_compare_as_canonical_decimal(self, tmp_path: Path) -> None:
        _, table = _assemble(tmp_path, fx.XOA_1_0)

        assert table.obs_names.tolist() == ["1", "2", "3", "4"]
        assert cast("pd.DataFrame", table.obs)["cell_id"].tolist() == [1, 2, 3, 4]


class TestLinkTable:
    """Linkage targets one element and validates exact instance membership."""

    @pytest.mark.parametrize(
        ("target", "instance_key", "expected"),
        [
            ("cell_labels", "label_id", pd.Index(np.arange(1, 5, dtype=np.uint32))),
            ("cell_boundaries", "cell_id", pd.Index(fx.HEX_IDS)),
        ],
    )
    def test_targets(self, tmp_path: Path, target: TableTarget, instance_key: str, expected: pd.Index) -> None:
        _, table = _assemble(tmp_path, fx.XOA_3_0)

        native = table.obsm["spatial"]
        expression = table.X
        linked = link_table(table, target, expected, context=TABLE)
        assert linked is table
        assert linked.obsm["spatial"] is native
        assert linked.X is expression

        assert get_table_keys(linked) == (target, "region", instance_key)
        assert isinstance(linked.X, sparse.csr_array)
        obs = cast("pd.DataFrame", linked.obs)
        assert not any(isinstance(dtype, pd.ArrowDtype) for dtype in obs.dtypes)

    def test_unlinked(self, tmp_path: Path) -> None:
        _, table = _assemble(tmp_path, fx.XOA_3_0)

        unlinked = link_table(table, None, None, context=TABLE)

        assert "spatialdata_attrs" not in unlinked.uns
        assert "region" not in unlinked.obs

    def test_membership_is_exact(self, tmp_path: Path) -> None:
        _, table = _assemble(tmp_path, fx.XOA_3_0)

        with pytest.raises(ReaderFormatError, match="exact target-instance membership"):
            link_table(table, "cell_labels", pd.Index(np.arange(1, 4, dtype=np.uint32)), context=TABLE)
