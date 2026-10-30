"""Counts, protein scaling, cell annotations, identity checks and final linkage."""

from __future__ import annotations

import io
import tarfile
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.xenium._cells import read_cells_store
from spatialdata_io.readers.xenium._layout import H5Counts, MtxArchive, MtxDirectory, TabularSource
from spatialdata_io.readers.xenium._metadata import ProteinPanel, ProteinTarget
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

    from spatialdata_io.readers.xenium._layout import TableTarget

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

    def test_default_selects_gene_expression_and_protein(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        _forbid_densification(mocker)

        counts = read_counts(_h5(bundle), None, PANEL)
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

        everything = read_counts(_h5(bundle), "all", None).table
        controls = read_counts(_h5(bundle), ("Negative Control Probe", "Genomic Control"), None).table

        assert everything.var_names.tolist() == list(bundle.features)
        np.testing.assert_array_equal(_dense(everything.X), bundle.counts.astype(np.float32))
        assert controls.var_names.tolist() == ["NegControlProbe_00001", "Intergenic_Region_1"]
        with pytest.raises(ReaderFormatError, match=r"feature types \['Protein Expression'\] absent"):
            read_counts(_h5(bundle), ("Protein Expression",), None)

    def test_matrix_market_twins_equal_the_h5(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("h5", "mtx", "mtx_archive")))
        reference = read_counts(_h5(bundle), "all", None).table

        for source in (
            MtxDirectory(ArtifactDirectory(tmp_path / "cell_feature_matrix")),
            MtxArchive(ArtifactFile(tmp_path / "cell_feature_matrix.tar.gz")),
        ):
            table = read_counts(source, "all", None).table
            np.testing.assert_array_equal(_dense(table.X), _dense(reference.X))
            assert table.obs_names.equals(reference.obs_names)
            assert cast("pd.DataFrame", table.var).equals(cast("pd.DataFrame", reference.var))

    def test_protein_needs_the_h5_factor_and_metric(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0, fx.Options(counts=("h5", "mtx")))
        mtx = MtxDirectory(ArtifactDirectory(tmp_path / "cell_feature_matrix"))

        with pytest.raises(ReaderFormatError, match="MatrixMarket matrix without the scaling factor"):
            read_counts(mtx, None, None)
        assert read_counts(mtx, ("Gene Expression",), None).protein is None

        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            handle.attrs["protein_metric"] = "MEDIAN"
        with pytest.raises(ReaderFormatError, match="MEAN_PER_CELL_STAIN"):
            read_counts(_h5(bundle), None, None)
        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            handle.attrs["protein_metric"] = b"MEAN_PER_CELL_STAIN"
            handle.attrs["protein_scaling_factor"] = 0
        with pytest.raises(ReaderFormatError, match="positive integer"):
            read_counts(_h5(bundle), None, None)
        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            del handle.attrs["protein_scaling_factor"]
        with pytest.raises(ReaderFormatError, match="positive integer"):
            read_counts(_h5(bundle), None, None)

    def test_protein_panel_must_cover_every_protein(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_4_0)
        partial = ProteinPanel({}, {"TXP000019": ProteinTarget("a", "b", 1)})

        with pytest.raises(ReaderFormatError, match=r"no targets for \['TXP000023'\]"):
            read_counts(_h5(bundle), None, partial)

    @pytest.mark.filterwarnings("ignore:Variable names are not unique")
    def test_repeated_features_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        with h5py.File(tmp_path / "cell_feature_matrix.h5", "r+") as handle:
            names = handle["matrix/features/name"][...]
            names[1] = names[0]
            handle["matrix/features/name"][...] = names

        with pytest.raises(ReaderFormatError, match="repeated feature names or barcodes"):
            read_counts(_h5(bundle), None, None)

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
                read_counts(source_spec, None, None)
        else:
            # Unrelated members are never extracted.
            read_counts(source_spec, None, None)
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
            read_feature_index(source)
        with pytest.raises(ReaderFormatError, match="exactly one"):
            read_counts(source, None, None)

    def test_metadata_only_archive_reads_never_extract_matrix(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(counts=("mtx_archive",)))
        extract = mocker.spy(tarfile.TarFile, "extractall")
        index = read_feature_index(MtxArchive(ArtifactFile(tmp_path / "cell_feature_matrix.tar.gz")))
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
            index = read_feature_index(source)
            assert index.features.tolist() == list(bundle.features)
            assert index.barcodes.tolist() == list(bundle.barcodes)


class TestReadCells:
    """Cell tables from parquet and CSV twins are typed, finite and identical."""

    @pytest.mark.parametrize("profile", [fx.XOA_1_0, fx.XOA_1_3, fx.XOA_2_0, fx.XOA_4_0])
    def test_parquet_and_csv_twins_agree(self, tmp_path: Path, profile: fx.Profile) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile, fx.Options(csv_twins=True))

        parquet = read_cells(TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet"), profile.family)
        csv = read_cells(TabularSource(ArtifactFile(tmp_path / "cells.csv.gz"), "csv_gz"), profile.family)

        assert parquet.ids.tolist() == list(bundle.cell_ids)
        assert parquet.ids.dtype == (np.int64 if profile.family == "integer_ids" else "str")
        pd.testing.assert_frame_equal(parquet.frame, csv.frame)
        assert parquet.frame["nucleus_area"].dtype == pd.ArrowDtype(pa.float64())

    def test_null_nucleus_area_becomes_nan(self, tmp_path: Path) -> None:
        fx.write_xenium_bundle(tmp_path, fx.XOA_2_0)
        table = pq.read_table(tmp_path / "cells.parquet")
        area = pa.array([None if np.isnan(v) else v for v in table["nucleus_area"].to_numpy()], pa.float64())
        pq.write_table(
            table.set_column(table.column_names.index("nucleus_area"), "nucleus_area", area), tmp_path / "cells.parquet"
        )
        assert pq.read_table(tmp_path / "cells.parquet")["nucleus_area"].null_count == 1

        cells = read_cells(TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet"), "polygon_sets")

        values = cells.frame["nucleus_area"].to_numpy(dtype=np.float64, na_value=np.nan)
        assert np.isnan(values[3])
        assert not cells.frame["nucleus_area"].isna().to_numpy()[:3].any()

    @pytest.mark.parametrize(
        ("column", "value", "match"),
        [
            ("cell_area", np.inf, "nonfinite cell_area"),
            ("x_centroid", np.nan, "nonfinite x_centroid"),
            ("nucleus_area", np.inf, "nonfinite nucleus_area"),
            ("total_counts", -1, "negative total_counts"),
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
            read_cells(TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet"), "polygon_sets")

    def test_csv_header_contracts(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,cell_id\n")
        with pytest.raises(ReaderFormatError, match="unique nonempty"):
            read_cells(TabularSource(ArtifactFile(path), "csv"), "hex_ids")

        path.write_text('"cell_id\n')
        with pytest.raises(ReaderFormatError, match="malformed CSV header"):
            read_header(TabularSource(ArtifactFile(path), "csv"))

    def test_csv_record_width(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(tables="csv"))
        path = bundle.root / "cells.csv"
        path.write_text(path.read_text() + "extra,1\n")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cells(TabularSource(ArtifactFile(path), "csv"), "polygon_sets")


def _assemble(
    tmp_path: Path, profile: fx.Profile, options: fx.Options | None = None
) -> tuple[fx.XeniumFixture, ad.AnnData]:
    bundle = fx.write_xenium_bundle(tmp_path, profile, options)
    cells = read_cells(TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet"), profile.family)
    store = read_cells_store(
        ArtifactFile(tmp_path / "cells.zarr.zip"),
        profile.family,
        pixel_size=fx.PIXEL_SIZE,
        context=ReaderErrorContext("xenium", "cells.zarr.zip"),
    )
    counts = read_counts(_h5(bundle), None, None)
    return bundle, build_table(cells, counts, store, path=tmp_path)


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
        assert obs["nucleus_count"].dtype == np.int64
        assert obs["segmentation_method"].tolist() == [fx.SEGMENTATION_METHODS[m] for m in fx.CELL_METHODS]
        assert np.isnan(obs["nucleus_centroid_x"].to_numpy()[3])
        np.testing.assert_array_equal(
            table.obsm["spatial"], obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
        )
        assert table.obsm["spatial"].dtype == np.float64
        assert bundle.cell_ids == tuple(obs["cell_id"])

    def test_identity_mismatches_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        cells = read_cells(TabularSource(ArtifactFile(tmp_path / "cells.parquet"), "parquet"), "polygon_sets")
        counts = read_counts(_h5(bundle), None, None)
        store = read_cells_store(
            ArtifactFile(tmp_path / "cells.zarr.zip"),
            "polygon_sets",
            pixel_size=fx.PIXEL_SIZE,
            context=ReaderErrorContext("x", "y"),
        )

        reordered = Cells(cells.frame.iloc[::-1].reset_index(drop=True), cells.ids[::-1])
        with pytest.raises(ReaderFormatError, match="differing from the count barcodes"):
            build_table(reordered, counts, None, path=tmp_path)

        shuffled_store = replace(store, cell_ids=store.cell_ids[::-1])
        with pytest.raises(ReaderFormatError, match="differing from the cells table"):
            build_table(cells, counts, shuffled_store, path=tmp_path)

        frame = cells.frame.assign(label_id=1)
        with pytest.raises(ReaderFormatError, match=r"reserved columns \['label_id'\]"):
            build_table(Cells(frame, cells.ids), counts, store, path=tmp_path)

        frame = cells.frame.assign(segmentation_method="other")
        with pytest.raises(ReaderFormatError, match="methods differing"):
            build_table(Cells(frame, cells.ids), counts, store, path=tmp_path)

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

        linked = link_table(table, target, expected)

        assert get_table_keys(linked) == (target, "region", instance_key)
        assert isinstance(linked.X, sparse.csr_array)
        obs = cast("pd.DataFrame", linked.obs)
        assert not any(isinstance(dtype, pd.ArrowDtype) for dtype in obs.dtypes)

    def test_unlinked(self, tmp_path: Path) -> None:
        _, table = _assemble(tmp_path, fx.XOA_3_0)

        unlinked = link_table(table, None, None)

        assert "spatialdata_attrs" not in unlinked.uns
        assert "region" not in unlinked.obs

    def test_membership_is_exact(self, tmp_path: Path) -> None:
        _, table = _assemble(tmp_path, fx.XOA_3_0)

        with pytest.raises(ReaderFormatError, match="exact target-instance membership"):
            link_table(table, "cell_labels", pd.Index(np.arange(1, 4, dtype=np.uint32)))
