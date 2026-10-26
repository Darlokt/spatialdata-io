"""Named source schemas, exact spans and annotation normalization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq import _schemas
from tests.integration.readers.stereoseq._fixtures import write_cell_gef, write_square_gef

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestRequireGroup:
    """Require groups without leaking HDF5 KeyError failures."""

    def test_missing_or_dataset(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle["array"] = np.ones(1)
            for name in ("missing", "array"):
                with pytest.raises(ReaderFormatError, match="HDF5 group"):
                    _schemas.require_group(handle, name, path)


class TestRequireDataset:
    """Reject missing, grouped and multidimensional storage."""

    def test_invalid_dataset_kinds(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle.create_group("group")
            handle["array"] = np.ones((2, 2))
            for name in ("missing", "group", "array"):
                with pytest.raises(ReaderFormatError, match="one-dimensional"):
                    _schemas.require_dataset(handle, name, path)


class TestFeatureSchema:
    """Dispatch observed field names independently of physical field order."""

    @pytest.mark.parametrize(
        "invalid", ["missing_identity", "float_count", "ambiguous_identity", "ambiguous_count", "string_type"]
    )
    def test_malformed_profiles(self, tmp_path: Path, invalid: str) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        fields: list[tuple[str, str]] = [("gene", "S5"), ("offset", "u4"), ("count", "u4")]
        kind = "square"
        if invalid == "missing_identity":
            fields[0] = ("unrecognized", "S5")
        elif invalid == "float_count":
            fields[-1] = ("count", "f4")
        elif invalid == "string_type":
            fields[0] = ("gene", "u4")
        elif invalid == "ambiguous_identity":
            fields.extend([("geneID", "S5"), ("geneName", "S5")])
        else:
            fields = [("geneName", "S5"), ("cellCount", "u4"), ("cellcount", "u4")]
            kind = "cell"
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("gene", data=np.empty(0, dtype=fields))

            with pytest.raises(ReaderFormatError):
                _schemas.feature_schema(dataset, path, kind="square" if kind == "square" else "cell")

    def test_lowercase_cellcount_and_reordered_fields(self, tmp_path: Path) -> None:
        path = ArtifactFile(write_cell_gef(tmp_path / "s.gef"))
        with h5py.File(path, "a") as handle:
            dataset = handle["cellBin/gene"]
            names = dataset.dtype.names

            assert names is not None
            fields = [("cellcount" if name == "cellCount" else name, dataset.dtype[name]) for name in reversed(names)]
            values = np.zeros(len(dataset), dtype=fields)
            for name in names:
                values["cellcount" if name == "cellCount" else name] = dataset[name][:]
            del handle["cellBin/gene"]
            dataset = handle.create_dataset("cellBin/gene", data=values)
            schema = _schemas.feature_schema(dataset, path, kind="cell")

            assert schema.count == "cellcount"


class TestExpressionSchema:
    """Require exact supported coordinate and feature-index storage widths."""

    @pytest.mark.parametrize(
        "dtype", [[("x", "i8"), ("y", "i4"), ("count", "u1")], [("geneID", "u1"), ("count", "u2")]]
    )
    def test_unsupported_integer_width(self, tmp_path: Path, dtype: list[tuple[str, str]]) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset("expression", data=np.empty(0, dtype=dtype))

            with pytest.raises(ReaderFormatError, match="byte widths"):
                _schemas.expression_schema(dataset, path, kind="square" if dtype[0][0] == "x" else "cell")


class TestValidateSpans:
    """Prove spans without unsigned overflow or trusting NumPy reductions."""

    @pytest.mark.parametrize(
        ("offsets", "lengths", "records"), [([1], [1], 2), ([0], [3], 2), ([0], [1], 2), ([0, 0], [2**64 - 1, 2], 2)]
    )
    def test_invalid_extents(self, tmp_path: Path, offsets: list[int], lengths: list[int], records: int) -> None:
        with pytest.raises(ReaderFormatError):
            _schemas.validate_spans(
                np.array(offsets, dtype="u8"), np.array(lengths, dtype="u8"), records, ArtifactFile(tmp_path), "spans"
            )

    def test_empty_features_and_zero_length_spans(self, tmp_path: Path) -> None:
        result = _schemas.validate_spans(np.array([0, 0, 2]), np.array([0, 2, 0]), 2, ArtifactFile(tmp_path), "spans")
        assert_array_equal(result, [0, 0, 2, 2])


class TestDecodeStrings:
    """Reject malformed encoding rather than replacing feature names."""

    def test_invalid_utf8(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="invalid UTF-8"):
            _schemas.decode_strings(np.array([b"\xff"], dtype="S1"), ArtifactFile(tmp_path), "text")


class TestFeatureAnnotations:
    """Separate identity uniqueness from accepted duplicate display labels."""

    @pytest.mark.parametrize("identity", [b"", b"z"])
    def test_empty_and_duplicate_identity(self, tmp_path: Path, identity: bytes) -> None:
        path = ArtifactFile(write_square_gef(tmp_path / "s.gef", bins=(1,)))
        with h5py.File(path, "a") as handle:
            dataset = handle["geneExp/bin1/gene"]
            records = dataset[:]
            records["gene"][1] = identity
            schema = _schemas.feature_schema(dataset, path, kind="square")

            with pytest.raises(ReaderFormatError, match="unique"):
                _schemas.feature_annotations(records, schema, path)

    def test_duplicate_display_names_preserve_unique_ids(self, tmp_path: Path) -> None:
        path = ArtifactFile(write_square_gef(tmp_path / "s.gef", profile="id_and_name", bins=(1,)))
        with h5py.File(path, "r") as handle:
            dataset = handle["geneExp/bin1/gene"]
            records = dataset[:]
            records["geneName"] = b"same"
            result = _schemas.feature_annotations(records, _schemas.feature_schema(dataset, path, kind="square"), path)

        assert result.index.tolist() == ["g_z", "g_a"]
        assert result["geneName"].tolist() == ["same", "same"]
        assert isinstance(result["count"].dtype, pd.ArrowDtype)


class TestArrowColumn:
    """HDF5 byte order must not change exact annotation values."""

    @pytest.mark.parametrize("dtype", ["<u8", ">u8"])
    def test_endianness_and_uint64_values(self, dtype: str) -> None:
        values = np.array([0, 2**64 - 1], dtype=dtype)
        column = _schemas.arrow_column(values)

        assert column.tolist() == [0, 2**64 - 1]
        assert_array_equal(values, np.array([0, 2**64 - 1], dtype=dtype))


class TestIntegerAttribute:
    """Validate scalar attributes and positive physical pitch."""

    @pytest.mark.parametrize("value", [None, [1, 2], 1.0, 0, -1])
    def test_invalid_positive_attribute(self, tmp_path: Path, value: float | list[int] | None) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            if value is not None:
                handle.attrs["resolution"] = value

            with pytest.raises(ReaderFormatError):
                _schemas.integer_attribute(handle.attrs, "resolution", path, "pitch", positive=True)


class TestValidateTranscriptomics:
    """Keep optional historical metadata separate from explicit unsupported modalities."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "Transcriptomics",
            np.bytes_(b"Transcriptomics"),
            ["Transcriptomics"],
            np.array([b"Transcriptomics"], dtype="S32"),
        ],
        ids=["missing", "scalar_text", "scalar_bytes", "text_vector", "fixed_bytes_vector"],
    )
    def test_supported_modality_representations(
        self, tmp_path: Path, value: str | np.bytes_ | list[str] | np.ndarray | None
    ) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            if value is not None:
                handle.attrs["omics"] = value
            _schemas.validate_transcriptomics(handle.attrs, path)

    @pytest.mark.parametrize(
        "value",
        [
            "Proteomics",
            "Microorganism",
            "",
            3,
            ["Transcriptomics", "Proteomics"],
            np.array([[b"Transcriptomics"]]),
            np.bytes_(b"\xff"),
        ],
        ids=["protein", "microorganism", "empty", "integer", "multiple", "rank_two", "invalid_utf8"],
    )
    def test_unsupported_and_malformed_modality(
        self, tmp_path: Path, value: str | int | list[str] | np.ndarray | np.bytes_
    ) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["omics"] = value

            with pytest.raises(ReaderFormatError, match="omics"):
                _schemas.validate_transcriptomics(handle.attrs, path)

    def test_reference_is_not_a_modality_name(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["omics"] = handle.ref

            with pytest.raises(ReaderFormatError, match="omics"):
                _schemas.validate_transcriptomics(handle.attrs, path)


class TestSampleId:
    """Accept scalar bytes/text and reject malformed source identities."""

    @pytest.mark.parametrize("value", [b"sample", "sample", np.array([b"sample"])], ids=["bytes", "str", "array"])
    def test_source_text(self, tmp_path: Path, value: bytes | str | np.ndarray) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["sn"] = value

            assert _schemas.sample_id(handle, path) == "sample"

    @pytest.mark.parametrize("value", ["", 3, ["a", "b"], np.bytes_(b"\xff")])
    def test_invalid_source_text(self, tmp_path: Path, value: str | int | list[str] | np.bytes_) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["sn"] = value

            with pytest.raises(ReaderFormatError):
                _schemas.sample_id(handle, path)

    def test_reference_is_not_a_text_identifier(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["sn"] = handle.ref

            with pytest.raises(ReaderFormatError, match="text sample"):
                _schemas.sample_id(handle, path)


class TestOptionalCounts:
    """Require aligned optional summaries and expression exons."""

    def test_misalignment_and_dtype(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "s.gef")
        with h5py.File(path, "w") as handle:
            handle["short"] = np.ones(1, dtype="u2")
            handle["float"] = np.ones(2)
            for name in ("short", "float"):
                with pytest.raises(ReaderFormatError):
                    _schemas.optional_counts(handle, name, 2, path)


def _native_analysis() -> ad.AnnData:
    table = ad.AnnData(
        X=csr_array([[2.5, 0], [1.0, 3.0]]),
        obs=pd.DataFrame({"id": np.array([90, 7], dtype=np.uint32)}, index=[str((20 << 32) | 10), str((9 << 32) | 13)]),
        var=pd.DataFrame(index=["g3", "g1"]),
    )
    table.obsm["spatial"] = np.array([[20, 10], [9, 13]], dtype=np.uint32)
    table.obs["x"] = np.array([20, 9], dtype=np.uint32)
    table.obs["y"] = np.array([10, 13], dtype=np.uint32)
    table.uns.update(bin_type="cell_bins", bin_size=1, resolution=500)
    return table


class TestValidateAnalysisTable:
    """Native annotation contracts establish identity and physical coordinate meaning."""

    def test_accepts_native_and_integer_float_storage_without_mutation(self) -> None:
        table = _native_analysis()
        table.obsm["spatial"] = np.asarray(table.obsm["spatial"], dtype=np.float64)
        result = _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

        assert result.bin_type == "cell_bins"
        assert result.bin_size == 1
        assert result.resolution == 500
        assert result.cell_ids == ("90", "7")
        assert result.sample_id is None
        assert result.coordinates.dtype == np.int64
        np.testing.assert_array_equal(result.coordinates, [[20, 10], [9, 13]])

        assert table.obs_names.tolist() == [str((20 << 32) | 10), str((9 << 32) | 13)]

    @pytest.mark.parametrize(
        "case", ["duplicates_obs", "duplicates_var", "duplicates_columns", "empty_obs", "nontext_var"]
    )
    def test_rejects_ambiguous_or_missing_source_identities(self, case: str) -> None:
        table = _native_analysis()
        if case == "duplicates_obs":
            table.obs_names = ["same", "same"]
        elif case == "duplicates_var":
            table.var_names = ["same", "same"]
        elif case == "duplicates_columns":
            table.obs.columns = ["same"] * 3
        elif case == "empty_obs":
            table.obs_names = ["", str((9 << 32) | 13)]
        else:
            table.var.index = pd.Index([1, 2])

        with pytest.raises(ReaderFormatError, match=r"unique|nonempty"):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    @pytest.mark.parametrize("case", ["none", "nan_sparse", "inf_dense", "complex"])
    def test_rejects_nonmaterialized_or_nonfinite_expression(self, case: str) -> None:
        table = _native_analysis()
        if case == "none":
            table.X = None
        elif case == "nan_sparse":
            table.X = csr_array([[float("nan"), 0], [1, 2]])
        elif case == "inf_dense":
            table.X = np.array([[1.0, float("inf")], [1, 2]])
        else:
            table.X = np.array([[1j, 0], [1, 2]])

        with pytest.raises(ReaderFormatError, match=r"expression|numerical"):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    @pytest.mark.parametrize(
        "case", ["missing", "shape", "text", "nan", "inf", "fractional", "negative", "overflow", "disagreement"]
    )
    def test_rejects_invalid_spatial_coordinates(self, case: str) -> None:
        table = _native_analysis()
        if case == "missing":
            del table.obsm["spatial"]
        elif case == "shape":
            table.obsm["spatial"] = np.zeros((2, 3))
        elif case == "text":
            table.obsm["spatial"] = np.array([["a", "b"], ["c", "d"]])
        elif case == "disagreement":
            table.obs["x"] = [21, 9]
        else:
            invalid = {"nan": np.nan, "inf": np.inf, "fractional": 1.5, "negative": -1.0, "overflow": float(2**32)}[
                case
            ]
            table.obsm["spatial"] = np.array([[invalid, 10], [9, 13]])

        with pytest.raises(ReaderFormatError, match=r"spatial|obs/x"):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    @pytest.mark.parametrize("key", ["resolution", "bin_size"])
    @pytest.mark.parametrize("value", [True, 1.5, 0, -1, "500", None])
    def test_rejects_nonpositive_or_inexact_integer_metadata(self, key: str, value: object) -> None:
        table = _native_analysis()
        table.uns[key] = value

        with pytest.raises(ReaderFormatError, match="positive integer"):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    @pytest.mark.parametrize(
        "case",
        [
            "array_type",
            "cell_size",
            "sn_type",
            "sn_columns",
            "sn_multiple",
            "sn_empty",
        ],
    )
    def test_rejects_malformed_native_metadata(self, case: str) -> None:
        table = _native_analysis()
        if case == "array_type":
            table.uns["bin_type"] = np.array(["bins", "cell_bins"])
        elif case == "cell_size":
            table.uns["bin_size"] = 50
        elif case == "sn_type":
            table.uns["sn"] = "sample"
        elif case == "sn_columns":
            table.uns["sn"] = pd.DataFrame({"sample": ["sample"]})
        elif case == "sn_multiple":
            table.uns["sn"] = pd.DataFrame({"sn": ["sample", "other"]})
        elif case == "sn_empty":
            table.uns["sn"] = pd.DataFrame({"sn": [""]})

        with pytest.raises(ReaderFormatError):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    @pytest.mark.parametrize("case", ["opaque_identity", "wrong_reindex", "float_id", "negative_id", "duplicate_id"])
    def test_rejects_malformed_native_identities(self, case: str) -> None:
        table = _native_analysis()
        if case == "opaque_identity":
            table.obs_names = ["unrelated", "identities"]
        elif case == "wrong_reindex":
            table.obs_names = ["sample:20_11", "sample:9_13"]
        elif case == "float_id":
            table.obs["id"] = [90.0, 7.5]
        elif case == "negative_id":
            table.obs["id"] = [-1, 7]
        else:
            table.obs["id"] = [7, 7]

        with pytest.raises(ReaderFormatError):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    def test_rejects_unknown_bin_type(self) -> None:
        table = _native_analysis()
        table.uns["bin_type"] = "unknown"

        with pytest.raises(ReaderFormatError, match="bin_type"):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))

    @pytest.mark.parametrize("original", ["sample", "arbitrary_label"])
    def test_validates_native_chip_serial_and_preserves_arbitrary_labels(self, original: str) -> None:
        table = _native_analysis()
        table.obs["orig.ident"] = [original, original]
        table.uns["sn"] = pd.DataFrame({"batch": ["-1"], "sn": ["real_chip"]})

        assert _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad"))).sample_id == "real_chip"

    def test_nongeneric_orig_ident_without_sn_does_not_claim_chip_identity(self) -> None:
        table = _native_analysis()
        table.obs["orig.ident"] = ["real_chip", "real_chip"]

        assert _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad"))).sample_id is None

    def test_rejects_nonmaterialized_annotations(self, mocker: MockerFixture) -> None:
        table = _native_analysis()
        mocker.patch.object(ad.AnnData, "obs", new_callable=mocker.PropertyMock, return_value=object())

        with pytest.raises(ReaderFormatError, match="pandas observation"):
            _schemas.validate_analysis_table(table, ArtifactFile(Path("analysis.h5ad")))
