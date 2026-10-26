"""Exact square records, first-occurrence coordinates and bounded reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, sparray, spmatrix

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq import _schemas, _square
from spatialdata_io.readers.stereoseq._layout import SquareSource
from spatialdata_io.readers.stereoseq._square import read_square
from tests.integration.readers.stereoseq._fixtures import write_square_gef

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadSquare:
    """Protect the stored-bin numerical and resource contract."""

    def test_unsupported_modality_fails_before_square_payload_discovery(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "protein.gef")
        with h5py.File(path, "w") as handle:
            handle.attrs["omics"] = "Proteomics"
        with pytest.raises(ReaderFormatError, match="omics"):
            read_square(SquareSource(path, (1,)))

    def test_native_transcriptomic_modality(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            handle.attrs["omics"] = np.array([b"Transcriptomics"], dtype="S32")

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [6, 3])

    @pytest.mark.parametrize("profile", ["name_only", "id_and_name"])
    def test_exact_order_zero_features_and_exon(
        self, tmp_path: Path, profile: Literal["name_only", "id_and_name"]
    ) -> None:
        path = write_square_gef(tmp_path / "s.gef", profile=profile, overflow=True, empty_feature=True)
        (result,) = read_square(SquareSource(ArtifactFile(path), (50,)))
        table = result.table
        assert isinstance(table.X, sparray)
        assert not isinstance(table.X, spmatrix)
        assert isinstance(table.X, csr_array)
        assert table.X.dtype == np.dtype(np.uint64)
        assert table.var_names.tolist() == (
            ["z", "a", "empty"] if profile == "name_only" else ["g_z", "g_a", "g_empty"]
        )
        assert table.obs_names.tolist() == ["0", "1"]
        assert_array_equal(result.coordinates, [[5, 6], [1, 2]])
        assert_array_equal(table.X.data, [300, 3])
        assert_array_equal(table.X.indices, [0, 1])
        assert_array_equal(table.layers["exon"].data, [3, 1])
        assert result.bin_size == 50
        assert result.origin == (0, 0)
        assert result.resolution == 500
        assert result.sample_id is None

    def test_all_stored_bins_numeric_order(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(50, 1))
        with h5py.File(path, "a") as handle:
            handle.attrs["sn"] = "sample.with.dots"
            handle["geneExp"].create_group("other")

        results = read_square(SquareSource(ArtifactFile(path), "all"))
        assert [result.bin_size for result in results] == [1, 50]
        assert results[0].sample_id == "sample.with.dots"

    def test_unselected_bin_and_absent_exons_are_not_requirements(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1, 50))
        with h5py.File(path, "a") as handle:
            del handle["geneExp/bin1/exon"]
            del handle["geneExp/bin50/expression"]

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert "exon" not in result.table.layers
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [6, 3])

    @pytest.mark.parametrize("names", [("bin01",), ()])
    def test_missing_and_ambiguous_bins(self, tmp_path: Path, names: tuple[str, ...]) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            for name in names:
                handle.copy("geneExp/bin1", f"geneExp/{name}")

        with pytest.raises(ReaderFormatError, match=r"bins|bin names"):
            read_square(SquareSource(ArtifactFile(path), (50,)))

    def test_empty_bin_and_explicit_zeros(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            group = handle["geneExp/bin1"]
            genes = group["gene"][:]
            genes["count"] = 0
            genes["offset"] = 0
            group["gene"][:] = genes
            expression = group["expression"]
            dtype, attrs = expression.dtype, dict(expression.attrs)
            del group["expression"]
            replacement = group.create_dataset("expression", data=np.empty(0, dtype=dtype))
            replacement.attrs.update(attrs)
            del group["exon"]

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert result.table.shape == (0, 2)
        assert isinstance(result.table.X, csr_array)
        assert result.table.X.nnz == 0

        path = write_square_gef(path, bins=(1,))
        with h5py.File(path, "a") as handle:
            expression = handle["geneExp/bin1/expression"][:]
            expression["count"] = 0
            handle["geneExp/bin1/expression"][:] = expression

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert result.table.shape == (2, 2)
        assert isinstance(result.table.X, csr_array)
        assert result.table.X.nnz == 2
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [0, 0])

    @pytest.mark.parametrize(
        ("coordinate_dtype", "coordinates"), [("i4", [-2147483648, 2147483647]), ("u4", [0, 4294967295])]
    )
    def test_coordinate_extremes(self, tmp_path: Path, coordinate_dtype: str, coordinates: list[int]) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            group = handle["geneExp/bin1"]
            attrs = dict(group["expression"].attrs)
            del group["expression"]
            dataset = group.create_dataset(
                "expression",
                data=np.array(
                    [
                        (coordinates[0], coordinates[1], 1),
                        (coordinates[0], coordinates[1], 2),
                        (coordinates[1], coordinates[0], 3),
                    ],
                    dtype=[("x", coordinate_dtype), ("y", coordinate_dtype), ("count", "u4")],
                ),
            )
            dataset.attrs.update(attrs)

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert result.coordinates.dtype == np.dtype(np.int64)
        assert_array_equal(result.coordinates, [coordinates, coordinates[::-1]])

    def test_mixed_signedness_uses_distinct_coordinate_keys(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            group = handle["geneExp/bin1"]
            attrs = dict(group["expression"].attrs)
            del group["expression"]
            dataset = group.create_dataset(
                "expression",
                data=np.array(
                    [(-1, 4294967295, 1), (-1, 0, 2), (0, 4294967295, 3)],
                    dtype=[("x", "i4"), ("y", "u4"), ("count", "u4")],
                ),
            )
            dataset.attrs.update(attrs)

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert_array_equal(result.coordinates, [[-1, 4294967295], [-1, 0], [0, 4294967295]])
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [1, 2, 3])

    def test_bounded_slices_and_no_densification(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        mocker.patch.object(_square, "_EXPRESSION_BUFFER_BYTES", 18)
        mocker.patch.object(csr_array, "toarray", side_effect=AssertionError("densified"))
        getitem = h5py.Dataset.__getitem__
        sizes: list[int] = []

        def tracked(dataset: h5py.Dataset, key: object) -> object:
            if dataset.name.endswith("/expression"):
                assert isinstance(key, slice)
                sizes.append((key.stop - key.start) * dataset.dtype.itemsize)
            return getitem(dataset, key)

        mocker.patch.object(h5py.Dataset, "__getitem__", tracked)

        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert sizes == [9, 9, 9]
        assert isinstance(result.table.X, csr_array)
        assert_array_equal(result.table.X.data, [6, 3])

    def test_source_closes_on_schema_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        handle = h5py.File(path, "r")
        mocker.patch.object(h5py, "File", return_value=handle)
        mocker.patch.object(_schemas, "feature_schema", side_effect=RuntimeError("injected"))

        with pytest.raises(RuntimeError, match="injected"):
            read_square(SquareSource(ArtifactFile(path), (1,)))
        assert not handle.id.valid

    def test_external_io_error_has_context(self, tmp_path: Path) -> None:
        with pytest.raises(OSError, match="No such file") as caught:
            read_square(SquareSource(ArtifactFile(tmp_path / "missing.gef"), (1,)))
        assert "stereoseq" in caught.value.__notes__[0]


class TestCheckAggregation:
    """Use independent Python integers at the uint64 limit."""

    @pytest.mark.parametrize(("dtype", "maximum"), [("u1", 255), ("u2", 65535), ("u4", 4294967295)])
    def test_native_count_widths_promote_before_summation(self, tmp_path: Path, dtype: str, maximum: int) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            group = handle["geneExp/bin1"]
            attrs = dict(group["expression"].attrs)
            del group["expression"]
            dataset = group.create_dataset(
                "expression",
                data=np.array(
                    [(9, maximum, 6, 5), (9, 1, 6, 5), (9, 3, 2, 1)],
                    dtype=[("unselected", "u2"), ("count", dtype), ("y", "i4"), ("x", "i4")],
                ),
            )
            dataset.attrs.update(attrs)
        (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
        assert isinstance(result.table.X, csr_array)
        assert int(result.table.X.data[0]) == maximum + 1

    def test_exon_overflow_is_checked_independently(self, tmp_path: Path) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            del handle["geneExp/bin1/exon"]
            handle["geneExp/bin1/exon"] = np.array([2**64 - 1, 1, 0], dtype="u8")
        with pytest.raises(ReaderFormatError, match="uint64"):
            read_square(SquareSource(ArtifactFile(path), (1,)))

    @pytest.mark.parametrize(("values", "fails"), [([2**64 - 2, 1], False), ([2**64 - 1, 1], True)])
    def test_exact_uint64_duplicates(self, tmp_path: Path, values: list[int], *, fails: bool) -> None:
        path = write_square_gef(tmp_path / "s.gef", bins=(1,))
        with h5py.File(path, "a") as handle:
            group = handle["geneExp/bin1"]
            attrs = dict(group["expression"].attrs)
            del group["expression"]
            dataset = group.create_dataset(
                "expression",
                data=np.array(
                    [(5, 6, values[0]), (5, 6, values[1]), (1, 2, 3)], dtype=[("x", "i4"), ("y", "i4"), ("count", "u8")]
                ),
            )
            dataset.attrs.update(attrs)

        if fails:
            with pytest.raises(ReaderFormatError, match="uint64"):
                read_square(SquareSource(ArtifactFile(path), (1,)))
        else:
            (result,) = read_square(SquareSource(ArtifactFile(path), (1,)))
            assert isinstance(result.table.X, csr_array)
            assert int(result.table.X.data[0]) == sum(values)
