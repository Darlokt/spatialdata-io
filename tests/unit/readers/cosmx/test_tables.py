"""Bounded CosMx counts, typed metadata, FOV positions, and exact table alignment."""

from __future__ import annotations

import gzip
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.cosmx import _tables
from spatialdata_io.readers.cosmx._tables import (
    build_table,
    read_counts,
    read_fov_positions,
    read_header,
    read_metadata,
)
from spatialdata_io.readers.cosmx._transforms import FovPosition

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_COUNTS = '"fov","cell_ID","B","HLA-A","NegPrb1"\n2,1,0,3,0\n1,0,9,9,9\n1,2,4,0,1\n1,1,0,0,2\n'
_DEFAULT_BATCH_BYTES = _tables._BATCH_BYTES  # noqa: SLF001, RUF100
_METADATA = (
    "fov,cell_ID,CenterX_local_px,CenterY_local_px,CenterX_global_px,CenterY_global_px,Area,Mean.DAPI,note\n"
    "1,1,2,5,102,15,4,1.5,01\n1,2,6,2,106,12,3,NA,\n2,1,2,5,202,25,5,,x\n"
)


def _write(tmp_path: Path, name: str, text: str, *, compressed: bool = False) -> ArtifactFile:
    path = tmp_path / (f"{name}.gz" if compressed else name)
    path.write_bytes(gzip.compress(text.encode(), mtime=0) if compressed else text.encode())
    return ArtifactFile(path)


class TestReadHeader:
    """Headers are UTF-8, and parsed with quoting."""

    @pytest.mark.parametrize("compressed", [False, True])
    def test_quoted_header(self, tmp_path: Path, *, compressed: bool) -> None:
        path = _write(tmp_path, "c.csv", _COUNTS, compressed=compressed)
        assert read_header(path, "count matrix") == ("fov", "cell_ID", "B", "HLA-A", "NegPrb1")

    def test_malformed(self, tmp_path: Path) -> None:
        path = tmp_path / "c.csv"
        path.write_bytes(b'"unterminated\n')
        with pytest.raises(ReaderFormatError, match="malformed CSV") as error:
            read_header(ArtifactFile(path), "cell metadata")
        assert error.value.context == ReaderErrorContext("cosmx", "cell metadata")

    def test_unreadable_source_preserves_exception_and_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = _write(tmp_path, "m.csv", _METADATA)
        failure = PermissionError(13, "denied", str(path))
        mocker.patch.object(type(path), "open", side_effect=failure)
        with pytest.raises(PermissionError) as caught:
            read_header(path, "cell metadata")
        assert caught.value is failure
        assert caught.value.errno == 13
        assert any("cosmx" in note and "cell metadata" in note for note in caught.value.__notes__)


class TestReadCounts:
    """Counts stream to exact float32/f64 CSR without background rows."""

    @pytest.mark.parametrize("compressed", [False, True])
    def test_background_removed_and_order_kept(
        self, tmp_path: Path, mocker: MockerFixture, *, compressed: bool
    ) -> None:
        mocker.patch("spatialdata_io.readers.cosmx._tables._BATCH_BYTES", 128)
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS, compressed=compressed))
        assert isinstance(counts.expression, sparse.csr_array)
        assert counts.expression.dtype == np.float32
        np.testing.assert_array_equal(counts.expression.toarray(), [[0, 3, 0], [4, 0, 1], [0, 0, 2]])
        assert counts.fovs.tolist() == [2, 1, 1]
        assert counts.cell_ids.tolist() == [1, 2, 1]

    @pytest.mark.parametrize(("value", "dtype"), [(7, "float32"), (2**24 + 1, "float64"), (2**24 + 2, "float32")])
    def test_adaptive_exact_counts(self, tmp_path: Path, value: int, dtype: str) -> None:
        text = f"fov,cell_ID,A,B\n1,1,1,0\n1,2,{value},2\n"
        counts = read_counts(_write(tmp_path, "c.csv", text))
        assert counts.expression.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(counts.expression.data, [1, value, 2])

    def test_rejects_inexact_float64_counts(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="not exactly representable") as caught:
            read_counts(_write(tmp_path, "c.csv", f"fov,cell_ID,A\n1,1,{2**53 + 1}\n"))
        assert caught.value.path == tmp_path / "c.csv"
        assert caught.value.context == ReaderErrorContext("cosmx", "count matrix")

    def test_native_batches_preserve_exact_counts(self, tmp_path: Path, mocker: MockerFixture) -> None:
        # Only the last row needs float64, so every earlier 1,000-row block is float32 on its own.
        rows = "".join(
            f"{1 + row % 3},{row // 3 + 1},{row % 5},{(row * 7) % 4},{2**24 + 1 if row == 20_000 else row % 2}\n"
            for row in range(20_001)
        )
        path = _write(tmp_path, "c.csv", "fov,cell_ID,A,B,C\n" + rows)
        blocks = mocker.spy(_tables, "as_csr_array")
        whole = read_counts(path)
        # The complete source fits the default parser block; smaller native batches keep exact CSR values.
        assert blocks.call_count == 1
        mocker.patch.object(_tables, "_BATCH_BYTES", 1024)
        batched = read_counts(path)
        assert blocks.call_count > 2
        assert whole.expression.dtype == batched.expression.dtype == np.float64
        assert whole.expression.nnz == batched.expression.nnz
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(whole.expression, name), getattr(batched.expression, name))
        np.testing.assert_array_equal(whole.fovs, batched.fovs)
        np.testing.assert_array_equal(whole.cell_ids, batched.cell_ids)

    def test_truncated_gzip_stream_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "c.csv", _COUNTS, compressed=True)
        path.write_bytes(path.read_bytes()[:-12])
        with pytest.raises(ReaderFormatError, match="invalid gzip data"):
            read_counts(path)

    def test_header_only(self, tmp_path: Path) -> None:
        counts = read_counts(_write(tmp_path, "c.csv", "fov,cell_ID,A\n"))
        assert counts.expression.shape == (0, 1)

    @pytest.mark.parametrize("compressed", [False, True])
    def test_background_counts_obey_the_typed_domain(self, tmp_path: Path, *, compressed: bool) -> None:
        path = _write(tmp_path, "c.csv", "fov,cell_ID,A\n1,0,-1\n", compressed=compressed)
        with pytest.raises(ReaderFormatError, match="malformed CSV") as caught:
            read_counts(path)
        assert caught.value.context == ReaderErrorContext("cosmx", "count matrix")
        assert caught.value.path == path

    @pytest.mark.parametrize(("count", "dtype"), [(2**63, "float32"), (2**63 + 2048, "float64")])
    def test_unsigned_counts_above_signed_range_are_exact(self, tmp_path: Path, count: int, dtype: str) -> None:
        source = f"fov,cell_ID,A\n{2**64 - 1},{2**64 - 1},{count}\n"
        counts = read_counts(_write(tmp_path, "c.csv", source))
        assert str(counts.fovs.dtype) == str(counts.cell_ids.dtype) == "uint64[pyarrow]"
        assert counts.fovs.tolist() == counts.cell_ids.tolist() == [2**64 - 1]
        assert counts.fovs.name is counts.cell_ids.name is None
        assert counts.expression.dtype == np.dtype(dtype)
        assert int(counts.expression.data[0]) == count

    @pytest.mark.parametrize("compressed", [False, True])
    def test_source_closes_when_a_record_is_malformed(
        self, tmp_path: Path, mocker: MockerFixture, *, compressed: bool
    ) -> None:
        path = _write(tmp_path, "c.csv", "fov,cell_ID,A\n1,1,1\n1,2\n", compressed=compressed)
        sources = mocker.spy(_tables.gzip, "GzipFile") if compressed else mocker.spy(type(path), "open")
        with pytest.raises(ReaderFormatError, match=r"malformed CSV \(") as caught:
            read_counts(path)
        assert caught.value.context == ReaderErrorContext("cosmx", "count matrix")
        assert sources.spy_return_list
        assert all(source.closed for source in sources.spy_return_list)

    @pytest.mark.parametrize("batch_bytes", [128, _DEFAULT_BATCH_BYTES])
    @pytest.mark.parametrize("cells", [(1, 2, 3), (0, 0, 0), (0, 2, 3)])
    def test_background_modes(
        self, tmp_path: Path, mocker: MockerFixture, batch_bytes: int, cells: tuple[int, ...]
    ) -> None:
        mocker.patch("spatialdata_io.readers.cosmx._tables._BATCH_BYTES", batch_bytes)
        text = "fov,cell_ID,A,B\n" + "".join(f"1,{cell},{i},0\n" for i, cell in enumerate(cells))
        counts = read_counts(_write(tmp_path, "c.csv", text))
        assert isinstance(counts.expression, sparse.csr_array)
        assert not isinstance(counts.expression, sparse.spmatrix)
        assert counts.expression.has_canonical_format
        assert counts.expression.dtype == np.float32
        expected = np.array([[i, 0] for i, cell in enumerate(cells) if cell], dtype=np.int64).reshape(-1, 2)
        np.testing.assert_array_equal(counts.expression.toarray(), expected)
        assert counts.cell_ids.tolist() == [cell for cell in cells if cell]

    def test_large_sparse_expression_is_never_densified(self, tmp_path: Path, mocker: MockerFixture) -> None:
        features = [f"Gene{i}" for i in range(512)]
        rows = [f"1,{row + 1}," + ",".join("1" if i == row % 512 else "0" for i in range(512)) for row in range(2048)]
        path = _write(tmp_path, "c.csv", "fov,cell_ID," + ",".join(features) + "\n" + "\n".join(rows) + "\n")
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("expression densified"))
        counts = read_counts(path)
        assert counts.expression.shape == (2048, 512)
        assert counts.expression.nnz == 2048
        np.testing.assert_array_equal(counts.expression.data, np.ones(2048, dtype=np.int64))

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("fov,cell_ID,A\n1,1,0\n1,1,2\n", "duplicate"),
            ("fov,cell_ID,A\n1,1,1.5\n", "malformed CSV"),
            ("fov,cell_ID,A\n0,1,1\n", "missing or zero 'fov'"),
            ("fov,cell_ID,A\n1,1\n", r"malformed CSV \("),
        ],
    )
    def test_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_counts(_write(tmp_path, "c.csv", text))


class TestReadMetadata:
    """Keys, centers, rounding bounds, and annotation types are exact."""

    @pytest.mark.parametrize("compressed", [False, True])
    def test_printed_coordinate_precision_is_independent_of_numeric_value(
        self, tmp_path: Path, *, compressed: bool
    ) -> None:
        text = (
            "fov,cell_ID,CenterX_local_px,CenterY_local_px,CenterX_global_px,CenterY_global_px\n"
            "1,1,2.00,5.000,102.0000,15.00000\n"
            "1,2,6.0,2.0,106.00,12.00\n"
        )
        metadata = read_metadata(_write(tmp_path, "m.csv", text, compressed=compressed))
        np.testing.assert_array_equal(metadata.local, [[2, 5], [6, 2]])
        np.testing.assert_array_equal(metadata.global_, [[102, 15], [106, 12]])
        np.testing.assert_array_equal(metadata.samples[0].local_half, [0.005, 0.05])
        np.testing.assert_array_equal(metadata.samples[0].global_half, [0.00005, 0.005])
        np.testing.assert_array_equal(metadata.samples[1].local_half, [0.0005, 0.05])
        np.testing.assert_array_equal(metadata.samples[1].global_half, [0.000005, 0.005])

    @pytest.mark.parametrize("compressed", [False, True])
    def test_typed_annotations_preserve_rounding_nulls_and_literal_strings(
        self, tmp_path: Path, *, compressed: bool
    ) -> None:
        text = _METADATA.replace("4,1.5,01", "0.10000000000000002,1e-400,NA")
        metadata = read_metadata(_write(tmp_path, "m.csv", text, compressed=compressed))
        assert str(metadata.obs["Area"].dtype) == "double[pyarrow]"
        assert metadata.obs["Area"].iloc[0] == float("0.10000000000000002")
        assert metadata.obs["Mean.DAPI"].iloc[0] == 0.0
        assert metadata.obs["Mean.DAPI"].isna().tolist() == [False, True, True]
        assert metadata.obs["note"].tolist() == ["NA", "", "x"]

    @pytest.mark.parametrize("token", ["NULL", "nan"])
    def test_numeric_annotations_allow_only_declared_null_spellings(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ReaderFormatError, match=r"malformed CSV|nonfinite"):
            read_metadata(_write(tmp_path, "m.csv", _METADATA.replace("4,1.5,01", f"4,{token},01")))

    def test_types_and_samples(self, tmp_path: Path) -> None:
        metadata = read_metadata(_write(tmp_path, "m.csv", _METADATA))
        assert metadata.fovs.tolist() == [1, 1, 2]
        assert metadata.cell_ids.tolist() == [1, 2, 1]
        np.testing.assert_array_equal(metadata.local, [[2, 5], [6, 2], [2, 5]])
        np.testing.assert_array_equal(metadata.global_, [[102, 15], [106, 12], [202, 25]])
        np.testing.assert_array_equal(metadata.samples[0].local_half, [0.5, 0.5, 0.5])
        for axis, samples in enumerate(metadata.samples):
            assert np.shares_memory(samples.local, metadata.local[:, axis])
            assert np.shares_memory(samples.global_, metadata.global_[:, axis])
            np.testing.assert_array_equal(samples.local, metadata.local[:, axis])
            np.testing.assert_array_equal(samples.global_, metadata.global_[:, axis])
        obs = metadata.obs
        assert obs.columns.tolist() == ["fov", "cell_ID", "Area", "Mean.DAPI", "note"]
        assert obs["Mean.DAPI"].isna().tolist() == [False, True, True]
        assert obs["note"].tolist() == ["01", "", "x"]

    def test_duplicate_vendor_cell_id_column_is_dropped(self, tmp_path: Path) -> None:
        text = "cell,cell_id," + _METADATA.replace("\n1,1,", "\nc_1_1,c_1_1,1,1,", 1).replace(
            "\n1,2,", "\nc_1_2,c_1_2,1,2,", 1
        ).replace("\n2,1,", "\nc_2_1,c_2_1,2,1,", 1)
        obs = read_metadata(_write(tmp_path, "m.csv", text)).obs
        assert "cell_id" not in obs
        assert obs["cell"].tolist() == ["c_1_1", "c_1_2", "c_2_1"]

    @pytest.mark.parametrize(
        ("replace", "match"),
        [
            (("\n1,2,", "\n1,1,"), "duplicate"),
            (("1,1,2,5,102", "1,0,2,5,102"), "missing or zero 'cell_ID'"),
            (("106,12", "inf,12"), "non-decimal"),
            (("1,2,6,2", "1,1.0,6,2"), "malformed CSV"),
            (("01\n", "01\n9,9,0x1,1,1,1,1,1,\n"), "non-decimal|nonnumeric"),
        ],
    )
    def test_rejects(self, tmp_path: Path, replace: tuple[str, str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_metadata(_write(tmp_path, "m.csv", _METADATA.replace(*replace, 1)))

    def test_unusable_column_names_are_sanitized(self, tmp_path: Path) -> None:
        header = "Region,Cell_ID,cell_id,my note,NOTE,cell,Cell,"
        rows = ("A,1,c_x,x,y,c_1_1,u,", "B,2,c_y,x,y,c_1_2,v,", "C,3,c_z,x,y,c_2_1,w,")
        lines = _METADATA.splitlines()
        text = "\n".join([header + lines[0], *(row + line for row, line in zip(rows, lines[1:], strict=True))]) + "\n"
        message = (
            "cosmx: the cell metadata CSV header has column names that a SpatialData table does not accept; using "
            "'Region' -> 'Region_1', 'Cell_ID' -> 'Cell_ID_1', 'cell_id' -> 'cell_id_2', 'my note' -> 'my_note', "
            "'Cell' -> 'Cell_1', 'note' -> 'note_1'. Rename the source columns to choose the names explicitly."
        )
        with pytest.warns(UserWarning, match=re.escape(message)) as caught:
            obs = read_metadata(_write(tmp_path, "m.csv", text)).obs
        assert [str(warning.message) for warning in caught] == [message]
        assert obs.columns.tolist() == [
            "fov", "cell_ID", "Area", "Mean.DAPI", "Region_1", "Cell_ID_1", "cell_id_2", "my_note", "NOTE", "cell",
            "Cell_1", "note_1",
        ]  # fmt: skip
        assert obs["note_1"].tolist() == ["01", "", "x"]
        assert obs["cell"].tolist() == ["c_1_1", "c_1_2", "c_2_1"]

    def test_truncated_gzip_stream_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "m.csv", _METADATA, compressed=True)
        path.write_bytes(path.read_bytes()[:-12])
        with pytest.raises(ReaderFormatError, match="invalid gzip data") as caught:
            read_metadata(path)
        assert caught.value.context == ReaderErrorContext("cosmx", "cell metadata")

    def test_repeated_vendor_ids(self, tmp_path: Path) -> None:
        text = "cell," + _METADATA.replace("\n1,", "\nc,1,").replace("\n2,", "\nc,2,")
        with pytest.raises(ReaderFormatError, match="vendor cell IDs"):
            read_metadata(_write(tmp_path, "m.csv", text))


class TestReadFovPositions:
    """Pixel positions keep their rounding bounds; millimetre-only files yield none."""

    def test_pixels(self, tmp_path: Path) -> None:
        positions = read_fov_positions(_write(tmp_path, "p.csv", '"fov","x_px","y_px"\n1,100,10.25\n'))
        assert positions == {1: FovPosition(100.0, 10.25, 0.5, 0.005)}

    def test_millimetres_only(self, tmp_path: Path) -> None:
        assert read_fov_positions(_write(tmp_path, "p.csv", "Slide,X_mm,Y_mm,FOV\n1,1.5,2.5,1\n")) == {}

    def test_repeated(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="repeated"):
            read_fov_positions(_write(tmp_path, "p.csv", "fov,x_px,y_px\n1,0,0\n1,1,1\n"))


class TestBuildTable:
    """Counts follow metadata order by exact keys; the populations must be identical."""

    def test_aligned_linked_table(self, tmp_path: Path) -> None:
        metadata_path = _write(tmp_path, "m.csv", _METADATA)
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS))
        table = build_table(counts, read_metadata(metadata_path), path=metadata_path, linked=True)
        assert isinstance(table.X, sparse.csr_array)
        np.testing.assert_array_equal(table.X.toarray(), [[0, 0, 2], [4, 0, 1], [0, 3, 0]])
        assert table.obs_names.tolist() == ["0", "1", "2"]
        assert table.var_names.tolist() == ["B", "HLA-A", "NegPrb1"]
        assert table.obs["region"].tolist() == ["1_labels", "1_labels", "2_labels"]
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[2, 5], [6, 2], [2, 5]])
        np.testing.assert_array_equal(table.obsm["spatial_global_pixels"], [[102, 15], [106, 12], [202, 25]])
        assert "spatial" not in table.obsm
        assert table.obs["spatial_pixel_frame"].tolist() == ["fov_1_pixels", "fov_1_pixels", "fov_2_pixels"]
        assert table.obs["spatial_pixel_frame"].cat.categories.tolist() == ["fov_1_pixels", "fov_2_pixels"]
        assert table.uns[POSITIONS_KEY] == {
            "schema_version": 1,
            "coordinates": {
                "spatial_pixels": {
                    "axes": ["x", "y"], "units": "pixel", "position": "cell center", "frame_key": "spatial_pixel_frame"
                },
                "spatial_global_pixels": {
                    "axes": ["x", "y"], "units": "pixel", "position": "cell center", "frame": "global_pixels"
                },
            },
        }  # fmt: skip
        assert get_table_keys(table) == (["1_labels", "2_labels"], "region", "cell_ID")

    def test_calibration_preserves_sparse_and_native_buffers(self, tmp_path: Path) -> None:
        metadata_path = _write(tmp_path, "m.csv", _METADATA)
        source = "fov,cell_ID,B,HLA-A,NegPrb1\n1,1,0,0,2\n1,2,4,0,1\n2,1,0,3,0\n"
        counts = read_counts(_write(tmp_path, "c.csv", source))
        metadata = read_metadata(metadata_path)
        table = build_table(counts, metadata, path=metadata_path, linked=False, pixel_size_um=0.5)
        assert isinstance(table.X, sparse.csr_array)
        assert np.shares_memory(table.X.data, counts.expression.data)
        assert table.obsm["spatial_pixels"] is metadata.local
        assert table.obsm["spatial_global_pixels"] is metadata.global_
        assert not np.shares_memory(table.obsm["spatial"], metadata.global_)
        np.testing.assert_array_equal(table.obsm["spatial"], metadata.global_ * 0.5)
        assert table.uns[POSITIONS_KEY]["coordinates"]["spatial"] == {
            "axes": ["x", "y"],
            "units": "micrometer",
            "position": "cell center",
            "frame": "global",
            "calibration": {
                "global": "Caller-declared 0.5 micrometres per validated global pixel; no stage registration."
            },
        }

    def test_unlinked_table(self, tmp_path: Path) -> None:
        metadata_path = _write(tmp_path, "m.csv", _METADATA)
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS))
        table = build_table(counts, read_metadata(metadata_path), path=metadata_path, linked=False)
        assert "spatialdata_attrs" not in table.uns

    def test_population_mismatch_fails_instead_of_intersecting(self, tmp_path: Path) -> None:
        metadata_path = _write(tmp_path, "m.csv", _METADATA)
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS.replace("2,1,0,3,0\n", "2,7,0,3,0\n")))
        with pytest.raises(ReaderFormatError, match="identical positive count and metadata populations"):
            build_table(counts, read_metadata(metadata_path), path=metadata_path, linked=True)

    @pytest.mark.parametrize("coordinate", [0.25, -0.25])
    def test_nonzero_physical_underflow_fails(self, tmp_path: Path, coordinate: float) -> None:
        metadata_path = _write(tmp_path, "m.csv", _METADATA)
        metadata = read_metadata(metadata_path)
        metadata.global_[0, 0] = coordinate
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS))
        with np.errstate(all="raise"), pytest.raises(ValueError, match=r"nonzero physical.*underflow to zero"):
            build_table(counts, metadata, path=metadata_path, linked=False, pixel_size_um=5e-324)

    def test_zero_and_signed_nonzero_subnormal_products_are_preserved(self, tmp_path: Path) -> None:
        metadata_path = _write(tmp_path, "m.csv", _METADATA)
        metadata = read_metadata(metadata_path)
        metadata.global_[:] = [[0, -0.0], [1, -1], [2, -2]]
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS))
        table = build_table(counts, metadata, path=metadata_path, linked=False, pixel_size_um=5e-324)
        physical = table.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        np.testing.assert_array_equal(physical, [[0, -0.0], [5e-324, -5e-324], [1e-323, -1e-323]])
        np.testing.assert_array_equal(np.signbit(physical), np.signbit(metadata.global_))
