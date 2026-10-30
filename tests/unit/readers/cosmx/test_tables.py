"""Bounded CosMx counts, typed metadata, FOV positions, and exact table alignment."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
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
        assert read_header(path) == ["fov", "cell_ID", "B", "HLA-A", "NegPrb1"]

    @pytest.mark.parametrize("payload", [b"\xff\xfe,a\n", b'"unterminated\n'])
    def test_malformed(self, tmp_path: Path, payload: bytes) -> None:
        path = tmp_path / "c.csv"
        path.write_bytes(payload)
        with pytest.raises(ReaderFormatError, match="malformed CSV header"):
            read_header(ArtifactFile(path))


class TestReadCounts:
    """Counts stream to exact int64 CSR without background rows."""

    @pytest.mark.parametrize("compressed", [False, True])
    def test_background_removed_and_order_kept(
        self, tmp_path: Path, mocker: MockerFixture, *, compressed: bool
    ) -> None:
        mocker.patch("spatialdata_io.readers.cosmx._tables._MAX_BATCH_ROWS", 1)
        counts = read_counts(_write(tmp_path, "c.csv", _COUNTS, compressed=compressed))
        assert isinstance(counts.expression, sparse.csr_array)
        assert counts.expression.dtype == np.int64
        np.testing.assert_array_equal(counts.expression.toarray(), [[0, 3, 0], [4, 0, 1], [0, 0, 2]])
        assert counts.fovs.tolist() == [2, 1, 1]
        assert counts.cell_ids.tolist() == [1, 2, 1]

    def test_truncated_gzip_stream_fails(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "c.csv", _COUNTS, compressed=True)
        path.write_bytes(path.read_bytes()[:-12])
        with pytest.raises(ReaderFormatError, match="invalid gzip data"):
            read_counts(path)

    def test_header_only(self, tmp_path: Path) -> None:
        counts = read_counts(_write(tmp_path, "c.csv", "fov,cell_ID,A\n"))
        assert counts.expression.shape == (0, 1)

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("fov,cell_ID,A\n1,1,0\n1,1,2\n", "duplicate"),
            ("fov,cell_ID,A\n1,1,1.5\n", "non-integer 'A'"),
            ("fov,cell_ID,A\n0,1,1\n", "'fov' values below 1"),
            ("fov,cell_ID,A\n1,1\n", "malformed CSV"),
        ],
    )
    def test_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_counts(_write(tmp_path, "c.csv", text))


class TestReadMetadata:
    """Keys, centers, rounding bounds, and annotation types are exact."""

    def test_types_and_samples(self, tmp_path: Path) -> None:
        metadata = read_metadata(_write(tmp_path, "m.csv", _METADATA))
        assert metadata.fovs.tolist() == [1, 1, 2]
        assert metadata.cell_ids.tolist() == [1, 2, 1]
        np.testing.assert_array_equal(metadata.local, [[2, 5], [6, 2], [2, 5]])
        np.testing.assert_array_equal(metadata.global_, [[102, 15], [106, 12], [202, 25]])
        np.testing.assert_array_equal(metadata.samples[0].local_half, [0.5, 0.5, 0.5])
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
            (("1,1,2,5,102", "1,0,2,5,102"), "below 1"),
            (("106,12", "inf,12"), "nonfinite"),
            (("01\n", "01\n9,9,0x1,1,1,1,1,1,\n"), "non-decimal|nonnumeric"),
        ],
    )
    def test_rejects(self, tmp_path: Path, replace: tuple[str, str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_metadata(_write(tmp_path, "m.csv", _METADATA.replace(*replace, 1)))

    def test_case_variant_names_fail(self, tmp_path: Path) -> None:
        text = _METADATA.replace("note\n", "note,NOTE\n").replace("01\n", "01,a\n").replace(",\n", ",,b\n", 1)
        text = text.replace(",x\n", ",x,c\n")
        with pytest.raises(ReaderFormatError, match="case-variant"):
            read_metadata(_write(tmp_path, "m.csv", text))

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
        np.testing.assert_array_equal(table.obsm["spatial"], [[2, 5], [6, 2], [2, 5]])
        np.testing.assert_array_equal(table.obsm["global"], [[102, 15], [106, 12], [202, 25]])
        assert get_table_keys(table) == (["1_labels", "2_labels"], "region", "cell_ID")

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
