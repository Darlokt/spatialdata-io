"""Test Stereo-seq reader resource ownership."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io._constants._constants import StereoseqKeys
from spatialdata_io.readers.stereoseq import stereoseq

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class _TrackingH5File:
    def __init__(self, contents: object) -> None:
        self.contents = contents
        self.closed = False

    def __enter__(self) -> object:
        return self.contents

    def __exit__(self, *_: object) -> None:
        self.closed = True


class _FailingH5Contents:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}

    def __getitem__(self, _: object) -> object:
        message = "injected HDF5 read failure"
        raise RuntimeError(message)


class _H5Contents(dict[object, object]):
    def __init__(self, initial: dict[object, object] | None = None) -> None:
        super().__init__({} if initial is None else initial)
        self.attrs: dict[str, object] = {}


class _Dataset:
    def __init__(self, data: np.ndarray, attrs: dict[object, object] | None = None) -> None:
        self.data = data
        self.attrs = attrs or {}

    def __getitem__(self, _: object) -> np.ndarray:
        return self.data


def _stereoseq_layout(path: Path) -> None:
    for directory in (
        StereoseqKeys.REGISTER,
        StereoseqKeys.CELLCUT,
        StereoseqKeys.TISSUECUT,
        StereoseqKeys.CELLCLUSTER,
    ):
        (path / directory).mkdir(parents=True)
    (path / StereoseqKeys.CELLCUT / f"sample{StereoseqKeys.CELLBIN_GEF}").touch()
    (path / StereoseqKeys.TISSUECUT / f"sample{StereoseqKeys.GEF_FILE}").touch()
    (path / StereoseqKeys.CELLCLUSTER / f"sample.{StereoseqKeys.CELL_CLUSTER_H5AD}").touch()


def _cell_bin_contents() -> _H5Contents:
    cell_bin = _H5Contents(
        {
            StereoseqKeys.CELL_DATASET: np.zeros((1, 10), dtype=np.int64),
            StereoseqKeys.CELL_EXON: np.zeros(1, dtype=np.int64),
            StereoseqKeys.FEATURE_KEY: np.array([[b"gene", 0, 1, 1, 1]], dtype=object),
            StereoseqKeys.GENE_EXON: np.zeros(1, dtype=np.int64),
            StereoseqKeys.CELL_EXP: {
                StereoseqKeys.GENE_ID: np.zeros(1, dtype=np.int64),
                StereoseqKeys.COUNT: np.zeros(1, dtype=np.int64),
            },
            StereoseqKeys.GENE_EXP: {
                StereoseqKeys.CELL_ID: np.zeros(1, dtype=np.int64),
                StereoseqKeys.COUNT: np.zeros(1, dtype=np.int64),
            },
            StereoseqKeys.CELL_EXP_EXON: np.zeros(1, dtype=np.int64),
            StereoseqKeys.GENE_EXP_EXON: np.zeros(1, dtype=np.int64),
            StereoseqKeys.BLOCK_INDEX: np.zeros(1, dtype=np.int64),
            StereoseqKeys.BLOCK_SIZE: np.zeros(1, dtype=np.int64),
            StereoseqKeys.CELL_TYPE_LIST: np.zeros(1, dtype=np.int64),
            StereoseqKeys.CELL_BORDER: np.zeros((1, 32, 2), dtype=np.int16),
        }
    )
    return _H5Contents({StereoseqKeys.CELL_BIN: cell_bin})


def _square_bin_contents() -> _H5Contents:
    expression = _Dataset(
        np.array([[0, 0, 2]], dtype=np.int64),
        attrs={StereoseqKeys.RESOLUTION: np.array(1)},
    )
    feature = np.array([[b"gene", 0, 1]], dtype=object)
    return _H5Contents(
        {
            StereoseqKeys.GENE_EXP: {
                "bin1": {
                    StereoseqKeys.EXPRESSION: expression,
                    StereoseqKeys.FEATURE_KEY: feature,
                }
            }
        }
    )


def _return_table(table: object, **_: object) -> object:
    return table


class TestStereoseq:
    """Test explicit ownership of Stereo-seq HDF5 inputs."""

    def test_image_model_options_are_not_mutated_on_layout_failure(self, tmp_path: Path) -> None:
        options = MappingProxyType({"chunks": (1, 8, 8), "scale_factors": [4, 4]})

        with pytest.raises(FileNotFoundError):
            stereoseq(tmp_path, dataset_id="sample", image_models_kwargs=options)

        assert dict(options) == {"chunks": (1, 8, 8), "scale_factors": [4, 4]}

    def test_cell_bin_file_closes_on_read_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _stereoseq_layout(tmp_path)
        resource = _TrackingH5File(_FailingH5Contents())
        mocker.patch("spatialdata_io.readers.stereoseq.ad.read_h5ad", return_value=ad.AnnData(np.zeros((1, 1))))
        mocker.patch("spatialdata_io.readers.stereoseq.h5py.File", return_value=resource)

        with pytest.raises(RuntimeError, match="injected HDF5 read failure"):
            stereoseq(tmp_path, dataset_id="sample", read_square_bin=False)

        assert resource.closed

    def test_cell_bin_file_closes_before_downstream_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _stereoseq_layout(tmp_path)
        resource = _TrackingH5File(_cell_bin_contents())
        adata = ad.AnnData(np.zeros((1, 1)), obs=pd.DataFrame(index=["cell"]), var=pd.DataFrame(index=["gene"]))
        mocker.patch("spatialdata_io.readers.stereoseq.ad.read_h5ad", return_value=adata)
        mocker.patch("spatialdata_io.readers.stereoseq.h5py.File", return_value=resource)
        mocker.patch("spatialdata_io.readers.stereoseq.TableModel.parse", side_effect=RuntimeError("injected"))

        with pytest.raises(RuntimeError, match="injected"):
            stereoseq(tmp_path, dataset_id="sample", read_square_bin=False)

        assert resource.closed

    def test_square_bin_file_closes_on_read_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _stereoseq_layout(tmp_path)
        cell_resource = _TrackingH5File(_cell_bin_contents())
        square_resource = _TrackingH5File(_FailingH5Contents())
        adata = ad.AnnData(np.zeros((1, 1)), obs=pd.DataFrame(index=["cell"]), var=pd.DataFrame(index=["gene"]))
        mocker.patch("spatialdata_io.readers.stereoseq.ad.read_h5ad", return_value=adata)
        mocker.patch(
            "spatialdata_io.readers.stereoseq.h5py.File",
            side_effect=[cell_resource, square_resource],
        )
        mocker.patch("spatialdata_io.readers.stereoseq.TableModel.parse", side_effect=_return_table)
        shape = mocker.MagicMock()
        mocker.patch("spatialdata_io.readers.stereoseq.ShapesModel.parse", return_value=shape)

        with pytest.raises(RuntimeError, match="injected HDF5 read failure"):
            stereoseq(tmp_path, dataset_id="sample")

        assert cell_resource.closed
        assert square_resource.closed

    def test_square_bin_file_closes_before_downstream_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _stereoseq_layout(tmp_path)
        cell_resource = _TrackingH5File(_cell_bin_contents())
        square_resource = _TrackingH5File(_square_bin_contents())
        adata = ad.AnnData(np.zeros((1, 1)), obs=pd.DataFrame(index=["cell"]), var=pd.DataFrame(index=["gene"]))
        mocker.patch("spatialdata_io.readers.stereoseq.ad.read_h5ad", return_value=adata)
        mocker.patch(
            "spatialdata_io.readers.stereoseq.h5py.File",
            side_effect=[cell_resource, square_resource],
        )
        mocker.patch("spatialdata_io.readers.stereoseq.TableModel.parse", side_effect=_return_table)
        shape = mocker.MagicMock()
        mocker.patch("spatialdata_io.readers.stereoseq.ShapesModel.parse", return_value=shape)
        mocker.patch("spatialdata_io.readers.stereoseq.PointsModel.parse", return_value=mocker.sentinel.points)
        mocker.patch("spatialdata_io.readers.stereoseq.tqdm", side_effect=RuntimeError("injected"))

        with pytest.raises(RuntimeError, match="injected"):
            stereoseq(tmp_path, dataset_id="sample")

        assert cell_resource.closed
        assert square_resource.closed
