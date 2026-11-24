"""Preserve HuBMAP panel annotations independently from QPI biomarkers through Zarr."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata import read_zarr

from spatialdata_io import phenocycler
from tests.integration.readers.phenocycler._fixtures import write_qptiff

if TYPE_CHECKING:
    from pathlib import Path


class TestHuBMAPChannelAnnotations:
    def test_channel_ids_segmentation_roles_and_qpi_channel_order(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "images"
        processed = tmp_path / "lab_processed" / "images"
        raw.mkdir(parents=True)
        processed.mkdir(parents=True)
        write_qptiff(raw / "Sample.qptiff", markers=("DAPI", "CD3"))
        (processed / "Sample.qptiff.channels.csv").write_text(
            "channel id,is channel used for nuclei segmentation,"
            "is channel used for cell segmentation,is antibody,notes\n"
            "Channel:0:1,No,Yes,Yes,extra\n"
            'Channel:0:0,Yes,No,No,"multi\nline"\n'
        )
        sdata = phenocycler(tmp_path)
        output = tmp_path / "out.zarr"
        sdata.write(output)
        stored = read_zarr(output)
        assert stored.images["Sample_image"].c.values.tolist() == ["DAPI", "CD3"]
        assert stored.attrs["akoya"]["regions"]["Sample"]["channels_csv"]["annotations"] == [
            {
                "channel id": "Channel:0:1",
                "is channel used for nuclei segmentation": False,
                "is channel used for cell segmentation": True,
                "is antibody": True,
            },
            {
                "channel id": "Channel:0:0",
                "is channel used for nuclei segmentation": True,
                "is channel used for cell segmentation": False,
                "is antibody": False,
            },
        ]
        assert not stored.labels
