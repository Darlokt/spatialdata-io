"""Cell identities and integer sparse aggregation of raw bins into nuclei."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.visium_hd._schemas import validate_mapping_schema, validate_mappings
from spatialdata_io.readers.visium_hd._tables import read_counts

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

_MAPPING_COLUMNS = ("square_002um", "cell_id", "in_cell", "in_nucleus")
_BATCH_SIZE = 65_536


def cell_ids(names: pd.Index, path: ArtifactFile) -> NDArray[np.int64]:
    """Decode canonical positive cell barcodes without accepting alternate spellings."""
    strings = names.astype("string")
    valid = strings.str.fullmatch(r"cellid_[0-9]{9}-1", na=False)
    if not valid.all():
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "cell barcodes"),
            path=path,
            found="noncanonical cell barcode",
            expected="cellid_{positive_id:09d}-1",
        )

    values = strings.str.slice(7, 16).to_numpy(dtype=np.int64)
    if np.any(values == 0):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "cell barcodes"),
            path=path,
            found="zero cell ID",
            expected="positive cell IDs",
        )

    return values


def read_cell_table(path: ArtifactFile, *, make_var_names_unique: bool) -> ad.AnnData:
    """Preserve the chosen raw or filtered cell population and source order."""
    table = read_counts(path)
    table.obs["instance_id"] = cell_ids(table.obs_names, path)
    if make_var_names_unique:
        table.var_names_make_unique()

    return table


def read_nucleus_table(
    raw_file: ArtifactFile, mapping_file: ArtifactFile, population: ad.AnnData, known_cell_ids: pd.Index
) -> ad.AnnData:
    """Sum raw 2 µm bins, including out-of-tissue bins, in selected cell order."""
    context = ReaderErrorContext("visium_hd", "nucleus aggregation")
    raw = read_counts(raw_file)
    if any(
        not np.array_equal(raw.var[column].to_numpy(), population.var[column].to_numpy())
        for column in ("gene_ids", "genome", "feature_types")
    ):
        raise ReaderFormatError(
            context=context,
            path=raw_file,
            found="different feature IDs/order",
            expected="the same features in raw bins and cell counts",
        )

    # read_counts guarantees canonical CSR; only the accumulation buffer changes dtype.
    imported = cast("csr_array", raw.X)
    # Compare Python integers so NumPy cannot round int64's upper bound when
    # imported counts use floating storage.
    maximum = int(imported.data.max(initial=0))
    if maximum > np.iinfo(np.int64).max:
        raise ReaderFormatError(
            context=context, path=raw_file, found="counts exceed int64", expected="safe int64 nucleus accumulation"
        )
    counts = imported.astype(np.int64)
    source_rows: list[NDArray[np.int64]] = []
    target_rows: list[NDArray[np.int64]] = []
    seen = np.zeros(raw.n_obs, dtype=np.bool_)

    with pq.ParquetFile(mapping_file) as source:
        validate_mapping_schema(source.schema_arrow, mapping_file)
        for batch in source.iter_batches(batch_size=_BATCH_SIZE, columns=list(_MAPPING_COLUMNS)):
            validate_mappings(batch, mapping_file)
            barcodes = batch.column("square_002um").to_pandas()
            indices = raw.obs_names.get_indexer(barcodes)
            if np.any(indices < 0) or np.unique(indices).size != indices.size or np.any(seen[indices]):
                raise ReaderFormatError(
                    context=context,
                    path=mapping_file,
                    found="unknown or repeated 2 µm barcode",
                    expected="one mapping per raw 2 µm bin",
                )
            seen[indices] = True

            members = batch.column("in_nucleus").to_numpy(zero_copy_only=False)
            assigned = batch.column("cell_id").to_pandas()
            nonnull = assigned.dropna()
            if not pd.Index(cell_ids(pd.Index(nonnull), mapping_file)).isin(known_cell_ids).all():
                raise ReaderFormatError(
                    context=context,
                    path=mapping_file,
                    found="unknown mapped cell ID",
                    expected="all assigned cells in nucleus geometry",
                )

            # Valid cells outside the selected raw/filtered population do not contribute.
            cells = population.obs_names.get_indexer(assigned)
            keep = members & (cells >= 0)
            source_rows.append(indices[keep])
            target_rows.append(cells[keep])

    if not seen.all():
        raise ReaderFormatError(
            context=context,
            path=mapping_file,
            found="incomplete barcode mappings",
            expected="one mapping per raw 2 µm bin",
        )

    rows = np.concatenate(target_rows) if target_rows else np.empty(0, dtype=np.int64)
    columns = np.concatenate(source_rows) if source_rows else np.empty(0, dtype=np.int64)
    contributing = np.bincount(rows, minlength=population.n_obs)
    if maximum * int(contributing.max(initial=0)) > np.iinfo(np.int64).max:
        raise ReaderFormatError(
            context=context, found="possible int64 sum overflow", expected="exact int64 accumulation"
        )

    # One operator row per selected cell sums its member bins without densifying counts.
    operator = csr_array((np.ones(rows.size, dtype=np.int64), (rows, columns)), shape=(population.n_obs, raw.n_obs))
    aggregated = operator @ counts
    # Both annotations are eager frames from the shared count boundary, never Dataset2D.
    return ad.AnnData(
        aggregated.astype(np.float32),
        obs=cast("pd.DataFrame", population.obs).copy(),
        var=cast("pd.DataFrame", population.var).copy(),
    )
