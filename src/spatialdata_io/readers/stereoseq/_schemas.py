"""Validate the GEF (HDF5) and analysis H5AD schemas that the Stereo-seq reader consumes.

GEF compound datasets are read by field name, never by column position, so
field order may vary. Integer compound fields and optional count vectors accept
either byte order at the supported unsigned widths; ``x``
and ``y`` are exact int32 or uint32 DNB-grid values. Root attributes ``omics``
(``"Transcriptomics"`` when present; absent in historical files) and ``sn``
(chip serial) are strict UTF-8 text. The optional root attribute ``version``
(uint32) is the GEF schema version, not a SAW release; STOmics documents it
as the gene expression file format version, and versions 1 to 4 exist
(version 4 stores ``geneID`` beside ``geneName``). It is normalized to
``gef-<n>``.

Square-bin GEF (``<sample>.raw.gef``, ``.tissue.gef`` or ``.gef``)::

    geneExp/bin<N>/
    ├── gene        (G,)  gene | geneID + geneName (fixed-width UTF-8), offset, count
    ├── expression  (R,)  x, y, count; attrs minX, minY, resolution (nm per grid unit)
    └── exon        (R,)  optional per-record exon counts

Cell-bin GEF (``<sample>[.adjusted].cellbin.gef``; root attrs offsetX,
offsetY, resolution)::

    cellBin/
    ├── cell          (C,)  id, x, y, offset, geneCount, expCount, dnbCount, area,
    │                       cellTypeID, clusterID
    ├── gene          (G,)  geneName | geneID + geneName, offset, cellCount | cellcount,
    │                       expCount, maxMIDcount
    ├── cellExp       (R,)  geneID (uint16/uint32 feature index), count
    ├── cellExpExon   (R,)  optional; cellExon (C,) and geneExon (G,) optional summaries
    ├── cellTypeList  (T,)  optional cell-type names
    └── cellBorder    (C, 32, 2) int16 contour offsets, read only for polygons

``offset`` and the count field of each feature (square) or cell (cell bins)
must tile the expression records contiguously. Stored ``x, y`` of every bin
size are bin-1 grid coordinates; ``resolution`` is the physical pitch of that
grid in nanometres (typically 500 nm). Analysis H5AD contracts are
described in ``_analysis``. Feature fields become ``var`` columns under their
GEF names, cell fields ``obs`` columns; all are fixed, valid column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import decode_hdf5_text

if TYPE_CHECKING:
    import numpy.typing as npt
    from anndata import AnnData

    from spatialdata_io.readers._utils.paths import ArtifactFile

EXPRESSION_LAYERS = ("exon", "counts", "log1p")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Identify the validated feature identity and native span fields."""

    identity: str
    name: str | None
    count: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisSchema:
    """Hold validated native AnnData identity and source-local grid coordinates."""

    coordinates: npt.NDArray[np.int64]
    bin_type: Literal["bins", "cell_bins"] | None
    bin_size: int | None
    resolution: int | None
    sample_id: str | None
    cell_ids: tuple[str, ...] | None


def schema_error(path: ArtifactFile, artifact: str, *, found: str, expected: str) -> ReaderFormatError:
    """Construct a contextual external-contract failure."""
    return ReaderFormatError(
        context=ReaderErrorContext("stereoseq", artifact), path=path, found=found, expected=expected
    )


def require_group(parent: h5py.Group, name: str, path: ArtifactFile) -> h5py.Group:
    """Require a named HDF5 group."""
    value = parent.get(name)
    if not isinstance(value, h5py.Group):
        raise schema_error(path, name, found=type(value).__name__, expected="an HDF5 group")
    return value


def require_dataset(parent: h5py.Group, name: str, path: ArtifactFile) -> h5py.Dataset:
    """Require a one-dimensional HDF5 dataset."""
    value = parent.get(name)
    if not isinstance(value, h5py.Dataset) or value.ndim != 1:
        raise schema_error(path, f"{parent.name}/{name}", found=str(value), expected="a one-dimensional HDF5 dataset")
    return value


def integer_dtype(
    dtype: np.dtype[np.generic],
    path: ArtifactFile,
    artifact: str,
    *,
    signed: bool = False,
    widths: tuple[int, ...] = (1, 2, 4, 8),
) -> None:
    """Validate an unsigned integer or explicitly signed coordinate dtype in either byte order."""
    kinds = "iu" if signed else "u"
    if dtype.kind not in kinds or dtype.itemsize not in widths:
        raise schema_error(
            path, artifact, found=str(dtype), expected=f"integer dtype kind {kinds!r}, byte widths {widths}"
        )


def _field(dataset: h5py.Dataset, name: str, path: ArtifactFile) -> np.dtype[np.generic]:
    fields = dataset.dtype.fields
    if fields is None or name not in fields:
        raise schema_error(path, dataset.name, found=str(dataset.dtype), expected=f"a named field {name!r}")
    return fields[name][0]


def feature_schema(dataset: h5py.Dataset, path: ArtifactFile, *, kind: Literal["square", "cell"]) -> FeatureSchema:
    """Validate name-only and stable-ID feature storage profiles by field name."""
    names = dataset.dtype.names or ()
    label: str | None
    if "geneID" in names:
        identity, label = "geneID", "geneName"
        if "gene" in names:
            raise schema_error(
                path, dataset.name, found=str(names), expected="one unambiguous feature identity profile"
            )
    else:
        identity, label = ("gene", None) if kind == "square" else ("geneName", None)

    text_fields = (identity,) if label is None else (identity, label)
    for name in text_fields:
        if _field(dataset, name, path).kind != "S":
            raise schema_error(
                path, dataset.name, found=str(dataset.dtype), expected=f"fixed-width UTF-8 bytes in {name!r}"
            )

    columns: tuple[str, ...]
    if kind == "square":
        count, columns = "count", ("offset", "count")
    else:
        alternatives = [name for name in ("cellCount", "cellcount") if name in names]
        if len(alternatives) != 1:
            raise schema_error(
                path, dataset.name, found=str(names), expected="exactly one cellCount or cellcount field"
            )
        count = alternatives[0]
        columns = ("offset", count, "expCount", "maxMIDcount")

    for name in columns:
        integer_dtype(_field(dataset, name, path), path, f"{dataset.name}/{name}")
    return FeatureSchema(identity, label, count, columns)


def expression_schema(dataset: h5py.Dataset, path: ArtifactFile, *, kind: Literal["square", "cell"]) -> None:
    """Validate primitive expression fields without positional interpretation."""
    integer_dtype(_field(dataset, "count", path), path, f"{dataset.name}/count")
    if kind == "square":
        for name in ("x", "y"):
            integer_dtype(_field(dataset, name, path), path, f"{dataset.name}/{name}", signed=True, widths=(4,))
    else:
        integer_dtype(_field(dataset, "geneID", path), path, f"{dataset.name}/geneID", widths=(2, 4))


def cell_schema(dataset: h5py.Dataset, path: ArtifactFile) -> tuple[str, ...]:
    """Validate the native cell IDs, positions, spans and annotations."""
    columns = ("id", "offset", "geneCount", "expCount", "dnbCount", "area", "cellTypeID", "clusterID")
    for name in columns:
        integer_dtype(_field(dataset, name, path), path, f"{dataset.name}/{name}")
    for name in ("x", "y"):
        integer_dtype(_field(dataset, name, path), path, f"{dataset.name}/{name}", signed=True, widths=(4,))
    return columns


def validate_spans(
    offsets: npt.NDArray[np.generic],
    lengths: npt.NDArray[np.generic],
    records: int,
    path: ArtifactFile,
    artifact: str,
) -> npt.NDArray[np.int64]:
    """Require contiguous spans with overflow-safe Python integer arithmetic."""
    indptr = np.empty(len(offsets) + 1, dtype=np.int64)
    indptr[0] = 0
    end = 0
    for index, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        if int(offset) != end:
            raise schema_error(
                path, artifact, found=f"offset {int(offset)} at row {index}", expected=f"contiguous offset {end}"
            )
        end += int(length)
        if end > records:
            raise schema_error(
                path, artifact, found=f"span ending at {end}", expected=f"an extent within {records} records"
            )
        indptr[index + 1] = end

    if end != records:
        raise schema_error(
            path, artifact, found=f"terminal extent {end}", expected=f"exactly {records} expression records"
        )
    return indptr


def decode_strings(values: npt.NDArray[np.bytes_], path: ArtifactFile, artifact: str) -> list[str]:
    """Decode fixed-width source strings strictly."""
    try:
        return [bytes(value).decode("utf-8") for value in values]
    except UnicodeDecodeError as error:
        raise schema_error(path, artifact, found="invalid UTF-8", expected="UTF-8 feature or category text") from error


def arrow_column(values: npt.NDArray[np.generic]) -> pd.api.extensions.ExtensionArray:
    """Create explicit Arrow-backed storage for owned primitive annotations."""
    # HDF5 may expose non-native endianness; Arrow requires native buffers.
    native = values.astype(values.dtype.newbyteorder("="), copy=False)
    return pd.arrays.ArrowExtensionArray(pa.array(native, type=pa.from_numpy_dtype(native.dtype)))


def feature_annotations(records: npt.NDArray[np.void], schema: FeatureSchema, path: ArtifactFile) -> pd.DataFrame:
    """Normalize feature identity while preserving source order and metadata."""
    identities = decode_strings(records[schema.identity], path, "feature identities")
    if any(not value for value in identities) or len(set(identities)) != len(identities):
        raise schema_error(
            path, "feature identities", found=repr(identities[:5]), expected="nonempty unique source identities"
        )

    frame = pd.DataFrame(index=pd.Index(identities))
    frame[schema.identity] = pd.array(identities, dtype="string[pyarrow]")
    if schema.name is not None:
        frame[schema.name] = pd.array(decode_strings(records[schema.name], path, "geneName"), dtype="string[pyarrow]")
    for name in schema.columns:
        frame[name] = arrow_column(records[name])
    return frame


def integer_attribute(
    attributes: h5py.AttributeManager, name: str, path: ArtifactFile, artifact: str, *, positive: bool = False
) -> int:
    """Read one required scalar integer attribute."""
    value = np.asarray(attributes.get(name))
    if value.size != 1 or value.dtype.kind not in "iu":
        raise schema_error(path, artifact, found=f"{name}={value!r}", expected="one integer attribute value")

    result = int(value.reshape(-1)[0])
    if positive and result <= 0:
        raise schema_error(path, artifact, found=f"{name}={result}", expected="a positive integer")
    return result


def format_version(attributes: h5py.AttributeManager, path: ArtifactFile) -> str | None:
    """Read the optional root GEF schema ``version`` as ``gef-<n>``; ``None`` when absent."""
    if "version" not in attributes:
        return None
    return f"gef-{integer_attribute(attributes, 'version', path, 'GEF version', positive=True)}"


def _text_attribute(attributes: h5py.AttributeManager, name: str, path: ArtifactFile) -> str:
    """Decode one text attribute stored as a scalar or as a one-element vector."""
    value = attributes[name]
    # Stereo-CITE writes ``omics`` as a one-element vector; other producers write scalars.
    if isinstance(value, np.ndarray) and value.shape == (1,):
        value = value[0]
    return decode_hdf5_text(value, attribute=name, context=ReaderErrorContext("stereoseq", name), path=path)


def validate_transcriptomics(attributes: h5py.AttributeManager, path: ArtifactFile) -> None:
    """Reject explicitly unsupported GEF modalities before reading payloads."""
    if "omics" not in attributes:
        return

    modality = _text_attribute(attributes, "omics", path)
    if modality != "Transcriptomics":
        raise schema_error(
            path,
            "omics",
            found=repr(modality),
            expected="'Transcriptomics', or an absent attribute in historical files",
        )


def sample_id(handle: h5py.File, path: ArtifactFile) -> str | None:
    """Read the optional producer sample identifier ``sn``."""
    return _text_attribute(handle.attrs, "sn", path) if "sn" in handle.attrs else None


def optional_counts(group: h5py.Group, name: str, size: int, path: ArtifactFile) -> h5py.Dataset | None:
    """Validate an optional aligned integer summary or expression layer."""
    if name not in group:
        return None
    dataset = require_dataset(group, name, path)
    integer_dtype(dataset.dtype, path, dataset.name)
    if len(dataset) != size:
        raise schema_error(path, dataset.name, found=f"{len(dataset)} rows", expected=f"exactly {size} aligned rows")
    return dataset


def _analysis_integer(value: object, path: ArtifactFile, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise schema_error(path, name, found=repr(value), expected="a positive integer")
    return int(value)


def validate_analysis_table(table: AnnData, path: ArtifactFile) -> AnalysisSchema:
    """Validate native Stereopy annotations without changing the owned table."""
    obs, var = table.obs, table.var
    if not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
        raise schema_error(
            path, "annotations", found="nonmaterialized annotations", expected="pandas observation and feature frames"
        )

    for name, frame in (("obs", obs), ("var", var)):
        if not frame.columns.is_unique or not frame.index.is_unique:
            raise schema_error(
                path, name, found="duplicate columns or index", expected="unique columns and source identities"
            )
        index = frame.index
        if len(index) and (
            pd.api.types.infer_dtype(index, skipna=False) != "string" or index.hasnans or bool(np.any(index == ""))
        ):
            raise schema_error(
                path, name, found="empty or nontext identity", expected="nonempty text source identities"
            )

    matrix = table.X
    if not isinstance(matrix, (np.ndarray, csr_array, csc_array, csr_matrix, csc_matrix)):
        raise schema_error(path, "X", found=type(matrix).__name__, expected="a materialized numerical expression array")
    values = matrix if isinstance(matrix, np.ndarray) else matrix.data
    if values.dtype.kind not in "iuf":
        raise schema_error(path, "X", found="nonnumerical values", expected="numerical analysis expression")
    # Expression stays untouched until final linkage, which validates sparse
    # storage and finiteness of X and every declared expression layer.

    coordinates = _analysis_coordinates(table, obs, path)
    bin_type, bin_size, resolution = _analysis_metadata(table, path)
    sample = _analysis_sample(table, path)
    cell_ids = _analysis_identities(obs, coordinates, path)
    return AnalysisSchema(coordinates, bin_type, bin_size, resolution, sample, cell_ids)


def _analysis_coordinates(table: AnnData, obs: pd.DataFrame, path: ArtifactFile) -> npt.NDArray[np.int64]:
    """Validate the native two-dimensional source grid and redundant annotations."""
    if "spatial" not in table.obsm:
        raise schema_error(
            path, "obsm/spatial", found="missing", expected="an observation-by-2 source-local DNB-grid array"
        )

    spatial = np.asarray(table.obsm["spatial"])
    if spatial.shape != (table.n_obs, 2) or spatial.dtype.kind not in "iuf":
        raise schema_error(
            path,
            "obsm/spatial",
            found=f"shape {spatial.shape}, dtype {spatial.dtype}",
            expected="a numerical (n_obs, 2) array",
        )
    if (
        not np.isfinite(spatial).all()
        or np.any(spatial < 0)
        or np.any(spatial > np.iinfo(np.uint32).max)
        or np.any(spatial != np.floor(spatial))
    ):
        raise schema_error(
            path,
            "obsm/spatial",
            found="nonfinite, fractional or out-of-range coordinates",
            expected="exact uint32 DNB-grid coordinates",
        )

    coordinates = spatial.astype(np.int64, copy=False)
    for axis, column in enumerate(("x", "y")):
        if column in obs and not np.array_equal(obs[column].to_numpy(), coordinates[:, axis]):
            raise schema_error(
                path, f"obs/{column}", found="coordinates disagree with spatial", expected="exact coordinate agreement"
            )
    return coordinates


def _analysis_metadata(
    table: AnnData, path: ArtifactFile
) -> tuple[Literal["bins", "cell_bins"] | None, int | None, int | None]:
    """Keep physical chip pitch separate from population and bin size."""
    # Stereopy exports chip pitch as uns['resolution']; it does not store the
    # filename's clustering resolution there:
    # https://github.com/STOmics/Stereopy/blob/2b21b0019777dc875ab624a1aed42392f227c2bb/stereo/io/reader.py#L989-L996
    metadata = cast("dict[str, object]", table.uns)
    bin_type = metadata.get("bin_type")
    if bin_type is not None and (not isinstance(bin_type, str) or bin_type not in ("bins", "cell_bins")):
        raise schema_error(path, "uns/bin_type", found=repr(bin_type), expected="'bins' or 'cell_bins'")
    normalized_type: Literal["bins", "cell_bins"] | None = None
    if bin_type == "bins":
        normalized_type = "bins"
    elif bin_type == "cell_bins":
        normalized_type = "cell_bins"

    bin_size = _analysis_integer(metadata["bin_size"], path, "uns/bin_size") if "bin_size" in metadata else None
    if normalized_type == "cell_bins" and bin_size is not None and bin_size != 1:
        raise schema_error(path, "uns/bin_size", found=str(bin_size), expected="1 for native cell bins")

    resolution = _analysis_integer(metadata["resolution"], path, "uns/resolution") if "resolution" in metadata else None
    return normalized_type, bin_size, resolution


def _analysis_sample(table: AnnData, path: ArtifactFile) -> str | None:
    """Read the native chip serial while preserving arbitrary sample labels."""
    metadata = cast("dict[str, object]", table.uns)
    sample: str | None = None
    if "sn" in metadata:
        serials = metadata["sn"]
        if not isinstance(serials, pd.DataFrame) or "sn" not in serials or len(serials) != 1:
            raise schema_error(path, "uns/sn", found=repr(serials), expected="one native single-sample sn row")
        serial: object = serials["sn"].iloc[0]
        if not isinstance(serial, str) or not serial:
            raise schema_error(path, "uns/sn", found=repr(serial), expected="a nonempty sample identifier")
        sample = serial
    # orig.ident is the converter's arbitrary sample_id label, not the chip serial.
    return sample


def _analysis_identities(
    obs: pd.DataFrame, coordinates: npt.NDArray[np.int64], path: ArtifactFile
) -> tuple[str, ...] | None:
    """Validate packed or explicitly reindexed observations and optional semantic IDs.

    Stereopy names an observation ``str((x << 32) | y)``; a reindexed table
    uses ``<prefix>:<x>_<y>``. Coordinates are validated uint32 values.
    """
    x, y = coordinates[:, 0], coordinates[:, 1]
    packed = (x.astype(np.uint64) << np.uint64(32)) | y.astype(np.uint64)
    other = obs.index != pd.Index(packed.astype(str))
    if other.any():
        # pandas string methods keep variable-width names; NumPy's fixed-width
        # strings would size every name like the longest one.
        names = pd.Series(obs.index[other])
        parts = names.str.rpartition(":")
        expected = np.strings.add(np.strings.add(x[other].astype(str), "_"), y[other].astype(str))
        reindexed = (
            (parts[0] != "").to_numpy() & (parts[1] == ":").to_numpy() & (parts[2].to_numpy(dtype=object) == expected)
        )
        if not reindexed.all():
            raise schema_error(
                path,
                "obs identity",
                found=repr(names[~reindexed].iloc[0]),
                expected="native packed-coordinate or sample:x_y source identities",
            )

    cell_ids: tuple[str, ...] | None = None
    if "id" in obs:
        ids = obs["id"].to_numpy()
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise schema_error(path, "obs/id", found=str(ids.dtype), expected="nonnegative exact integer cell IDs")
        cell_ids = tuple(ids.astype(str).tolist())
        if len(set(cell_ids)) != len(cell_ids):
            raise schema_error(path, "obs/id", found="duplicate cell IDs", expected="unique source cell IDs")
    return cell_ids
