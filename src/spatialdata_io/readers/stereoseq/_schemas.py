"""Named-field GEF storage and annotation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import numpy.typing as npt
    from anndata import AnnData

    from spatialdata_io.readers._utils.paths import ArtifactFile


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


def format_error(path: ArtifactFile, artifact: str, found: str, expected: str) -> ReaderFormatError:
    """Construct a contextual external-contract failure."""
    return ReaderFormatError(
        context=ReaderErrorContext("stereoseq", artifact), path=path, found=found, expected=expected
    )


def require_group(parent: h5py.Group, name: str, path: ArtifactFile) -> h5py.Group:
    """Require a named HDF5 group."""
    value = parent.get(name)
    if not isinstance(value, h5py.Group):
        raise format_error(path, name, type(value).__name__, "an HDF5 group")
    return value


def require_dataset(parent: h5py.Group, name: str, path: ArtifactFile) -> h5py.Dataset:
    """Require a one-dimensional HDF5 dataset."""
    value = parent.get(name)
    if not isinstance(value, h5py.Dataset) or value.ndim != 1:
        raise format_error(path, f"{parent.name}/{name}", str(value), "a one-dimensional HDF5 dataset")
    return value


def integer_dtype(
    dtype: np.dtype[np.generic],
    path: ArtifactFile,
    artifact: str,
    *,
    signed: bool = False,
    widths: tuple[int, ...] = (1, 2, 4, 8),
) -> None:
    """Validate an unsigned integer or explicitly signed coordinate dtype."""
    kinds = "iu" if signed else "u"
    if dtype.kind not in kinds or dtype.itemsize not in widths:
        raise format_error(path, artifact, str(dtype), f"integer dtype kind {kinds!r}, byte widths {widths}")


def _field(dataset: h5py.Dataset, name: str, path: ArtifactFile) -> np.dtype[np.generic]:
    fields = dataset.dtype.fields
    if fields is None or name not in fields:
        raise format_error(path, dataset.name, str(dataset.dtype), f"a named field {name!r}")
    return fields[name][0]


def feature_schema(dataset: h5py.Dataset, path: ArtifactFile, *, kind: Literal["square", "cell"]) -> FeatureSchema:
    """Validate name-only and stable-ID feature storage profiles by field name."""
    names = dataset.dtype.names or ()
    label: str | None
    if "geneID" in names:
        identity, label = "geneID", "geneName"
        if "gene" in names:
            raise format_error(path, dataset.name, str(names), "one unambiguous feature identity profile")
    else:
        identity, label = ("gene", None) if kind == "square" else ("geneName", None)

    text_fields = (identity,) if label is None else (identity, label)
    for name in text_fields:
        if _field(dataset, name, path).kind != "S":
            raise format_error(path, dataset.name, str(dataset.dtype), f"fixed-width UTF-8 bytes in {name!r}")

    columns: tuple[str, ...]
    if kind == "square":
        count, columns = "count", ("offset", "count")
    else:
        alternatives = [name for name in ("cellCount", "cellcount") if name in names]
        if len(alternatives) != 1:
            raise format_error(path, dataset.name, str(names), "exactly one cellCount or cellcount field")
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
            raise format_error(path, artifact, f"offset {int(offset)} at row {index}", f"contiguous offset {end}")
        end += int(length)
        if end > records:
            raise format_error(path, artifact, f"span ending at {end}", f"an extent within {records} records")
        indptr[index + 1] = end

    if end != records:
        raise format_error(path, artifact, f"terminal extent {end}", f"exactly {records} expression records")
    return indptr


def decode_strings(values: npt.NDArray[np.bytes_], path: ArtifactFile, artifact: str) -> list[str]:
    """Decode fixed-width source strings strictly."""
    try:
        return [bytes(value).decode("utf-8") for value in values]
    except UnicodeDecodeError as error:
        raise format_error(path, artifact, "invalid UTF-8", "UTF-8 feature or category text") from error


def arrow_column(values: npt.NDArray[np.generic]) -> pd.api.extensions.ExtensionArray:
    """Create explicit Arrow-backed storage for owned primitive annotations."""
    # HDF5 may expose non-native endianness; Arrow requires native buffers.
    native = values.astype(values.dtype.newbyteorder("="), copy=False)
    return pd.arrays.ArrowExtensionArray(pa.array(native, type=pa.from_numpy_dtype(native.dtype)))


def feature_annotations(records: npt.NDArray[np.void], schema: FeatureSchema, path: ArtifactFile) -> pd.DataFrame:
    """Normalize feature identity while preserving source order and metadata."""
    identities = decode_strings(records[schema.identity], path, "feature identities")
    if any(not value for value in identities) or len(set(identities)) != len(identities):
        raise format_error(path, "feature identities", repr(identities[:5]), "nonempty unique source identities")

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
        raise format_error(path, artifact, f"{name}={value!r}", "one integer attribute value")

    result = int(value.reshape(-1)[0])
    if positive and result <= 0:
        raise format_error(path, artifact, f"{name}={result}", "a positive integer")
    return result


def validate_transcriptomics(attributes: h5py.AttributeManager, path: ArtifactFile) -> None:
    """Reject explicitly unsupported GEF modalities before reading payloads."""
    if "omics" not in attributes:
        return

    value = np.asarray(attributes["omics"])
    expected = "scalar or one-element text 'Transcriptomics', or an absent historical attribute"
    if value.shape not in ((), (1,)) or value.dtype.kind not in "SUO":
        raise format_error(path, "omics", repr(value), expected)

    scalar = value.reshape(-1)[0]
    if isinstance(scalar, (bytes, np.bytes_)):
        try:
            modality = bytes(scalar).decode("utf-8")
        except UnicodeDecodeError as error:
            raise format_error(path, "omics", "invalid UTF-8", expected) from error
    elif isinstance(scalar, (str, np.str_)):
        modality = str(scalar)
    else:
        raise format_error(path, "omics", repr(scalar), expected)

    if modality != "Transcriptomics":
        raise format_error(path, "omics", repr(modality), expected)


def sample_id(handle: h5py.File, path: ArtifactFile) -> str | None:
    """Read a bounded optional producer sample identifier."""
    if "sn" not in handle.attrs:
        return None

    value = np.asarray(handle.attrs["sn"])
    if value.size != 1 or value.dtype.kind not in "SUO":
        raise format_error(path, "sn", repr(value), "one nonempty text sample identifier")

    scalar = value.reshape(-1)[0]
    if isinstance(scalar, (bytes, np.bytes_)):
        try:
            result = bytes(scalar).decode("utf-8")
        except UnicodeDecodeError as error:
            raise format_error(path, "sn", repr(scalar), "UTF-8 sample identifier") from error
    elif isinstance(scalar, (str, np.str_)):
        result = str(scalar)
    else:
        raise format_error(path, "sn", repr(scalar), "one text sample identifier")

    if not result:
        raise format_error(path, "sn", "empty string", "a nonempty sample identifier")
    return result


def optional_counts(group: h5py.Group, name: str, size: int, path: ArtifactFile) -> h5py.Dataset | None:
    """Validate an optional aligned integer summary or expression layer."""
    if name not in group:
        return None
    dataset = require_dataset(group, name, path)
    integer_dtype(dataset.dtype, path, dataset.name)
    if len(dataset) != size:
        raise format_error(path, dataset.name, f"{len(dataset)} rows", f"exactly {size} aligned rows")
    return dataset


def _analysis_integer(value: object, path: ArtifactFile, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise format_error(path, name, repr(value), "a positive integer")
    return int(value)


def validate_analysis_table(table: AnnData, path: ArtifactFile) -> AnalysisSchema:
    """Validate native Stereopy annotations without changing the owned table."""
    obs, var = table.obs, table.var
    if not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
        raise format_error(path, "annotations", "nonmaterialized annotations", "pandas observation and feature frames")

    for name, frame in (("obs", obs), ("var", var)):
        if not frame.columns.is_unique or not frame.index.is_unique:
            raise format_error(path, name, "duplicate columns or index", "unique columns and source identities")
        if any(not isinstance(value, str) or not value for value in frame.index):
            raise format_error(path, name, "empty or nontext identity", "nonempty text source identities")

    matrix = table.X
    if not isinstance(matrix, (np.ndarray, csr_array, csc_array, csr_matrix, csc_matrix)):
        raise format_error(path, "X", type(matrix).__name__, "a materialized numerical expression array")
    values = matrix if isinstance(matrix, np.ndarray) else matrix.data
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise format_error(path, "X", "nonnumerical or nonfinite values", "finite numerical analysis expression")

    coordinates = _analysis_coordinates(table, obs, path)
    bin_type, bin_size, resolution = _analysis_metadata(table, path)
    sample = _analysis_sample(table, path)
    cell_ids = _analysis_identities(obs, coordinates, path)
    return AnalysisSchema(coordinates, bin_type, bin_size, resolution, sample, cell_ids)


def _analysis_coordinates(table: AnnData, obs: pd.DataFrame, path: ArtifactFile) -> npt.NDArray[np.int64]:
    """Validate the native two-dimensional source grid and redundant annotations."""
    if "spatial" not in table.obsm:
        raise format_error(path, "obsm/spatial", "missing", "an observation-by-2 source-local DNB-grid array")

    spatial = np.asarray(table.obsm["spatial"])
    if spatial.shape != (table.n_obs, 2) or spatial.dtype.kind not in "iuf":
        raise format_error(
            path, "obsm/spatial", f"shape {spatial.shape}, dtype {spatial.dtype}", "a numerical (n_obs, 2) array"
        )
    if (
        not np.isfinite(spatial).all()
        or np.any(spatial < 0)
        or np.any(spatial > np.iinfo(np.uint32).max)
        or np.any(spatial != np.floor(spatial))
    ):
        raise format_error(
            path,
            "obsm/spatial",
            "nonfinite, fractional or out-of-range coordinates",
            "exact uint32 DNB-grid coordinates",
        )

    coordinates = spatial.astype(np.int64, copy=False)
    for axis, column in enumerate(("x", "y")):
        if column in obs and not np.array_equal(obs[column].to_numpy(), coordinates[:, axis]):
            raise format_error(path, f"obs/{column}", "coordinates disagree with spatial", "exact coordinate agreement")
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
        raise format_error(path, "uns/bin_type", repr(bin_type), "'bins' or 'cell_bins'")
    normalized_type: Literal["bins", "cell_bins"] | None = None
    if bin_type == "bins":
        normalized_type = "bins"
    elif bin_type == "cell_bins":
        normalized_type = "cell_bins"

    bin_size = _analysis_integer(metadata["bin_size"], path, "uns/bin_size") if "bin_size" in metadata else None
    if normalized_type == "cell_bins" and bin_size is not None and bin_size != 1:
        raise format_error(path, "uns/bin_size", str(bin_size), "1 for native cell bins")

    resolution = _analysis_integer(metadata["resolution"], path, "uns/resolution") if "resolution" in metadata else None
    return normalized_type, bin_size, resolution


def _analysis_sample(table: AnnData, path: ArtifactFile) -> str | None:
    """Read the native chip serial while preserving arbitrary sample labels."""
    metadata = cast("dict[str, object]", table.uns)
    sample: str | None = None
    if "sn" in metadata:
        serials = metadata["sn"]
        if not isinstance(serials, pd.DataFrame) or "sn" not in serials or len(serials) != 1:
            raise format_error(path, "uns/sn", repr(serials), "one native single-sample sn row")
        serial: object = serials["sn"].iloc[0]
        if not isinstance(serial, str) or not serial:
            raise format_error(path, "uns/sn", repr(serial), "a nonempty sample identifier")
        sample = serial
    # orig.ident is the converter's arbitrary sample_id label, not the chip serial.
    return sample


def _analysis_identities(
    obs: pd.DataFrame, coordinates: npt.NDArray[np.int64], path: ArtifactFile
) -> tuple[str, ...] | None:
    """Validate packed or explicitly reindexed observations and optional semantic IDs."""
    for name, (x, y) in zip(obs.index, coordinates, strict=True):
        packed = str((int(x) << 32) | int(y))
        prefix, separator, coordinate_name = name.rpartition(":")
        is_reindexed = bool(separator and prefix) and coordinate_name == f"{x}_{y}"
        if name != packed and not is_reindexed:
            raise format_error(
                path, "obs identity", repr(name), "native packed-coordinate or sample:x_y source identities"
            )

    cell_ids: tuple[str, ...] | None = None
    if "id" in obs:
        ids = obs["id"].to_numpy()
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise format_error(path, "obs/id", str(ids.dtype), "nonnegative exact integer cell IDs")
        cell_ids = tuple(str(int(value)) for value in ids)
        if len(set(cell_ids)) != len(cell_ids):
            raise format_error(path, "obs/id", "duplicate cell IDs", "unique source cell IDs")
    return cell_ids
