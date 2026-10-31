"""Select one legacy library and import its stored image grids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale, Sequence, Translation

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray
    from xarray import DataArray, DataTree

_SPATIAL_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class LegacyImage:
    """Identify a validated image and its relative pixel resolution."""

    key: str
    scale: float


@dataclass(frozen=True, slots=True)
class LegacyLibrary:
    """Carry selected row ordinals and validated image interpretation."""

    name: str | None
    rows: NDArray[np.int64] | None
    images: tuple[LegacyImage, ...]


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) or not key for key in value):
        message = f"{location} must be a mapping with nonempty string keys."
        raise TypeError(message)
    # Runtime key validation establishes the generic mapping contract.
    return cast("Mapping[str, object]", value)


def _option(value: str | None, *, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        message = f"{name} must be a nonempty string or None."
        raise TypeError(message)


def _scale(scales: Mapping[str, object], *, library: str, image: str) -> float:
    key = image if image in scales else f"tissue_{image}_scalef"
    value = scales.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        message = f"Legacy library {library!r} image {image!r} requires a positive finite scale factor at {key!r}."
        raise TypeError(message)
    try:
        scale = float(value)
    except OverflowError as error:
        message = f"Legacy library {library!r} image {image!r} has an invalid scale factor."
        raise ValueError(message) from error
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(1.0 / scale):
        message = f"Legacy library {library!r} image {image!r} has an invalid scale factor {scale!r}."
        raise ValueError(message)
    return scale


def _image(value: object, *, library: str, key: str) -> None:
    if not isinstance(value, np.ndarray):
        message = f"Legacy library {library!r} image {key!r} requires an eager NumPy image."
        raise TypeError(message)
    if value.ndim not in (2, 3) or any(size == 0 for size in value.shape) or value.dtype.kind not in "buif":
        message = f"Legacy library {library!r} image {key!r} requires nonempty YX or YXC real numerical pixels."
        raise ValueError(message)


def _memberships(adata: AnnData, key: str) -> pd.Series:
    observations = cast("pd.DataFrame", adata.obs)
    if key not in observations:
        message = f"Legacy library observation column {key!r} is missing."
        raise ValueError(message)
    if observations.columns.tolist().count(key) != 1:
        message = f"Legacy library observation column {key!r} is ambiguous; expected one column."
        raise ValueError(message)
    values = observations[key]
    observed = pd.unique(values)
    if values.isna().any() or any(not isinstance(value, str) or not value for value in observed):
        message = f"Legacy library column {key!r} requires non-null, nonempty string identifiers."
        raise ValueError(message)
    return values


def _library_images(dataset: Mapping[str, object], *, name: str | None) -> tuple[LegacyImage, ...]:
    images = _mapping(dataset.get("images", {}), location=f"Legacy library {name!r} images")
    scales = _mapping(dataset.get("scalefactors", {}), location=f"Legacy library {name!r} scalefactors")
    selections: list[LegacyImage] = []
    for key, value in images.items():
        _image(value, library=str(name), key=key)
        selections.append(LegacyImage(key, _scale(scales, library=str(name), image=key)))
    return tuple(selections)


def select_library(adata: AnnData, *, library_id: str | None, library_key: str | None) -> LegacyLibrary:
    """Validate one legacy library and explicitly associated observation rows."""
    _option(library_id, name="library_id")
    _option(library_key, name="library_key")
    libraries = _mapping(adata.uns.get("spatial", {}), location="uns['spatial']")
    memberships = _memberships(adata, library_key) if library_key is not None else None
    names = tuple(libraries)
    if memberships is not None:
        if not names:
            names = tuple(memberships.unique().tolist())
        if not memberships.isin(names).all():
            message = f"Legacy library column {library_key!r} contains unknown library identifiers; expected {names!r}."
            raise ValueError(message)
    if library_id is None:
        if len(names) > 1:
            message = f"Multiple legacy libraries {names!r}; specify library_id and library_key."
            raise ValueError(message)
        library_id = names[0] if names else None
    elif library_id not in names:
        message = f"Unknown legacy library {library_id!r}; available libraries are {names!r}."
        raise ValueError(message)
    if len(names) > 1 and memberships is None:
        message = "Selecting among multiple legacy libraries requires library_key observation associations."
        raise ValueError(message)
    rows = None
    if memberships is not None and library_id is not None:
        rows = np.flatnonzero(memberships.eq(library_id).to_numpy(dtype=bool))
        if rows.size == 0:
            message = f"Legacy library {library_id!r} has no associated observations."
            raise ValueError(message)
    dataset = _mapping(
        libraries.get(library_id, {}) if library_id is not None else {}, location=f"Library {library_id!r}"
    )
    return LegacyLibrary(library_id, rows, _library_images(dataset, name=library_id))


def read_images(
    adata: AnnData, library: LegacyLibrary, *, frame: str, rgb: bool | None
) -> dict[str, DataArray | DataTree]:
    """Wrap owned image pixels with their exact image-index-to-legacy mapping."""
    images: dict[str, DataArray | DataTree] = {}
    if not library.images:
        return images
    # select_library validated these dictionaries before the table ownership copy.
    dataset = cast("Mapping[str, Mapping[str, np.ndarray]]", adata.uns["spatial"][library.name])
    for image in library.images:
        values = dataset["images"][image.key]
        if values.ndim == _SPATIAL_DIMENSIONS:
            values = values[:, :, None]
        transformation = Sequence(
            [Translation([-0.5, -0.5], axes=("x", "y")), Scale([1.0 / image.scale] * 2, axes=("x", "y"))]
        )
        name = f"{library.name}_{image.key}"
        try:
            images[name] = Image2DModel.parse(
                values, dims=("y", "x", "c"), transformations={frame: transformation}, rgb=rgb
            )
        except (TypeError, ValueError) as error:
            error.add_note(f"while importing legacy library {library.name!r}, image {image.key!r}")
            raise
    return images
