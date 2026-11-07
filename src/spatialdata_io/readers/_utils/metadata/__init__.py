"""Strict structured-metadata utilities for private reader layouts.

Strict JSON objects from files or embedded text, and strict UTF-8 text from
HDF5 attribute values.
"""

from __future__ import annotations

from spatialdata_io.readers._utils.metadata._hdf5 import decode_hdf5_text
from spatialdata_io.readers._utils.metadata._json import parse_json_object, read_json_object

__all__ = ["decode_hdf5_text", "parse_json_object", "read_json_object"]
