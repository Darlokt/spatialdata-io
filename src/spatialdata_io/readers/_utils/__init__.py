"""Private shared foundations for vendor readers.

The subpackages hold the count-container, strict CSV, GeoJSON geometry, lazy raster,
bounded JSON and HDF5-text metadata, local path, provenance and sparse-table boundaries.
``errors`` holds the contextual exceptions they share and ``warn_user``, which
attributes a reader's ``UserWarning`` to the user's call site. ``names`` validates
SpatialData element names and repairs element names, table column names and
top-level ``uns`` keys derived from files. Readers keep all vendor semantics.
"""
