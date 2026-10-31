# Coordinate systems and calibration

Readers retain source pixels, vertices, centroids and imported table embeddings.
A transformation attached to a spatial element expresses its coordinates in the
named frame. Canonical table `obsm["spatial"]`, when present, contains finite
float64 XY micrometres with a position/frame declaration in
`uns["spatialdata_io_positions"]`. Native values remain in source fields or
explicit `spatial_pixels`, `spatial_grid` and `spatial_source` embeddings.
Morphology annotations keep their documented source units; calibrating positions
does not rescale area, axis lengths or orientation.

One registered calibrated group normally uses `global` in micrometres. Multiple
independent samples, ROIs, cores or stacks retain distinct frames; names do not
establish registration. A split MACSima nuclei image shares its parent stack's
frame. No universal stage origin or half-pixel correction is inferred.

| Reader | Calibration and fallback |
| --- | --- |
| Generic | Complete selected base-series OME spatial sizes calibrate retained XY or XYZ axes. A volume missing Z spacing, an explicitly reinterpreted spatial axis, or a selected nonzero stored level retains source pixels. GeoJSON coordinates have source-defined units and require explicit registration when appended to an existing store. |
| DBiT | Barcode-grid coordinates retain their documented grid units; an image's OME spacing alone cannot establish barcode pitch or orientation. |
| seqFISH | Verified source micrometre coordinates; one selected ROI uses `global`, independent ROIs keep their own frames. |
| Stereo-seq | GEF grid pitch divided by 1000 maps the common source grid to micrometres. Native bins and original origins keep source values. Tables retain exact `spatial_grid` and expose physical `spatial`; source bin reference positions are not shifted to footprint centres. Image-only reads use an exact-sample GEF companion when available, otherwise source pixels. |
| MERSCOPE | Native vectors are micrometres. The inverse vendor micron-to-mosaic affine calibrates images without changing vector coordinates. Missing image calibration keeps images and native vectors in separate frames when both are returned. |
| MACSima | Validated isotropic OME spacing always calibrates the stack. One parent stack uses `global`; independent stacks retain their names. The transformation switch has been removed. |
| MCMICRO | Complete base OME XY spacing calibrates each local image and same-grid segmentations. Independent map/core/tissue-mask grids remain separate. Table regionprops-index centres map as `(source_xy + 0.5) * spacing_xy_um`, with exact centroids retained in `spatial_pixels`. Several segmentations of one image share its frame. Mixed-calibration aggregate tables omit canonical `spatial`; no core crop placement is reconstructed. ASHLAR may write fallback metadata, so declared spacing is not independent evidence of acquisition accuracy. |
| ISS | The caller must supply raw image and labels registered on one base grid. Equal extents alone do not prove alignment. Complete raw-image OME spacing calibrates the pair; labels may inherit missing spacing, but conflicting declarations fail. Unknown imported H5AD `spatial` moves exactly to `spatial_source`. Calibration of the raster pair alone does not establish table units or origin. |
| Steinbock | Optional IMC `images.csv` acquisition extents and emitted dimensions calibrate exact matching images and masks. Complete OME spacing is also supported, with agreement required. Complete per-sample calibration gives physical table positions `(source_xy + 0.5) * spacing_xy_um`; exact native centroids remain in `spatial_pixels` and source fields. Samples remain independent. Mixed-calibration aggregate tables omit canonical `spatial`. |
| Classic Visium | Full-resolution image pixels remain the reference. `spot_diameter_fullres` is a display diameter, not a physical ruler; it does not establish a conversion from the nominal capture diameter. Arbitrary selected full-resolution images are not verified original microscope acquisitions. Hires/lowres factors register those image pixels only. |
| CODEX | Source `obs` XY and radius-one display circles use `table_pixels` without calibration. `pixel_size_um` calibrates the established table frame; explicit `experiment_file` supports caller-associated Akoya 1.5.0.38/1.7.0.6 `xyResolution` in nm/pixel. Simultaneous declarations must agree. A selected image keeps `image_pixels` unless explicit `image_to_table` supplies its XY affine; physical scale follows that mapping. No automatic image association or universal montage factor is inferred. |
| CosMx | Native local `spatial_pixels` and global `global` retain exact pixels. Optional `pixel_size_um` calibrates the validated global pixel frame consistently for tables, rasters and transcripts, after existing signed placement. Stage-only millimetres do not establish a stage-to-pixel mapping. Current paired/optical source profiles need separate association and mapping evidence; no universal scale or fitted correction is substituted. |
| Curio | Supported Seeker primary exports use manufacturer-declared approximately 0.647 µm/pixel. Preserve native XY in `spatial_pixels` and stored orientation; canonical `spatial` and display circles use this approximate scale. This does not establish a coordinate contract for arbitrary reprocessed/merged or Trekker products. |

Optional OME calibration supports µm, nm and mm and preserves anisotropy. All
retained spatial axes must have finite positive representable sizes; incomplete
calibration leaves the entire grid in source units. Present malformed values or
unsupported units fail. Linked-grid declarations must agree, including partial
ones. Calibration applies at level zero and retains the native raster origin;
OME plane stage positions are not used to invent registration.

Steinbock discovers only root `images.csv`, parsed from an owned CSV stream.
Its named fields are `image`, `width_px`, `height_px`, and optional paired
`acquisition_width_um`, `acquisition_height_um`. Fields may be reordered and
additional fields are allowed. The exact association is `<sample>.tiff` to
`ome/<sample>.ome.tiff` and the selected mask. Duplicate names, malformed CSV,
partial physical pairs, nonpositive values, unrepresentable scales, stale
raster dimensions and conflicting OME declarations fail. Missing sample rows,
absent physical columns or two empty extent fields provide no IMC calibration.
The scale is acquisition width/width in pixels for X and height/height for Y;
there is no hardcoded one-micrometre pixel size or stage-coordinate placement.

## Table positions and independent frames

One calibrated sample normally uses `global`. Independent calibrated samples,
cores or ROIs keep distinct frames, even when their numerical origins coincide.
Varying table frames are categorical string observation columns referenced by
`frame_key` in the position declaration. Linkage regions name targets; they are
not automatically separate physical frames. A table containing both calibrated
and uncalibrated samples preserves every row and its aggregate key, retains
native coordinates, and omits canonical physical `spatial` entirely. An unrelated
image without table rows does not decide that table's calibration.

The half-pixel offset above belongs specifically to the producer's zero-based
regionprops centroid convention in Steinbock and MCMICRO. Their native position
descriptions identify pixel-index frames separately from the unchanged raster
corner-coordinate frame. Do not apply that offset to vendor cell centres, chip
reference positions or arbitrary imported embeddings.

Missing calibration is a native-only output. Present malformed or conflicting
supported declarations fail. Nominal bead/spot diameter, image dimensions,
nearest-neighbour spacing and matching bounds do not establish physical units.
Declared source calibration is used as supplied; it is not an independent
measurement of instrument accuracy. CODEX's explicit experiment selection is a
caller assertion of table association, rather than automatic recognition of a
complete processor export or registration of its display image.

Table coordinates are snapshots. Later registration of spatial elements does
not refresh these arrays. Re-export positions explicitly after registration.
Ordinary AnnData concatenation does not reconcile coordinate declarations;
separate samples with overlapping local origins must remain partitioned for
physical analysis. See [table contracts](table_contracts.md) for native keys,
position meanings and persistence limitations.

## Legacy AnnData boundaries

Plain legacy AnnData `obsm["spatial"]` uses full-resolution image-pixel indices,
not micrometres. `from_legacy_anndata()` preserves it as `spatial_pixels` and
creates linked Points without inferred spot radii. Legacy resolution factors
register stored image levels to that pixel frame; they are not a physical
calibration. Declared physical positions are supported without legacy images;
physical positions with legacy images require a mapping those scalar factors
cannot establish and fail.

`to_legacy_anndata()` preserves stored positions by default. Explicit centroid
mode applies linked-element transformations in one selected frame and declares
the resulting position definition and units. A frame name alone cannot establish
micrometres. Native centroids stay in `spatial_centroids`, and original physical
positions replaced by physical centroids remain in `spatial_original`.
Label centroids can require computing the complete raster. No images are
exported or resampled. See [AnnData access and legacy conversion](legacy_anndata.md)
for explicit selection, ownership and source-snapshot limitations.
