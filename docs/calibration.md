# Coordinate systems and calibration

Readers retain source pixels, vertices, centroids and imported table embeddings.
A transformation attached to a spatial element expresses its coordinates in the
named frame. Table embeddings and morphology annotations retain their documented
source units even when the linked image, labels or shapes have a physical
transformation.

One registered calibrated group normally uses `global` in micrometres. Multiple
independent samples, ROIs, cores or stacks retain distinct frames; names do not
establish registration. A split MACSima nuclei image shares its parent stack's
frame. No universal stage origin or half-pixel correction is inferred.

| Reader | Calibration and fallback |
| --- | --- |
| Generic | Complete selected base-series OME spatial sizes calibrate retained XY or XYZ axes. A volume missing Z spacing, an explicitly reinterpreted spatial axis, or a selected nonzero stored level retains source pixels. GeoJSON coordinates have source-defined units and require explicit registration when appended to an existing store. |
| DBiT | Barcode-grid coordinates retain their documented grid units; an image's OME spacing alone cannot establish barcode pitch or orientation. |
| seqFISH | Verified source micrometre coordinates; one selected ROI uses `global`, independent ROIs keep their own frames. |
| Stereo-seq | GEF grid pitch divided by 1000 maps the common source grid to micrometres. Bins, original origins and table embeddings keep source values. Image-only reads use an exact-sample GEF companion when available, otherwise source pixels. |
| MERSCOPE | Native vectors are micrometres. The inverse vendor micron-to-mosaic affine calibrates images without changing vector coordinates. Missing image calibration keeps images and native vectors in separate frames when both are returned. |
| MACSima | Validated isotropic OME spacing always calibrates the stack. One parent stack uses `global`; independent stacks retain their names. The transformation switch has been removed. |
| MCMICRO | Complete base OME XY spacing calibrates each local image and same-grid segmentations. Independent map/core/tissue-mask grids remain separate. Missing calibration retains pixels; no core crop placement is reconstructed. ASHLAR may write fallback metadata, so declared spacing is not independent evidence of acquisition accuracy. |
| ISS | The caller must supply raw image and labels registered on one base grid. Equal extents alone do not prove alignment. Complete raw-image OME spacing calibrates the pair; labels may inherit missing spacing, but conflicting declarations fail. Imported H5AD embeddings remain unchanged. |
| Steinbock | Optional IMC `images.csv` acquisition extents and emitted dimensions calibrate exact matching images and masks. Complete OME spacing is also supported, with agreement required. Missing calibration retains pixels; samples remain independent. |
| Classic Visium | Full-resolution image pixels remain the reference. `spot_diameter_fullres` is a display diameter, not a physical ruler; it does not establish a conversion from the nominal capture diameter. Arbitrary selected full-resolution images are not verified original microscope acquisitions. Hires/lowres factors register those image pixels only. |
| CODEX | Supported tables and display circles retain full-resolution source pixels. A Nolan `Experiment.json` can record acquisition spacing in nm, but the supported stripped region layout does not establish its table association or the independently selected montage's reduction/origin. No sidecar calibration or universal montage factor is inferred. |
| CosMx | Local and global coordinates remain pixels. Millimetre-only stage positions do not establish a stage-to-pixel mapping; no universal pixel size or fit from a single FOV is substituted. |
| Curio | Source-defined coordinate units are retained. They are not asserted to be pixels or micrometres without calibration evidence. |

Optional OME calibration supports µm, nm and mm and preserves anisotropy. All
retained spatial axes must have finite positive representable sizes; incomplete
calibration leaves the entire grid in source units. Present malformed values or
unsupported units fail. Linked-grid declarations must agree, including partial
ones. Calibration applies at level zero and retains the native raster origin;
OME plane stage positions are not used to invent registration.

Steinbock discovers only root `images.csv`, with a maximum input size of 8 MiB.
Its named fields are `image`, `width_px`, `height_px`, and optional paired
`acquisition_width_um`, `acquisition_height_um`. Fields may be reordered and
additional fields are allowed. The exact association is `<sample>.tiff` to
`ome/<sample>.ome.tiff` and the selected mask. Duplicate names, malformed CSV,
partial physical pairs, nonpositive values, unrepresentable scales, stale
raster dimensions and conflicting OME declarations fail. Missing sample rows,
absent physical columns or two empty extent fields provide no IMC calibration.
The scale is acquisition width/width in pixels for X and height/height for Y;
there is no hardcoded one-micrometre pixel size or stage-coordinate placement.
