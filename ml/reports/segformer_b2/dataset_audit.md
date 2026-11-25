# CoastWatch SegFormer Dataset Audit

Generated: `2026-11-24T04:17:40.344805+00:00`

## Verdict: **PASS_WITH_BLOCKED_DATASET**

Usable datasets: ['sam3']. Blocked datasets: ['sclabels']. Training may proceed only on the usable datasets listed in `training_dataset`.

### Dataset usage

| dataset | usable | images | masks | valid pairs | blocked reason |
|---|---:|---:|---:|---:|---|
| `sam3` | yes | 132 | 132 | 132 | — |
| `sclabels` | NO | 1717 | 0 | 0 | mask directory is empty or missing; 1717 images have no mask; no valid image/mask pairs |

## CoastWatch SAM3 coastal CCTV annotations (`sam3`)

- Root: `C:\Users\bitui\Documents\coastwatch\ml\datasets`
- Images: `C:\Users\bitui\Documents\coastwatch\ml\datasets\image_sam3`
- Masks: `C:\Users\bitui\Documents\coastwatch\ml\datasets\image_sam3_mask`
- Image formats: `{'JPEG': 67, 'PNG': 65}`
- Image modes: `{'RGB': 67, 'RGBA': 65}`
- Mask modes: `{'L': 132}`
- Image bit depths: `[8]`
- Mask bit depths: `[8]`
- Image sizes: 1280x720 (88), 1920x1080 (44)
- Alpha channels: `65` (non-opaque: `0`, range `255`..`255`)
- Raw mask values: `{'0': 78774223, '1': 63567835, '2': 29997142}`
- Mapped class counts: `{'Beach': 29997142, 'Ocean': 63567835, 'Sky': 78774223, 'ignore': 0}`
- Mapped class fractions: Beach=17.4059%, Ocean=36.8853%, Sky=45.7088%, ignore=0.0000%
- Mapping verified from polygon JSON: `True`
- Polygon records checked: `132`
- Inter-class polygon overlap: `{'beach+ocean': 6822058}` (the delivered raster resolves overlaps; this is an annotation ambiguity warning)
- Polygon union coverage of the image: median `78.7%`, min `41.8%`, mean `78.3%`; images below 80%: `75`
- Pixels outside every polygon: `39,824,457` with mask values `{'0': 39549301, '1': 143497, '2': 131659}` (unlabeled policy: `ignore_outside_polygons`)
- **Documentation conflict**: `CONFLICT: classes.json describes a different class schema/encoding than the delivered PNG rasters. The verified polygon labels and actual pixel values are authoritative; classes.json is stale and is not used for conversion.` (C:\Users\bitui\Documents\coastwatch\ml\datasets\image_sam3\labels\classes.json)
- Exact duplicate groups: `1`; near-duplicate groups: `12`

![sam3 pairs](sam3_pairs.png)

## SCLabels Spanish CoastSnap (`sclabels`)

- Root: `C:\Users\bitui\Documents\coastwatch\ml\datasets`
- Images: `C:\Users\bitui\Documents\coastwatch\ml\datasets\images`
- Masks: `C:\Users\bitui\Documents\coastwatch\ml\datasets\masks`
- Image formats: `{'PNG': 1717}`
- Image modes: `{'RGB': 1717}`
- Mask modes: `{}`
- Image bit depths: `[8]`
- Mask bit depths: `[]`
- Image sizes: 901x215 (27), 901x214 (27), 901x212 (27), 901x206 (25), 901x208 (25), 901x216 (24), … 383 more
- Alpha channels: `0` (non-opaque: `0`, range `None`..`None`)
- Raw mask values: `{}`
- Mapped class counts: `{'Beach': 0, 'Ocean': 0, 'Sky': 0, 'ignore': 0}`
- Mapped class fractions: 
- Mapping verified from polygon JSON: `None`
- Polygon records checked: `0`
- Exact duplicate groups: `0`; near-duplicate groups: `32`

![sclabels images (no masks)](sclabels_images.png)

## Verified class conversion (usable dataset)

| source value | source meaning | target id | target class |
|---:|---|---:|---|
| 0 | sky | 2 | Sky |
| 1 | ocean | 1 | Ocean |
| 2 | beach | 0 | Beach |
| 255 | ignore | 255 | ignore |

## Known limitations

- `sclabels` is blocked: mask directory is empty or missing; 1717 images have no mask; no valid image/mask pairs. It is excluded rather than labelled synthetically.
- Only 132 labeled pairs are usable. This is a small dataset; validation/test metrics will have wide confidence intervals.
- No session/camera metadata is available for the UUID/videoframe images. Leakage protection is limited to exact duplicates, perceptual near-duplicates, and keeping timestamp-named frames together.
- No foam/whitewater class exists in the source annotations; wave foam will be learned as Ocean or Beach, not as a separate class.
- Beach/Ocean polygon overlap exists in the JSON annotations. The delivered single-label PNG resolves it; no overlapping pixels remain in training masks.
- **Unlabeled background is encoded as 0 (Sky).** 39,824,457 pixels (50.2% of all source Sky pixels) lie outside every annotation polygon, i.e. the raster background defaults to the Sky id. This includes wet sand / swash-zone gaps between the Beach and Ocean polygons. With `unlabeled_policy: ignore_outside_polygons` these pixels are converted to ignore 255 rather than trained as Sky.
- Label policy in effect: `ignore_outside_polygons`. Pixels outside the verified polygon union are excluded from loss and metrics.
- Nighttime, low-light, cloud, sun-glare, wet-sand, vegetation, rock and tide coverage is not annotated as metadata. Condition breakdowns in evaluation are image-statistics heuristics, not ground-truth meteorological labels.

## Full records

Per-pair records (dimensions, modes, raw value counts, hashes) are in `dataset_audit.json`.
