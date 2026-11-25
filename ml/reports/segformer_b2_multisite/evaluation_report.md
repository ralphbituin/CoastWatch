# CoastWatch SegFormer Evaluation

Generated: `2026-11-25T06:50:46.041845+00:00`

- Checkpoint: `C:\Users\bitui\Documents\coastwatch\ml\checkpoints\segformer_b2_multisite\best.pt`
- Split: `test` (100 images)
- Metric resolution: native image resolution after logit upsampling

## Primary metrics

Beach IoU and shoreline MAE are the primary quality signals for the application. Mean IoU is retained for continuity but is dominated by the large, easy Ocean and Sky classes.

| metric | value |
|---|---:|
| Beach IoU | 0.8562 |
| Shoreline MAE, mean | 45.0904 px |
| Shoreline MAE, median | 7.6568 px |
| Shoreline MAE, p95 | 270.9244 px |
| mean IoU (dominated by easy classes) | 0.8678 |

## Overall test metrics

| class | IoU | Dice | precision | recall | support px |
|---|---:|---:|---:|---:|---:|
| Beach | 0.8562 | 0.9225 | 0.8697 | 0.9822 | 27,485,659 |
| Ocean | 0.8091 | 0.8945 | 0.9384 | 0.8545 | 35,125,611 |
| Sky | 0.9380 | 0.9680 | 0.9729 | 0.9631 | 41,111,677 |
| **mean (dominated by easy classes)** | **0.8678** | **0.9283** | | | **103,722,947** px |

Pixel accuracy: **0.9314**

## Confusion matrix

![confusion matrix](confusion_matrix.png)

## Condition coverage (image-statistics heuristics)

These tags are **not** ground-truth weather labels. They are derived from luminance, glare and saturation. A condition with zero images is not represented and must not be claimed as supported.

| tag | images | mean IoU | pixel accuracy |
|---|---:|---:|---:|
| night | 0 | n/a | n/a |
| low_light | 26 | 0.7111 | 0.8378 |
| normal_light | 71 | 0.9036 | 0.9536 |
| bright_sunlight | 3 | 0.9965 | 0.9991 |
| strong_glare | 23 | 0.7754 | 0.8989 |
| overcast_or_low_saturation | 25 | 0.9031 | 0.9498 |
| high_saturation | 0 | n/a | n/a |
| small_beach_visible_fraction | 7 | 0.9888 | 0.9985 |
| ocean_dominant | 8 | 0.9753 | 0.9912 |

Unrepresented condition tags: `night`, `high_saturation`.

## Shoreline extraction

- Predicted shoreline available: 100/100 test images
- GT shoreline available: 91/100
- Mean per-column MAE: 45.0904 px; median: 7.6568 px; p95: 270.9244 px
- Per-column MAE compares only image columns where both the ground-truth and predicted polylines exist. Some frames contain a far shore as well as the near waterline, so the single-valued polyline can switch branches; treat this as an image-space diagnostic, not a physical accuracy claim.
- These are image-pixel distances only. No tide, bathymetry, or metric world-coordinate claim is made.

## Prediction visualisations

![prediction grid](prediction_grid.png)

Per-image four-panel outputs are in `predictions/`.

## Honest limitations

- Training data: 371 labelled images in the training split (the baseline model used 131). Training masks are automatic SAM3 annotations, so metrics measure agreement with those annotations rather than human ground truth.
- Dataset B (SCLabels, 1,717 images) has no masks and is absent from training.
- The test split contains 100 images; metrics have uncertainty and are not a production or cross-camera accuracy claim.
- Nighttime is supported only if the test split actually contains night images. Check the condition coverage table; do not infer nighttime capability from a daytime-only evaluation.
- Condition tags are luminance/glare/saturation heuristics, not meteorological or tide ground truth. Waves, sea foam, wet sand, vegetation, rocks and tide are not separately annotated or labelled.
- Shoreline error is in image pixels. No camera calibration or world-coordinate conversion was applied, so no metre/tide claim is made.
- Source annotations are SAM3-generated polygons with documented Beach/Ocean overlap and visible coarse boundaries; some annotation noise is inherited.
- The annotation polygons leave the wet-sand swash zone unlabelled in many frames. Those pixels are ignore 255, not Sky. In 6/20 test images the remaining Beach and Ocean polygons do not touch, so no ground-truth shoreline can be extracted and that image is excluded from shoreline error.
- Some frames contain a far shore in addition to the near waterline. The single-valued per-column shoreline can switch branches, so shoreline MAE is an image-space diagnostic reported over shared columns, not a physical accuracy claim.

## Per-image worst cases (by mean IoU)

| image | mean IoU | Beach IoU | Ocean IoU | Sky IoU | tags |
|---|---:|---:|---:|---:|---|
| `site_japan_shirahama_2026-11-25_04-32-00.png` | 0.3052 | 0.2977 | 0.0014 | 0.6165 | low_light |
| `site_japan_shirahama_2026-11-25_04-22-00.png` | 0.3483 | 0.3679 | 0.0388 | 0.6381 | low_light |
| `site_japan_shirahama_2026-11-25_03-32-00.png` | 0.4608 | 0.3995 | 0.0466 | 0.9364 | normal_light, strong_glare |
| `site_usa_turquoise_beach_alabama_2026-11-24_18-04-00.png` | 0.4618 | 0.9672 | 0.0325 | 0.3857 | low_light, overcast_or_low_saturation |
| `site_japan_shirahama_2026-11-25_03-52-00.png` | 0.4645 | 0.3540 | 0.1235 | 0.9158 | normal_light, strong_glare |
| `site_japan_shirahama_2026-11-25_04-02-00.png` | 0.4693 | 0.4005 | 0.1142 | 0.8933 | normal_light, strong_glare |
| `site_japan_shirahama_2026-11-25_03-22-00.png` | 0.5269 | 0.5024 | 0.1624 | 0.9160 | normal_light, strong_glare |
| `site_usa_turquoise_beach_alabama_2026-11-24_18-24-00.png` | 0.5642 | 0.9981 | 0.2513 | 0.4433 | low_light, overcast_or_low_saturation |
