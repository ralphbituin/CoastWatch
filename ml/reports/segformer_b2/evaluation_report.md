# CoastWatch SegFormer Evaluation

Generated: `2026-11-24T05:18:18.227897+00:00`

- Checkpoint: `C:\Users\bitui\Documents\coastwatch\ml\checkpoints\segformer_b2\best.pt`
- Split: `test` (20 images)
- Metric resolution: native image resolution after logit upsampling

## Primary metrics

Beach IoU and shoreline MAE are the primary quality signals for the application. Mean IoU is retained for continuity but is dominated by the large, easy Ocean and Sky classes.

| metric | value |
|---|---:|
| Beach IoU | 0.9580 |
| Shoreline MAE, mean | 47.4253 px |
| Shoreline MAE, median | 14.0329 px |
| Shoreline MAE, p95 | 157.3949 px |
| mean IoU (dominated by easy classes) | 0.9778 |

## Overall test metrics

| class | IoU | Dice | precision | recall | support px |
|---|---:|---:|---:|---:|---:|
| Beach | 0.9580 | 0.9785 | 0.9647 | 0.9928 | 4,819,362 |
| Ocean | 0.9777 | 0.9887 | 0.9965 | 0.9811 | 10,052,985 |
| Sky | 0.9976 | 0.9988 | 0.9977 | 1.0000 | 6,371,301 |
| **mean (dominated by easy classes)** | **0.9778** | **0.9887** | | | **21,243,648** px |

Pixel accuracy: **0.9894**

## Confusion matrix

![confusion matrix](confusion_matrix.png)

## Condition coverage (image-statistics heuristics)

These tags are **not** ground-truth weather labels. They are derived from luminance, glare and saturation. A condition with zero images is not represented and must not be claimed as supported.

| tag | images | mean IoU | pixel accuracy |
|---|---:|---:|---:|
| night | 0 | n/a | n/a |
| low_light | 0 | n/a | n/a |
| normal_light | 17 | 0.9753 | 0.9883 |
| bright_sunlight | 3 | 0.9926 | 0.9982 |
| strong_glare | 3 | 0.9926 | 0.9982 |
| overcast_or_low_saturation | 3 | 0.9929 | 0.9982 |
| high_saturation | 0 | n/a | n/a |
| small_beach_visible_fraction | 4 | 0.9916 | 0.9984 |
| ocean_dominant | 8 | 0.9675 | 0.9880 |

Unrepresented condition tags: `night`, `low_light`, `high_saturation`.

## Shoreline extraction

- Predicted shoreline available: 20/20 test images
- GT shoreline available: 14/20
- Mean per-column MAE: 47.4253 px; median: 14.0329 px; p95: 157.3949 px
- Per-column MAE compares only image columns where both the ground-truth and predicted polylines exist. Some frames contain a far shore as well as the near waterline, so the single-valued polyline can switch branches; treat this as an image-space diagnostic, not a physical accuracy claim.
- These are image-pixel distances only. No tide, bathymetry, or metric world-coordinate claim is made.

## Prediction visualisations

![prediction grid](prediction_grid.png)

Per-image four-panel outputs are in `predictions/`.

## Honest limitations

- Dataset B (SCLabels, 1,717 images) has no masks and is absent from training; only the 131-pair SAM3 dataset was trainable.
- The test split is small (20 images), so all metrics have wide uncertainty.
- Nighttime is supported only if the test split actually contains night images. Check the condition coverage table; do not infer nighttime capability from a daytime-only evaluation.
- Condition tags are luminance/glare/saturation heuristics, not meteorological or tide ground truth. Waves, sea foam, wet sand, vegetation, rocks and tide are not separately annotated or labelled.
- Shoreline error is in image pixels. No camera calibration or world-coordinate conversion was applied, so no metre/tide claim is made.
- Source annotations are SAM3-generated polygons with documented Beach/Ocean overlap and visible coarse boundaries; some annotation noise is inherited.
- The annotation polygons leave the wet-sand swash zone unlabelled in many frames. Those pixels are ignore 255, not Sky. In 6/20 test images the remaining Beach and Ocean polygons do not touch, so no ground-truth shoreline can be extracted and that image is excluded from shoreline error.
- Some frames contain a far shore in addition to the near waterline. The single-valued per-column shoreline can switch branches, so shoreline MAE is an image-space diagnostic reported over shared columns, not a physical accuracy claim.

## Per-image worst cases (by mean IoU)

| image | mean IoU | Beach IoU | Ocean IoU | Sky IoU | tags |
|---|---:|---:|---:|---:|---|
| `sam3_7312152a-56a9-4bde-867d-a30277e20ccd.jpg` | 0.8838 | 0.6821 | 0.9705 | 0.9987 | normal_light, ocean_dominant |
| `sam3_5ae846f5-a4c2-4f8a-b4c9-00dbb465a90d.jpg` | 0.9245 | 0.8696 | 0.9039 | 1.0000 | normal_light |
| `sam3_c032c6c4-0ee8-43f5-9f00-103cf2f91291.jpg` | 0.9351 | 0.8849 | 0.9205 | 1.0000 | normal_light, ocean_dominant |
| `sam3_d8cdec16-8fc1-4844-8291-b6fd0936aa17.jpg` | 0.9594 | 0.8797 | 0.9987 | 0.9998 | normal_light, overcast_or_low_saturation, small_beach_visible_fraction, ocean_dominant |
| `sam3_e9855b95-c632-4651-a700-16b28bc0514e.jpg` | 0.9624 | 0.9448 | 0.9423 | 1.0000 | normal_light |
| `sam3_videoframe_2320477.png` | 0.9760 | 0.9954 | 0.9399 | 0.9925 | normal_light |
| `sam3_videoframe_2117435.png` | 0.9800 | 0.9909 | 0.9533 | 0.9959 | normal_light |
| `sam3_90814531-60dc-4498-accd-9f73ef02ad26.jpg` | 0.9823 | 0.9513 | 0.9955 | 1.0000 | normal_light, ocean_dominant |
