# Split policy

- Images: 131; groups: 79
- Target: train 70% / val 15% / test 15%
- Actual: train 91 / val 20 / test 20
- Group integrity violations: 0

Whole groups are assigned to one split. Exact duplicates are
excluded in `prepare_dataset.py`; perceptual near-duplicates are
clustered; timestamp-named frames stay together. No UUID/videoframe
camera or session metadata exists, so cross-camera generalisation
cannot be proven from this split alone.

The test split is never used for checkpoint selection.
