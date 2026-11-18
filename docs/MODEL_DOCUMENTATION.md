# Model Documentation

> Segmentation and anomaly models used by CoastWatch. **Model weights are never
> committed to git** — `.gitignore` excludes `*.pt`, `*.pth`, `*.onnx`, `*.h5`,
> `*.keras`, `*.safetensors`, `*.tflite`, `*.engine`, and `ml/models/*`.

## 1. Where models live

Model artifacts are loaded from the path in the `MODEL_PATH` environment
variable (default `./ml/models`; in Docker the directory is mounted at
`/app/models`). The directory is a deploy-time input, not source code.

```
ml/models/
└── <model_name>/
    ├── model.onnx          # or model.pt for PyTorch
    ├── config.json         # input size, class map, preprocessing, version
    └── README.md           # provenance, training data, metrics
```

`config.json` must record at minimum: model architecture, input resolution,
class mapping (`water`, `land`, `foam`, `unknown`), normalization, the training
dataset identifier, and the evaluation metrics (IoU / Dice).

## 2. Segmentation (Phase 4)

- Architectures: **U-Net** and **DeepLabV3+**.
- A pretrained model is used as a starting point but is **not trusted** until
  validated on actual coastal footage. Generic segmentation models frequently
  confuse wet sand, foam and glare with water.
- Inference is served through **ONNX Runtime** for portability (CPU on servers,
  CUDA/TensorRT on Jetson in Phase 9).
- Output classes: `water`, `land`/`sand`, `foam` (optional), `unknown`/uncertain.

## 3. Anomaly models (Phase 6)

- Initial: **Z-score**, rolling statistical thresholds, **CUSUM** — these need
  no training and are fully explainable.
- Advanced: **Isolation Forest**, **One-Class SVM**, and **LSTM** once enough
  historical data exists.
- No ML model is enabled for detection until it has been validated
  chronologically (see Phase 8) and beats the statistical baseline.

## 4. Retrain-and-redeploy path

1. **Collect & label** coastal frames in `ml/datasets/` (never commit real data
   beyond tiny, clearly-labelled samples).
2. **Train** with a script in `ml/training/`; log metrics and dataset version.
3. **Evaluate** in `ml/evaluation/`: IoU, Dice, precision, recall, and shoreline
   localization error against a held-out, **chronologically** split set.
4. **Export** to ONNX (or TorchScript) and write `config.json`.
5. **Place** the artifact under `MODEL_PATH/<model_name>/` and set
   `MODEL_PATH` (or the mounted volume) so the backend loads it at startup.
6. **Record** provenance and metrics in that folder's `README.md`.

## 5. Evaluation metrics (Phase 8)

| Task | Metrics |
|---|---|
| Segmentation | IoU, Dice coefficient, precision, recall |
| Shoreline | Shoreline localization error (px and, when calibrated, metres) |
| Anomaly | Precision, recall, F1, false-positive/negative rate, detection latency |

Synthetic tests are always labelled as synthetic; they are reported separately
from any real-world validation.

## 6. Reproducibility & honesty

- Training/eval data and random seeds are recorded with each artifact.
- A model card (`README.md` next to the artifact) states limitations and the
  date/version of the data it was trained on.
- No model output is presented as a validated tsunami probability.
