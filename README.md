# CardioLens

Cardiomegaly detection from chest X-rays using a fine-tuned DenseNet-121, with Grad-CAM explainability, a FastAPI inference service, and a Streamlit demo.

**Live demo:** https://cardiolenskp65.streamlit.app/
**API (Swagger docs):** https://cardiolens-kp65.onrender.com/docs

> **Note:** the API runs on a free hosting tier and sleeps after inactivity. The first request after idle time can take 30+ seconds while it wakes up.

---

## Disclaimer

**This is an educational/portfolio project. It is not a diagnostic tool and has not been validated for clinical use.** Do not use it to make medical decisions. The model was trained on a public research dataset with noisy labels and has never been tested on data from a real clinical workflow.

---

## Overview

CardioLens classifies frontal chest X-rays as **Cardiomegaly** or **No Cardiomegaly**. The focus of the project is a correct evaluation methodology and a complete serving pipeline, not a state-of-the-art score:

- Patient-level train/validation/test split, with an explicit check that no patient appears in more than one split
- Model selection and decision-threshold selection on the validation set only; the test set is evaluated once with the frozen threshold
- Grad-CAM heatmaps showing which region of the image drove each prediction
- FastAPI service, Dockerized and deployed, with a Streamlit front end

## Results

Held-out test set, patient-level split, threshold frozen from validation:

| Metric | Value |
|---|---|
| AUROC | **0.8344** |
| AUPRC | 0.7977 |
| Accuracy | 0.7826 |
| Sensitivity (recall) | 0.8207 |
| Specificity | 0.7446 |
| Precision (PPV) | 0.7626 |
| NPV | 0.8059 |
| F1 | 0.7906 |
| Decision threshold | 0.678 |

Best validation AUROC was 0.8336, close to the test AUROC. That agreement is what you would expect once patient-level leakage is removed.

Plots: `assets/roc_curve.png`, `assets/confusion_matrix.png`. Per-image test predictions are saved to `assets/test_predictions.csv`.

### How to read these numbers

- **AUROC/AUPRC** measure ranking quality across all thresholds. **Accuracy, sensitivity, specificity, precision, NPV, and F1** depend on the chosen threshold.
- The test set is **class-balanced (50/50)**, so precision, NPV, and AUPRC reflect a 50% prevalence. Real-world prevalence of cardiomegaly is much lower, so precision would be lower in practice. AUROC, sensitivity, and specificity are not affected by prevalence.
- Published ChestX-ray14 work generally reports higher cardiomegaly AUROC (roughly 0.90 and above), typically using the full dataset and more training. This project used a subset and a lightweight fine-tuning setup, so a lower number is expected. Those results also use different splits and are not directly comparable.

## Methodology

### Data

- **Dataset:** NIH ChestX-ray14 (112,120 frontal X-rays, 14 labels). This project uses the first 6 of the 12 image archives (`images_001`–`images_006`) plus `Data_Entry_2017.csv`.
- **Task:** Cardiomegaly (1) vs. everything else (0). The negative class includes both "No Finding" and images with other diseases, so this is *Cardiomegaly vs. non-Cardiomegaly*, not *Cardiomegaly vs. healthy*.
- **Split:** 80/10/10 by **patient** (`GroupShuffleSplit` on `Patient ID`), so a patient's images all land in one split. The script asserts the three patient sets are disjoint.
- **Balancing:** after splitting, negatives are undersampled to match positives *within each split*. This changes the training distribution and, for the test set, the evaluation prevalence (see above). Using the full negative set with `pos_weight` is a planned follow-up experiment.

| Split | Images | Patients | Positives | Negatives |
|---|---|---|---|---|
| Train | _fill from `prepare_data.py` output_ | | | |
| Validation | | | | |
| Test | | | | |

### Model

- **Backbone:** DenseNet-121, ImageNet-pretrained (torchvision)
- **Head:** single-logit linear layer, `BCEWithLogitsLoss`
- **Fine-tuning:** early layers frozen; `denseblock4`, `norm5`, and the classifier head trained
- **Training:** Adam (lr 1e-4), batch size 16, 15 epochs, mixed precision, 224×224 input, random horizontal flip
- **Checkpointing:** best validation AUROC

### Threshold selection

The default 0.5 was not assumed. The threshold (0.678) was chosen on the **validation set** by maximizing Youden's J (sensitivity + specificity − 1), then frozen before the test set was scored.

### Explainability

Grad-CAM (`pytorch-grad-cam`) on the final DenseNet feature layer (`features.norm5`) produces a heatmap overlay for each prediction. On spot checks of known positive cases, the activation concentrates over the cardiac silhouette. Grad-CAM is used for interpretability only, not as a performance metric, and a plausible-looking heatmap does not prove the model is reasoning correctly.

## Architecture

```
Streamlit (Streamlit Community Cloud)
        │  POST /predict (image)
        ▼
FastAPI (Docker, Render)
        │
        ▼
inference.predict()  ──►  DenseNet-121  ──►  probability + label
        │
        └──►  Grad-CAM overlay
```

`src/inference.py` is the single code path for prediction and Grad-CAM. The model is loaded once at API startup.

## API

**`GET /health`**
```json
{"status": "ok"}
```

**`POST /predict`** — multipart image upload.

```bash
curl -X POST https://cardiolens-kp65.onrender.com/predict \
  -F "file=@chest_xray.png"
```

Response:
```json
{
  "probability": 0.9912,
  "label": "Cardiomegaly",
  "threshold": 0.678,
  "heatmap_base64": "<base64-encoded PNG>"
}
```

## Project structure

```
cardiolens/
├── api/                  # FastAPI app (main.py, schemas.py, utils.py)
├── src/                  # config, dataset, model, train, evaluate, gradcam, inference
├── streamlit_app/        # Streamlit front end
├── data/                 # download_data.py, prepare_data.py (raw data and splits are gitignored)
├── models/               # trained weights
├── assets/               # ROC curve, confusion matrix, test predictions, screenshots
├── Dockerfile
├── requirements.txt
└── keep_alive.py         # optional pinger for free-tier hosting
```

## Run locally

```bash
git clone https://github.com/AyanJaved/CardioLens.git
cd CardioLens
pip install -r requirements.txt
pip install streamlit requests
```

**Serve the API**
```bash
python -m uvicorn api.main:app --port 8000
```

**Run the demo** (in a second terminal)
```bash
streamlit run streamlit_app/app.py
```
Set `CARDIOLENS_API_URL` to point the demo at a different API (defaults to `http://localhost:8000`).

**Docker**
```bash
docker build -t cardiolens-api .
docker run -p 7860:7860 cardiolens-api
```

### Reproduce training

```bash
python data/download_data.py --out data/raw --n 6
# extract each images_00X.tar.gz into data/raw/, so images end up in data/raw/images/
python data/prepare_data.py
python src/train.py
python src/evaluate.py
```

Training requires a CUDA GPU for reasonable speed; a PyTorch build matching your CUDA version is needed.

## Limitations

- Trained on a subset of ChestX-ray14 (6 of 12 archives) with a small balanced training set
- ChestX-ray14 labels were extracted from radiology reports with NLP and contain noise
- Balanced test set: precision, NPV, and AUPRC are not representative of real-world prevalence
- No external validation on another hospital or dataset
- Single train/val/test split with one random seed; no confidence intervals
- Frontal views only; no handling of poor image quality or out-of-distribution inputs
- Probabilities are not calibrated
- Not clinically validated

## Possible next steps

- Full negative set with `pos_weight` instead of undersampling, evaluated on a prevalence-realistic test set
- Bootstrap confidence intervals and multiple seeds
- Unfreeze more layers, learning-rate scheduling, higher resolution
- External validation on CheXpert or another dataset
- Probability calibration

## Acknowledgments and citation

Data: NIH Clinical Center, ChestX-ray14 — https://nihcc.app.box.com/v/ChestXray-NIHCC

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* CVPR 2017.

Built with PyTorch, torchvision, pytorch-grad-cam, FastAPI, Streamlit, and Docker.