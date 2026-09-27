# CardioLens — Build Plan

Cardiomegaly detection from chest X-rays: fine-tuned model, served via FastAPI, with a Streamlit demo frontend and Grad-CAM explainability.

## Project Structure

```
cardiolens/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/                       # NIH ChestX-ray14 zip archives + Data_Entry_2017.csv
│   ├── processed/                 # binary-labeled train/val/test CSVs
│   ├── download_data.py           # downloads images_001-003.tar.gz + labels CSV
│   └── prepare_data.py            # relabeling + balanced subset + stratified split
│
├── notebooks/
│   └── 01_train_colab.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── inference.py
│
├── models/
│   └── cardiolens_densenet121.pt
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── utils.py
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│   ├── test_api.py
│   └── test_inference.py
│
└── assets/
    └── demo_screenshots/
```

## Step 1 — Dataset

- Source: official NIH ChestX-ray14 zip archives (`images_001.tar.gz` – `images_003.tar.gz`) + `Data_Entry_2017.csv`, downloaded via `data/download_data.py`.
- Binary target: 1 if "Cardiomegaly" appears in an image's `Finding Labels`, else 0 (includes "No Finding" images and other-disease-but-not-cardiomegaly images as negatives).
- Cardiomegaly is a minority label. From the extracted 3 archives, take all available cardiomegaly-positive images and undersample negatives to match, building a balanced subset.
- Split into `data/processed/{train,val,test}` with an 80/10/10 stratified split.
- `data/prepare_data.py` handles relabeling, balancing, and the split — run after extracting the zips.

## Step 2 — Fine-tune the model

- Backbone: DenseNet-121 pretrained on ImageNet, fine-tuned on the CardioLens dataset.
- Freeze early layers, fine-tune later blocks + classifier head to save VRAM and training time.
- Mixed precision, small batch size (8–16) for a laptop/Colab GPU.
- Weighted loss if the final split isn't perfectly balanced.
- Save weights to `models/cardiolens_densenet121.pt`.

## Step 3 — Evaluate

- Primary metric: AUROC.
- Also report sensitivity and specificity at a chosen threshold.
- Save a confusion matrix and ROC curve plot for the README.

## Step 4 — Grad-CAM explainability

- Heatmap overlay showing which region of the X-ray drove each prediction.
- Reusable function in `src/gradcam.py` (image + model → overlay image), shared by API and Streamlit.

## Step 5 — FastAPI backend

- `POST /predict` — accepts an uploaded X-ray, returns probability, binary label, Grad-CAM heatmap.
- `GET /health`.
- Model loads once at startup.

## Step 6 — Streamlit frontend

- Upload an X-ray → "Analyze" → probability, label, Grad-CAM heatmap.
- Visible disclaimer: educational project, not a diagnostic tool, not validated for clinical use.
- Points at the FastAPI `/predict` endpoint (local during dev, deployed URL after Step 7).

## Step 7 — Containerize and deploy

- Dockerfile for the FastAPI app.
- Deploy API to Hugging Face Spaces (Docker SDK) or Render free tier.
- Deploy Streamlit frontend as a second Space or alongside the API, pointed at the deployed API URL.

## Step 8 — README

- Dataset source + NIH attribution + citation, model architecture, training setup, evaluation numbers, API usage example, Grad-CAM screenshots, disclaimer.