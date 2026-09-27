import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from inference import load_model, predict  # noqa: E402
from .schemas import HealthResponse, PredictResponse
from .utils import decode_upload, encode_image_base64

app = FastAPI(title="CardioLens API")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image = decode_upload(await file.read())
    result = predict(image)

    return {
        "probability": result["probability"],
        "label": result["label"],
        "threshold": result["threshold"],
        "heatmap_base64": encode_image_base64(result["heatmap"]),
    }