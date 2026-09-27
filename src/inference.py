import torch
from PIL import Image

from config import DEVICE, MODEL_PATH, THRESHOLD
from dataset import EVAL_TRANSFORM
from gradcam import generate_gradcam
from model import build_model

_model = None


def load_model():
    global _model
    if _model is None:
        _model = build_model().to(DEVICE)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.eval()
    return _model


def predict(pil_image: Image.Image) -> dict:
    model = load_model()
    image = pil_image.convert("RGB")

    input_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor).squeeze()).item()

    label = "Cardiomegaly" if prob >= THRESHOLD else "No Cardiomegaly"
    heatmap = generate_gradcam(model, image)

    return {
        "probability": prob,
        "label": label,
        "threshold": THRESHOLD,
        "heatmap": heatmap,
    }