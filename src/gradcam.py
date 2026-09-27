import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import DEVICE, IMAGE_SIZE
from dataset import EVAL_TRANSFORM


def generate_gradcam(model, pil_image: Image.Image) -> Image.Image:
    """Returns a PIL image: original X-ray with a Grad-CAM heatmap overlay."""
    model.eval()
    target_layer = model.features.norm5

    pil_image = pil_image.convert("RGB")
    input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    rgb_img = np.array(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return Image.fromarray(overlay)