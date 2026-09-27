import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardiolens_densenet121.pt")

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# frozen threshold selected on validation set via Youden's J (see evaluate.py)
THRESHOLD = 0.678