import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DEVICE, MODEL_PATH, NUM_WORKERS, PROCESSED_DIR
from dataset import CardiomegalyDataset, EVAL_TRANSFORM
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def get_predictions(model, loader):
    probs, targets = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images).squeeze(1)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(labels.numpy())
    return np.array(targets), np.array(probs)


def best_threshold_youden(targets, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs)
    return thresholds[np.argmax(tpr - fpr)]


def compute_metrics(targets, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    def safe_div(a, b):
        return a / b if b else 0.0

    metrics = {
        "auroc": roc_auc_score(targets, probs),
        "auprc": average_precision_score(targets, probs),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(targets),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
    }
    metrics["f1"] = safe_div(2 * metrics["precision"] * metrics["sensitivity"],
                              metrics["precision"] + metrics["sensitivity"])
    return metrics, preds, (tn, fp, fn, tp)


def main():
    val_ds = CardiomegalyDataset(os.path.join(PROCESSED_DIR, "val.csv"), EVAL_TRANSFORM)
    test_ds = CardiomegalyDataset(os.path.join(PROCESSED_DIR, "test.csv"), EVAL_TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # threshold selected on val ONLY, then frozen before touching test
    val_targets, val_probs = get_predictions(model, val_loader)
    threshold = best_threshold_youden(val_targets, val_probs)

    test_targets, test_probs = get_predictions(model, test_loader)
    metrics, test_preds, (tn, fp, fn, tp) = compute_metrics(test_targets, test_probs, threshold)

    print("Test Results")
    print("-------------------------")
    for key in ["auroc", "auprc", "threshold", "accuracy", "sensitivity", "specificity", "precision", "npv", "f1"]:
        print(f"{key:12s}: {metrics[key]:.4f}")

    os.makedirs(ASSETS_DIR, exist_ok=True)

    with open(os.path.join(ASSETS_DIR, "test_results.txt"), "w") as f:
        for key, val in metrics.items():
            f.write(f"{key}: {val:.4f}\n")

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    test_df["probability"] = test_probs
    test_df["prediction"] = test_preds
    test_df = test_df.rename(columns={"label": "true_label"})
    test_df.to_csv(os.path.join(ASSETS_DIR, "test_predictions.csv"), index=False)

    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, "roc_curve.png"))

    plt.figure()
    plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    plt.xticks([0, 1], ["Pred Negative", "Pred Positive"])
    plt.yticks([0, 1], ["Actual Negative", "Actual Positive"])
    for i, row in enumerate([[tn, fp], [fn, tp]]):
        for j, val in enumerate(row):
            plt.text(j, i, str(val), ha="center", va="center")
    plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()