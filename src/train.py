import os

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DEVICE, EPOCHS, LR, MODEL_PATH, NUM_WORKERS, PROCESSED_DIR
from dataset import CardiomegalyDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM
from model import build_model


def evaluate(model, loader):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images).squeeze(1)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(labels.numpy())
    return roc_auc_score(targets, probs)


def main():
    train_ds = CardiomegalyDataset(os.path.join(PROCESSED_DIR, "train.csv"), TRAIN_TRANSFORM)
    val_ds = CardiomegalyDataset(os.path.join(PROCESSED_DIR, "val.csv"), EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = GradScaler()

    best_auc = 0.0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_auc = evaluate(model, val_loader)
        print(f"epoch {epoch+1}/{EPOCHS} - loss {train_loss:.4f} - val_auc {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"best val AUROC: {best_auc:.4f}")


if __name__ == "__main__":
    main()