import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_model():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    # freeze everything, then unfreeze the last dense block + final norm
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True

    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model