import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights


def load_model_architecture(weights=None, device="mps", hidden_dim=512, num_classes=101):
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, num_classes)
    )
    return model.to(device)

