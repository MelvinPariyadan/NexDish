import torch
import numpy as np
from dotenv import load_dotenv
import os

PATH = "models/food_classification_model/checkpoints/best_model.pth"

def test_model_accuracy_threshold():
    load_dotenv()
    model_data = torch.load(PATH)
    test_acc = model_data.get('test_acc', 0.0)
    assert test_acc >= 0.80, f"Top-1 Accuracy too low: {test_acc:.2%}"