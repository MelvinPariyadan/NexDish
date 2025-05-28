import torch
import numpy as np
from dotenv import load_dotenv
import os

def test_model_accuracy_threshold():
    load_dotenv()
    path_to_model = os.getenv("PATH_TO_MODEL")
    model_data = torch.load(path_to_model)
    test_acc = model_data.get('test_acc', 0.0)

    assert test_acc >= 0.80, f"Top-1 Accuracy too low: {test_acc:.2%}"