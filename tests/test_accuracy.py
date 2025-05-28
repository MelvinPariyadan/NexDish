import torch
import numpy as np
from dotenv import load_dotenv
import os
import sys
import time
from PIL import Image
import yaml


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from food_classification_model.src.transform import get_transforms
from food_classification_model.src.model import load_model_architecture


PATH = "models/food_classification_model/checkpoints/best_model.pth"
INFERENCE_IMAGE_PATH = "tests/inference_img.jpg"
CONFIG_PATH = "models/food_classification_model/src/params.yaml"


def test_model_accuracy_threshold():
    load_dotenv()
    model_data = torch.load(PATH)
    test_acc = model_data.get('test_acc', 0.0)
    assert test_acc >= 0.80, f"Top-1 Accuracy too low: {test_acc:.2%}"


def test_model_inference_time():

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    hidden_dim = config["train"]["hidden_dim"]
    num_classes = config["dataset"]["num_classes"]

    # Load checkpoint and model
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    classes = checkpoint["classes"]

    model = load_model_architecture(weights=None, device="cpu", hidden_dim=hidden_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get validation transforms
    _, val_tfms = get_transforms()

    # Load and transform image
    img = Image.open(INFERENCE_IMAGE_PATH).convert("RGB")
    input_tensor = val_tfms(img).unsqueeze(0)

    # Inference timing
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        end = time.time()

    inference_time = end - start
    assert inference_time <= 0.2, f"Inference too slow: {inference_time:.3f} seconds"