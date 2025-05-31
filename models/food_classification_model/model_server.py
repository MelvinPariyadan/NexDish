from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
import io
import json
from offline_src.transform import get_transforms
from offline_src.model import load_model_architecture
from torchvision.models import ResNet50_Weights
import yaml


app = Flask(__name__)

with open("models/food_classification_model/src/params.yaml") as f:
    config = yaml.safe_load(f)

HIDDEN_DIM = config["train"]["hidden_dim"]
NUM_CLASSES = config["dataset"]["num_classes"]

# Active Model
checkpoint = torch.load('checkpoints/resnet50_full_dataset.pth', map_location=torch.device('cpu'))
classes = checkpoint['classes']

model = load_model_architecture(weights=None, device="cpu", hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_tfms, val_tfms = get_transforms()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_bytes = request.files['file'].read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = val_tfms(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, predicted_idx = outputs.max(1)
            predicted_class = classes[predicted_idx.item()]
        
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8002, debug=True)
