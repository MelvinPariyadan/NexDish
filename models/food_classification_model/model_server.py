from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
import io
import json

app = Flask(__name__)


# Active Model
checkpoint = torch.load('checkpoints/resnet50_full_dataset.pth', map_location=torch.device('cpu'))
classes = checkpoint['classes']




# Model Architecture
model = resnet50(pretrained=False)  
hidden_dim = 512
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(hidden_dim, len(classes))
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# image to tensor tranformation
transformation_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_bytes = request.files['file'].read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = transformation_pipeline(img).unsqueeze(0)

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
