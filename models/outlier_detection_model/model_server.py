from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
import yaml
import numpy as np
from alibi_detect.saving import load_detector


app = Flask(__name__)

with open("models/outlier_detection_model/src/params.yaml") as f:
    config = yaml.safe_load(f)

MAX_BATCH = config["output"]["max_batch"]
SAVE_DIR = config["output"]["model_checkpoint"]
IMG_SIZE = 64 # encoder trained on 64x64x3 images

#TODO - load model from DVC ?
outlier_detector = load_detector(SAVE_DIR)


@app.route('/detect_outlier', methods=['POST'])
def predict():
    try:
        files = request.files.getlist("file")
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        images = []
        for file in files[:MAX_BATCH]:  # limit batch size
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_np = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_np)

        images_np = np.stack(images)  # shape: (B, H, W, C)

        preds = outlier_detector.predict(
            images_np,
            outlier_type='instance',
            return_instance_score=True,
            return_feature_score=False  # change to True if needed
        )

        instance_scores = preds['data']['instance_score'].tolist()
        is_outlier = preds['data']['is_outlier'].tolist()

        results = [{'score': score, 'is_outlier': bool(flag)}
                   for score, flag in zip(instance_scores, is_outlier)]

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8003, debug=True)