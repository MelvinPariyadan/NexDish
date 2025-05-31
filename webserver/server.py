# web_server.py
from flask import Flask, request, jsonify
import requests
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# local
#CLASSIFICATION_MODEL_SERVER_URL = 'http://localhost:8002'
#LLM_MODEL_SERVER_URL = 'http://localhost:8001'

#docker
CLASSIFICATION_MODEL_SERVER_URL = 'http://classification_model:8002'
LLM_MODEL_SERVER_URL = 'http://llm_model:8001'
OUTLIER_SERVER_URL = 'http://outlier_detection_model:8003'

def log_outlier_detection(file, score):
    log_dir = Path("outliers")
    log_dir.mkdir(exist_ok=True)

    # Save log entry
    with open("log.txt", "a") as f:
        f.write(f"[{datetime.now()}] Outlier detected: {file.filename}, score={score}\n")

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    image_path = log_dir / safe_name

    file.seek(0)
    with open(image_path, "wb") as out_img:
        out_img.write(file.read())

def fetch_food_info(food_label):
    try:
        response = requests.get(f"{LLM_MODEL_SERVER_URL}/info/{food_label}")
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Failed to fetch food info. Status code: {response.status_code}"}
    except Exception as e:
        return {
            "error": f"Exception during food info fetch: {str(e)}"        }

def check_outlier(file):
    try:
        file.seek(0)  # Reset read pointer before reuse
        response = requests.post(
            f"{OUTLIER_SERVER_URL}/detect_outlier",
            files={'file': (file.filename, file.read(), file.content_type)}
        )
        return response.json()
    except Exception as e:
        return {"error": f"Outlier detection error: {str(e)}"}
    
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Call OD for prediction
    outlier_result = check_outlier(file)
    if 'error' in outlier_result:
        return jsonify(outlier_result), 500

    is_outlier = outlier_result.get("results", [{}])[0].get("is_outlier", None)
    score = outlier_result.get("results", [{}])[0].get("score", None)

    if is_outlier is None:
        return jsonify({'error': 'Invalid response from outlier detection model'}), 500

    if is_outlier:
        log_outlier_detection(file, score)
        return jsonify({'warning': 'Outlier image detected', 'outlier_score': score}), 400

    # If not an outlier, proceed with classification
    try:
        file.seek(0)
        response = requests.post(
            f"{CLASSIFICATION_MODEL_SERVER_URL}/predict",
            files={'file': (file.filename, file.read(), file.content_type)}
        )
    
        output = response.json()
        predicted_class = output["predicted_class"]
        food_info = fetch_food_info(predicted_class)


        return jsonify({'predicted_class': predicted_class, 'food_info':food_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000, debug=True)
