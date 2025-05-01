# web_server.py
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# local
#CLASSIFICATION_MODEL_SERVER_URL = 'http://localhost:8002'
#LLM_MODEL_SERVER_URL = 'http://localhost:8001'

#docker
CLASSIFICATION_MODEL_SERVER_URL = 'http://classification_model:8002'
LLM_MODEL_SERVER_URL = 'http://llm_model:8001'



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

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        file_bytes = file.read()
        response = requests.post(
            f"{CLASSIFICATION_MODEL_SERVER_URL}/predict",
            files={'file': (file.filename, file_bytes, file.content_type)}
        )
    
        output = response.json()
        predicted_class = output["predicted_class"]
        food_info = fetch_food_info(predicted_class)


        return jsonify({'predicted_class': predicted_class, 'food_info':food_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000, debug=True)
