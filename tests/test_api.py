import requests

BASE_URL = "http://localhost:8000"
LLM_URL = "http://localhost:8001"
CLASSIFIER_URL = "http://localhost:8002"
IMG_PATH = "tests/inference_img.jpg"

def test_upload_endpoint_success():
    with open(IMG_PATH, "rb") as img:
        files = {"file": ("test.jpg", img, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "food_info" in data
    assert isinstance(data["food_info"], dict)

def test_upload_no_file():
    response = requests.post(f"{BASE_URL}/upload", files={})
    assert response.status_code == 400
    assert "error" in response.json()

def test_llm_basic_response():
    response = requests.get(f"{LLM_URL}/info/pizza")
    assert response.status_code == 200
    data = response.json()
    assert "description" in data
