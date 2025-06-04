FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py .
COPY checkpoints/ checkpoints/
COPY src/ models/food_classification_model/src
COPY offline_src/ models/food_classification_model/offline_src

ENV PYTHONPATH=/app/models/food_classification_model

EXPOSE 8002

CMD ["python", "model_server.py"]
