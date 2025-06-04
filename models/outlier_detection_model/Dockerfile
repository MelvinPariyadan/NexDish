FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py .
COPY checkpoints/ checkpoints/
COPY src/ models/outlier_detection_model/src

ENV PYTHONPATH=/app/models/outlier_detection_model

EXPOSE 8003

CMD ["python", "model_server.py"]
