FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py .
COPY checkpoints/ checkpoints/

EXPOSE 8002

CMD ["python", "model_server.py"]
