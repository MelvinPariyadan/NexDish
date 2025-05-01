FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py .

EXPOSE 8001

CMD ["python", "model_server.py"]
