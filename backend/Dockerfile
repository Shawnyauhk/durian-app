# DurianAI Backend — FastAPI + TFLite + pydub(ffmpeg)
FROM python:3.11-slim

# Install system dependencies (ffmpeg for pydub webm/opus decoding)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes model files)
COPY backend/ .

# Verify model file exists and log it
RUN ls -la /app/models/acoustic/ 2>/dev/null || echo "No model files yet"

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
