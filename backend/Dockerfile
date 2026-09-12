# ==============================================================================
# AI Wellbeing Buffer — Production Backend Dockerfile
# Optimized for self-hosted deployment with persistent weight caches.
# ==============================================================================

FROM python:3.11-slim

# Prevent python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TORCH_HOME=/app/cache/torch \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    NUDENET_CACHE_DIR=/app/cache/nudenet \
    MODEL_DIR=/app/models \
    MODEL_PATH=/app/models/custom_toxic_model.joblib

WORKDIR /app

# Install minimal OS runtime packages for image processing & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Persistent storage directories
RUN mkdir -p /app/models /app/cache /app/data

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY . /app

# Pre-train initial models if model artifact is not yet present
RUN if [ ! -f "$MODEL_PATH" ] && [ -f "train_model.py" ]; then \
        python train_model.py || echo "Initial training skipped; can be triggered via /train-custom-classifier"; \
    fi

EXPOSE 8000

# Healthcheck for container orchestration (Fly.io / Render / Docker Compose)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
