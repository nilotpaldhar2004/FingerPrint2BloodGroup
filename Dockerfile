FROM python:3.10-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY static/ static/

# Copy model files — these must exist in the root of your HF Space repo
# If missing, the container will build but startup will fail with FileNotFoundError
COPY blood_group_resnet50_best.pth .
COPY blood_group_classes.npy .

# Set environment variables so main.py finds the files
ENV MODEL_PATH=/app/blood_group_resnet50_best.pth
ENV CLASSES_PATH=/app/blood_group_classes.npy

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
