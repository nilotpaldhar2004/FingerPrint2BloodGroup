FROM python:3.10-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and frontend
COPY main.py .
COPY index.html .

# Copy model files
COPY blood_group_resnet50_best.pth .
COPY blood_group_classes.npy .

# Set environment variables
ENV MODEL_PATH=/app/blood_group_resnet50_best.pth
ENV CLASSES_PATH=/app/blood_group_classes.npy
ENV PORT=7860

EXPOSE 7860

CMD ["python", "main.py"]
