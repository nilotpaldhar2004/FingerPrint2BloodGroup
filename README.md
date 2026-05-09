---
title: FingerPrint2BloodGroup
emoji: 🩸
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🩸 FingerPrint2BloodGroup

> **Research Demo** — Predicting blood group from fingerprint images using Transfer Learning (ResNet-50).

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/nilotpaldhar2004/fingerprint-blood-group)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)](https://pytorch.org/)

---

## ⚠️ Disclaimer

This project is a **research and portfolio demonstration only**.  
Current validation accuracy is ~51% on 8 classes — this model is **not suitable for clinical or diagnostic use**. Do not make any medical decisions based on its output.

---

## 🧠 What It Does

Upload a fingerprint image → the model predicts the corresponding blood group along with:

- **Predicted class** with confidence score
- **Class probability bar chart** (all 8 blood groups)
- **Grad-CAM heatmap** showing which regions of the fingerprint the model focused on

---

## 🏗️ Model Architecture

| Component     | Detail                              |
|---------------|-------------------------------------|
| Backbone      | ResNet-50 (ImageNet pretrained)     |
| Backbone      | Frozen (feature extractor only)     |
| FC Head       | 2048 → 1024 → 512 → 128 → 8        |
| Activations   | ReLU + Dropout (0.3, 0.2)           |
| Loss          | CrossEntropyLoss                    |
| Optimizer     | Adam (lr=0.0001)                    |
| Input size    | 448 × 448 × 3                       |
| FC Head       | 2048 → 1024 → BN → 512 → 128 → 64 → 8 |
| Dropout       | 0.4, 0.2, 0.1                       |

---

## 📊 Training Results

| Epoch | Val Accuracy |
|-------|-------------|
| 1     | 25.42%      |
| 12    | 49.17%      |
| 20    | 50.50%      |
| 24    | **51.58%** ⭐ |
| 25    | 51.08%      |

Training set: 4800 samples | Validation set: 1200 samples | Batch size: 32 | Epochs: 25

---

## 🗂️ Project Structure

```
fingerprint-blood-group/
│
├── main.py                        # FastAPI backend + Grad-CAM logic
├── static/
│   └── index.html                 # Frontend UI
├── blood_group_resnet50_best.pth  # Trained model weights
├── blood_group_classes.npy        # Label encoder classes
├── requirements.txt
├── Dockerfile
├── .gitignore
└── .github/
    └── workflows/
        └── deploy_to_hf.yml       # GitHub Actions → HF Space auto-deploy
```

---

## 🚀 Local Setup

```bash
# Clone
git clone https://github.com/YOUR_GITHUB_USERNAME/fingerprint-blood-group.git
cd fingerprint-blood-group

# Install dependencies
pip install -r requirements.txt

# Place model files in root directory:
#   blood_group_resnet50_best.pth
#   blood_group_classes.npy

# Run
uvicorn main:app --reload --port 8000

# Open browser
# http://localhost:8000
```

---

## 🐳 Docker

```bash
docker build -t fingerprint-blood-group .
docker run -p 7860:7860 fingerprint-blood-group
# Open: http://localhost:7860
```

---

## ☁️ Deploy to HuggingFace Space

1. Create a new Space on HuggingFace: `fingerprint-blood-group` with **Docker** SDK
2. Add these GitHub Secrets to your repository:
   - `HF_TOKEN` → your HuggingFace write token
   - `HF_USERNAME` → your HuggingFace username
3. Push to `main` branch — GitHub Actions handles the rest automatically

> **Important:** Model weight files (`.pth`, `.npy`) are in `.gitignore`. Upload them manually to your HF Space via the web UI or use [Git LFS](https://git-lfs.github.com/).

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, PyTorch, torchvision, OpenCV
- **Frontend:** Vanilla HTML/CSS/JS (no framework)
- **Explainability:** Grad-CAM on `layer4[-1]` of ResNet-50
- **Deployment:** Docker · HuggingFace Spaces · GitHub Actions

---

## 👤 Author

**Nilotpal** — CSBS, Semester 6 | AI/ML Enthusiast  
[HuggingFace](https://huggingface.co/nilotpaldhar2004) · [GitHub](https://github.com/nilotpaldhar2004)

---

## 📄 License

MIT License — free to use for educational and research purposes.
