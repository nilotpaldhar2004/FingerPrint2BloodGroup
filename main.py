import os
import io
import base64
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("fingerprint2bloodgroup-api")

MODEL_PATH   = os.getenv("MODEL_PATH",   "blood_group_resnet50_best.pth")
CLASSES_PATH = os.getenv("CLASSES_PATH", "blood_group_classes.npy")
PORT         = int(os.getenv("PORT", 7860))
IMG_SIZE     = 448

ml = {}  # shared model store

# ── 2. TRANSFORMS ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 3. MODEL ARCHITECTURE ─────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)

    # Freeze backbone — must match training setup
    for p in model.parameters():
        p.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ── 4. GRAD-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, mdl: nn.Module, target_layer: nn.Module):
        self.mdl         = mdl
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda _, __, out: setattr(self, "activations", out.detach())
        )
        target_layer.register_full_backward_hook(
            lambda _, __, grad_out: setattr(self, "gradients", grad_out[0].detach())
        )

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.mdl.zero_grad()
        tensor = tensor.clone().requires_grad_(True)
        out    = self.mdl(tensor)
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(pil_img: Image.Image, cam: np.ndarray) -> str:
    """Returns base64 JPEG string of Grad-CAM overlay."""
    img_np  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    cam_up  = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 255).astype(np.uint8)
    _, buf  = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")

# ── 5. LIFESPAN ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI resources...")
    try:
        classes      = np.load(CLASSES_PATH, allow_pickle=True)
        model        = build_model(num_classes=len(classes))
        grad_cam     = GradCAM(model, target_layer=model.layer4[-1])
        ml["model"]   = model
        ml["classes"] = classes
        ml["gradcam"] = grad_cam
        logger.info(f"Model loaded. Classes: {list(classes)}")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    yield
    ml.clear()
    logger.info("Resources released.")

# ── 6. FASTAPI APP ────────────────────────────────────────────────────────────
app = FastAPI(title="FingerPrint2BloodGroup API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 7. ROUTES ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    ready = "model" in ml
    return {
        "status":  "ok" if ready else "loading",
        "classes": list(ml["classes"]) if ready else [],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model   = ml.get("model")
    classes = ml.get("classes")
    gradcam = ml.get("gradcam")

    if model is None or classes is None:
        raise HTTPException(status_code=503, detail="Model not ready. Try again in a moment.")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Only JPEG / PNG / WEBP accepted.")

    try:
        t0        = time.perf_counter()
        raw       = await file.read()
        pil_img   = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor    = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)

        pred_idx   = idx.item()
        pred_label = str(classes[pred_idx])

        all_probs = {
            str(classes[i]): round(float(probs[i]) * 100, 2)
            for i in range(len(classes))
        }

        # Grad-CAM
        cam         = gradcam.generate(tensor, pred_idx)
        gradcam_b64 = overlay_cam(pil_img, cam)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Predicted: {pred_label} ({conf.item()*100:.2f}%) | {latency}ms")

        return JSONResponse({
            "predicted_class":   pred_label,
            "confidence":        round(conf.item() * 100, 2),
            "all_probabilities": all_probs,
            "gradcam_image":     gradcam_b64,
            "latency_ms":        latency,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
