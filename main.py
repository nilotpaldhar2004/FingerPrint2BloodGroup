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
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("fingerprint2bloodgroup-api")

MODEL_PATH       = os.getenv("MODEL_PATH",   "blood_group_resnet50_best.pth")
CLASSES_PATH     = os.getenv("CLASSES_PATH", "blood_group_classes.npy")
PORT             = int(os.getenv("PORT", 7860))
IMG_SIZE         = 448
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

VAL_ACCURACY_PCT = float(os.getenv("VAL_ACCURACY_PCT", 87.22))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ml = {}

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

    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True

    num_features = model.fc.in_features          # 2048
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

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ── 4. GRAD-CAM ───────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, mdl: nn.Module, target_layer: nn.Module):
        self.mdl         = mdl
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, output):
        # Keep tensor attached during forward pass so backward computes correctly
        self.activations = output

    def _bwd_hook(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.mdl.zero_grad()

        with torch.enable_grad():
            tensor = tensor.clone().detach().requires_grad_(True)
            out    = self.mdl(tensor)
            score  = out[0, class_idx]
            score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to extract activations/gradients for Grad-CAM.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam     = cam.detach().cpu().numpy()
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
    logger.info(f"Loading AI resources on device={DEVICE} ...")
    try:
        classes  = np.load(CLASSES_PATH, allow_pickle=True)
        model    = build_model(num_classes=len(classes))
        grad_cam = GradCAM(model, target_layer=model.layer4[-1])

        ml["model"]   = model
        ml["classes"] = classes
        ml["gradcam"] = grad_cam
        logger.info(f"Model loaded successfully. Classes: {list(classes)}")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
    yield
    ml.clear()
    logger.info("Resources released.")

# ── 6. FASTAPI APP ────────────────────────────────────────────────────────────
app = FastAPI(title="FingerPrint2BloodGroup API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 7. ROUTES ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "FingerPrint2BloodGroup API — GET /docs for interactive reference."}


@app.get("/health")
async def health():
    ready = "model" in ml
    return {
        "status":  "ok" if ready else "loading",
        "device":  str(DEVICE),
        "classes": [str(c) for c in ml["classes"]] if ready else [],
    }


@app.get("/model-info")
async def model_info():
    ready = "model" in ml
    return {
        "architecture":   "ResNet-50 (ImageNet-pretrained backbone, layer3+layer4 fine-tuned)",
        "input_size":     IMG_SIZE,
        "num_classes":    int(len(ml["classes"])) if ready else None,
        "classes":        [str(c) for c in ml["classes"]] if ready else [],
        "val_accuracy":   VAL_ACCURACY_PCT,
        "device":         str(DEVICE),
        "explainability": "Grad-CAM on the last block of layer4",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model   = ml.get("model")
    classes = ml.get("classes")
    gradcam = ml.get("gradcam")

    if model is None or classes is None or gradcam is None:
        raise HTTPException(status_code=503,
                            detail="Model not ready. Try again in a moment.")

    allowed_mimes = (
        "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
        "application/octet-stream"
    )
    if file.content_type and file.content_type not in allowed_mimes:
        raise HTTPException(status_code=400,
                            detail="Unsupported file format. Please upload a standard image.")

    try:
        t0  = time.perf_counter()
        raw = await file.read()

        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large — max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
            )

        try:
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Could not read image file — is it corrupted?")

        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # ── Inference ─────────────────────────────────────────
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)

        pred_idx   = idx.item()
        pred_label = str(classes[pred_idx])
        confidence = round(float(conf.item()) * 100, 2)

        all_probs = {
            str(classes[i]): round(float(probs[i]) * 100, 2)
            for i in range(len(classes))
        }

        # ── Grad-CAM ──────────────────────────────────────────
        cam         = gradcam.generate(tensor, pred_idx)
        gradcam_b64 = overlay_cam(pil_img, cam)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"Predicted: {pred_label} | Confidence: {confidence}% | {latency}ms")

        return JSONResponse({
            "predicted_class":   pred_label,
            "confidence":        confidence,
            "all_probabilities": all_probs,
            "gradcam_image":     gradcam_b64,
            "latency_ms":        latency,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
