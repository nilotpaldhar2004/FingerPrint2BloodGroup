import io
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "blood_group_resnet50_best.pth"
LABELS_PATH = "blood_group_classes.npy"
IMG_SIZE    = 448

# ── Load label classes ───────────────────────────────────────────────────────
classes = np.load(LABELS_PATH, allow_pickle=True).tolist()
NUM_CLASSES = len(classes)          # should be 8

# ── Build model (must match training architecture) ───────────────────────────
def build_model():
    m = models.resnet50(weights=None)
    for p in m.parameters():
        p.requires_grad = False
    num_features = m.fc.in_features
    m.fc = nn.Sequential(
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
        nn.Linear(64, NUM_CLASSES),
    )
    return m

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ── Image transform (same as test_transform in training) ────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Grad-CAM helper ─────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, mdl, target_layer):
        self.mdl          = mdl
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx):
        self.mdl.zero_grad()
        # temporarily unfreeze for gradient flow
        input_tensor.requires_grad_(True)
        output = self.mdl(input_tensor)           # forward
        score  = output[0, class_idx]
        score.backward()                          # backward

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


grad_cam = GradCAM(model, target_layer=model.layer4[-1])


def overlay_cam(pil_img: Image.Image, cam: np.ndarray) -> str:
    """Return base64-encoded JPEG of the Grad-CAM overlay."""
    img_np  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    cam_up  = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)
    _, buf   = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="FingerPrint2BloodGroup API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate ──────────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400,
                            detail="Only JPEG / PNG / WEBP images accepted.")

    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # ── Inference ─────────────────────────────────────────────────────────
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # All class probabilities for the bar chart
    all_probs = {cls: round(float(p) * 100, 2)
                 for cls, p in zip(classes, probs)}

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    cam_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    cam        = grad_cam.generate(cam_tensor, pred_idx)
    gradcam_b64 = overlay_cam(pil_img, cam)

    return JSONResponse({
        "predicted_class": pred_label,
        "confidence":      round(confidence, 2),
        "all_probabilities": all_probs,
        "gradcam_image":   gradcam_b64,
    })
