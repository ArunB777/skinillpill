# --------- COMBINED FASTAPI BACKEND (HF HUB READY, ENV TOKEN) ---------
import os
import io
import time
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import matplotlib.cm as cm
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from huggingface_hub import hf_hub_download, login

# ============================================
# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# HUGGING FACE LOGIN (via environment variable)
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
else:
    print("Warning: HF_TOKEN not set. Make sure the repo is public or the token is provided as a secret.")

# ============================================
# HF MODEL PATHS
# Using token for private repo access
HAM_EFF_PATHS = [
    hf_hub_download(
        repo_id="arunbaigra/skinillpill",
        filename=f"final_model_fold{i}.pth",
        use_auth_token=HF_TOKEN
    ) for i in range(5)
]

HAM_EFF_CLASSES = [
    "Actinic keratoses and intraepithelial carcinoma (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic nevi (nv)",
    "Vascular lesions (vasc)"
]

CONVNEXT_MASSIVE_PATH = hf_hub_download(
    repo_id="arunbaigra/skinillpill",
    filename="best_finetune.pth",
    use_auth_token=HF_TOKEN
)
CONVNEXT_MASSIVE_CLASSES = [
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions",
    "Malignant",
    "Melanoma Skin Cancer Nevi And Moles",
    "Seborrheic Keratoses And Other Benign Tumors",
    "Vascular Tumors",
    "Warts Molluscum And Other Viral Infections",
    "Acne And Rosacea",
    "Eczema",
    "Psoriasis Pictures Lichen Planus And Related Diseases",
    "Poison Ivy And Other Contact Dermatitis",
]

CONVNEXT_HAM_PATH = hf_hub_download(
    repo_id="arunbaigra/skinillpill",
    filename="fine_tuned_convnext_ham.pth",
    use_auth_token=HF_TOKEN
)
CONVNEXT_HAM_CLASSES = HAM_EFF_CLASSES

# ============================================
# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================
# MODELS
class SkinEffNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

def load_eff_model(path):
    model = SkinEffNet(num_classes=len(HAM_EFF_CLASSES))
    state_dict = torch.load(path, map_location=DEVICE)
    # Remove "module." prefix if using DataParallel
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

eff_models = [load_eff_model(p) for p in HAM_EFF_PATHS]

class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.convnext_base(weights=None)
        base.classifier = nn.Identity()
        self.base = base
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.base.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x)); x = self.dropout1(x)
        x = torch.relu(self.fc2(x)); x = self.dropout2(x)
        return self.fc3(x)

def load_convnext(path, num_classes):
    model = CustomConvNeXt(num_classes=num_classes)
    state_dict = torch.load(path, map_location=DEVICE)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

convnext_massive = load_convnext(CONVNEXT_MASSIVE_PATH, len(CONVNEXT_MASSIVE_CLASSES))
convnext_ham = load_convnext(CONVNEXT_HAM_PATH, len(CONVNEXT_HAM_CLASSES))

# ============================================
# GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        def fwd(m, i, o): self.activations = o.detach()
        def bwd(m, gi, go): self.gradients = go[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_backward_hook(bwd)
    def generate(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward(retain_graph=True)
        g = self.gradients.mean(dim=(2, 3), keepdim=True)
        w = (g * self.activations).sum(dim=1, keepdim=True)
        h = torch.relu(w).squeeze().cpu().numpy()
        return (h - h.min()) / (h.max() + 1e-8)

def get_heatmap(x, model, layer, idx, orig_img):
    gc = GradCAM(model, layer)
    h = gc.generate(x, idx)
    h = Image.fromarray(np.uint8(255 * h)).resize(orig_img.size, Image.BILINEAR)
    h = np.array(h)
    c = cm.jet(h / 255.0)[:, :, :3]
    c = (c * 255).astype(np.uint8)
    overlay = Image.blend(orig_img.convert("RGB"), Image.fromarray(c), alpha=0.5)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ============================================
# FASTAPI APP
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"msg": "Combined Skin AI API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_choice: str = Form(...)):
    start = time.time()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    xt = transform(img).unsqueeze(0).to(DEVICE)

    if model_choice == "efficientnet":
        probs = []
        with torch.no_grad():
            for m in eff_models:
                p = torch.softmax(m(xt), dim=1).cpu().numpy()
                probs.append(p)
        probs = np.mean(probs, axis=0)[0]
        classes = HAM_EFF_CLASSES
        heatmap = get_heatmap(xt, eff_models[0], eff_models[0].model.blocks[-1], np.argmax(probs), img)

    elif model_choice == "convnext_massive":
        with torch.no_grad():
            out = convnext_massive(xt)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        classes = CONVNEXT_MASSIVE_CLASSES
        heatmap = get_heatmap(xt, convnext_massive, convnext_massive.base.features[-1], np.argmax(probs), img)

    elif model_choice == "convnext_ham":
        with torch.no_grad():
            out = convnext_ham(xt)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        classes = CONVNEXT_HAM_CLASSES
        heatmap = get_heatmap(xt, convnext_ham, convnext_ham.base.features[-1], np.argmax(probs), img)

    else:
        return JSONResponse({"error": "Invalid model_choice"}, status_code=400)

    pred_idx = int(np.argmax(probs))
    return JSONResponse({
        "predicted_class": classes[pred_idx],
        "confidence_score": round(float(probs[pred_idx]) * 100, 2),
        "prediction_time": round(time.time() - start, 2),
        "class_probs": probs.tolist(),
        "class_names": classes,
        "heatmap": heatmap
    })
