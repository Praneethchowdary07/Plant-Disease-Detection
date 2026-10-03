"""Flask web app: plant disease detection with a ResNet18 model (ONNX Runtime)."""
from __future__ import annotations

import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from disease_info import DEFAULT_INFO, DISEASE_INFO

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads

_session = None
_classes: list[str] = []
_metrics: dict = {}


def load_model():
    """Load the ONNX model once (lazily, so the page works while the model uploads)."""
    global _session, _classes, _metrics
    if _session is None:
        import onnxruntime as ort

        path = os.path.join(MODEL_DIR, "plant_resnet18.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError("Model not uploaded yet. Run the Colab notebook to add model/plant_resnet18.onnx.")
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
            _classes = json.load(f)
        mpath = os.path.join(MODEL_DIR, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                _metrics = json.load(f)
    return _session


def preprocess(img: Image.Image) -> np.ndarray:
    """Same transform as training: resize 224x224, scale to [0,1], ImageNet normalize, NCHW."""
    x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def estimate_severity(img: Image.Image) -> float:
    """Percent of the leaf area that is discoloured (not green).

    Leaf pixels = saturated, non-background pixels in HSV space; diseased pixels =
    leaf pixels whose hue falls outside the healthy-green band (yellow/brown spots, lesions).
    """
    hsv = np.asarray(img.resize((256, 256)).convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    background = h > 185                         # blue/purple/magenta backdrops and shadows are never leaf tissue
    leaf = (s > 0.20) & (v > 0.18) & ~background
    if leaf.sum() < 500:
        return 0.0
    green = (h >= 65) & (h <= 170)
    diseased = leaf & ~green
    return round(float(diseased.sum() / leaf.sum() * 100), 1)


DISPLAY = {
    "Apple___Apple_scab": ("Apple", "Apple scab"),
    "Apple___Black_rot": ("Apple", "Black rot"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar apple rust"),
    "Apple___healthy": ("Apple", "Healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Corn (maize)", "Cercospora / gray leaf spot"),
    "Corn_(maize)___Common_rust_": ("Corn (maize)", "Common rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Corn (maize)", "Northern leaf blight"),
    "Corn_(maize)___healthy": ("Corn (maize)", "Healthy"),
    "Pepper__bell___Bacterial_spot": ("Bell pepper", "Bacterial spot"),
    "Pepper__bell___healthy": ("Bell pepper", "Healthy"),
    "Potato___Early_blight": ("Potato", "Early blight"),
    "Potato___Late_blight": ("Potato", "Late blight"),
    "Potato___healthy": ("Potato", "Healthy"),
    "Tomato_Bacterial_spot": ("Tomato", "Bacterial spot"),
    "Tomato_Early_blight": ("Tomato", "Early blight"),
    "Tomato_Late_blight": ("Tomato", "Late blight"),
    "Tomato_Leaf_Mold": ("Tomato", "Leaf mold"),
    "Tomato_Septoria_leaf_spot": ("Tomato", "Septoria leaf spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Two-spotted spider mites"),
    "Tomato__Target_Spot": ("Tomato", "Target spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomato", "Yellow leaf curl virus"),
    "Tomato__Tomato_mosaic_virus": ("Tomato", "Mosaic virus"),
    "Tomato_healthy": ("Tomato", "Healthy"),
}


def pretty(label: str) -> tuple[str, str]:
    return DISPLAY.get(label, (label.split("_")[0], label.replace("_", " ")))


@app.get("/")
def index():
    examples = sorted(os.listdir(os.path.join(BASE, "static", "examples")))
    return render_template("index.html", examples=examples)


@app.post("/api/predict")
def predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Please choose an image."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.read()))).convert("RGB")
    except Exception:
        return jsonify(error="That file isn't a readable image."), 400
    try:
        session = load_model()
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503

    probs = session.run(None, {"input": preprocess(img)})[0][0]
    top = np.argsort(probs)[::-1][:3]
    label = _classes[int(top[0])]
    crop, disease = pretty(label)
    healthy = "healthy" in label.lower()
    info = DISEASE_INFO.get(label, DEFAULT_INFO)
    return jsonify(
        label=label,
        crop=crop,
        disease=disease,
        healthy=healthy,
        confidence=round(float(probs[top[0]]) * 100, 1),
        severity=0.0 if healthy else estimate_severity(img),
        yield_rate=info["yield_rate"],
        solution=info["solution"],
        top3=[{"crop": pretty(_classes[int(i)])[0], "disease": pretty(_classes[int(i)])[1],
               "p": round(float(probs[int(i)]) * 100, 1)} for i in top],
    )


@app.get("/api/model")
def model_info():
    try:
        load_model()
    except FileNotFoundError as exc:
        return jsonify(ready=False, error=str(exc))
    return jsonify(ready=True, classes=len(_classes), val_accuracy=_metrics.get("val_accuracy"),
                   val_images=_metrics.get("val_images"))


@app.get("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
