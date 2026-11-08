import hmac
import io
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = 224
MAX_BYTES = 4_000_000                      # Vercel rejects request bodies above about 4.5 MB
MODEL_PATH = Path(__file__).parent / "model.onnx"
MODEL_NAME = os.environ.get("MODEL_NAME", "mobilenetv2_finetuned")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Loaded once per container (cold start), never per request.
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

app = FastAPI(title="VisionGuard model API")


def preprocess(data: bytes) -> np.ndarray:
    """Same steps as training: RGB, 224x224 bilinear resize, float32 in [0, 255].
    Rescaling to the range the network expects happens inside the ONNX model."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File is not a decodable image")
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]       # (1, 224, 224, 3)


def check_key(x_api_key):
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server has no API_KEY configured")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid x-api-key header")


@app.get("/")
def root():
    return {"service": "VisionGuard model API", "endpoints": ["GET /health", "POST /predict"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD,
            "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_BYTES} bytes")
    t0 = time.perf_counter()
    x = preprocess(data)
    p_ai = float(session.run(None, {INPUT_NAME: x})[0].reshape(-1)[0])
    label = "ai" if p_ai >= THRESHOLD else "real"
    return {
        "label": label,
        "confidence": round(p_ai if label == "ai" else 1.0 - p_ai, 4),
        "probability_ai": round(p_ai, 4),
        "threshold": THRESHOLD,
        "model": MODEL_NAME,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }