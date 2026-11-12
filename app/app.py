import base64
import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from . import db, inference

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "5"))
ALLOWED_EXT = {"jpg", "jpeg", "png"}
ALLOWED_MIME = {"image/jpeg", "image/png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
db.init_db()


def error_page(code, title, message):
    return render_template("error.html", code=code, title=title, message=message), code


@app.errorhandler(413)
def too_large(_e):
    return error_page(413, "File too large", f"The image must be smaller than {MAX_UPLOAD_MB} MB.")


@app.errorhandler(404)
def not_found(_e):
    return error_page(404, "Page not found", "That page does not exist.")


@app.errorhandler(500)
def server_error(_e):
    return error_page(500, "Something went wrong", "An unexpected error occurred. Please try again.")


def thumbnail_b64(data: bytes) -> str:
    """Preview shown on the result page, built in memory, never written to disk."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((480, 480))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return error_page(400, "No file chosen", "Please choose an image to upload.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT or file.mimetype not in ALLOWED_MIME:
        return error_page(400, "Unsupported file type", "Please upload a JPG or PNG image.")
    filename = secure_filename(file.filename) or "upload"

    data = file.read()
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return error_page(400, "Not a valid image",
                          "That file has an image extension but cannot be read as an image.")

    try:
        label, confidence = inference.predict(data)
    except Exception:
        app.logger.exception("inference failed")
        return error_page(500, "Prediction failed",
                          "The model could not process this image. Please try another one.")

    db.add_prediction(filename=filename, label=label, confidence=confidence,
                      threshold=inference.THRESHOLD, model_name=inference.MODEL_NAME, source="local")
    return render_template("result.html", filename=filename, label=label, confidence=confidence,
                           threshold=inference.THRESHOLD, model_name=inference.MODEL_NAME,
                           preview=thumbnail_b64(data))


@app.get("/history")
def history():
    return render_template("history.html", rows=db.recent(100))


@app.get("/health")
def health():
    try:
        db.ping()
        db_status = "ok"
    except Exception:
        db_status = "error"
    body = {"status": "ok" if db_status == "ok" else "degraded", "model": inference.MODEL_NAME,
            "threshold": inference.THRESHOLD, "database": db_status}
    return jsonify(body), (200 if db_status == "ok" else 503)
