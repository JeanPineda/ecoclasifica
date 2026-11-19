"""EcoClasifica web interface and public inference API."""
import io
import json
import os
import warnings
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
with (ROOT / "models" / "metadata.json").open(encoding="utf-8") as f:
    metadata = json.load(f)
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(ROOT / "models" / "model.onnx"),
                             sess_options=options, providers=["CPUExecutionProvider"])
def preprocess(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    shorter = metadata["resize_shorter"]
    if width < height:
        size = (shorter, int(shorter * height / width))
    else:
        size = (int(shorter * width / height), shorter)
    image = image.resize(size, Image.Resampling.BILINEAR)
    crop = metadata["crop"]
    left, top = round((size[0] - crop) / 2), round((size[1] - crop) / 2)
    image = image.crop((left, top, left + crop, top + crop))
    array = np.asarray(image, dtype=np.float32) / 255.
    array = (array - np.array(metadata["mean"], dtype=np.float32)) / np.array(metadata["std"], dtype=np.float32)
    return np.transpose(array, (2, 0, 1))[None].astype(np.float32)
@app.get("/")
def home():
    return render_template("index.html")
@app.get("/health")
def health():
    return jsonify(status="ok", model_version=metadata["version"])
@app.get("/api")
def api_info():
    return jsonify(method="POST", endpoint="/predict", encoding="multipart/form-data",
                   file_field="images", maximum_files=10, maximum_file_bytes=5*1024*1024,
                   classes=metadata["classes"], model_version=metadata["version"])
@app.errorhandler(RequestEntityTooLarge)
def too_large(error):
    return jsonify(error="La solicitud supera 30 MB. Divide las imágenes en lotes más pequeños."), 413
@app.post("/predict")
def predict():
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="Selecciona al menos una imagen en el campo images."), 400
    if len(files) > 10:
        return jsonify(error="Se permiten hasta 10 imágenes por solicitud."), 400
    results = []
    for uploaded in files:
        name = Path(uploaded.filename or "imagen").name
        result = {"file": name, "model_version": metadata["version"]}
        try:
            content = uploaded.read(5 * 1024 * 1024 + 1)
            if len(content) > 5 * 1024 * 1024:
                raise ValueError("La imagen supera 5 MB.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as im:
                    if im.format not in ("JPEG", "PNG"):
                        raise ValueError("Usa imágenes JPG o PNG.")
                    x = preprocess(im)
            probabilities = session.run(None, {"images": x})[0][0]
            index = int(np.argmax(probabilities))
            result.update(status="ok", predicted_class=metadata["classes"][index],
                          label=metadata["display_names"][index],
                          score=float(probabilities[index]),
                          scores={name: float(p) for name,p in zip(metadata["display_names"], probabilities)})
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            result.update(status="error", error=str(error) if isinstance(error, ValueError) else "El archivo no es una imagen JPG o PNG válida.")
        results.append(result)
    return jsonify(results=results)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
