"""Automatic Gold Packet Counter - Flask application."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "model" / "best.pt"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_RETENTION_DAYS = 7

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

model = None


def allowed_file(filename):
    """Return True when the filename has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_expired_files(folder):
    """Remove stored files older than the configured retention period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=UPLOAD_RETENTION_DAYS)
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
        if modified_at < cutoff:
            file_path.unlink(missing_ok=True)


def get_model():
    """Load the trained model once, only when the first prediction is requested."""
    global model
    if model is None:
        if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
            raise FileNotFoundError("Place the trained model at model/best.pt before detecting packets.")
        model = YOLO(str(MODEL_PATH))
    return model


def make_result_image(image_path, result, output_path):
    """Draw YOLO detections with OpenCV and save the annotated image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded image could not be read.")

    for box in result.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (55, 174, 103), 3)

    if not cv2.imwrite(str(output_path), image):
        raise ValueError("The detected image could not be saved.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive an image, run YOLO, and return the result data as JSON."""
    remove_expired_files(UPLOAD_FOLDER)
    remove_expired_files(RESULT_FOLDER)

    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify(success=False, message="Please choose an image to upload."), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify(success=False, message="Only JPG, JPEG, and PNG images are supported."), 400

    safe_name = secure_filename(uploaded_file.filename)
    unique_name = f"{uuid4().hex}_{safe_name}"
    image_path = UPLOAD_FOLDER / unique_name
    uploaded_file.save(image_path)
    result_name = f"detected_{Path(unique_name).stem}.jpg"
    result_path = RESULT_FOLDER / result_name

    try:
        detection_results = get_model().predict(
            source=str(image_path),
            conf=0.32,
            iou=0.7,
            imgsz=832,
            save=False,
        )
        detection = detection_results[0]
        count = len(detection.boxes)
        make_result_image(image_path, detection, result_path)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        image_path.unlink(missing_ok=True)
        return jsonify(success=False, message=str(error)), 500

    return jsonify(
        success=True,
        count=count,
        original_url=url_for("uploaded_file", filename=unique_name),
        detected_url=url_for("static", filename=f"results/{result_name}"),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve the original image for the result comparison."""
    from flask import send_from_directory

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(success=False, message="The image must be smaller than 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
