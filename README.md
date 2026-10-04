# Automatic Gold Packet Counter

An offline Flask web application for detecting and counting gold packets in one uploaded image. It uses a trained Ultralytics YOLOv11 Nano model, OpenCV, and a responsive HTML/CSS/JavaScript interface.

## Project Structure

```text
Gold_Packet_Counter/
|-- app.py
|-- model/
|   `-- best.pt
|-- templates/
|   `-- index.html
|-- static/
|   |-- css/style.css
|   |-- js/script.js
|   `-- results/
|-- uploads/
|-- requirements.txt
`-- README.md
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your trained YOLOv11 weights file at `model/best.pt`.

4. Start the Flask application:

   ```bash
   python app.py
   ```

5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000/) in your browser.

## How It Works

The browser previews the selected image locally. Clicking **Detect packets** sends it to `/predict`. Flask saves the upload securely, loads `model/best.pt`, and runs:

```python
results = model.predict(source=image_path, conf=0.5, save=False)
count = len(results[0].boxes)
```

OpenCV draws the bounding boxes and saves the annotated image under `static/results/`. No Roboflow API or cloud service is used.

Before counting, the application runs an image gate. A clear image must contain at least
three detections at confidence `0.45`. Difficult views, such as dark or overlapping
packets, may pass with at least two detections at confidence `0.35`. If neither check
finds enough evidence, the request is rejected. Images that pass the gate are counted
using the normal detection confidence of `0.32`.

This gate reduces false counts from unrelated images, but it is not a replacement for a
dedicated image classifier. For production accuracy, add negative training images
(books, people, empty scenes, and other common uploads) and train a separate
`gold_packet`/`not_gold_packet` classifier or a two-class detector. That classifier should
be used as the gateway before this packet detector.
# python -c "from pathlib import Path; p=Path(r'F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\labels\PHOTO-2026-09-26-16-20-15-2_jpg.rf.1ba9017dbd9df12a50f27242fb7edfb4.txt'); print('Ground-truth annotations:', len(p.read_text().splitlines()))"  