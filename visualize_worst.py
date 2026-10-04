from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "model/best.pt"

IMAGE_DIR = Path(
    r"F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\images"
)

OUTPUT_DIR = Path("worst_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CONF = 0.35
IOU = 0.7

WORST_IMAGES = [
    "IMG-20260828-WA0055_jpg.rf.a8690c5f3b1b2daa308f43688d31f5e8.jpg",
    "PHOTO-2026-09-26-16-19-08-3_jpg.rf.242a8df64cbedae38446ab172ee986a2.jpg",
    "PHOTO-2026-09-26-16-20-50-2_jpg.rf.5484ced168f2dd6e93687bdefe079e93.jpg",
    "PHOTO-2026-09-26-16-16-11-2_jpg.rf.8cada221ff46e5bfcac467f78aa47ef8.jpg",
    "PHOTO-2026-09-26-16-20-17-2_jpg.rf.3562bd97644c866bd524ec9ab0b692e5.jpg",
]

model = YOLO(MODEL_PATH)

for filename in WORST_IMAGES:

    image_path = IMAGE_DIR / filename

    results = model.predict(
        source=str(image_path),
        conf=CONF,
        iou=IOU,
        save=False,
        verbose=False
    )

    result = results[0]

    output_path = OUTPUT_DIR / filename

    result.save(filename=str(output_path))

    print(f"Saved: {output_path}")
    print(f"Detections: {len(result.boxes)}")

print("\nDone.")