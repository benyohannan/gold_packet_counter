from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "model/best.pt"

IMAGE_DIR = Path(
    r"F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\images"
)

LABEL_DIR = Path(
    r"F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\labels"
)

CONF = 0.35
IOU = 0.7

model = YOLO(MODEL_PATH)

results_list = []

for image_path in sorted(IMAGE_DIR.glob("*")):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    gt_count = len(label_path.read_text().splitlines())

    results = model.predict(
        source=str(image_path),
        conf=CONF,
        iou=IOU,
        save=False,
        verbose=False
    )

    pred_count = len(results[0].boxes)

    error = pred_count - gt_count
    abs_error = abs(error)

    results_list.append(
        (abs_error, image_path.name, gt_count, pred_count, error)
    )

results_list.sort(reverse=True)

print("\n" + "=" * 80)
print("TOP 5 WORST TEST IMAGES")
print("=" * 80)

for i, (abs_error, name, gt, pred, error) in enumerate(results_list[:5], 1):
    print(f"\n{i}. {name}")
    print(f"   Ground truth : {gt}")
    print(f"   Prediction   : {pred}")
    print(f"   Error        : {error:+d}")
    print(f"   Absolute err : {abs_error}")