from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "model/best.pt"

IMAGE_DIR = Path(
    r"F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\images"
)

LABEL_DIR = Path(
    r"F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\labels"
)

CONF_VALUES = [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40]

IOU = 0.7
IMG_SIZE = 832

model = YOLO(MODEL_PATH)

for CONF in CONF_VALUES:

    total_gt = 0
    total_pred = 0
    total_abs_error = 0
    exact_count = 0
    image_count = 0

    print("\n" + "=" * 60)
    print(f"CONF = {CONF} | IOU = {IOU} | IMAGE SIZE = {IMG_SIZE}")
    print("=" * 60)

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
            imgsz=IMG_SIZE,
            save=False,
            verbose=False
        )

        pred_count = len(results[0].boxes)

        error = pred_count - gt_count

        total_gt += gt_count
        total_pred += pred_count
        total_abs_error += abs(error)

        if pred_count == gt_count:
            exact_count += 1

        image_count += 1

    mae = total_abs_error / image_count
    exact_percentage = exact_count / image_count * 100

    print(f"Images evaluated       : {image_count}")
    print(f"Total ground truth     : {total_gt}")
    print(f"Total predictions      : {total_pred}")
    print(f"Mean Absolute Error    : {mae:.2f} packets/image")
    print(f"Exact-count images     : {exact_count}/{image_count}")
    print(f"Exact-count percentage : {exact_percentage:.2f}%")