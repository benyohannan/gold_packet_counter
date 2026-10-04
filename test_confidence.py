from ultralytics import YOLO

model = YOLO("model/best.pt")

image_path = r"F:\mini-project\Gold Packet Counter.v3-gold-packet-dataset-final.yolov11\test\images\PHOTO-2026-09-26-16-16-13_jpg.rf.3e29635dc0a437b9d89a5e5d7e6aa133.jpg"
for conf in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    results = model.predict(
        source=image_path,
        conf=conf,
        save=False,
        verbose=False
    )

    count = len(results[0].boxes)

    print(f"Confidence {conf:.2f} -> {count} detections")