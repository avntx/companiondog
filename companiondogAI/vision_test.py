import os
import sys
import cv2
from ultralytics import YOLO

# Path to your image
IMAGE_PATH = "dog_frame.jpeg"   # Make sure the name matches exactly
RESULTS_DIR = "results"

def main():
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("Make sure dog_frame.jpeg is in the same folder as vision_test.py")
        sys.exit(1)

    # Create results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("✅ Loading YOLOv8n tiny model...")
    model = YOLO("yolov8n.pt")  # downloads weights automatically

    print(f"🔍 Running detection on {IMAGE_PATH} ...")
    results = model(IMAGE_PATH)
    r = results[0]

    # Get annotated image (NumPy array in BGR)
    annotated = r.plot()

    # Save detection result
    out_path = os.path.join(RESULTS_DIR, "dog_detection.png")
    cv2.imwrite(out_path, annotated)

    # Extract detected class names
    names = model.names
    detected_classes = [names[int(cls)] for cls in r.boxes.cls]

    print("📌 Detected classes:", detected_classes)
    print(f"💾 Saved annotated detection image at: {out_path}")

if __name__ == "__main__":
    main()



def run_vision(image_path: str) -> dict:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model(image_path)
    r = results[0]
    names = model.names

    detected = []
    dog_candidates = []
    car_conf = 0.0

    # image size for area ratio
    h, w = r.orig_shape
    img_area = w * h

    # thresholds
    DOG_STRONG = 0.70
    DOG_GREY = 0.60
    CAR_VETO = 0.60
    MIN_AREA_RATIO = 0.03  # dog box must cover at least 3% of image

    if r.boxes is not None and len(r.boxes) > 0:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            label = names[int(cls_id)]
            conf = float(conf)

            x1, y1, x2, y2 = [float(v) for v in box]
            area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio = area / img_area

            detected.append((label, round(conf, 3), round(area_ratio, 3)))

            if label == "car":
                car_conf = max(car_conf, conf)

            if label == "dog":
                dog_candidates.append((conf, area_ratio))

    dog_conf = max([c for c, _ in dog_candidates], default=0.0)
    best_area = max([a for _, a in dog_candidates], default=0.0)

    # --- Decision logic ---
    dog_detected = False

    # strong detection
    if dog_conf >= DOG_STRONG and best_area >= MIN_AREA_RATIO:
        dog_detected = True

    # grey zone: allow slightly lower confidence if box is big enough
    elif dog_conf >= DOG_GREY and best_area >= 0.06:  # require larger box in grey zone
        dog_detected = True

    # veto for strong car scene (reduces car->dog false positives)
    if car_conf >= CAR_VETO and dog_conf < 0.85:
        dog_detected = False

    return {
        "dog_detected": dog_detected,
        "dog_conf": round(dog_conf, 3),
        "car_conf": round(car_conf, 3),
        "best_area_ratio": round(best_area, 3),
        "detected": detected
    }


