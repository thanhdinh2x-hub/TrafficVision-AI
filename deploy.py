import argparse
import cv2
import pytesseract
import re
import os
from pathlib import Path
from ultralytics import YOLO


# ============================================
# 1. CONFIG
# ============================================

ROOT = Path(__file__).resolve().parent

VEHICLE_MODEL_PATH = ROOT / "yolo26n.pt"

PLATE_MODEL_PATH = (
    ROOT / "runs" / "detect"
    / "train-3" / "weights" / "best.pt"
)


if os.name == "nt":
    # Chạy trên Windows
    TESSERACT_PATH = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    # Chạy trong Docker Linux
    TESSERACT_PATH = "/usr/bin/tesseract"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# COCO class IDs
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

VEHICLE_CONF = 0.35
PLATE_CONF = 0.20

# Nguong phan biet bien 1 dong / 2 dong
RATIO_THRESHOLD = 2.5


# ============================================
# 2. OCR - TESSERACT
# ============================================

def ocr_text(img, psm):

    gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.copyMakeBorder(
        gray,
        30, 30, 30, 30,
        cv2.BORDER_CONSTANT,
        value=255
    )

    config = (
        f"--oem 1 --psm {psm} "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    raw = pytesseract.image_to_string(
        gray,
        lang="eng",
        config=config
    )

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        raw.upper()
    )

    return text


def read_plate(plate_img):

    h, w = plate_img.shape[:2]

    if h == 0 or w == 0:
        return {
            "type": "UNKNOWN",
            "lines": [],
            "text": ""
        }

    ratio = w / h

    # Bien 1 dong
    if ratio >= RATIO_THRESHOLD:

        plate_text = ocr_text(
            plate_img,
            psm=7
        )

        return {
            "type": "ONE_LINE",
            "lines": [plate_text],
            "text": plate_text
        }

    # Bien 2 dong
    split_y = h // 2

    top = plate_img[:split_y, :]
    bottom = plate_img[split_y:, :]

    text_top = ocr_text(top, psm=8)
    text_bottom = ocr_text(bottom, psm=8)

    return {
        "type": "TWO_LINES",
        "lines": [text_top, text_bottom],
        "text": "-".join(
            x for x in [text_top, text_bottom] if x
        )
    }


# ============================================
# 3. HELPER - CROP ANH
# ============================================

def get_crop(img, box, padding=0):

    h, w = img.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img[y1:y2, x1:x2].copy()

    return crop, (x1, y1, x2, y2)


# ============================================
# 4. DEPLOY PIPELINE
# ============================================

def deploy(image_path):

    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not VEHICLE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Khong tim thay model detect xe: {VEHICLE_MODEL_PATH}"
        )

    if not PLATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay model bien so: {PLATE_MODEL_PATH}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Khong doc duoc anh: {image_path}"
        )

    # Thu muc output rieng cho moi anh
    output_dir = ROOT / "outputs" / image_path.stem

    vehicle_dir = output_dir / "vehicles"
    plate_dir = output_dir / "plates"

    vehicle_dir.mkdir(parents=True, exist_ok=True)
    plate_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")

    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    plate_model = YOLO(str(PLATE_MODEL_PATH))

    annotated = image.copy()
    records = []

    # ========================================
    # MODEL 1 - DETECT VEHICLES
    # ========================================

    print("\n[MODEL 1] Detecting vehicles...")

    vehicle_result = vehicle_model.predict(
        source=image,
        conf=VEHICLE_CONF,
        classes=list(VEHICLE_CLASSES.keys()),
        imgsz=1280,
        verbose=False
    )[0]
    print("Vehicle model device:", vehicle_result.boxes.data.device)
    vehicle_boxes = vehicle_result.boxes

    print(f"Detected vehicles: {len(vehicle_boxes)}")

    # ========================================
    # LAP QUA TUNG XE
    # ========================================

    for vehicle_idx, vehicle_box in enumerate(
        vehicle_boxes, start=1
    ):

        vehicle_class = int(
            vehicle_box.cls[0].item()
        )

        vehicle_type = VEHICLE_CLASSES.get(
            vehicle_class, "unknown"
        )

        vehicle_conf = float(
            vehicle_box.conf[0].item()
        )

        box = (
            vehicle_box.xyxy[0].cpu().numpy()
        )

        # Them padding quanh xe
        vehicle_crop, vehicle_coords = get_crop(
            image, box, padding=8
        )

        if vehicle_crop is None:
            continue

        vx1, vy1, vx2, vy2 = vehicle_coords

        vehicle_id = f"vehicle_{vehicle_idx:03d}"

        cv2.imwrite(
            str(vehicle_dir / f"{vehicle_id}.jpg"),
            vehicle_crop
        )

        print(
            f"\n[{vehicle_id}] "
            f"{vehicle_type} ({vehicle_conf:.2f})"
        )

        # Ve bbox xe
        cv2.rectangle(
            annotated,
            (vx1, vy1),
            (vx2, vy2),
            (0, 255, 0),
            2
        )

        # ====================================
        # MODEL 2 - DETECT LICENSE PLATES
        # ====================================

        plate_result = plate_model.predict(
            source=vehicle_crop,
            conf=PLATE_CONF,
            imgsz=960,
            verbose=False
        )[0]
        print("Plate model device:", plate_result.boxes.data.device)
        plate_boxes = plate_result.boxes

        # Neu khong tim thay bien
        if len(plate_boxes) == 0:

            print("  Plate: NOT DETECTED")

            records.append({
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle_type,
                "vehicle_conf": vehicle_conf,
                "plate_type": "UNKNOWN",
                "plate_text": "",
                "lines": [],
                "status": "PLATE_NOT_DETECTED"
            })

            cv2.putText(
                annotated,
                f"{vehicle_id}: NO PLATE",
                (vx1, max(20, vy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            continue

        # Lay bien co confidence cao nhat
        best_plate_box = max(
            plate_boxes,
            key=lambda b: float(b.conf[0].item())
        )

        plate_box = (
            best_plate_box.xyxy[0].cpu().numpy()
        )

        plate_conf = float(
            best_plate_box.conf[0].item()
        )

        # Crop bien tu anh xe
        plate_crop, plate_coords = get_crop(
            vehicle_crop,
            plate_box,
            padding=0
        )

        if plate_crop is None:
            continue

        px1, py1, px2, py2 = plate_coords

        plate_path = (
            plate_dir / f"{vehicle_id}_plate.jpg"
        )

        cv2.imwrite(
            str(plate_path),
            plate_crop
        )

        # ====================================
        # MODEL 3 - OCR
        # ====================================

        ocr_result = read_plate(
            plate_crop
        )

        plate_text = ocr_result["text"]

        print(
            f"  Plate confidence: {plate_conf:.2f}"
        )
        print(
            f"  Plate type: {ocr_result['type']}"
        )
        print(
            f"  Lines: {ocr_result['lines']}"
        )
        print(
            f"  OCR: {plate_text or 'UNREADABLE'}"
        )

        # Ghi du lieu vao records
        records.append({
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "vehicle_conf": vehicle_conf,
            "plate_type": ocr_result["type"],
            "plate_text": plate_text,
            "lines": ocr_result["lines"],
            "status": (
                "OCR_OK" if plate_text
                else "OCR_EMPTY"
            )
        })

        # ====================================
        # VE KET QUA LEN ANH GOC
        # ====================================

        # Chuyen toa do bien ve anh goc
        gx1 = vx1 + px1
        gy1 = vy1 + py1
        gx2 = vx1 + px2
        gy2 = vy1 + py2

        cv2.rectangle(
            annotated,
            (gx1, gy1),
            (gx2, gy2),
            (0, 0, 255),
            2
        )

        label = (
            f"{vehicle_id} {vehicle_type}: "
            f"{plate_text or 'UNREADABLE'}"
        )

        text_y = max(
            20, vy1 - 10
        )

        cv2.putText(
            annotated,
            label,
            (vx1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================
    # 5. XUAT FILE TEXT
    # ========================================

    txt_path = output_dir / "result.txt"

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(f"Image: {image_path.name}\n")
        f.write(f"Total vehicles: {len(records)}\n")
        f.write("=" * 50 + "\n\n")

        for record in records:

            f.write(
                f"Vehicle ID: {record['vehicle_id']}\n"
            )

            f.write(
                f"Vehicle type: {record['vehicle_type']}\n"
            )

            f.write(
                f"Vehicle confidence: "
                f"{record['vehicle_conf']:.3f}\n"
            )

            f.write(
                f"Plate type: {record['plate_type']}\n"
            )

            f.write(
                f"Plate text: "
                f"{record['plate_text'] or 'UNKNOWN'}\n"
            )

            f.write(
                f"Lines: {record['lines']}\n"
            )

            f.write(
                f"Status: {record['status']}\n"
            )

            f.write("-" * 50 + "\n\n")

    # ========================================
    # 6. LUU ANH ANNOTATED
    # ========================================

    annotated_path = (
        output_dir / "annotated.jpg"
    )

    cv2.imwrite(
        str(annotated_path),
        annotated
    )

    print("\n" + "=" * 50)
    print(f"Total vehicles: {len(records)}")
    print(f"TXT saved: {txt_path}")
    print(f"Image saved: {annotated_path}")
    print("=" * 50)

    # Hien thi anh thu nho neu qua lon
    display = annotated.copy()

    h, w = display.shape[:2]

    if w > 1200:
        scale = 1200 / w

        display = cv2.resize(
            display,
            None,
            fx=scale,
            fy=scale
        )

    cv2.imshow("Traffic ALPR Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return records


# ============================================
# 7. MAIN
# ============================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default=str(
            ROOT / "picture" / "gt2.png"
        ),
        help="Path to traffic image"
    )

    args = parser.parse_args()

    deploy(args.image)
