
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PROJECT PATH
# ==========================================

# Tu dong lay thu muc chua file Python
ROOT = Path(__file__).resolve().parent

# Duong dan dataset
DATASET_DIR = ROOT / "licence plate" / "train"

# Ten file anh va label
FILE_NAME = "mn_0001-_1_jpg.rf.3b6896ab4402d1fa7e933044732a8301"

image_path = DATASET_DIR / "images" / f"{FILE_NAME}.jpg"
label_path = DATASET_DIR / "labels" / f"{FILE_NAME}.txt"

# ==========================================
# 2. READ IMAGE AND LABEL
# ==========================================

img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(image_path)

if not label_path.is_file():
    raise FileNotFoundError(label_path)

H, W = img.shape[:2]

with open(label_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print("Image size:", W, H)
print("Label:")
print("\n".join(lines))


# ==========================================
# 3. CONVERT BOUNDING BOX
# ==========================================

def convert_box(coords, fmt):
    a, b, c, d = coords

    if fmt == "YOLO normalized":
        xc, yc, w, h = a * W, b * H, c * W, d * H
        return xc-w/2, yc-h/2, xc+w/2, yc+h/2

    if fmt == "XYXY normalized":
        return a*W, b*H, c*W, d*H

    if fmt == "XYXY pixels":
        return a, b, c, d

    if fmt == "XYWH pixels":
        return a, b, a+c, b+d


formats = [
    "YOLO normalized",
    "XYXY normalized",
    "XYXY pixels",
    "XYWH pixels"
]

# ==========================================
# 4. VISUALIZE BOUNDING BOX
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for ax, fmt in zip(axes.flat, formats):
    canvas = img.copy()

    for line in lines:
        values = list(map(float, line.replace(",", " ").split()))

        # Ho tro: class + 4 toa do hoac chi 4 toa do
        if len(values) == 5:
            cls = int(values[0])
            coords = values[1:]
        elif len(values) == 4:
            cls = None
            coords = values
        else:
            print("Bo qua label khong co 4 toa do:", line)
            continue

        x1, y1, x2, y2 = convert_box(coords, fmt)

        cv2.rectangle(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        if cls is not None:
            cv2.putText(
                canvas,
                str(cls),
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(fmt)
    ax.axis("off")

plt.tight_layout()
plt.show()
