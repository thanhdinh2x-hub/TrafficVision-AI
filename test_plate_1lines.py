
import cv2
import pytesseract
import re
from pathlib import Path

# Duong dan Tesseract OCR Engine
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Thu muc chua file Python
ROOT = Path(__file__).resolve().parent

# Doi sang plate_motorbike.jpg khi test xe may
IMAGE_PATH = ROOT / "picture"/ "plate_car2.PNG"


def main():

    # 1. Doc anh bien so
    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        raise FileNotFoundError(IMAGE_PATH)

    print("Image shape:", img.shape)

    # 2. Chuyen anh sang grayscale
    gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )

    # 3. Phong to anh 3 lan
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # 4. Them vien trang
    gray = cv2.copyMakeBorder(
        gray, 15, 15, 15, 15,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # 5. Nhi phan hoa
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 6. Cau hinh OCR
    config = (
        "--oem 1 --psm 7 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    # 7. OCR
    raw_text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config=config
    )

    # 8. Lam sach text
    plate_text = re.sub(
        r"[^A-Z0-9]",
        "",
        raw_text.upper()
    )

    print("Raw OCR:", repr(raw_text))
    print("License Plate:", plate_text)

    # 9. Hien thi anh
    cv2.imshow("Original Plate", img)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Binary", binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
