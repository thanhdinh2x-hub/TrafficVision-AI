
import cv2
import pytesseract
import re
from pathlib import Path

# 1. Cau hinh Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

ROOT = Path(__file__).resolve().parent

# CHI CAN DOI DUONG DAN ANH O DAY
IMAGE_PATH = ROOT / "picture" / "plate_car1.PNG"

OUTPUT_PATH = ROOT / "ocr_result.jpg"


def ocr_text(img, psm):
    # Chuyen anh sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phong to 4 lan
    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Them vien trang
    gray = cv2.copyMakeBorder(
        gray, 30, 30, 30, 30,
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


def main():

    # 2. Doc anh
    plate = cv2.imread(str(IMAGE_PATH))

    if plate is None:
        raise FileNotFoundError(IMAGE_PATH)

    h, w = plate.shape[:2]
    ratio = w / h

    print("Image size:", w, "x", h)
    print("Ratio:", round(ratio, 2))

    # 3. Xac dinh bien 1 dong hay 2 dong
    if ratio >= 2.5:

        plate_type = "ONE_LINE"

        # Bien 1 dong: OCR toan bo anh
        plate_text = ocr_text(plate, psm=7)

        plate_result = {
            "type": plate_type,
            "lines": [plate_text],
            "text": plate_text
        }

    else:

        plate_type = "TWO_LINES"

        # Bien 2 dong: cat anh lam 2
        split_y = h // 2

        top = plate[:split_y, :]
        bottom = plate[split_y:, :]

        # OCR tung dong bang PSM 8
        text_top = ocr_text(top, psm=8)
        text_bottom = ocr_text(bottom, psm=8)

        # Ghep 2 dong bang dau "-"
        plate_text = text_top + "-" + text_bottom

        plate_result = {
            "type": plate_type,
            "lines": [text_top, text_bottom],
            "text": plate_text
        }

        print("Top line:", text_top)
        print("Bottom line:", text_bottom)

    # 4. In ket qua dictionary
    print("=" * 40)
    print("OCR Result:", plate_result)
    print("Plate type:", plate_result["type"])
    print("License plate:", plate_result["text"])
    print("=" * 40)

    # 5. Tao anh hien thi ket qua
    display = cv2.copyMakeBorder(
        plate,
        20, 110, 20, 20,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    # Hien thi loai bien
    cv2.putText(
        display,
        "Plate Type: " + plate_result["type"],
        (20, h + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Hien thi text OCR
    cv2.putText(
        display,
        "License Plate: " + (plate_text or "UNKNOWN"),
        (20, h + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # 6. Luu ket qua
    cv2.imwrite(str(OUTPUT_PATH), display)

    # 7. Show anh
    cv2.imshow("License Plate OCR", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return plate_result


if __name__ == "__main__":
    result = main()
    print("Final dictionary:", result)
