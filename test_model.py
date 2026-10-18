from ultralytics import YOLO

# Load model đã train
model = YOLO("runs/detect/train-3/weights/best.pt")

# Test với ảnh mới
results = model.predict(
    source="test1.PNG",
    conf=0.25,
    imgsz=640,
    save=True
)

# Hiển thị kết quả
results[0].show()