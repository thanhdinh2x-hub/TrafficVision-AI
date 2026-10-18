from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model(
    "https://ultralytics.com/images/bus.jpg",
    classes=[2, 5, 7]
)

results[0].show()
result = results[0]

for box in result.boxes:
    xyxy = box.xyxy[0].tolist()
    confidence = box.conf[0].item()
    class_id = int(box.cls[0].item())

    class_name = result.names[class_id]

    print("Bounding box:", xyxy)
    print("Confidence:", confidence)
    print("Class:", class_name)
print(results)