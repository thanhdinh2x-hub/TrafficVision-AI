from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    # Load pretrained model
    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(ROOT / "licence plate" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=False,
        resume=False
    )