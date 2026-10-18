
from ultralytics import YOLO, settings
from tensorboard import program
from pathlib import Path
import webbrowser


def start_tensorboard(log_dir):
    """Khoi dong TensorBoard tu dong."""

    tb = program.TensorBoard()

    tb.configure(
        argv=[
            "tensorboard",
            "--logdir", str(log_dir),
            "--port", "6006",
            "--host", "127.0.0.1"
        ]
    )

    url = tb.launch()

    print(f"TensorBoard running at: {url}")

    # Tu dong mo Chrome/trinh duyet
    webbrowser.open(url)

    return tb


def main():

    # 1. Bat TensorBoard logging
    settings.update({"tensorboard": True})

    # 2. Duong dan project
    root = Path(__file__).resolve().parent

    train_dir = root / "runs" / "detect" / "train-3"
    tensorboard_dir = root / "runs" / "detect" / "train-3" / "tensorboard_import"

    # 3. Duong dan checkpoint
    weight_path = train_dir / "weights" / "last.pt"

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay weight: {weight_path}"
        )

    # 4. Khoi dong TensorBoard
    tb_server = start_tensorboard(tensorboard_dir)

    # 5. Load YOLO26 da train
    model = YOLO(str(weight_path))

    # 6. Resume training
    results = model.train(
        resume=True,
        workers=4
    )

    print("Training completed!")
    print("Weights folder:", train_dir / "weights")

    # Giu TensorBoard mo sau khi train xong
    input("Press Enter to close TensorBoard...")


if __name__ == "__main__":
    main()
