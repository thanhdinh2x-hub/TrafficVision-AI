import csv
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

run_dir = Path("runs/detect/train-3")

csv_path = run_dir / "results.csv"
log_dir = run_dir / "tensorboard_import"

writer = SummaryWriter(log_dir=str(log_dir))

with open(csv_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader):
        row = {k.strip(): v for k, v in row.items()}

        epoch = int(float(row.get("epoch", i)))

        for key, value in row.items():
            if key == "epoch" or not value.strip():
                continue

            try:
                writer.add_scalar(key, float(value), epoch)
            except ValueError:
                pass

writer.close()

print("TensorBoard logs saved to:", log_dir)