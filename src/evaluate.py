from utils import load_model
from config import DATA_CONFIG
import os

def evaluate_split(model, split_name):
    metrics = model.val(
        data=DATA_CONFIG,
        split=split_name,
        save_json=True,
        name=f"evaluation_{split_name}"   # separate folders
    )

    print(f"\n===== {split_name.upper()} METRICS =====")
    print(f"P: {metrics.box.mp:.4f}")
    print(f"R: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    output_dir = metrics.save_dir
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"{split_name.upper()} METRICS\n")
        f.write(f"P: {metrics.box.mp:.4f}\n")
        f.write(f"R: {metrics.box.mr:.4f}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")

    return metrics


def main():
    model = load_model()

    val_metrics = evaluate_split(model, "val")
    test_metrics = evaluate_split(model, "test")


if __name__ == "__main__":
    main()
