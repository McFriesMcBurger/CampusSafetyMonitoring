from utils import load_model
from config import DATA_CONFIG
import os

def main():
    model = load_model()

    metrics = model.val(
        data=DATA_CONFIG,
        split="val",
        save_json=True,
        name="evaluation"
    )

    output_dir = metrics.save_dir

    print("\n===== EVALUATION =====")
    print(f"P: {metrics.box.mp:.4f}")
    print(f"R: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    metrics_path = os.path.join(output_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write(f"P: {metrics.box.mp:.4f}\n")
        f.write(f"R: {metrics.box.mr:.4f}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")

    print(f"\nSaved everything to: {output_dir}")


if __name__ == "__main__":
    main()