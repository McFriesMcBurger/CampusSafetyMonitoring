import os
from ultralytics import YOLO
from config import DATA_CONFIG


def evaluate_split(model, split_name, run_name):
    metrics = model.val(
        data=DATA_CONFIG,
        split=split_name,
        save_json=True,
        name=f"{run_name}_{split_name}"
    )

    print(f"\n===== {run_name.upper()} | {split_name.upper()} =====")
    print(f"P: {metrics.box.mp:.4f}")
    print(f"R: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n===== PER-CLASS METRICS =====")

    class_names = metrics.names

    per_class_results = {}

    for class_id, class_name in class_names.items():

        precision = metrics.box.p[class_id]
        recall = metrics.box.r[class_id]
        ap50 = metrics.box.ap50[class_id]
        ap = metrics.box.ap[class_id]

        per_class_results[class_name] = {
            "precision": precision,
            "recall": recall,
            "map50": ap50,
            "map5095": ap,
        }

        print(f"\nClass: {class_name}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  mAP50:     {ap50:.4f}")
        print(f"  mAP50-95:  {ap:.4f}")

    output_dir = metrics.save_dir

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:

        f.write(f"{run_name.upper()} - {split_name.upper()}\n")

        f.write(f"P: {metrics.box.mp:.4f}\n")
        f.write(f"R: {metrics.box.mr:.4f}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")

        f.write("\n===== PER-CLASS METRICS =====\n")

        for class_name, class_metrics in per_class_results.items():

            f.write(f"\nClass: {class_name}\n")
            f.write(
                f"  Precision: {class_metrics['precision']:.4f}\n"
            )
            f.write(
                f"  Recall:    {class_metrics['recall']:.4f}\n"
            )
            f.write(
                f"  mAP50:     {class_metrics['map50']:.4f}\n"
            )
            f.write(
                f"  mAP50-95:  {class_metrics['map5095']:.4f}\n"
            )

    return {
        "overall": {
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "map50": metrics.box.map50,
            "map5095": metrics.box.map,
        },
        "per_class": per_class_results
    }


def evaluate_model(model_path, run_name):

    model = YOLO(model_path)

    val_metrics = evaluate_split(model, "val", run_name)
    test_metrics = evaluate_split(model, "test", run_name)

    return {
        "val": val_metrics,
        "test": test_metrics,
    }