from config import *
from train import train_model
from evaluate import evaluate_model

START_FROM = "category_c"

EXPERIMENTS = {

    "baseline": {},

    # Category A
    "category_a": {
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
    },

    # Category B
    "category_b": {
        "degrees": DEGREES,
        "mixup": MIXUP,
    },

    # Category C
    "category_c": {
        "cos_lr": COS_LR,
    },

    # Final combined model
    "final_combined": {
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,

        "degrees": DEGREES,
        "mixup": MIXUP,

        "cos_lr": COS_LR,
    }
}


def print_summary(results):

    print("\n\n============================")
    print("FINAL COMPARISON")
    print("============================")

    for experiment_name, metrics in results.items():

        test_metrics = metrics["test"]["overall"]

        print(f"\n{experiment_name.upper()}")
        print(f"mAP50:     {test_metrics['map50']:.4f}")
        print(f"mAP50-95:  {test_metrics['map5095']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")


def main():

    all_results = {}

    start_running = False

    for experiment_name, experiment_args in EXPERIMENTS.items():

        if experiment_name == START_FROM:
            start_running = True

        if not start_running:
            continue

        print("\n\n########################################")
        print(f"RUNNING EXPERIMENT: {experiment_name}")
        print("########################################")

        best_model_path = train_model(
            run_name=experiment_name,
            extra_args=experiment_args
        )

        metrics = evaluate_model(
            model_path=best_model_path,
            run_name=experiment_name
        )

        all_results[experiment_name] = metrics

    print_summary(all_results)


if __name__ == "__main__":
    main()