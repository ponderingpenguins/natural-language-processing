"""
Label-Noise Sensitivity Analysis: Scaling Laws for BERT and LSTM Models

Trains both models with fixed best hyperparameters on different data fractions
(25%, 50%, 100%) to measure scaling behavior and sensitivity to training data size.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data
from utils.training import compute_metrics, set_seed

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE

# Data fractions to test
DATA_FRACTIONS = [0.25, 0.5, 1.0]
OUTPUT_DIR = "scaling_law_results"
RUNS_PER_FRACTION = 3  # Multiple runs to get error bars


def load_best_config(model_type: str) -> dict:
    """Load the best config from hyperparameter search output."""
    config_path = f"{model_type}_output/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found at {config_path}. Run hyperparameter tuning first."
        )
    with open(config_path) as f:
        return json.load(f)


def train_model(
    model,
    train_data,
    eval_data,
    config: dict,
    model_type: str,
    fraction: float,
    run_id: int,
):
    """Train a model on a data subset with fixed hyperparameters."""
    output_dir = os.path.join(
        OUTPUT_DIR, f"{model_type}_frac{fraction:.2f}_run{run_id}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        num_train_epochs=config.get("num_train_epochs", 15),
        warmup_steps=config.get("warmup_steps", 500),
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 3)
            ),
        ],
    )

    logger.info(
        "Training %s on %.0f%% data (run %d/%d)",
        model_type,
        fraction * 100,
        run_id + 1,
        RUNS_PER_FRACTION,
    )
    trainer.train()

    # Save the best model checkpoint path for evaluation
    return trainer.state.best_model_checkpoint


def evaluate_on_test(model, test_data, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    batch_size = 64
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i : i + batch_size]
        labels = torch.tensor(batch["labels"]).to(device)

        model_inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
        }
        if "attention_mask" in batch:
            model_inputs["attention_mask"] = torch.tensor(batch["attention_mask"]).to(
                device
            )
        if "token_type_ids" in batch:
            model_inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(
                device
            )
        if "lengths" in batch:
            model_inputs["lengths"] = torch.tensor(batch["lengths"]).to(device)

        with torch.no_grad():
            outputs = model(**model_inputs, labels=labels)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {"accuracy": acc, "f1": f1}


def run_scaling_law_experiment(model_type: str):
    """Run the full scaling law experiment for a model type."""
    logger.info("Starting scaling law experiment for %s", model_type)

    # Load config
    best_config = load_best_config(model_type)
    logger.info(
        "Loaded best config: lr=%f, batch_size=%d",
        best_config.get("learning_rate"),
        best_config.get("per_device_train_batch_size"),
    )

    # Load dataset
    dataset_cfg = OmegaConf.load("configs/dataset.yaml")
    data = dataset_prep(dataset_cfg.dataset)

    # Initialize model to get tokenizer
    if model_type == "bert":
        model_template = BertClassifier(OmegaConf.create(best_config), device=DEVICE)
    else:
        model_template = LSTMClassifier(OmegaConf.create(best_config), device=DEVICE)

    # Tokenize data
    tokenized_data_path = f"./{dataset_cfg.dataset.hf_dataset}_tokenized"
    data = try_load_tokenized_data(tokenized_data_path, data, model_template.tokenize)

    # Clean up
    data = DatasetDict(
        {
            split: ds.remove_columns(
                [c for c in dataset_cfg.dataset.cols_to_drop if c in ds.column_names]
            )
            for split, ds in data.items()
        }
    )

    # Store results
    results = {frac: [] for frac in DATA_FRACTIONS}

    # Run experiments
    for fraction in DATA_FRACTIONS:
        subset_size = int(len(data["train"]) * fraction)
        train_subset = data["train"].select(range(subset_size))

        logger.info(
            "Running %d runs for %.0f%% (%d samples)",
            RUNS_PER_FRACTION,
            fraction * 100,
            subset_size,
        )

        for run_id in range(RUNS_PER_FRACTION):
            # Set seed for reproducibility
            set_seed(42 + run_id)

            # Create fresh model
            if model_type == "bert":
                model = BertClassifier(OmegaConf.create(best_config), device=DEVICE)
            else:
                model = LSTMClassifier(OmegaConf.create(best_config), device=DEVICE)

            # Train
            try:
                train_model(
                    model,
                    train_subset,
                    data["dev"],
                    best_config,
                    model_type,
                    fraction,
                    run_id,
                )

                # Evaluate
                test_metrics = evaluate_on_test(model, data["test"], DEVICE)
                results[fraction].append(test_metrics)
                logger.info(
                    "  Run %d: Accuracy=%.4f, F1=%.4f",
                    run_id + 1,
                    test_metrics["accuracy"],
                    test_metrics["f1"],
                )
            except Exception as e:
                logger.error("Error in run %d for fraction %f: %s", run_id, fraction, e)
                results[fraction].append({"accuracy": None, "f1": None})

    return results


def plot_scaling_laws(bert_results, lstm_results):
    """Plot scaling laws for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fractions = sorted(DATA_FRACTIONS)
    fraction_sizes = [
        int(f * 120000) for f in fractions
    ]  # Assuming ~120k training samples

    # Extract means and stds
    for model_name, results in [("BERT", bert_results), ("LSTM", lstm_results)]:
        accs = []
        acc_stds = []
        f1s = []
        f1_stds = []

        for f in fractions:
            acc_values = [
                r["accuracy"] for r in results[f] if r["accuracy"] is not None
            ]
            f1_values = [r["f1"] for r in results[f] if r["f1"] is not None]

            if acc_values:
                accs.append(float(np.mean(acc_values)))
                acc_stds.append(float(np.std(acc_values)))
            else:
                accs.append(np.nan)
                acc_stds.append(0.0)

            if f1_values:
                f1s.append(float(np.mean(f1_values)))
                f1_stds.append(float(np.std(f1_values)))
            else:
                f1s.append(np.nan)
                f1_stds.append(0.0)

        valid_acc_idx = [i for i, v in enumerate(accs) if not np.isnan(v)]
        valid_f1_idx = [i for i, v in enumerate(f1s) if not np.isnan(v)]

        if not valid_acc_idx and not valid_f1_idx:
            logger.warning("No valid points to plot for %s", model_name)
            continue

        # Plot accuracy
        if valid_acc_idx:
            ax1.errorbar(
                [fraction_sizes[i] for i in valid_acc_idx],
                [accs[i] for i in valid_acc_idx],
                yerr=[acc_stds[i] for i in valid_acc_idx],
                marker="o",
                label=model_name,
                capsize=5,
                linewidth=2,
            )

        # Plot F1
        if valid_f1_idx:
            ax2.errorbar(
                [fraction_sizes[i] for i in valid_f1_idx],
                [f1s[i] for i in valid_f1_idx],
                yerr=[f1_stds[i] for i in valid_f1_idx],
                marker="s",
                label=model_name,
                capsize=5,
                linewidth=2,
            )

    ax1.set_xlabel("Training Data Size (samples)")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Scaling Laws: Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    ax2.set_xlabel("Training Data Size (samples)")
    ax2.set_ylabel("Test F1 (weighted)")
    ax2.set_title("Scaling Laws: F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scaling_laws.png"), dpi=300)
    logger.info("Saved plot to %s", os.path.join(OUTPUT_DIR, "scaling_laws.png"))
    plt.close()


def save_results_table(bert_results, lstm_results):
    """Save results as a formatted table."""
    output_file = os.path.join(OUTPUT_DIR, "scaling_law_results.txt")

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LABEL-NOISE SENSITIVITY: SCALING LAW RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for model_name, results in [("BERT", bert_results), ("LSTM", lstm_results)]:
            f.write(f"\n{model_name}\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Data Frac':<15} {'Accuracy (mean)':<20} {'Accuracy (std)':<20} "
                f"{'F1 (mean)':<20} {'F1 (std)':<20}\n"
            )
            f.write("-" * 80 + "\n")

            for frac in sorted(results.keys()):
                accs = [
                    r["accuracy"] for r in results[frac] if r["accuracy"] is not None
                ]
                f1s = [r["f1"] for r in results[frac] if r["f1"] is not None]

                if accs and f1s:
                    f.write(
                        f"{frac:.2f}{'':<11} "
                        f"{np.mean(accs):.4f}{'':<15} "
                        f"{np.std(accs):.4f}{'':<15} "
                        f"{np.mean(f1s):.4f}{'':<15} "
                        f"{np.std(f1s):.4f}\n"
                    )

    logger.info("Results saved to %s", output_file)


def main():
    """Run scaling law experiments for both models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LABEL-NOISE SENSITIVITY SCALING LAW ANALYSIS")
    logger.info("=" * 80)

    # Run experiments
    logger.info("\nRunning BERT scaling law experiment...")
    bert_results = run_scaling_law_experiment("bert")

    logger.info("\nRunning LSTM scaling law experiment...")
    lstm_results = run_scaling_law_experiment("lstm")

    # Save results
    logger.info("\nGenerating results...")
    save_results_table(bert_results, lstm_results)
    plot_scaling_laws(bert_results, lstm_results)

    logger.info("\n" + "=" * 80)
    logger.info("SCALING LAW ANALYSIS COMPLETE")
    logger.info("Results saved to %s/", OUTPUT_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
