"""
Label-Noise Sensitivity Analysis: Scaling Laws for BERT and LSTM Models
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data
from utils.training import compute_metrics, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE

DATA_FRACTIONS = [0.25, 0.5, 1.0]
RUNS_PER_FRACTION = 1
OUTPUT_DIR = "scaling_law_results"


def load_best_config(model_type: str) -> dict:
    path = f"{model_type}_output/config.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config not found at {path}. Run hyperparameter tuning first."
        )
    with open(path) as f:
        return json.load(f)


def make_model(model_type: str, config: dict):
    cls = BertClassifier if model_type == "bert" else LSTMClassifier
    return cls(OmegaConf.create(config), device=DEVICE)


def load_data(config, tokenize_fn) -> DatasetDict:
    data = dataset_prep(config)
    max_samples = config.get("max_samples")
    cache_path = f"./{config.get('hf_dataset', '').replace('/', '__')}_tokenized_{'full' if max_samples is None else f'max{max_samples}'}"
    data = try_load_tokenized_data(cache_path, data, tokenize_fn)
    return DatasetDict(
        {
            split: ds.remove_columns(
                [c for c in config.get("cols_to_drop", []) if c in ds.column_names]
            )
            for split, ds in data.items()
        }
    )


def train_and_evaluate(
    model, train_data, eval_data, test_data, config: dict, output_dir: str
) -> dict:
    steps_per_epoch = max(
        1, math.ceil(len(train_data) / config.get("per_device_train_batch_size", 16))
    )
    total_steps = steps_per_epoch * config.get("num_train_epochs", 15)
    warmup_steps = min(config.get("warmup_steps", 500), int(total_steps * 0.1))

    args = TrainingArguments(
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
        warmup_steps=warmup_steps,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 3)
            )
        ],
    )

    trainer.train()
    metrics = trainer.evaluate(test_data, metric_key_prefix="test")

    accuracy_key = next(
        (key for key in ("test_accuracy", "test_eval_accuracy") if key in metrics),
        None,
    )
    f1_key = next(
        (key for key in ("test_f1", "test_eval_f1") if key in metrics),
        None,
    )
    if accuracy_key is None or f1_key is None:
        raise KeyError(f"Unexpected test metric keys: {sorted(metrics.keys())}")

    return {"accuracy": metrics[accuracy_key], "f1": metrics[f1_key]}


def run_experiment(model_type: str) -> tuple[dict, dict]:
    config = load_best_config(model_type)
    dataset_cfg = OmegaConf.load("configs/dataset.yaml")

    tokenizer_model = make_model(model_type, config)
    data = load_data(
        dataset_cfg.dataset,
        tokenizer_model.tokenize,
    )

    results = {}
    subset_sizes = {}

    for fraction in DATA_FRACTIONS:
        n = int(len(data["train"]) * fraction)
        subset_sizes[fraction] = n
        fraction_results = []

        for run_id in range(RUNS_PER_FRACTION):
            set_seed(42 + run_id)
            train_subset = data["train"].shuffle(seed=42 + run_id).select(range(n))
            model = make_model(model_type, config)
            out_dir = os.path.join(
                OUTPUT_DIR, f"{model_type}_frac{fraction:.2f}_run{run_id}"
            )

            try:
                metrics = train_and_evaluate(
                    model, train_subset, data["dev"], data["test"], config, out_dir
                )
                logger.info(
                    "  %s frac=%.2f run=%d | acc=%.4f f1=%.4f",
                    model_type,
                    fraction,
                    run_id + 1,
                    metrics["accuracy"],
                    metrics["f1"],
                )
            except Exception as e:
                logger.error(
                    "Error: %s frac=%.2f run=%d | %s", model_type, fraction, run_id, e
                )
                metrics = {"accuracy": None, "f1": None}

            fraction_results.append(metrics)

        results[fraction] = fraction_results

    return results, subset_sizes


def aggregate(results: dict, metric: str) -> tuple[list, list, list]:
    fractions = sorted(results)
    means, stds = [], []
    for f in fractions:
        vals = [r[metric] for r in results[f] if r[metric] is not None]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if vals else 0.0)
    return fractions, means, stds


def plot_results(bert_results: dict, lstm_results: dict, subset_sizes: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, results in [("BERT", bert_results), ("LSTM", lstm_results)]:
        for ax, metric, marker in [(ax1, "accuracy", "o"), (ax2, "f1", "s")]:
            fractions, means, stds = aggregate(results, metric)
            sizes = [subset_sizes[f] for f in fractions]
            valid = [
                (s, m, e) for s, m, e in zip(sizes, means, stds) if not np.isnan(m)
            ]
            if valid:
                s, m, e = zip(*valid)
                ax.errorbar(
                    s,
                    m,
                    yerr=e,
                    marker=marker,
                    label=model_name,
                    capsize=5,
                    linewidth=2,
                )

    for ax, metric in [(ax1, "Accuracy"), (ax2, "F1 (weighted)")]:
        ax.set_xlabel("Training Data Size (samples)")
        ax.set_ylabel(f"Test {metric}")
        ax.set_title(f"Scaling Laws: {metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scaling_laws.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info("Plot saved to %s", path)


def save_results(bert_results: dict, lstm_results: dict):
    path = os.path.join(OUTPUT_DIR, "scaling_law_results.txt")
    with open(path, "w") as f:
        f.write("LABEL-NOISE SENSITIVITY: SCALING LAW RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for model_name, results in [("BERT", bert_results), ("LSTM", lstm_results)]:
            f.write(f"{model_name}\n{'-' * 40}\n")
            f.write(
                f"{'Frac':<8} {'Acc (mean)':<16} {'Acc (std)':<16} {'F1 (mean)':<16} {'F1 (std)':<16}\n"
            )
            for frac in sorted(results):
                _, acc_means, acc_stds = aggregate({frac: results[frac]}, "accuracy")
                _, f1_means, f1_stds = aggregate({frac: results[frac]}, "f1")
                f.write(
                    f"{frac:<8.2f} {acc_means[0]:<16.4f} {acc_stds[0]:<16.4f} {f1_means[0]:<16.4f} {f1_stds[0]:<16.4f}\n"
                )
            f.write("\n")
    logger.info("Results saved to %s", path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Starting scaling law analysis")

    bert_results, subset_sizes = run_experiment("bert")
    lstm_results, _ = run_experiment("lstm")

    save_results(bert_results, lstm_results)
    plot_results(bert_results, lstm_results, subset_sizes)

    logger.info("Done. Results in %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
