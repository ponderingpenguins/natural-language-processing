"""Run data-fraction sensitivity for BERT and LSTM."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data
from utils.training import compute_metrics, set_seed

SRC_DIR = Path(__file__).resolve().parent
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if os.getenv("REQUIRE_CUDA", "0").lower() in {"1", "true", "yes"} and DEVICE.type != "cuda":
    raise RuntimeError(
        "REQUIRE_CUDA is enabled but CUDA is unavailable. Aborting to prevent CPU fallback."
    )

DEFAULT_DATA_FRACTIONS = [0.25, 0.5, 1.0]
DEFAULT_RUNS_PER_FRACTION = 1


def _build_tokenized_cache_path(dataset_cfg, model_type: str, model_cfg) -> str:
    dataset_tag = dataset_cfg.hf_dataset.replace("/", "__")
    sample_tag = (
        "full"
        if getattr(dataset_cfg, "max_samples", None) is None
        else f"max{dataset_cfg.max_samples}"
    )
    eval_tag = (
        "fulleval"
        if getattr(dataset_cfg, "eval_max_samples", None) is None
        else f"eval{dataset_cfg.eval_max_samples}"
    )
    if model_type == "bert":
        len_tag = f"maxlen{model_cfg.max_length}"
    else:
        len_tag = f"seq{model_cfg.sequence_length}"
    cache_name = f"{dataset_tag}_tokenized_{model_type}_{len_tag}_{sample_tag}_{eval_tag}"
    return str(SRC_DIR / cache_name)


def _load_model_config(model_type: str):
    """Load model config for the requested model type."""
    if model_type == "bert":
        curated_json = SRC_DIR / "bert_output/sanity_distilbert/config.json"
        if curated_json.exists():
            return OmegaConf.create(json.loads(curated_json.read_text(encoding="utf-8")))
        return OmegaConf.load(SRC_DIR / "configs/bert.yaml").bert_model

    return OmegaConf.load(SRC_DIR / "configs/lstm.yaml").lstm_model


def make_model(model_type: str, config):
    cls = BertClassifier if model_type == "bert" else LSTMClassifier
    return cls(config, device=DEVICE)


def load_data(dataset_cfg, model_type: str, model_cfg) -> DatasetDict:
    """Load and tokenize data with cache-keyed settings."""
    data = dataset_prep(dataset_cfg)
    tokenizer_probe = make_model(model_type, model_cfg)
    cache_path = _build_tokenized_cache_path(dataset_cfg, model_type, model_cfg)
    tokenization_config = {
        "hf_dataset": dataset_cfg.hf_dataset,
        "model_type": model_type,
        "model": tokenizer_probe.__class__.__name__,
        "tokenizer_name": getattr(tokenizer_probe, "tokenizer", None).__class__.__name__,
        "max_samples": dataset_cfg.max_samples,
        "eval_max_samples": dataset_cfg.eval_max_samples,
        "sequence_length": getattr(model_cfg, "sequence_length", None),
        "max_length": getattr(model_cfg, "max_length", None),
    }
    data = try_load_tokenized_data(
        cache_path, data, tokenizer_probe.tokenize, tokenization_config
    )
    return DatasetDict(
        {
            split: ds.remove_columns(
                [c for c in dataset_cfg.cols_to_drop if c in ds.column_names]
            )
            for split, ds in data.items()
        }
    )


def train_and_evaluate(
    model,
    train_data,
    eval_data,
    test_data,
    run_cfg,
    output_dir: Path,
) -> dict:
    """Train one run and return test accuracy and macro-F1."""
    steps_per_epoch = max(
        1, math.ceil(len(train_data) / int(run_cfg.get("per_device_train_batch_size", 16)))
    )
    total_steps = steps_per_epoch * int(run_cfg.get("num_train_epochs", 8))
    warmup_steps = min(int(run_cfg.get("warmup_steps", 0)), int(total_steps * 0.1))

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=float(run_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(run_cfg.get("weight_decay", 0.0)),
        per_device_train_batch_size=int(run_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(run_cfg.get("per_device_eval_batch_size", 16)),
        num_train_epochs=float(run_cfg.get("num_train_epochs", 8)),
        warmup_steps=warmup_steps,
        report_to="none",
        logging_steps=max(1, steps_per_epoch // 4),
    )

    tokenizer = getattr(model, "tokenizer", None)
    data_collator = (
        DataCollatorWithPadding(tokenizer=tokenizer) if tokenizer is not None else None
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(test_data, metric_key_prefix="test")

    accuracy_key = next(
        (k for k in ("test_accuracy", "test_eval_accuracy") if k in metrics), None
    )
    macro_f1_key = next(
        (
            k
            for k in (
                "test_macro_f1",
                "test_eval_macro_f1",
                "test_f1",
                "test_eval_f1",
            )
            if k in metrics
        ),
        None,
    )
    if accuracy_key is None or macro_f1_key is None:
        raise KeyError(f"Unexpected test metric keys: {sorted(metrics.keys())}")

    return {"accuracy": metrics[accuracy_key], "macro_f1": metrics[macro_f1_key]}


def run_experiment(
    model_type: str,
    dataset_cfg,
    training_cfg,
    fractions: list[float],
    runs_per_fraction: int,
    output_dir: Path,
) -> tuple[dict, dict]:
    """Run a fraction sweep for one model."""
    model_cfg = _load_model_config(model_type)
    run_cfg = OmegaConf.merge(training_cfg, model_cfg)
    data = load_data(dataset_cfg, model_type, run_cfg)

    results = {}
    subset_sizes = {}

    for fraction in fractions:
        n = max(1, int(len(data["train"]) * fraction))
        subset_sizes[fraction] = n
        fraction_results = []

        for run_id in range(runs_per_fraction):
            run_seed = int(dataset_cfg.seed) + run_id
            set_seed(run_seed)
            train_subset = data["train"].shuffle(seed=run_seed).select(range(n))

            model = make_model(model_type, run_cfg)
            run_dir = output_dir / f"{model_type}_frac{fraction:.2f}_run{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                metrics = train_and_evaluate(
                    model=model,
                    train_data=train_subset,
                    eval_data=data["dev"],
                    test_data=data["test"],
                    run_cfg=run_cfg,
                    output_dir=run_dir,
                )
                logger.info(
                    "%s frac=%.2f run=%d | acc=%.4f macro_f1=%.4f",
                    model_type,
                    fraction,
                    run_id + 1,
                    metrics["accuracy"],
                    metrics["macro_f1"],
                )
            except Exception as exc:
                logger.error(
                    "Error: %s frac=%.2f run=%d | %s",
                    model_type,
                    fraction,
                    run_id + 1,
                    exc,
                )
                metrics = {"accuracy": None, "macro_f1": None}

            fraction_results.append(metrics)

        results[fraction] = fraction_results

    return results, subset_sizes


def aggregate(results: dict, metric: str) -> tuple[list, list, list]:
    """Compute mean/std over runs for each fraction."""
    fractions = sorted(results)
    means, stds = [], []
    for fraction in fractions:
        vals = [row[metric] for row in results[fraction] if row[metric] is not None]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if vals else 0.0)
    return fractions, means, stds


def plot_results(results_by_model: dict, subset_sizes: dict, output_dir: Path) -> None:
    """Plot scaling curves for accuracy and macro-F1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, results in results_by_model.items():
        for ax, metric, marker in [(
            ax1,
            "accuracy",
            "o",
        ), (ax2, "macro_f1", "s")]:
            fractions, means, stds = aggregate(results, metric)
            sizes = [subset_sizes[f] for f in fractions]
            valid = [
                (size, mean, std)
                for size, mean, std in zip(sizes, means, stds)
                if not np.isnan(mean)
            ]
            if not valid:
                continue
            x, y, err = zip(*valid)
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker=marker,
                label=model_name.upper(),
                capsize=4,
                linewidth=2,
            )

    for ax, metric_label in [(ax1, "Accuracy"), (ax2, "Macro-F1")]:
        ax.set_xlabel("Train size")
        ax.set_ylabel(f"Test {metric_label}")
        ax.set_title(f"{metric_label} vs train size")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.tight_layout()
    plot_path = output_dir / "scaling_laws.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Saved plot: %s", plot_path)


def save_results(results_by_model: dict, output_dir: Path) -> None:
    """Save text and JSON summaries."""
    txt_path = output_dir / "scaling_law_results.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Data-fraction sensitivity results\n")
        handle.write("=" * 80 + "\n\n")
        for model_name, results in results_by_model.items():
            handle.write(f"{model_name.upper()}\n{'-' * 40}\n")
            handle.write(
                f"{'Frac':<8} {'Acc (mean)':<16} {'Acc (std)':<16} "
                f"{'Macro-F1 (mean)':<16} {'Macro-F1 (std)':<16}\n"
            )
            for fraction in sorted(results):
                _, acc_means, acc_stds = aggregate({fraction: results[fraction]}, "accuracy")
                _, f1_means, f1_stds = aggregate({fraction: results[fraction]}, "macro_f1")
                handle.write(
                    f"{fraction:<8.2f} {acc_means[0]:<16.4f} {acc_stds[0]:<16.4f} "
                    f"{f1_means[0]:<16.4f} {f1_stds[0]:<16.4f}\n"
                )
            handle.write("\n")
    logger.info("Saved text results: %s", txt_path)

    raw_path = output_dir / "scaling_law_results.json"
    raw_path.write_text(json.dumps(results_by_model, indent=2), encoding="utf-8")
    logger.info("Saved JSON results: %s", raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data-fraction sensitivity.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert", "lstm"],
        default=["bert", "lstm"],
        help="Models to run.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_DATA_FRACTIONS,
        help="Train fractions, e.g. 0.25 0.5 1.0.",
    )
    parser.add_argument(
        "--runs-per-fraction",
        type=int,
        default=DEFAULT_RUNS_PER_FRACTION,
        help="Runs per fraction.",
    )
    parser.add_argument(
        "--output-dir",
        default="scaling_law_results",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional train cap.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Optional dev/test cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SRC_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = OmegaConf.load(SRC_DIR / "configs/dataset.yaml").dataset
    dataset_cfg.max_samples = args.max_samples
    dataset_cfg.eval_max_samples = args.eval_max_samples
    training_cfg = OmegaConf.load(SRC_DIR / "configs/training.yaml").training

    logger.info("Starting run on device=%s", DEVICE)
    logger.info("Fractions=%s runs_per_fraction=%d", args.fractions, args.runs_per_fraction)

    results_by_model = {}
    subset_sizes = None
    for model_type in args.models:
        results, sizes = run_experiment(
            model_type=model_type,
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            fractions=args.fractions,
            runs_per_fraction=args.runs_per_fraction,
            output_dir=output_dir,
        )
        results_by_model[model_type] = results
        if subset_sizes is None:
            subset_sizes = sizes

    save_results(results_by_model, output_dir)
    if subset_sizes is not None:
        plot_results(results_by_model, subset_sizes, output_dir)

    logger.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()
