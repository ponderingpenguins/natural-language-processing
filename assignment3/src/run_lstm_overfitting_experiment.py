"""Run one LSTM overfitting experiment and save a markdown note."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data
from utils.training import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Return accuracy and both weighted and macro F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy,
        # Keep f1 as an alias because the training config expects eval_f1.
        "f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }


def load_config() -> Any:
    """Load the default configs and CLI overrides."""
    dataset_cfg = OmegaConf.load("configs/dataset.yaml")
    lstm_cfg = OmegaConf.load("configs/lstm.yaml")
    training_cfg = OmegaConf.load("configs/training.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(dataset_cfg, lstm_cfg, training_cfg, cli_cfg)

    if "experiment" not in cfg:
        cfg.experiment = {}
    cfg.experiment.setdefault("name", "baseline")
    cfg.experiment.setdefault(
        "change_summary", "No config change. Baseline run for overfitting analysis."
    )
    cfg.experiment.setdefault(
        "hypothesis", "Baseline reference for train/eval loss gap."
    )
    cfg.experiment.setdefault("notes_dir", "overfitting_notes")
    cfg.experiment.setdefault("seed", cfg.dataset.seed)
    return cfg


def prepare_data(cfg: Any, model: LSTMClassifier) -> DatasetDict:
    """Load, preprocess, tokenize, and trim the dataset columns."""
    data = dataset_prep(cfg.dataset)

    sample_tag = (
        "full"
        if getattr(cfg.dataset, "max_samples", None) is None
        else f"max{cfg.dataset.max_samples}"
    )
    eval_tag = (
        "fulleval"
        if getattr(cfg.dataset, "eval_max_samples", None) is None
        else f"eval{cfg.dataset.eval_max_samples}"
    )
    tokenized_data_path = (
        f"./{cfg.dataset.hf_dataset.replace('/', '__')}_"
        f"lstm_seq{cfg.lstm_model.sequence_length}_{sample_tag}_{eval_tag}"
    )
    tokenization_config = {
        "hf_dataset": cfg.dataset.hf_dataset,
        "model": model.__class__.__name__,
        "tokenizer_name": getattr(model, "tokenizer", None).__class__.__name__,
        "max_samples": cfg.dataset.max_samples,
        "eval_max_samples": getattr(cfg.dataset, "eval_max_samples", None),
        "sequence_length": cfg.lstm_model.sequence_length,
    }
    data = try_load_tokenized_data(
        tokenized_data_path, data, model.tokenize, tokenization_config
    )

    return DatasetDict(
        {
            split: ds.remove_columns(
                [c for c in cfg.dataset.cols_to_drop if c in ds.column_names]
            )
            for split, ds in data.items()
        }
    )


def extract_loss_summary(log_history: list[dict[str, Any]]) -> dict[str, float | None]:
    """Extract the key train/eval loss numbers from a trainer log history."""
    train_losses = [entry for entry in log_history if "loss" in entry]
    eval_losses = [entry for entry in log_history if "eval_loss" in entry]

    def _min_entry(entries: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        return min(entries, key=lambda entry: float(entry[key])) if entries else None

    best_train = _min_entry(train_losses, "loss")
    best_eval = _min_entry(eval_losses, "eval_loss")
    last_train = train_losses[-1] if train_losses else None
    last_eval = eval_losses[-1] if eval_losses else None

    return {
        "best_train_loss": None if best_train is None else float(best_train["loss"]),
        "best_train_step": None if best_train is None else int(best_train["step"]),
        "last_train_loss": None if last_train is None else float(last_train["loss"]),
        "last_train_step": None if last_train is None else int(last_train["step"]),
        "best_eval_loss": (
            None if best_eval is None else float(best_eval["eval_loss"])
        ),
        "best_eval_step": None if best_eval is None else int(best_eval["step"]),
        "last_eval_loss": None if last_eval is None else float(last_eval["eval_loss"]),
        "last_eval_step": None if last_eval is None else int(last_eval["step"]),
        "loss_gap_best_eval_minus_best_train": (
            None
            if best_train is None or best_eval is None
            else float(best_eval["eval_loss"]) - float(best_train["loss"])
        ),
        "loss_gap_last_eval_minus_last_train": (
            None
            if last_train is None or last_eval is None
            else float(last_eval["eval_loss"]) - float(last_train["loss"])
        ),
    }


def markdown_lines(cfg: Any, metrics: dict[str, Any]) -> list[str]:
    """Format one experiment note in a compact style."""

    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _as_text(value: Any) -> str:
        if isinstance(value, dict):
            return "; ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    lines = [
        f"# {cfg.experiment.name}",
        "",
        f"Run time: {datetime.now(timezone.utc).isoformat()} | device: `{DEVICE}`",
        f"Change tried: {_as_text(cfg.experiment.change_summary)}",
        f"Hypothesis: {_as_text(cfg.experiment.hypothesis)}",
        "",
        "## Key Settings",
        "",
        (
            f"- Data caps: train={_fmt(cfg.dataset.max_samples)}, "
            f"eval={_fmt(cfg.dataset.eval_max_samples)}"
        ),
        (
            f"- Model: embed={_fmt(cfg.lstm_model.embed_dim)}, "
            f"hidden={_fmt(cfg.lstm_model.hidden_dim)}, "
            f"layers={_fmt(cfg.lstm_model.num_layers)}, "
            f"bidirectional={_fmt(cfg.lstm_model.bidirectional)}, "
            f"pooling={_fmt(getattr(cfg.lstm_model, 'pooling_type', 'mean'))}, "
            f"pack={_fmt(getattr(cfg.lstm_model, 'pack_sequences', False))}, "
            f"dropout={_fmt(cfg.lstm_model.dropout)}, "
            f"seq_len={_fmt(cfg.lstm_model.sequence_length)}"
        ),
        (
            f"- Optimizer: lr={_fmt(cfg.lstm_model.learning_rate)}, "
            f"weight_decay={_fmt(cfg.lstm_model.weight_decay)}"
        ),
        (
            f"- Training: epochs={_fmt(cfg.training.num_train_epochs)}, "
            f"eval={_fmt(cfg.training.eval_strategy)}, "
            f"save={_fmt(cfg.training.save_strategy)}, "
            f"patience={_fmt(cfg.training.early_stopping_patience)}"
        ),
        "",
        "## Data Sizes",
        "",
        (
            f"- train={_fmt(metrics['dataset_sizes']['train'])}, "
            f"dev={_fmt(metrics['dataset_sizes']['dev'])}, "
            f"test={_fmt(metrics['dataset_sizes']['test'])}"
        ),
        "",
        "## Results",
        "",
        (
            f"- Loss (best train / best eval): "
            f"{_fmt(metrics['loss_summary']['best_train_loss'])} / "
            f"{_fmt(metrics['loss_summary']['best_eval_loss'])}"
        ),
        (
            f"- Loss (last train / last eval): "
            f"{_fmt(metrics['loss_summary']['last_train_loss'])} / "
            f"{_fmt(metrics['loss_summary']['last_eval_loss'])}"
        ),
        (
            f"- Gap (best): "
            f"{_fmt(metrics['loss_summary']['loss_gap_best_eval_minus_best_train'])}"
        ),
        (
            f"- Gap (last): "
            f"{_fmt(metrics['loss_summary']['loss_gap_last_eval_minus_last_train'])}"
        ),
        (
            f"- Dev acc / macro-F1: "
            f"{_fmt(metrics['dev_metrics'].get('eval_accuracy'))} / "
            f"{_fmt(metrics['dev_metrics'].get('eval_macro_f1'))}"
        ),
        (
            f"- Test acc / macro-F1: "
            f"{_fmt(metrics['test_metrics'].get('test_accuracy'))} / "
            f"{_fmt(metrics['test_metrics'].get('test_macro_f1'))}"
        ),
        "",
        f"Takeaway: {metrics['interpretation']}",
        "",
    ]
    return lines


def build_interpretation(metrics: dict[str, Any]) -> str:
    """Generate a short interpretation for the note."""
    best_gap = metrics["loss_summary"]["loss_gap_best_eval_minus_best_train"]
    test_macro_f1 = metrics["test_metrics"].get("test_macro_f1")
    if best_gap is None:
        return "Could not read the train/eval loss gap from trainer logs."
    if best_gap > 0.4:
        return f"The gap is still large ({best_gap:.4f}), so overfitting is still visible."
    if test_macro_f1 is None:
        return f"The gap improved to {best_gap:.4f}, but test macro-F1 is missing."
    return (
        f"Gap is {best_gap:.4f} with test macro-F1 {test_macro_f1:.4f}. "
        "Compare it against baseline in the summary table."
    )


def main() -> None:
    """Run the configured experiment and save metrics and a markdown note."""
    cfg = load_config()
    set_seed(int(cfg.experiment.seed))

    output_dir = Path(cfg.lstm_model.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    notes_dir = Path(cfg.experiment.notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running LSTM overfitting experiment: %s", cfg.experiment.name)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    model = LSTMClassifier(cfg.lstm_model, device=DEVICE)
    data = prepare_data(cfg, model)

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        learning_rate=cfg.lstm_model.learning_rate,
        weight_decay=cfg.lstm_model.weight_decay,
        per_device_train_batch_size=cfg.lstm_model.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.lstm_model.per_device_eval_batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        warmup_steps=cfg.training.warmup_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_dir=cfg.lstm_model.logging_dir,
        logging_steps=cfg.training.logging_steps,
        logging_first_step=cfg.training.logging_first_step,
        report_to=cfg.training.report_to,
        save_total_limit=cfg.training.save_total_limit,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=DataCollatorWithPadding(tokenizer=model.tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            )
        ],
    )

    trainer.train()
    trainer.remove_callback(EarlyStoppingCallback)
    dev_metrics = trainer.evaluate(data["dev"])
    test_metrics = trainer.predict(data["test"], metric_key_prefix="test").metrics
    loss_summary = extract_loss_summary(trainer.state.log_history)

    metrics = {
        "experiment_name": cfg.experiment.name,
        "dataset_sizes": {split: len(data[split]) for split in ("train", "dev", "test")},
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "loss_summary": loss_summary,
    }
    metrics["interpretation"] = build_interpretation(metrics)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    note_path = notes_dir / f"{cfg.experiment.name}.md"
    note_path.write_text("\n".join(markdown_lines(cfg, metrics)), encoding="utf-8")

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved note to %s", note_path)


if __name__ == "__main__":
    main()
