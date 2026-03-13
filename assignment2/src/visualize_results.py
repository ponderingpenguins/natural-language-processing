"""Aggregate and visualize Assignment 2 experiment results.

This script compares CNN vs LSTM runs across max sequence lengths (64/128/256)
using:
- hyperparameter tuning results
- final test-set classification reports
- full-training epoch logs (parsed from experiments/experiment_*.txt)

Run from `assignment2`:
src/.venv/bin/python src/visualize_results.py --src-dir src --out-dir report/figures/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TrainingPoint:
    """Epoch-level training metrics."""

    epoch: int
    train_loss: float
    train_acc: float
    train_f1: float
    val_loss: float
    val_acc: float
    val_f1: float
    train_precision: Optional[float] = None
    train_recall: Optional[float] = None
    val_precision: Optional[float] = None
    val_recall: Optional[float] = None
    epoch_time_seconds: Optional[float] = None


@dataclass
class RunResult:
    """Container for one experiment run."""

    model: str
    seq_len: int
    run_dir: Path
    best_config: dict
    best_dev_f1: float
    all_dev_f1: list[float]
    test_accuracy: float
    macro_f1: float
    weighted_f1: float
    class_f1: dict[str, float]
    training_history: list[TrainingPoint]

    @property
    def run_label(self) -> str:
        return f"{self.model.upper()}-seq{self.seq_len}"


RUN_DIR_PATTERN = re.compile(r"experiment_(cnn|lstm)_seq(\d+)$")
LOG_RUN_PATTERN = re.compile(
    r"Running:\s*model_type=(cnn|lstm),\s*max_seq_length=(\d+)"
)
HEADER_PATTERN = re.compile(
    r"^\s*epoch\s+train_loss\s+train_acc\s+val_loss\s+val_acc\s*$"
)
ROW_PATTERN = re.compile(
    r"\b(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\b"
)
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

CLASS_LABELS = {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    this_file = Path(__file__).resolve()
    src_dir = this_file.parent
    assignment2_dir = src_dir.parent
    default_output_dir = assignment2_dir / "report" / "figures" / "generated"

    parser = argparse.ArgumentParser(
        description="Visualize CNN vs LSTM experiment runs for Assignment 2."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=src_dir,
        help="Path to assignment2/src (default: current script directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where plots and CSV summaries will be written.",
    )
    return parser.parse_args()


def parse_classification_report(report_path: Path) -> dict:
    """Parse test classification report txt into a metrics dictionary."""
    text = report_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    class_f1: dict[str, float] = {}
    test_accuracy = None
    macro_f1 = None
    weighted_f1 = None

    class_row = re.compile(
        r"^(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)$"
    )
    acc_row = re.compile(r"^accuracy\s+([0-9]*\.?[0-9]+)\s+(\d+)$")
    avg_row = re.compile(
        r"^(macro avg|weighted avg)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)$"
    )

    for line in lines:
        class_match = class_row.match(line)
        if class_match:
            class_id = class_match.group(1)
            class_f1[class_id] = float(class_match.group(4))
            continue

        acc_match = acc_row.match(line)
        if acc_match:
            test_accuracy = float(acc_match.group(1))
            continue

        avg_match = avg_row.match(line)
        if avg_match:
            avg_name = avg_match.group(1)
            avg_f1 = float(avg_match.group(4))
            if avg_name == "macro avg":
                macro_f1 = avg_f1
            elif avg_name == "weighted avg":
                weighted_f1 = avg_f1

    if test_accuracy is None or macro_f1 is None or weighted_f1 is None:
        raise ValueError(f"Failed to parse classification metrics from {report_path}")

    return {
        "test_accuracy": test_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_f1": class_f1,
    }


def parse_training_tables_from_log(log_path: Path) -> list[list[TrainingPoint]]:
    """Extract all epoch tables from one experiment log file."""
    blocks: list[list[TrainingPoint]] = []
    current_block: list[TrainingPoint] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
        for raw_line in file_handle:
            clean_line = ANSI_PATTERN.sub("", raw_line).replace("\x1b[A", "").strip()

            if HEADER_PATTERN.match(clean_line):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                continue

            row_match = ROW_PATTERN.search(clean_line)
            if row_match:
                point = TrainingPoint(
                    epoch=int(row_match.group(1)),
                    train_loss=float(row_match.group(2)),
                    train_acc=float(row_match.group(3)),
                    train_f1=float(row_match.group(3)),
                    train_precision=None,
                    train_recall=None,
                    val_loss=float(row_match.group(4)),
                    val_acc=float(row_match.group(5)),
                    val_f1=float(row_match.group(5)),
                    val_precision=None,
                    val_recall=None,
                    epoch_time_seconds=None,
                )

                if current_block and point.epoch <= current_block[-1].epoch:
                    blocks.append(current_block)
                    current_block = [point]
                else:
                    current_block.append(point)
                continue

            if current_block and "Test Set Classification Report:" in clean_line:
                blocks.append(current_block)
                current_block = []

    if current_block:
        blocks.append(current_block)

    return blocks


def map_run_to_log(src_dir: Path) -> dict[tuple[str, int], Path]:
    """Map (model, seq_len) to matching log file in experiments/ directory."""
    experiments_dir = src_dir / "experiments"
    mapping: dict[tuple[str, int], Path] = {}

    for log_path in sorted(experiments_dir.glob("experiment_*.txt")):
        run_model = None
        run_seq = None

        with log_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
            for line in file_handle:
                match = LOG_RUN_PATTERN.search(line)
                if match:
                    run_model = match.group(1)
                    run_seq = int(match.group(2))
                    break

        if run_model is not None and run_seq is not None:
            mapping[(run_model, run_seq)] = log_path

    return mapping


def parse_training_history_from_results(results_path: Path) -> list[TrainingPoint]:
    """Read epoch-level metrics from training_results.json history."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    points: list[TrainingPoint] = []

    for item in history:
        points.append(
            TrainingPoint(
                epoch=int(item["epoch"]),
                train_loss=float(item["train_loss"]),
                train_acc=float(item["train_acc"]),
                train_f1=float(item.get("train_f1", item["train_acc"])),
                train_precision=(
                    float(item["train_precision"])
                    if item.get("train_precision") is not None
                    else None
                ),
                train_recall=(
                    float(item["train_recall"])
                    if item.get("train_recall") is not None
                    else None
                ),
                val_loss=float(item["val_loss"]),
                val_acc=float(item["val_acc"]),
                val_f1=float(item.get("val_f1", item["val_acc"])),
                val_precision=(
                    float(item["val_precision"])
                    if item.get("val_precision") is not None
                    else None
                ),
                val_recall=(
                    float(item["val_recall"])
                    if item.get("val_recall") is not None
                    else None
                ),
                epoch_time_seconds=(
                    float(item["epoch_time_seconds"])
                    if item.get("epoch_time_seconds") is not None
                    else None
                ),
            )
        )

    return points


def load_run_result(run_dir: Path, log_map: dict[tuple[str, int], Path]) -> RunResult:
    """Load all artifacts for one run directory."""
    run_match = RUN_DIR_PATTERN.match(run_dir.name)
    if run_match is None:
        raise ValueError(f"Unexpected run directory name: {run_dir.name}")

    model = run_match.group(1)
    seq_len = int(run_match.group(2))

    tuning_path = run_dir / f"hyperparameter_tuning_{model}.json"
    report_path = run_dir / "classification_report.txt"
    report_metrics = parse_classification_report(report_path)

    training_history: list[TrainingPoint] = []
    training_results_path = run_dir / "training_results.json"
    if training_results_path.exists():
        training_history = parse_training_history_from_results(training_results_path)
    else:
        log_path = log_map.get((model, seq_len))
        if log_path is not None:
            blocks = parse_training_tables_from_log(log_path)
            if blocks:
                blocks = sorted(blocks, key=len)
                training_history = blocks[-1]

    if tuning_path.exists():
        tuning_payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        best_config = tuning_payload.get("best_config", {})
        best_dev_f1 = float(tuning_payload.get("best_f1", 0.0))
        all_dev_f1 = [
            float(item["dev_f1"]) for item in tuning_payload.get("all_results", [])
        ]
    else:
        model_config_path = run_dir / "model_config.json"
        model_config = {}
        if model_config_path.exists():
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))

        if model == "cnn":
            best_config = {
                "lr": model_config.get("learning_rate"),
                "num_filters": model_config.get("cnn_num_filters"),
                "kernel_sizes": model_config.get("cnn_kernel_sizes"),
                "embed_dim": model_config.get("embed_dim"),
                "weight_decay": model_config.get("weighted_decay"),
                "dropout": model_config.get("cnn_dropout"),
            }
        else:
            best_config = {
                "lr": model_config.get("learning_rate"),
                "hidden_dim": model_config.get("lstm_hidden_dim"),
                "embed_dim": model_config.get("embed_dim"),
                "bidirectional": model_config.get("lstm_bidirectional"),
                "weight_decay": model_config.get("weighted_decay"),
                "dropout": model_config.get("lstm_dropout"),
            }

        if training_history:
            best_dev_f1 = max(point.val_f1 for point in training_history)
        else:
            best_dev_f1 = report_metrics["macro_f1"]
        all_dev_f1 = [best_dev_f1]

    return RunResult(
        model=model,
        seq_len=seq_len,
        run_dir=run_dir,
        best_config=best_config,
        best_dev_f1=float(best_dev_f1),
        all_dev_f1=all_dev_f1,
        test_accuracy=report_metrics["test_accuracy"],
        macro_f1=report_metrics["macro_f1"],
        weighted_f1=report_metrics["weighted_f1"],
        class_f1=report_metrics["class_f1"],
        training_history=training_history,
    )


def write_csv_summaries(out_dir: Path, runs: list[RunResult]) -> None:
    """Write machine-readable result summaries for report tables."""
    run_rows = []
    history_rows = []

    for run in runs:
        run_rows.append(
            {
                "run_label": run.run_label,
                "model": run.model,
                "seq_len": run.seq_len,
                "best_dev_f1": run.best_dev_f1,
                "test_accuracy": run.test_accuracy,
                "macro_f1": run.macro_f1,
                "weighted_f1": run.weighted_f1,
            }
        )

        for point in run.training_history:
            history_rows.append(
                {
                    "run_label": run.run_label,
                    "model": run.model,
                    "seq_len": run.seq_len,
                    "epoch": point.epoch,
                    "train_loss": point.train_loss,
                    "train_acc": point.train_acc,
                    "train_f1": point.train_f1,
                    "train_precision": point.train_precision,
                    "train_recall": point.train_recall,
                    "val_loss": point.val_loss,
                    "val_acc": point.val_acc,
                    "val_f1": point.val_f1,
                    "val_precision": point.val_precision,
                    "val_recall": point.val_recall,
                    "epoch_time_seconds": point.epoch_time_seconds,
                }
            )

    run_csv = out_dir / "runs_summary.csv"
    with run_csv.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(
            sorted(run_rows, key=lambda row: (row["model"], row["seq_len"]))
        )

    if history_rows:
        history_csv = out_dir / "full_training_history.csv"
        with history_csv.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle, fieldnames=list(history_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(
                sorted(
                    history_rows,
                    key=lambda row: (row["model"], row["seq_len"], row["epoch"]),
                )
            )


def write_best_config_summary(out_dir: Path, runs: list[RunResult]) -> None:
    """Write flattened best-hyperparameter configurations for each run."""
    all_param_keys = sorted({key for run in runs for key in run.best_config.keys()})
    fieldnames = [
        "run_label",
        "model",
        "seq_len",
        "best_dev_f1",
        *all_param_keys,
    ]

    rows = []
    for run in sorted(runs, key=lambda item: (item.model, item.seq_len)):
        row = {
            "run_label": run.run_label,
            "model": run.model,
            "seq_len": run.seq_len,
            "best_dev_f1": run.best_dev_f1,
        }
        for key in all_param_keys:
            value = run.best_config.get(key)
            row[key] = value if not isinstance(value, list) else str(value)
        rows.append(row)

    output_path = out_dir / "best_config_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_latex_number(value: object) -> str:
    """Format numeric values for compact LaTeX table output."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0.0"
        if abs(value) < 1e-4:
            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _resolve_value(run: RunResult, *keys: str) -> object | None:
    """Get first present value from run best_config or run model_config.json."""
    for key in keys:
        if key in run.best_config:
            return run.best_config[key]

    model_config_path = run.run_dir / "model_config.json"
    if model_config_path.exists():
        model_cfg = json.loads(model_config_path.read_text(encoding="utf-8"))
        for key in keys:
            if key in model_cfg:
                return model_cfg[key]

    return None


def _best_config_table_row(run: RunResult) -> list[str]:
    """Create one row for the best-config LaTeX table."""
    model = run.model.lower()

    lr = _resolve_value(run, "lr", "learning_rate")
    embed = _resolve_value(run, "embed_dim")
    weight_decay = _resolve_value(run, "weight_decay", "weighted_decay")

    if model == "cnn":
        filters = _resolve_value(run, "num_filters", "cnn_num_filters")
        hidden = "---"
        bilstm = "---"
        dropout = _resolve_value(run, "dropout", "cnn_dropout")
    else:
        filters = "---"
        hidden = _resolve_value(run, "hidden_dim", "lstm_hidden_dim")
        bilstm = _resolve_value(run, "bidirectional", "lstm_bidirectional")
        dropout = _resolve_value(run, "dropout", "lstm_dropout")

    return [
        model.upper(),
        str(run.seq_len),
        _format_latex_number(embed) if embed is not None else "---",
        _format_latex_number(filters) if filters != "---" else "---",
        _format_latex_number(hidden) if hidden != "---" else "---",
        _format_latex_number(bilstm) if bilstm != "---" else "---",
        _format_latex_number(lr) if lr is not None else "---",
        _format_latex_number(dropout) if dropout is not None else "---",
        _format_latex_number(weight_decay) if weight_decay is not None else "---",
    ]


def _tuning_best_config_row(model: str, best_config: dict) -> list[str]:
    """Create one row for the tuning-only best-config LaTeX table."""
    lr = best_config.get("lr")
    embed = best_config.get("embed_dim")
    weight_decay = best_config.get("weight_decay")

    if model == "cnn":
        filters = best_config.get("num_filters")
        hidden = "---"
        bilstm = "---"
        dropout = best_config.get("dropout")
    else:
        filters = "---"
        hidden = best_config.get("hidden_dim")
        bilstm = best_config.get("bidirectional")
        dropout = best_config.get("dropout")

    return [
        model.upper(),
        _format_latex_number(embed) if embed is not None else "---",
        _format_latex_number(filters) if filters != "---" else "---",
        _format_latex_number(hidden) if hidden is not None else "---",
        _format_latex_number(bilstm) if bilstm is not None else "---",
        _format_latex_number(lr) if lr is not None else "---",
        _format_latex_number(dropout) if dropout is not None else "---",
        _format_latex_number(weight_decay) if weight_decay is not None else "---",
    ]


def _load_tuning_best_configs(src_dir: Path) -> dict[str, dict]:
    """Load best_config for CNN/LSTM from dedicated tuning experiment folders."""
    tuning_files = {
        "cnn": src_dir / "experiment_cnn_tuning" / "hyperparameter_tuning_cnn.json",
        "lstm": src_dir / "experiment_lstm_tuning" / "hyperparameter_tuning_lstm.json",
    }

    payloads: dict[str, dict] = {}
    for model, json_path in tuning_files.items():
        if not json_path.exists():
            raise FileNotFoundError(f"Missing tuning file: {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payloads[model] = payload.get("best_config", {})

    return payloads


def build_tuning_best_config_latex_table(
    src_dir: Path,
    caption: str = (
        "Best hyperparameter configurations from dedicated tuning runs "
        r"(\texttt{experiment\_cnn\_tuning} and "
        r"\texttt{experiment\_lstm\_tuning}). "
        "Embed: embedding dimension, WD: weight decay."
    ),
    label: str = "tab:best_configs",
) -> str:
    """Build LaTeX table from tuning-only best configurations."""
    best_configs = _load_tuning_best_configs(src_dir)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Embed} & \textbf{Filters} & \textbf{Hidden} & \textbf{BiLSTM} & \textbf{LR} & \textbf{Dropout} & \textbf{WD} \\",
        r"\midrule",
    ]

    for model in ("cnn", "lstm"):
        row = _tuning_best_config_row(model, best_configs[model])
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_tuning_best_config_latex_table(out_dir: Path, src_dir: Path) -> Path:
    """Write tuning-only best-config LaTeX table for report inclusion."""
    table_text = build_tuning_best_config_latex_table(src_dir=src_dir)
    output_path = out_dir / "best_config_tuning_only.tex"
    output_path.write_text(table_text + "\n", encoding="utf-8")
    return output_path


def build_main_vs_ablation_latex_table(
    runs: list[RunResult],
    main_seq_len: int = 128,
    caption: str = (
        "Main comparison at sequence length 128 (CNN vs. LSTM), "
        "followed by sequence-length ablations. "
        "Embed: embedding dimension, WD: weight decay."
    ),
    label: str = "tab:main_vs_ablation",
) -> str:
    """Build LaTeX table with main seq-length comparison and ablation rows."""
    main_rows: list[RunResult] = []
    for model in ("cnn", "lstm"):
        match = next(
            (
                run
                for run in runs
                if run.model.lower() == model and run.seq_len == main_seq_len
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"Missing {model.upper()} run for seq_len={main_seq_len}; cannot build main table."
            )
        main_rows.append(match)

    ablation_rows = sorted(
        [run for run in runs if run.seq_len != main_seq_len],
        key=lambda item: (0 if item.model.lower() == "cnn" else 1, item.seq_len),
    )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllllllll}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Seq} & \textbf{Embed} & \textbf{Filters} & \textbf{Hidden} & \textbf{BiLSTM} & \textbf{LR} & \textbf{Dropout} & \textbf{WD} \\",
        r"\midrule",
        rf"\multicolumn{{9}}{{l}}{{\textit{{Main model comparison (seq length = {main_seq_len})}}}} \\",
    ]

    for run in main_rows:
        row = _best_config_table_row(run)
        lines.append(" & ".join(f"\\textbf{{{value}}}" for value in row) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Ablation: sequence length}} \\")

    for run in ablation_rows:
        lines.append(" & ".join(_best_config_table_row(run)) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _select_main_and_ablation_runs(
    runs: list[RunResult], main_seq_len: int
) -> tuple[list[RunResult], list[RunResult]]:
    """Split runs into main rows (CNN/LSTM at one seq) and ablation rows."""
    main_rows: list[RunResult] = []
    for model in ("cnn", "lstm"):
        match = next(
            (
                run
                for run in runs
                if run.model.lower() == model and run.seq_len == main_seq_len
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"Missing {model.upper()} run for seq_len={main_seq_len}; cannot build table."
            )
        main_rows.append(match)

    ablation_rows = sorted(
        [run for run in runs if run.seq_len != main_seq_len],
        key=lambda item: (0 if item.model.lower() == "cnn" else 1, item.seq_len),
    )
    return main_rows, ablation_rows


def write_main_vs_ablation_latex_table(
    out_dir: Path, runs: list[RunResult], main_seq_len: int = 128
) -> Path:
    """Write LaTeX table for report inclusion via \\input{}."""
    table_text = build_main_vs_ablation_latex_table(runs, main_seq_len=main_seq_len)
    output_path = out_dir / "best_config_main_vs_ablation.tex"
    output_path.write_text(table_text + "\n", encoding="utf-8")
    return output_path


def _results_table_row(
    run: RunResult,
    best_scores: dict[str, float],
    emphasize_main: bool = False,
) -> list[str]:
    """Create one row for the results LaTeX table."""

    def format_metric(value: float, metric_key: str) -> str:
        text = f"{value:.4f}"
        is_best = abs(value - best_scores[metric_key]) < 1e-12
        return f"\\textbf{{{text}}}" if is_best else text

    model = run.model.upper()
    seq_len = str(run.seq_len)
    if emphasize_main:
        model = f"\\textbf{{{model}}}"
        seq_len = f"\\textbf{{{seq_len}}}"

    return [
        model,
        seq_len,
        format_metric(run.best_dev_f1, "best_dev_f1"),
        format_metric(run.test_accuracy, "test_accuracy"),
        format_metric(run.macro_f1, "macro_f1"),
    ]


def build_results_main_vs_ablation_latex_table(
    runs: list[RunResult],
    main_seq_len: int = 128,
    caption: str = (
        "Main comparison at sequence length 128 (CNN vs. LSTM), "
        "followed by sequence-length ablations. "
        "Dev F1 is the best development macro-F1 found during tuning. "
        "Best scores per metric column are bolded."
    ),
    label: str = "tab:test_results",
) -> str:
    """Build LaTeX results table with main and ablation sections."""
    main_rows, ablation_rows = _select_main_and_ablation_runs(runs, main_seq_len)
    best_scores = {
        "best_dev_f1": max(run.best_dev_f1 for run in runs),
        "test_accuracy": max(run.test_accuracy for run in runs),
        "macro_f1": max(run.macro_f1 for run in runs),
    }

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Seq Len} & \textbf{Dev F1} & \textbf{Test Acc} & \textbf{Macro F1} \\",
        r"\midrule",
        rf"\multicolumn{{5}}{{l}}{{\textit{{Main model comparison (seq length = {main_seq_len})}}}} \\",
    ]

    for run in main_rows:
        lines.append(
            " & ".join(_results_table_row(run, best_scores, emphasize_main=True))
            + r" \\",
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Ablation: sequence length}} \\")

    for run in ablation_rows:
        lines.append(" & ".join(_results_table_row(run, best_scores)) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_results_main_vs_ablation_latex_table(
    out_dir: Path, runs: list[RunResult], main_seq_len: int = 128
) -> Path:
    """Write LaTeX results table for report inclusion via \\input{}."""
    table_text = build_results_main_vs_ablation_latex_table(
        runs,
        main_seq_len=main_seq_len,
    )
    output_path = out_dir / "test_results_main_vs_ablation.tex"
    output_path.write_text(table_text + "\n", encoding="utf-8")
    return output_path


def build_tokenizer_coverage_latex_table(
    tokenizer_coverage_path: Path,
    caption: str = (
        "Tokenizer coverage summary for the shared BPE tokenizer used in all runs. "
        "OOV metrics are omitted. Tokens/Word indicates average subword split rate."
    ),
    label: str = "tab:tokenizer_coverage",
) -> str:
    """Build LaTeX table summarizing tokenizer coverage and split characteristics."""
    payload = json.loads(tokenizer_coverage_path.read_text(encoding="utf-8"))
    tokenizer_type = payload.get("tokenizer_type", "Tokenizer")
    vocab_size = int(payload.get("vocab_size", 0))
    splits = payload.get("splits", {})

    def _split_stats(split_name: str) -> dict:
        split = splits.get(split_name, {})
        return {
            "docs": int(split.get("total_documents", 0)),
            "types": int(split.get("total_types", 0)),
            "tokens": int(split.get("total_tokens", 0)),
            "words": int(split.get("total_words", 0)),
            "tokens_per_word": float(split.get("tokens_per_word", 0.0)),
        }

    def _format_int(value: int) -> str:
        return f"{value}"

    def _format_float(value: float, decimals: int = 4) -> str:
        return f"{value:.{decimals}f}"

    train_stats = _split_stats("train")
    dev_stats = _split_stats("dev")
    test_stats = _split_stats("test")

    test_bucket_stats = (
        splits.get("test", {}).get("oov_by_length_bucket", {})
        if isinstance(splits.get("test", {}), dict)
        else {}
    )
    test_total_docs = max(test_stats["docs"], 1)
    test_total_tokens = max(test_stats["tokens"], 1)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Split} & \textbf{Docs} & \textbf{Types} & \textbf{Tokens} & \textbf{Words} & \textbf{Tokens/Word} \\",
        r"\midrule",
        (
            "Train"
            f" & {_format_int(train_stats['docs'])}"
            f" & {_format_int(train_stats['types'])}"
            f" & {_format_int(train_stats['tokens'])}"
            f" & {_format_int(train_stats['words'])}"
            f" & {_format_float(train_stats['tokens_per_word'])} \\\\"
        ),
        (
            "Dev"
            f" & {_format_int(dev_stats['docs'])}"
            f" & {_format_int(dev_stats['types'])}"
            f" & {_format_int(dev_stats['tokens'])}"
            f" & {_format_int(dev_stats['words'])}"
            f" & {_format_float(dev_stats['tokens_per_word'])} \\\\"
        ),
        (
            "Test"
            f" & {_format_int(test_stats['docs'])}"
            f" & {_format_int(test_stats['types'])}"
            f" & {_format_int(test_stats['tokens'])}"
            f" & {_format_int(test_stats['words'])}"
            f" & {_format_float(test_stats['tokens_per_word'])} \\\\"
        ),
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Test split by length bucket}} \\",
        r"\textbf{Bucket} & \textbf{Docs} & \textbf{Doc Share (\%)} & \textbf{Tokens} & \textbf{Token Share (\%)} & \textbf{Tokens/Doc} \\",
        r"\midrule",
    ]

    for bucket_name in ("short", "medium", "long"):
        bucket = test_bucket_stats.get(bucket_name, {})
        bucket_docs = int(bucket.get("total_documents", 0))
        bucket_tokens = int(bucket.get("total_tokens", 0))
        doc_share = (bucket_docs / test_total_docs) * 100
        token_share = (bucket_tokens / test_total_tokens) * 100
        tokens_per_doc = (bucket_tokens / bucket_docs) if bucket_docs else 0.0

        lines.append(
            f"{bucket_name.capitalize()}"
            f" & {bucket_docs}"
            f" & {doc_share:.1f}"
            f" & {bucket_tokens}"
            f" & {token_share:.1f}"
            f" & {tokens_per_doc:.1f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                f"\\caption{{{caption} "
                f"(Tokenizer: {tokenizer_type}, vocab size = {vocab_size}.)}}"
            ),
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_tokenizer_coverage_latex_table(
    out_dir: Path,
    tokenizer_coverage_path: Path,
) -> Path:
    """Write tokenizer coverage LaTeX table for report inclusion via \\input{}."""
    table_text = build_tokenizer_coverage_latex_table(tokenizer_coverage_path)
    output_path = out_dir / "tokenizer_coverage_summary.tex"
    output_path.write_text(table_text + "\n", encoding="utf-8")
    return output_path


def plot_test_metrics_vs_seq(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot test accuracy and macro-F1 as a function of sequence length."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for model in ["cnn", "lstm"]:
        model_runs = sorted(
            [run for run in runs if run.model == model], key=lambda run: run.seq_len
        )
        seq = [run.seq_len for run in model_runs]
        accuracy = [run.test_accuracy for run in model_runs]
        macro_f1 = [run.macro_f1 for run in model_runs]

        axes[0].plot(seq, accuracy, marker="o", linewidth=2, label=model.upper())
        axes[1].plot(seq, macro_f1, marker="o", linewidth=2, label=model.upper())

    axes[0].set_title("Test Accuracy vs Max Sequence Length")
    axes[0].set_xlabel("Max Sequence Length")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test Macro-F1 vs Max Sequence Length")
    axes[1].set_xlabel("Max Sequence Length")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(alpha=0.3)

    for axis in axes:
        axis.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "test_metrics_vs_seq.png", dpi=200)
    plt.close(fig)


def plot_dev_vs_test_f1(out_dir: Path, runs: list[RunResult]) -> None:
    """Compare best tuning dev-F1 with final test macro-F1 per run."""
    runs_sorted = sorted(runs, key=lambda run: (run.model, run.seq_len))
    labels = [run.run_label for run in runs_sorted]
    x = list(range(len(labels)))
    dev = [run.best_dev_f1 for run in runs_sorted]
    test = [run.macro_f1 for run in runs_sorted]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.38
    ax.bar([value - width / 2 for value in x], dev, width=width, label="Best Dev F1")
    ax.bar([value + width / 2 for value in x], test, width=width, label="Test Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("Best Dev F1 vs Final Test Macro F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dev_vs_test_f1.png", dpi=200)
    plt.close(fig)


def plot_cnn_lstm_delta_by_seq(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot CNN-LSTM deltas (CNN minus LSTM) per sequence length."""
    cnn_runs = {run.seq_len: run for run in runs if run.model == "cnn"}
    lstm_runs = {run.seq_len: run for run in runs if run.model == "lstm"}
    common_seq = sorted(set(cnn_runs.keys()) & set(lstm_runs.keys()))

    if not common_seq:
        return

    acc_delta = [
        cnn_runs[seq].test_accuracy - lstm_runs[seq].test_accuracy for seq in common_seq
    ]
    f1_delta = [cnn_runs[seq].macro_f1 - lstm_runs[seq].macro_f1 for seq in common_seq]

    x = list(range(len(common_seq)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.bar(
        [value - width / 2 for value in x],
        acc_delta,
        width=width,
        label="Accuracy Δ (CNN - LSTM)",
    )
    ax.bar(
        [value + width / 2 for value in x],
        f1_delta,
        width=width,
        label="Macro-F1 Δ (CNN - LSTM)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(seq) for seq in common_seq])
    ax.set_xlabel("Max Sequence Length")
    ax.set_ylabel("Delta")
    ax.set_title("CNN vs LSTM Performance Delta by Sequence Length")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cnn_lstm_delta_by_seq.png", dpi=220)
    plt.close(fig)


def _is_numeric_value(value: object) -> bool:
    """Return True for numeric (non-bool) values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def plot_best_hyperparameter_trends(
    out_dir: Path, runs: list[RunResult], model: str
) -> None:
    """Plot how each model's best hyperparameters vary with sequence length."""
    model_runs = sorted(
        [run for run in runs if run.model == model], key=lambda run: run.seq_len
    )
    if not model_runs:
        return

    param_keys = sorted(model_runs[0].best_config.keys())
    fig, axes = plt.subplots(
        len(param_keys),
        1,
        figsize=(10, 2.5 * len(param_keys)),
        sharex=True,
    )
    if len(param_keys) == 1:
        axes = [axes]

    seq = [run.seq_len for run in model_runs]

    for axis, param in zip(axes, param_keys):
        values = [run.best_config[param] for run in model_runs]

        if all(_is_numeric_value(value) for value in values):
            axis.plot(seq, values, marker="o", linewidth=2)
            axis.set_ylabel(param)
        else:
            categories = []
            for value in values:
                label = str(value)
                if label not in categories:
                    categories.append(label)
            codes = [categories.index(str(value)) for value in values]
            axis.plot(seq, codes, marker="o", linewidth=2)
            axis.set_yticks(range(len(categories)))
            axis.set_yticklabels(categories)
            axis.set_ylabel(param)

        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Max Sequence Length")
    fig.suptitle(f"Best Hyperparameters vs Sequence Length ({model.upper()})", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"best_hyperparams_trend_{model}.png", dpi=220)
    plt.close(fig)


def plot_generalization_gap_by_seq(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot best-dev minus test-macro F1 gap across sequence lengths."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model in ["cnn", "lstm"]:
        model_runs = sorted(
            [run for run in runs if run.model == model], key=lambda run: run.seq_len
        )
        seq = [run.seq_len for run in model_runs]
        gaps = [run.best_dev_f1 - run.macro_f1 for run in model_runs]
        ax.plot(seq, gaps, marker="o", linewidth=2, label=model.upper())

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xlabel("Max Sequence Length")
    ax.set_ylabel("Generalization Gap (Best Dev F1 - Test Macro F1)")
    ax.set_title("Generalization Gap by Sequence Length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap_by_seq.png", dpi=220)
    plt.close(fig)


def plot_class_delta_by_seq(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot class-wise CNN-LSTM F1 deltas across sequence lengths."""
    cnn_runs = {run.seq_len: run for run in runs if run.model == "cnn"}
    lstm_runs = {run.seq_len: run for run in runs if run.model == "lstm"}
    common_seq = sorted(set(cnn_runs.keys()) & set(lstm_runs.keys()))
    if not common_seq:
        return

    class_ids = sorted({class_id for run in runs for class_id in run.class_f1.keys()})

    fig, ax = plt.subplots(figsize=(10, 5))
    for class_id in class_ids:
        deltas = [
            cnn_runs[seq].class_f1.get(class_id, 0.0)
            - lstm_runs[seq].class_f1.get(class_id, 0.0)
            for seq in common_seq
        ]
        ax.plot(
            common_seq,
            deltas,
            marker="o",
            linewidth=2,
            label=CLASS_LABELS.get(class_id, f"Class {class_id}"),
        )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_xlabel("Max Sequence Length")
    ax.set_ylabel("Class F1 Delta (CNN - LSTM)")
    ax.set_title("Class-wise CNN vs LSTM Delta by Sequence Length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "class_delta_by_seq.png", dpi=220)
    plt.close(fig)


def plot_hyperparameter_distributions(out_dir: Path, src_dir: Path) -> None:
    """Plot Dev-F1 distributions from dedicated model tuning runs."""
    labels: list[str] = []
    distributions: list[list[float]] = []
    models_present: list[str] = []

    for model in ["cnn", "lstm"]:
        tuning_json = (
            src_dir
            / f"experiment_{model}_tuning"
            / f"hyperparameter_tuning_{model}.json"
        )
        if not tuning_json.exists():
            continue

        payload = json.loads(tuning_json.read_text(encoding="utf-8"))
        all_results = payload.get("all_results", [])
        dev_f1_scores = [
            float(item["dev_f1"]) for item in all_results if "dev_f1" in item
        ]
        if not dev_f1_scores:
            continue

        labels.append(f"{model.upper()} tuning")
        distributions.append(dev_f1_scores)
        models_present.append(model)

    if not distributions:
        return

    width = max(4, len(distributions) * 2.5)
    fig, ax = plt.subplots(figsize=(width, 5))
    bp = ax.boxplot(distributions, patch_artist=True, tick_labels=labels)

    palette = sns.color_palette("Set2", 2)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(palette[0] if models_present[i] == "cnn" else palette[1])
        patch.set_alpha(0.7)

    ax.set_ylabel("Dev F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Hyperparameter Search Dev-F1 Distributions (Tuning Only)")
    ax.tick_params(axis="x")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hyperparameter_devf1_distributions.png", dpi=200)
    plt.close(fig)


def plot_class_f1_heatmap(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot class-wise test F1 heatmap across all runs."""
    runs_sorted = sorted(runs, key=lambda run: (run.model, run.seq_len))
    class_ids = sorted({class_id for run in runs_sorted for class_id in run.class_f1})

    matrix = []
    for run in runs_sorted:
        matrix.append([run.class_f1.get(class_id, 0.0) for class_id in class_ids])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        xticklabels=[
            CLASS_LABELS.get(class_id, f"Class {class_id}") for class_id in class_ids
        ],
        yticklabels=[run.run_label for run in runs_sorted],
        vmin=0.80,
        vmax=1.00,
        ax=ax,
    )
    ax.set_title("Class-wise Test F1 Across Runs")
    fig.tight_layout()
    fig.savefig(out_dir / "class_f1_heatmap.png", dpi=220)
    plt.close(fig)


def plot_full_training_curves(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot full training trajectories for final training runs."""
    if not any(run.training_history for run in runs):
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    metric_specs = [
        ("train_loss", "Train Loss", axes[0][0]),
        ("val_loss", "Validation Loss", axes[0][1]),
        ("train_acc", "Train Accuracy", axes[1][0]),
        ("val_acc", "Validation Accuracy", axes[1][1]),
    ]

    for run in sorted(runs, key=lambda item: (item.model, item.seq_len)):
        if not run.training_history:
            continue

        epochs = [point.epoch for point in run.training_history]
        style = "-" if run.model == "cnn" else "--"

        for metric_name, title, axis in metric_specs:
            values = [getattr(point, metric_name) for point in run.training_history]
            axis.plot(
                epochs,
                values,
                linestyle=style,
                marker="o",
                markersize=3,
                linewidth=1.7,
                label=run.run_label,
            )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "full_training_curves_comparison.png", dpi=220)
    plt.close(fig)


def plot_f1_per_epoch(out_dir: Path, runs: list[RunResult]) -> None:
    """Plot epoch-level train/validation F1 curves for final training runs."""
    if not any(run.training_history for run in runs):
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=True)
    metric_specs = [
        ("train_f1", "Train F1 per Epoch", axes[0]),
        ("val_f1", "Validation F1 per Epoch", axes[1]),
    ]

    for run in sorted(runs, key=lambda item: (item.model, item.seq_len)):
        if not run.training_history:
            continue

        epochs = [point.epoch for point in run.training_history]
        style = "-" if run.model == "cnn" else "--"

        for metric_name, title, axis in metric_specs:
            values = [getattr(point, metric_name) for point in run.training_history]
            axis.plot(
                epochs,
                values,
                linestyle=style,
                marker="o",
                markersize=3,
                linewidth=1.7,
                label=run.run_label,
            )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("F1")
            axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "f1_per_epoch_comparison.png", dpi=220)
    plt.close(fig)


def collect_runs(src_dir: Path) -> list[RunResult]:
    """Load all run directories in assignment2/src."""
    log_map = map_run_to_log(src_dir)
    run_dirs = sorted(
        path
        for path in src_dir.iterdir()
        if path.is_dir() and RUN_DIR_PATTERN.match(path.name)
    )

    runs = [load_run_result(run_dir, log_map) for run_dir in run_dirs]
    if not runs:
        raise RuntimeError(f"No run directories found in {src_dir}")
    return runs


def main() -> None:
    """Main entry point."""
    args = parse_args()
    src_dir = args.src_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    runs = collect_runs(src_dir)
    tokenizer_coverage_path = (
        src_dir / "experiment_lstm_seq256" / "tokenizer_coverage.json"
    )
    if not tokenizer_coverage_path.exists():
        raise FileNotFoundError(
            f"Missing tokenizer coverage file: {tokenizer_coverage_path}"
        )

    write_csv_summaries(out_dir, runs)
    write_best_config_summary(out_dir, runs)
    write_tuning_best_config_latex_table(out_dir, src_dir)
    write_results_main_vs_ablation_latex_table(out_dir, runs, main_seq_len=128)
    write_tokenizer_coverage_latex_table(out_dir, tokenizer_coverage_path)
    plot_test_metrics_vs_seq(out_dir, runs)
    plot_dev_vs_test_f1(out_dir, runs)
    plot_cnn_lstm_delta_by_seq(out_dir, runs)
    plot_generalization_gap_by_seq(out_dir, runs)
    plot_class_delta_by_seq(out_dir, runs)
    plot_best_hyperparameter_trends(out_dir, runs, model="cnn")
    plot_best_hyperparameter_trends(out_dir, runs, model="lstm")
    plot_hyperparameter_distributions(out_dir, src_dir)
    plot_class_f1_heatmap(out_dir, runs)
    plot_full_training_curves(out_dir, runs)
    plot_f1_per_epoch(out_dir, runs)

    print(f"Loaded {len(runs)} runs.")
    print(f"Saved visualizations and CSV summaries to: {out_dir}")
    print("Generated files:")
    for path in sorted(out_dir.glob("*")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()


"""
Best hyperparameters do change with sequence length, especially for LSTM.
CNN is mostly stable across seq lengths (same core best config; dropout shifts from 0.3 at 64 to 0.5 at 128/256).
LSTM best config shifts more (hidden_dim/embed_dim/weight_decay move with seq length), suggesting stronger interaction with context length.
Both models peak at seq=256; seq=128 looks like a local dip.
Business and Sci/Tech remain hardest classes; Sports is consistently easiest.
CNN’s class-wise advantage narrows at seq=256 (and may reverse on some classes), which is worth discussing.
"""
