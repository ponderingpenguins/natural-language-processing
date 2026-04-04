"""Generate Assignment 3 comparison tables from saved metric artifacts."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
OUT_DIR = SRC_DIR / "comparison_tables"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_assignment2_report(path: Path) -> tuple[float, float]:
    """Parse test accuracy and macro-F1 from sklearn text report."""
    text = path.read_text(encoding="utf-8")
    acc_match = re.search(r"^\s*accuracy\s+([0-9.]+)\s+\d+\s*$", text, flags=re.MULTILINE)
    macro_match = re.search(
        r"^\s*macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not acc_match or not macro_match:
        raise ValueError(f"Could not parse A2 metrics from {path}")
    return float(acc_match.group(1)), float(macro_match.group(1))


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def build_rows() -> list[dict]:
    a2_report = Path(
        os.getenv(
            "A2_REPORT_PATH",
            str(REPO_ROOT / "assignment2/src/experiment_cnn_seq64/classification_report.txt"),
        )
    )
    a2_train = Path(
        os.getenv(
            "A2_TRAIN_RESULTS_PATH",
            str(REPO_ROOT / "assignment2/src/experiment_cnn_seq64/training_results.json"),
        )
    )
    bert_final = Path(
        os.getenv(
            "BERT_FINAL_METRICS_PATH",
            str(SRC_DIR / "bert_output/sanity_distilbert/final_metrics.json"),
        )
    )
    lstm_metrics = Path(
        os.getenv(
            "LSTM_METRICS_PATH",
            str(SRC_DIR / "experiment_outputs/updated_default_full/metrics.json"),
        )
    )

    a2_test_acc, a2_test_macro = _parse_assignment2_report(a2_report)
    a2_train_data = _read_json(a2_train)
    a2_dev_acc = float(a2_train_data["val_metrics"]["acc"])
    a2_dev_macro = float(a2_train_data["val_metrics"]["f1"])

    bert = _read_json(bert_final)
    bert_dev_acc = float(bert["dev_metrics"]["eval_accuracy"])
    bert_dev_macro = float(bert["dev_metrics"]["eval_macro_f1"])
    bert_test_acc = float(bert["test_metrics"]["test_eval_accuracy"])
    bert_test_macro = float(bert["test_metrics"]["test_eval_macro_f1"])

    lstm = _read_json(lstm_metrics)
    lstm_dev_acc = float(lstm["dev_metrics"]["eval_accuracy"])
    lstm_dev_macro = float(lstm["dev_metrics"]["eval_macro_f1"])
    lstm_test_acc = float(lstm["test_metrics"]["test_accuracy"])
    lstm_test_macro = float(lstm["test_metrics"]["test_macro_f1"])

    return [
        {
            "model": "A2 CNN (seq64)",
            "category": "baseline",
            "dev_accuracy": a2_dev_acc,
            "dev_macro_f1": a2_dev_macro,
            "test_accuracy": a2_test_acc,
            "test_macro_f1": a2_test_macro,
        },
        {
            "model": "A3 LSTM (updated_full)",
            "category": "neural",
            "dev_accuracy": lstm_dev_acc,
            "dev_macro_f1": lstm_dev_macro,
            "test_accuracy": lstm_test_acc,
            "test_macro_f1": lstm_test_macro,
        },
        {
            "model": "A3 DistilBERT (fine-tuned)",
            "category": "transformer",
            "dev_accuracy": bert_dev_acc,
            "dev_macro_f1": bert_dev_macro,
            "test_accuracy": bert_test_acc,
            "test_macro_f1": bert_test_macro,
        },
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "category",
                "dev_accuracy",
                "dev_macro_f1",
                "test_accuracy",
                "test_macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# Model Comparison (Dev + Test)",
        "",
        "| Model | Category | Dev Accuracy | Dev Macro-F1 | Test Accuracy | Test Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['category']} | {_fmt(row['dev_accuracy'])} | "
            f"{_fmt(row['dev_macro_f1'])} | {_fmt(row['test_accuracy'])} | {_fmt(row['test_macro_f1'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[dict], path: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\begin{tabular}{l l r r r r}",
        "\\hline",
        "Model & Category & Dev Acc & Dev Macro-F1 & Test Acc & Test Macro-F1 \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['category']} & {_fmt(row['dev_accuracy'])} & "
            f"{_fmt(row['dev_macro_f1'])} & {_fmt(row['test_accuracy'])} & {_fmt(row['test_macro_f1'])} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Assignment 3 model comparison on AG News (same split).}",
            "\\label{tab:a3_model_comparison}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, OUT_DIR / "model_comparison.csv")
    write_markdown(rows, OUT_DIR / "model_comparison.md")
    write_latex(rows, OUT_DIR / "model_comparison.tex")
    print(f"Saved comparison tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
