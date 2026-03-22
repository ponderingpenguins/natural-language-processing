"""Generate appendix section with misclassified examples for all final runs.

Run from repository root:
    src/.venv/bin/python src/generate_appendix.py

By default this writes:
    report/sections/8_appendix.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

RUN_ORDER: list[tuple[str, int]] = [
    ("cnn", 64),
    ("cnn", 128),
    ("cnn", 256),
    ("lstm", 64),
    ("lstm", 128),
    ("lstm", 256),
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    this_file = Path(__file__).resolve()
    src_dir = this_file.parent
    assignment_dir = src_dir.parent
    default_output = assignment_dir / "report" / "sections" / "8_appendix.tex"

    parser = argparse.ArgumentParser(
        description="Generate 8_appendix.tex from misclassified JSONL files."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=src_dir,
        help="Path to assignment2/src directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output .tex path (default: report/sections/8_appendix.tex).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum examples to include per run (default: 20).",
    )
    return parser.parse_args()


def sanitize_for_verbatim(text: str) -> str:
    """Prepare text for LaTeX Verbatim blocks."""
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return clean_text.replace(r"\end{Verbatim}", r"\end{Verba tim}")


def load_jsonl(path: Path) -> list[dict]:
    """Load newline-delimited JSON records from disk."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_heading_index(model: str, seq_len: int) -> int:
    """Return 1-based run number in the appendix section order."""
    for index, entry in enumerate(RUN_ORDER, start=1):
        if entry == (model, seq_len):
            return index
    raise ValueError(f"Unexpected run: {model}-{seq_len}")


def build_run_section(
    model: str, seq_len: int, rows: list[dict], max_examples: int
) -> str:
    """Build one run subsection block in the appendix format."""
    model_upper = model.upper()
    run_id = run_heading_index(model, seq_len)
    shown_rows = rows[:max_examples]

    lines: list[str] = [
        rf"\subsection{{{model_upper} (Max Sequence Length: {seq_len})}}",
        rf"\label{{app:run{run_id}}}",
        "",
        f"Total misclassified examples: {len(rows)}",
        "",
    ]

    for example_number, row in enumerate(shown_rows, start=1):
        gold = LABELS.get(int(row["true_label"]), str(row["true_label"]))
        predicted = LABELS.get(
            int(row["misclassified_as"]), str(row["misclassified_as"])
        )
        text = sanitize_for_verbatim(str(row["text"]))
        dataset_index = row["index"]

        lines.extend(
            [
                r"\noindent",
                (
                    rf"\textbf{{Example {example_number}}} \quad "
                    rf"\textit{{(Index {dataset_index}: Gold: {gold}, Predicted: {predicted})}}"
                ),
                "",
                r"\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\small]",
                text,
                r"\end{Verbatim}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def build_appendix(src_dir: Path, max_examples: int) -> str:
    """Build complete appendix text from all configured runs."""
    sections: list[str] = [
        "This appendix provides a listing of misclassified examples from the test set across all six experimental runs.",
        "Each example is labeled with its dataset index, gold label, and predicted label for reference during error analysis.",
        "",
    ]

    for model, seq_len in RUN_ORDER:
        run_dir = src_dir / f"experiment_{model}_seq{seq_len}"
        jsonl_path = run_dir / f"misclassified_examples_{model}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing file: {jsonl_path}")

        rows = load_jsonl(jsonl_path)
        sections.append(
            build_run_section(model, seq_len, rows, max_examples=max_examples)
        )
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    """Generate and write appendix file."""
    args = parse_args()
    src_dir = args.src_dir.resolve()
    output_path = args.output.resolve()

    appendix_text = build_appendix(src_dir, max_examples=args.max_examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(appendix_text, encoding="utf-8")

    print(f"Wrote appendix to: {output_path}")


if __name__ == "__main__":
    main()
