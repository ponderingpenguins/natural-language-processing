"""Run evaluation artifacts for assignment 3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent


def run_step(args: list[str], env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, *args]
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SRC_DIR, check=True, env=env)


def ensure_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but was not detected. "
            "Use --allow-cpu only if you want CPU fallback."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-bert",
        action="store_true",
        help="Train BERT before evaluation steps.",
    )
    parser.add_argument(
        "--train-lstm",
        action="store_true",
        help="Train LSTM before evaluation steps.",
    )
    parser.add_argument(
        "--with-label-noise",
        action="store_true",
        help="Run reduced-training-size sensitivity after main steps.",
    )
    parser.add_argument(
        "--label-noise-output-dir",
        default="scaling_law_results",
        help="Output directory for sensitivity results.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback (default requires CUDA).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_cpu:
        ensure_cuda_available()

    step_env = os.environ.copy()
    if not args.allow_cpu:
        step_env["REQUIRE_CUDA"] = "1"

    if args.train_bert:
        run_step(
            [
                "main.py",
                "run.model_type=bert",
                "run.use_hp_search=false",
                "dataset.max_samples=null",
                "dataset.eval_max_samples=null",
            ],
            env=step_env,
        )

    if args.train_lstm:
        run_step(
            [
                "main.py",
                "run.model_type=lstm",
                "run.use_hp_search=false",
                "dataset.max_samples=null",
                "dataset.eval_max_samples=null",
            ],
            env=step_env,
        )

    run_step(["evaluation.py", "--model-type", "bert"], env=step_env)
    run_step(["evaluation.py", "--model-type", "lstm"], env=step_env)
    run_step(["robustness_slices.py"], env=step_env)
    run_step(["generate_comparison_table.py"], env=step_env)

    if args.with_label_noise:
        run_step(
            [
                "label_noise_sensitivity.py",
                "--output-dir",
                args.label_noise_output_dir,
            ],
            env=step_env,
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
