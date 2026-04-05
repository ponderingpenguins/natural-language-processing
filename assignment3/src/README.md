# Assignment 3: Transformer Fine-tuning, Robustness, and Limitations

This repository contains AG News text classification code with two model families:

- DistilBERT classifier (fine-tuned pretrained encoder)
- LSTM baseline classifier

The robustness part includes:

- Length-bucket + keyword-masking slices (`robustness_slices.py`)
- Reduced-training-size sensitivity (`label_noise_sensitivity.py`)

## Project Scope

Assignment requirements are in [instructions.md](instructions.md).
This codebase supports:

- model training and optional hyperparameter search
- evaluation on regular test (and masked test when provided)
- robustness experiments
- report artifact generation

## Repository Layout

- [main.py](main.py): train BERT/LSTM; optional hyperparameter search
- [evaluation.py](evaluation.py): evaluate saved BERT/LSTM checkpoint
- [robustness_slices.py](robustness_slices.py): length buckets + keyword masking (+ random control)
- [label_noise_sensitivity.py](label_noise_sensitivity.py): 25/50/100 percent train-size sensitivity
- [generate_comparison_table.py](generate_comparison_table.py): build final comparison tables
- [run_assignment3_pipeline.py](run_assignment3_pipeline.py): run eval + slices + tables in one command
- [configs/](configs): dataset/model/training configs
- [models/](models): model implementations
- [utils/](utils): dataset + training utilities

Main generated artifacts:

- `bert_output/`
- `bert_evaluation_full/`
- `lstm_evaluation_updated_full/`
- `robustness_slices/`
- `comparison_tables/`
- `scaling_law_results/`

## Environment Setup

Use Python 3.12+ and `uv`:

```bash
uv sync
```

Local dependency path expected by scripts:

- `../../penguinlp`

Run commands with:

```bash
uv run python <script>.py ...
```

## Data and Preprocessing

Dataset handling is in [utils/dataset.py](utils/dataset.py):

- loads `sh0416/ag_news` from Hugging Face
- uses official AG News train/test
- creates stratified train/dev split from train (`seed=67`)
- constructs `text = title + description`
- normalizes text and converts labels to 0-based when needed

Current defaults in [configs/dataset.yaml](configs/dataset.yaml):

- `max_samples: null`
- `eval_max_samples: null`

For quick debug runs:

```bash
uv run python main.py dataset.max_samples=1000 dataset.eval_max_samples=200
```

## Training and Hyperparameter Search

Single-run training:

```bash
uv run python main.py run.model_type=bert run.use_hp_search=false dataset.max_samples=null dataset.eval_max_samples=null
uv run python main.py run.model_type=lstm run.use_hp_search=false dataset.max_samples=null dataset.eval_max_samples=null
```

Hyperparameter search (optional):

```bash
uv run python main.py run.model_type=bert run.use_hp_search=true
uv run python main.py run.model_type=lstm run.use_hp_search=true
```

## Robustness Evaluation 1: Slices

Run:

```bash
uv run python robustness_slices.py --models lstm bert cnn_a2
```

Outputs:

- `robustness_slices/length_buckets.json`
- `robustness_slices/keyword_masking.json`
- `robustness_slices/summary.md`

## Robustness Evaluation 2: Reduced Training Size

Run:

```bash
uv run python label_noise_sensitivity.py --models bert lstm --fractions 0.25 0.5 1.0 --runs-per-fraction 1
```

Outputs are saved under `scaling_law_results/`.

## Standard Evaluation and Tables

Evaluate saved checkpoints:

```bash
uv run python evaluation.py --model-type bert
uv run python evaluation.py --model-type lstm
```

Generate final comparison table:

```bash
uv run python generate_comparison_table.py
```

One-command run (eval + slices + tables):

```bash
uv run python run_assignment3_pipeline.py
```

## Reproducibility Notes

- Seeds are set in [utils/training.py](utils/training.py).
- Device selection is CUDA -> MPS -> CPU.
- `REQUIRE_CUDA=1` can be used to block CPU fallback.
