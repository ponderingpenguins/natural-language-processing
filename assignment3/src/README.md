# Assignment 3: Transformer Fine-tuning, Robustness, and Limitations

This repository contains code for AG News text classification with two model families:

- BERT-based classifier (fine-tuning a pretrained encoder)
- LSTM baseline classifier

The project also includes two robustness analyses required by the assignment:

- Keyword masking probe
- Label-noise sensitivity / scaling-law style data-fraction study

## Project Scope

The assignment requirements are described in [instructions.md](instructions.md). This codebase is organized to support:

- model training and hyperparameter tuning
- model evaluation on regular and masked test sets
- robustness experiments with reduced training fractions
- reporting artifacts for the final write-up

## Repository Layout

- [main.py](main.py): runs hyperparameter tuning for BERT and LSTM
- [evaluation.py](evaluation.py): evaluates a saved LSTM checkpoint on regular vs masked test sets
- [label_noise_sensitivity.py](label_noise_sensitivity.py): runs 25/50/100 percent data-fraction experiments for both models
- [data-exploration.ipynb](data-exploration.ipynb): TF-IDF exploration and keyword masking dataset creation
- [configs/bert.yaml](configs/bert.yaml): BERT training and search config
- [configs/lstm.yaml](configs/lstm.yaml): LSTM training and search config
- [configs/dataset.yaml](configs/dataset.yaml): dataset split/preprocessing config
- [models/](models): model implementations
- [utils/](utils): dataset + training utilities

Generated experiment artifacts:

- [bert_output/](bert_output): BERT search outputs/checkpoints
- [lstm_output/](lstm_output): LSTM search outputs/checkpoints
- [masked_test_set/](masked_test_set): keyword-masked test split saved to disk
- [scaling_law_results/](scaling_law_results): data-fraction experiment checkpoints/plots/tables

## Environment Setup

This project uses Python 3.12+ and a local package dependency on `penguinlp`.

1. Create and sync the environment with uv:

```bash
uv sync
```

2. Confirm that the local dependency path exists:

- [../../penguinlp](../../penguinlp)

3. Run scripts through uv:

```bash
uv run python main.py
```

## Data and Preprocessing

Dataset loading and preprocessing is handled in [utils/dataset.py](utils/dataset.py):

- loads `sh0416/ag_news` via Hugging Face datasets
- creates train/dev split from the original train set using stratification
- constructs `text = title + description`
- applies HTML/escaping cleanup
- converts 1-based labels to 0-based if needed
- renames `label` to `labels` for Trainer compatibility

Important default in [configs/dataset.yaml](configs/dataset.yaml):

- `max_samples: 100`

This is useful for quick debugging but produces tiny-data results. For full training, override from CLI.

Example:

```bash
uv run python main.py dataset.max_samples=null
```

## Training and Hyperparameter Tuning

Run both BERT and LSTM hyperparameter search from [main.py](main.py):

```bash
uv run python main.py dataset.max_samples=null
```

What this does:

- loads and merges config from [configs/bert.yaml](configs/bert.yaml), [configs/lstm.yaml](configs/lstm.yaml), and [configs/dataset.yaml](configs/dataset.yaml)
- tokenizes and caches data to `./<hf_dataset>_tokenized`
- runs Optuna-backed search (learning rate + train batch size)
- saves best config to:
	- `bert_output/config.json`
	- `lstm_output/config.json`

## Robustness Evaluation 1: Keyword Masking Probe

Keyword masking dataset creation is implemented in [data-exploration.ipynb](data-exploration.ipynb):

- compute class-wise TF-IDF terms
- export scores to [tfidf_scores_by_label.json](tfidf_scores_by_label.json)
- replace top keywords in test text with tokenizer mask token
- re-tokenize and save masked split to [masked_test_set/](masked_test_set)

Then evaluate a trained LSTM checkpoint on regular and masked test sets:

```bash
uv run python evaluation.py
```

Note: [evaluation.py](evaluation.py) currently points to a fixed checkpoint path (`lstm_output/run-0/checkpoint-1`). Update `model_path` and `config_path` in that file if your best run differs.

## Robustness Evaluation 2: Label-Noise Sensitivity

Run scaling-law style data-fraction experiments for both models:

```bash
uv run python label_noise_sensitivity.py
```

Default behavior in [label_noise_sensitivity.py](label_noise_sensitivity.py):

- fractions: 0.25, 0.50, 1.00
- runs per fraction: 3
- metrics: test accuracy and weighted F1
- outputs saved under [scaling_law_results/](scaling_law_results)

Main summary table:

- [scaling_law_results/scaling_law_results.txt](scaling_law_results/scaling_law_results.txt)

## Reproducibility Notes

- Seeds are set in [utils/training.py](utils/training.py) for random/NumPy/PyTorch.
- Device selection prefers CUDA, then Apple MPS, else CPU.
- Tokenized datasets are cached to disk for repeatability and speed.

## Current Status and Caveats

- Existing scaling-law summary values are very low and likely reflect the `max_samples: 100` setting.
- For report-quality numbers, rerun with full dataset and track the best checkpoints used for each comparison.
