"""
Evaluation script for the trained LSTM model on regular and masked test sets.

This script loads the trained LSTM model from disk, evaluates it on both the regular test set and a masked test set (where top-20 TF-IDF keywords have been replaced), and prints classification reports for both evaluations.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import ClassLabel, load_from_disk
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


def load_model(model_type: str, model_path: str, config_path: str):
    """Load the trained model and its configuration from disk."""

    with open(config_path, encoding="utf-8") as f:
        config = OmegaConf.create(json.load(f))

    model = (
        LSTMClassifier(config, device=DEVICE)
        if model_type == "lstm"
        else BertClassifier(config, device=DEVICE)
    )

    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    return model


def evaluate(model, dataset, batch_size=16, suffix="", output_dir="."):
    """Evaluate the model on the given dataset and return a classification report."""
    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i : i + batch_size]
        input_tensors = [
            torch.tensor(ids, dtype=torch.long) for ids in batch["input_ids"]
        ]
        lengths = torch.tensor(
            [len(ids) for ids in batch["input_ids"]], dtype=torch.long
        )
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=0).to(
            DEVICE
        )
        labels = torch.tensor(batch["labels"], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, lengths=lengths)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Generate a classification report with 4 decimal places and handle zero division cases
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    # Save the classification report to a text file
    with open(
        f"{output_dir}/classification_report{suffix}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(report)

    # Confusion matrix

    cm = confusion_matrix(all_labels, all_preds)

    # plot it using seaborn

    names = ["World", "Sports", "Business", "Sci/Tech"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion-matrix{suffix}.png")
    plt.close()

    # save all misclassified examples to a file for error analysis
    misclassified = []
    for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
        if true_label != pred_label:
            misclassified.append(
                {
                    "text": dataset[i]["text"],
                    "true_label": true_label.item(),
                    "predicted_label": pred_label.item(),
                    "predicted_label_name": names[pred_label],
                    "true_label_name": names[true_label],
                }
            )

    with open(
        f"{output_dir}/misclassified_examples{suffix}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(misclassified, f, indent=4, ensure_ascii=False)


def main():
    """Main function to load the model, prepare the datasets, and evaluate on both regular and masked test sets."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per evaluation split (regular/masked test).",
    )
    args = parser.parse_args()

    MODEL_PATH = "lstm_output/run-0/checkpoint-626"
    CONFIG_PATH = "lstm_output/config.json"
    MODEL_TYPE = "lstm"
    OUTPUT_DIR = f"{MODEL_TYPE}_evaluation"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = load_model(MODEL_TYPE, MODEL_PATH, CONFIG_PATH)

    masked_test_set = load_from_disk("masked_test_set")

    dataset_cfg = OmegaConf.load("configs/dataset.yaml")
    dataset_cfg.dataset.max_samples = None

    eval_max_samples = args.eval_max_samples
    if eval_max_samples is None:
        eval_max_samples = getattr(dataset_cfg.dataset, "eval_max_samples", None)

    data = dataset_prep(dataset_cfg.dataset)
    tokenization_config = {
        "hf_dataset": dataset_cfg.dataset.hf_dataset,
        "model": model.__class__.__name__,
        "tokenizer_name": getattr(model, "tokenizer", None).__class__.__name__,
        "max_samples": dataset_cfg.dataset.max_samples,
        "eval_max_samples": eval_max_samples,
    }
    regular_test_set = try_load_tokenized_data(
        f"./{dataset_cfg.dataset.hf_dataset.replace('/', '__')}_tokenized_full",
        data,
        model.tokenize,
        tokenization_config,
    )["test"]

    # Needed for the stratified sampling later
    names = [
        dataset_cfg.dataset.label_mapping[k]
        for k in sorted(dataset_cfg.dataset.label_mapping)
    ]
    regular_test_set = regular_test_set.cast_column(
        "labels",
        ClassLabel(
            num_classes=len(dataset_cfg.dataset.label_mapping),
            names=names,
        ),
    )
    masked_test_set = masked_test_set.cast_column(
        "labels",
        ClassLabel(
            num_classes=len(dataset_cfg.dataset.label_mapping),
            names=names,
        ),
    )

    if eval_max_samples:
        logger.info(
            "Subsampling the test sets to %d examples each for faster evaluation.",
            eval_max_samples,
        )

        if len(regular_test_set) > eval_max_samples:
            regular_test_set = regular_test_set.train_test_split(
                test_size=eval_max_samples, stratify_by_column="labels"
            )["test"]

        if len(masked_test_set) > eval_max_samples:
            masked_test_set = masked_test_set.train_test_split(
                test_size=eval_max_samples, stratify_by_column="labels"
            )["test"]
    else:
        logger.info("Evaluating on full regular and masked test sets.")

    print("=== Regular test set ===")
    evaluate(
        model,
        regular_test_set,
        suffix=f"-regular-{MODEL_TYPE}",
        output_dir=OUTPUT_DIR,
    )

    print("=== Masked test set (top-20 TF-IDF keywords replaced) ===")
    evaluate(
        model,
        masked_test_set,
        suffix=f"-masked-{MODEL_TYPE}",
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
