"""
Evaluation script for the trained LSTM model on regular and masked test sets.

This script loads the trained LSTM model from disk, evaluates it on both the regular test set and a masked test set (where top-20 TF-IDF keywords have been replaced), and prints classification reports for both evaluations.
"""

import json

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
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


def evaluate(model, dataset, batch_size=16):
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

    # Confusion matrix

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # plot it using seaborn

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["World", "Sports", "Business", "Sci/Tech"],
        yticklabels=["World", "Sports", "Business", "Sci/Tech"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # save all misclassified examples to a file for error analysis
    misclassified = []
    for i in range(len(all_labels)):
        if all_labels[i] != all_preds[i]:
            misclassified.append(
                {
                    "text": dataset[i]["text"],
                    "true_label": all_labels[i],
                    "predicted_label": all_preds[i],
                }
            )

    # with open("misclassified_examples.json", "w", encoding="utf-8") as f:
    # TypeError: Object of type int64 is not JSON serializable
    # for example in misclassified:
    #     example["true_label"] = example["true_label"].item()
    #     example["predicted_label"] = example["predicted_label"].item()
    # json.dump(misclassified, f, indent=4, ensure_ascii=False)

    return report


def main():
    """Main function to load the model, prepare the datasets, and evaluate on both regular and masked test sets."""

    model_path = "lstm_output/run-0/checkpoint-1"
    config_path = "lstm_output/config.json"

    model = load_model("lstm", model_path, config_path)

    masked_test_set = load_from_disk("masked_test_set")

    dataset_cfg = OmegaConf.load("configs/dataset.yaml")
    dataset_cfg.dataset.max_samples = None
    data = dataset_prep(dataset_cfg.dataset)
    regular_test_set = try_load_tokenized_data(
        f"./{dataset_cfg.dataset.hf_dataset.replace('/', '__')}_tokenized_full",
        data,
        model.tokenize,
    )["test"]

    label_names = ["World", "Sports", "Business", "Sci/Tech"]  # AG News

    print("=== Regular test set ===")
    print(evaluate(model, regular_test_set))

    print("=== Masked test set (top-20 TF-IDF keywords replaced) ===")
    print(evaluate(model, masked_test_set))


if __name__ == "__main__":
    main()
