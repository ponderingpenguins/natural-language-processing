import json

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from safetensors.torch import load_file
from sklearn.metrics import classification_report

from models.lstm import LSTMClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

model_path = "lstm_output/run-0/checkpoint-1"
config_path = f"lstm_output/config.json"

with open(config_path) as f:
    config = OmegaConf.create(json.load(f))

model = LSTMClassifier(config, device=DEVICE)
state_dict = load_file(f"{model_path}/model.safetensors")
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

masked_test_set = load_from_disk("masked_test_set")
regular_test_set = load_from_disk("sh0416/ag_news_tokenized")["test"]


def evaluate(model, dataset, batch_size=64):
    """Evaluate the model on the given dataset and return a classification report."""
    all_preds = []
    all_labels = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
        labels = torch.tensor(batch["labels"]).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return classification_report(all_labels, all_preds, digits=4, zero_division=0)


label_names = ["World", "Sports", "Business", "Sci/Tech"]  # AG News

print("=== Regular test set ===")
print(evaluate(model, regular_test_set))

print("=== Masked test set (top-20 TF-IDF keywords replaced) ===")
print(evaluate(model, masked_test_set))
