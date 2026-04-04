"""Evaluate a saved LSTM/BERT checkpoint on regular and masked AG News test sets."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import ClassLabel, Dataset, load_from_disk
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data

SRC_DIR = Path(__file__).resolve().parent
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if os.getenv("REQUIRE_CUDA", "0").lower() in {"1", "true", "yes"} and DEVICE.type != "cuda":
    raise RuntimeError(
        "REQUIRE_CUDA is enabled but CUDA is unavailable. Aborting to prevent CPU fallback."
    )

DEFAULT_MODEL_SPECS = {
    "bert": {
        "model_path": "bert_output/sanity_distilbert/checkpoint-20250",
        "config_path": "bert_output/sanity_distilbert/config.json",
        "output_dir": "bert_evaluation_full",
    },
    "lstm": {
        "model_path": "experiment_outputs/updated_default_full/checkpoint-10125",
        "config_path": "configs/lstm.yaml",
        "output_dir": "lstm_evaluation_updated_full",
    },
}


def _infer_lstm_architecture_from_state_dict(state_dict: dict) -> dict:
    """Infer LSTM architecture values needed to instantiate the model."""
    layer_indices = []
    bidirectional = False

    for key in state_dict:
        match = re.match(r"lstm\.weight_ih_l(\d+)(?:_reverse)?$", key)
        if not match:
            continue
        layer_indices.append(int(match.group(1)))
        if key.endswith("_reverse"):
            bidirectional = True

    inferred = {}
    if layer_indices:
        inferred["num_layers"] = max(layer_indices) + 1
    inferred["bidirectional"] = bidirectional
    return inferred


def _build_tokenized_cache_path(dataset_cfg, model_type: str, model_config) -> str:
    dataset_tag = dataset_cfg.hf_dataset.replace("/", "__")
    sample_tag = (
        "full" if getattr(dataset_cfg, "max_samples", None) is None else "sampled"
    )
    eval_tag = (
        "fulleval"
        if getattr(dataset_cfg, "eval_max_samples", None) is None
        else f"eval{dataset_cfg.eval_max_samples}"
    )
    if model_type == "bert":
        len_tag = f"maxlen{model_config.max_length}"
    else:
        len_tag = f"seq{model_config.sequence_length}"
    return f"./{dataset_tag}_tokenized_{model_type}_{len_tag}_{sample_tag}_{eval_tag}"


def load_model(model_type: str, model_path: str, config_path: str):
    """Load trained model and configuration from disk."""
    config_file = Path(config_path)
    if config_file.suffix in {".yaml", ".yml"}:
        raw_cfg = OmegaConf.load(str(config_file))
        key = f"{model_type}_model"
        config = raw_cfg[key] if key in raw_cfg else raw_cfg
    else:
        config = OmegaConf.create(json.loads(config_file.read_text(encoding="utf-8")))
    state_dict = load_file(f"{model_path}/model.safetensors")

    if model_type == "lstm":
        if hasattr(config, "init_embeddings_from_bert"):
            config.init_embeddings_from_bert = False
        inferred = _infer_lstm_architecture_from_state_dict(state_dict)
        for field, inferred_value in inferred.items():
            setattr(config, field, inferred_value)

    model = (
        LSTMClassifier(config, device=DEVICE)
        if model_type == "lstm"
        else BertClassifier(config, device=DEVICE)
    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Model/checkpoint mismatch after alignment. "
            f"Missing keys: {missing_keys}; Unexpected keys: {unexpected_keys}."
        )
    model.eval()
    model.to(DEVICE)
    return model, config


def evaluate(
    model,
    dataset: Dataset,
    label_names: list[str],
    batch_size: int,
    suffix: str,
    output_dir: Path,
) -> dict:
    """Evaluate a checkpoint and save reports."""
    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {suffix}"):
        batch = dataset[i : i + batch_size]
        input_tensors = [
            torch.tensor(ids, dtype=torch.long) for ids in batch["input_ids"]
        ]
        if "attention_mask" in batch:
            mask_tensors = [
                torch.tensor(mask, dtype=torch.long) for mask in batch["attention_mask"]
            ]
        else:
            mask_tensors = [
                torch.tensor([1 if token_id != 0 else 0 for token_id in ids])
                for ids in batch["input_ids"]
            ]

        attention_mask = pad_sequence(
            mask_tensors, batch_first=True, padding_value=0
        ).to(DEVICE)
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=0).to(
            DEVICE
        )
        labels = torch.tensor(batch["labels"], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    report_text = classification_report(
        all_labels, all_preds, target_names=label_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    (output_dir / f"classification_report{suffix}.txt").write_text(
        report_text, encoding="utf-8"
    )
    (output_dir / f"confusion_matrix{suffix}.json").write_text(
        json.dumps(cm.tolist(), indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion-matrix{suffix}.png")
    plt.close()

    misclassified = []
    for idx, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
        if int(true_label) == int(pred_label):
            continue
        sample = dataset[idx]
        misclassified.append(
            {
                "index": idx,
                "text": sample.get("text", ""),
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "predicted_label_name": label_names[int(pred_label)],
                "true_label_name": label_names[int(true_label)],
            }
        )

    (output_dir / f"misclassified_examples{suffix}.json").write_text(
        json.dumps(misclassified, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = {
        "accuracy": report_dict["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "num_examples": len(all_labels),
        "num_misclassified": len(misclassified),
    }
    (output_dir / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def _prepare_regular_and_masked_sets(
    model,
    model_type: str,
    model_config,
    eval_max_samples: int | None,
    masked_test_set_path: str,
):
    dataset_cfg = OmegaConf.load(SRC_DIR / "configs/dataset.yaml").dataset
    dataset_cfg.max_samples = None
    dataset_cfg.eval_max_samples = None

    data = dataset_prep(dataset_cfg)
    tokenized_data_path = _build_tokenized_cache_path(
        dataset_cfg=dataset_cfg, model_type=model_type, model_config=model_config
    )
    tokenization_config = {
        "hf_dataset": dataset_cfg.hf_dataset,
        "model_type": model_type,
        "model": model.__class__.__name__,
        "tokenizer_name": getattr(model, "tokenizer", None).__class__.__name__,
        "max_samples": dataset_cfg.max_samples,
        "eval_max_samples": dataset_cfg.eval_max_samples,
        "sequence_length": getattr(model_config, "sequence_length", None),
        "max_length": getattr(model_config, "max_length", None),
    }
    tokenized = try_load_tokenized_data(
        tokenized_data_path, data, model.tokenize, tokenization_config
    )
    regular_test_set = tokenized["test"]

    label_names = [dataset_cfg.label_mapping[k] for k in sorted(dataset_cfg.label_mapping)]
    regular_test_set = regular_test_set.cast_column(
        "labels",
        ClassLabel(num_classes=len(label_names), names=label_names),
    )

    masked_test_set = None
    masked_path = Path(masked_test_set_path)
    if not masked_path.is_absolute():
        masked_path = SRC_DIR / masked_path
    if masked_path.exists():
        masked_test_set = load_from_disk(str(masked_path))
        masked_test_set = masked_test_set.map(
            model.tokenize, batched=False, load_from_cache_file=False
        )
        masked_test_set = masked_test_set.cast_column(
            "labels",
            ClassLabel(num_classes=len(label_names), names=label_names),
        )
    else:
        logger.warning("Masked test set not found at %s, skipping masked evaluation.", masked_path)

    if eval_max_samples:
        if len(regular_test_set) > eval_max_samples:
            regular_test_set = regular_test_set.train_test_split(
                test_size=eval_max_samples,
                stratify_by_column="labels",
                seed=dataset_cfg.seed,
            )["test"]
        if masked_test_set is not None and len(masked_test_set) > eval_max_samples:
            masked_test_set = masked_test_set.train_test_split(
                test_size=eval_max_samples,
                stratify_by_column="labels",
                seed=dataset_cfg.seed,
            )["test"]

    return regular_test_set, masked_test_set, label_names


def _resolve_path(path: str | None, *, default: str) -> Path:
    """Resolve a user path. fall back to src relative default when omitted."""
    if path:
        provided = Path(path)
        if provided.is_absolute() or provided.exists():
            return provided
        src_relative = SRC_DIR / provided
        return src_relative
    return SRC_DIR / default


def main():
    """Load checkpoint, evaluate regular + masked test sets, and persist artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint and save reports/metrics."
    )
    parser.add_argument("--model-type", choices=["lstm", "bert"], default="lstm")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Checkpoint directory (default: model-specific).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Config file (.json/.yaml) (default: model-specific).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: model-specific).",
    )
    parser.add_argument(
        "--masked-test-set-path",
        default="masked_test_set",
        help="Masked test set path.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Optional stratified sample cap per split.",
    )
    args = parser.parse_args()

    defaults = DEFAULT_MODEL_SPECS[args.model_type]
    model_path = _resolve_path(args.model_path, default=defaults["model_path"])
    config_path = _resolve_path(args.config_path, default=defaults["config_path"])
    output_dir = _resolve_path(args.output_dir, default=defaults["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_config = load_model(
        model_type=args.model_type,
        model_path=str(model_path),
        config_path=str(config_path),
    )
    regular_test_set, masked_test_set, label_names = _prepare_regular_and_masked_sets(
        model=model,
        model_type=args.model_type,
        model_config=model_config,
        eval_max_samples=args.eval_max_samples,
        masked_test_set_path=args.masked_test_set_path,
    )

    batch_size = int(
        getattr(model_config, "per_device_eval_batch_size", 16)
    )

    regular_metrics = evaluate(
        model=model,
        dataset=regular_test_set,
        label_names=label_names,
        batch_size=batch_size,
        suffix=f"-regular-{args.model_type}",
        output_dir=output_dir,
    )
    logger.info("Regular metrics: %s", regular_metrics)

    if masked_test_set is not None:
        masked_metrics = evaluate(
            model=model,
            dataset=masked_test_set,
            label_names=label_names,
            batch_size=batch_size,
            suffix=f"-masked-{args.model_type}",
            output_dir=output_dir,
        )
        logger.info("Masked metrics: %s", masked_metrics)


if __name__ == "__main__":
    main()
