"""
Main training script for text classification using BERT and LSTM models with hyperparameter tuning.

This script loads the dataset, initializes the model, and runs hyperparameter tuning using Hugging Face's Trainer API. The configuration is managed using OmegaConf with support for YAML files and CLI overrides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, cast

import torch
from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
from penguinlp.helpers import logger

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data
from utils.training import create_trainer, hyperparameter_tuning, set_seed

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


def _resolve_model_settings(cfg: DictConfig) -> tuple[str, bool]:
    """Resolve model type and whether to run hyperparameter search."""
    model_type = str(cfg.run.model_type).lower()
    if model_type not in {"bert", "lstm"}:
        raise ValueError(
            f"Unsupported run.model_type={model_type!r}. Choose from 'bert' or 'lstm'."
        )
    return model_type, bool(cfg.run.use_hp_search)


def _build_model_factory(
    cfg: DictConfig, model_type: str
) -> tuple[Callable[[], Any], DictConfig]:
    """Return a fresh-model factory plus model-specific config node."""
    if model_type == "bert":
        model_cfg = cfg.bert_model
        return lambda: BertClassifier(model_cfg, device=DEVICE), model_cfg

    model_cfg = cfg.lstm_model
    return lambda: LSTMClassifier(model_cfg, device=DEVICE), model_cfg


def _build_tokenized_cache_path(
    cfg: DictConfig, model_type: str, model_cfg: DictConfig
) -> str:
    """Build a cache path that avoids collisions across model/tokenization setups."""
    dataset_tag = cfg.dataset.hf_dataset.replace("/", "__")
    sample_tag = (
        "full"
        if getattr(cfg.dataset, "max_samples", None) is None
        else f"max{cfg.dataset.max_samples}"
    )
    eval_tag = (
        "fulleval"
        if getattr(cfg.dataset, "eval_max_samples", None) is None
        else f"eval{cfg.dataset.eval_max_samples}"
    )
    if model_type == "bert":
        len_tag = f"maxlen{model_cfg.max_length}"
    else:
        len_tag = f"seq{model_cfg.sequence_length}"
    cache_name = (
        f"{dataset_tag}_tokenized_{model_type}_{len_tag}_{sample_tag}_{eval_tag}"
    )
    return str(SRC_DIR / cache_name)


def _srcify_path(path_value: str) -> str:
    """Interpret relative paths as assignment3/src-relative paths."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(SRC_DIR / path)


def main() -> None:
    """Run the selected training flow with YAML+CLI configuration."""
    defaults = OmegaConf.create({"run": {"model_type": "bert", "use_hp_search": False}})
    cfg = OmegaConf.merge(
        defaults,
        OmegaConf.load(SRC_DIR / "configs/dataset.yaml"),
        OmegaConf.load(SRC_DIR / "configs/bert.yaml"),
        OmegaConf.load(SRC_DIR / "configs/lstm.yaml"),
        OmegaConf.load(SRC_DIR / "configs/training.yaml"),
        OmegaConf.from_cli(),
    )
    # Keep artifacts in assignment3/src even when launched from repo root.
    for node_name in ("bert_model", "lstm_model"):
        node = cfg[node_name]
        if "output_dir" in node:
            node.output_dir = _srcify_path(str(node.output_dir))
        if "logging_dir" in node:
            node.logging_dir = _srcify_path(str(node.logging_dir))

    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    model_type, use_hp_search = _resolve_model_settings(cfg)
    model_factory, model_cfg = _build_model_factory(cfg, model_type)
    model = model_factory()

    data = dataset_prep(cfg.dataset)
    tokenized_data_path = _build_tokenized_cache_path(cfg, model_type, model_cfg)
    tokenization_config = {
        "hf_dataset": cfg.dataset.hf_dataset,
        "model_type": model_type,
        "model": model.__class__.__name__,
        "tokenizer_name": getattr(model, "tokenizer", None).__class__.__name__,
        "max_samples": cfg.dataset.max_samples,
        "eval_max_samples": cfg.dataset.eval_max_samples,
        "sequence_length": getattr(model_cfg, "sequence_length", None),
        "max_length": getattr(model_cfg, "max_length", None),
    }
    data = try_load_tokenized_data(
        tokenized_data_path, data, model.tokenize, tokenization_config
    )
    data = cast(DatasetDict, data)

    data = DatasetDict(
        {
            split: ds.remove_columns(
                [c for c in cfg.dataset.cols_to_drop if c in ds.column_names]
            )
            for split, ds in data.items()
        }
    )
    run_cfg = OmegaConf.merge(model_cfg, cfg.training)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Initialized %s model with %d parameters on %s",
        model_type,
        num_params,
        DEVICE,
    )

    if use_hp_search:
        logger.info("Starting %s hyperparameter search...", model_type)
        search_results = hyperparameter_tuning(
            cfg=run_cfg,
            data=data,
            model_fn=model_factory,
        )
        logger.info(
            "Hyperparameter search completed. Best trial: %s", search_results["best"]
        )
        return

    logger.info("Starting single-run training for %s...", model_type)
    output_dir = Path(run_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(OmegaConf.to_container(model_cfg, resolve=True), indent=2),
        encoding="utf-8",
    )

    trainer = create_trainer(
        model_init=lambda _trial: model_factory(),
        data=data,
        cfg=run_cfg,
    )
    trainer.train()
    dev_metrics = trainer.evaluate(data["dev"])
    test_metrics = trainer.predict(data["test"], metric_key_prefix="test").metrics

    metrics_path = output_dir / "final_metrics.json"
    metrics_path.write_text(
        json.dumps({"dev_metrics": dev_metrics, "test_metrics": test_metrics}, indent=2),
        encoding="utf-8",
    )

    logger.info("Single-run finished. Dev metrics: %s", dev_metrics)
    logger.info("Single-run finished. Test metrics: %s", test_metrics)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
