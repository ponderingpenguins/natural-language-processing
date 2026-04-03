"""
Main training script for text classification using BERT and LSTM models with hyperparameter tuning.

This script loads the dataset, initializes the model, and runs hyperparameter tuning using Hugging Face's Trainer API. The configuration is managed using OmegaConf with support for YAML files and CLI overrides.
"""

from typing import cast

import torch
from datasets import DatasetDict
from omegaconf import OmegaConf
from penguinlp.helpers import logger

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import dataset_prep, try_load_tokenized_data
from utils.training import hyperparameter_tuning

# Set the device for training (GPU if available, otherwise CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


def main() -> None:
    """Main function to run the training pipeline with OmegaConf YAML config and CLI overrides."""
    # Load config from YAML file (default: configs/bert.yaml)
    bert_model_cfg = OmegaConf.load("configs/bert.yaml")
    lstm_model_cfg = OmegaConf.load("configs/lstm.yaml")
    dataset_cfg = OmegaConf.load("configs/dataset.yaml")
    training_cfg = OmegaConf.load("configs/training.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(lstm_model_cfg, bert_model_cfg, dataset_cfg, cli_cfg)
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    data = dataset_prep(cfg.dataset)
    logger.info("Initializing model and tokenizer...")
    model = BertClassifier(cfg.bert_model, device=DEVICE)
    model = LSTMClassifier(cfg.lstm_model, device=DEVICE)

    # Load tokenized data from disk if it exists, otherwise tokenize and save to disk for future runs
    tokenized_data_path = f"./{cfg.dataset.hf_dataset}_tokenized"
    tokenization_config = {
        "hf_dataset": cfg.dataset.hf_dataset,
        "model": model.__class__.__name__,
        "tokenizer_name": getattr(model, "tokenizer", None).__class__.__name__,
        "max_samples": cfg.dataset.max_samples,
        "sequence_length": getattr(cfg.lstm_model, "sequence_length", None),
        "max_length": getattr(cfg.bert_model, "max_length", None),
    }
    data = try_load_tokenized_data(
        tokenized_data_path, data, model.tokenize, tokenization_config
    )
    data = cast(DatasetDict, data)

    logger.info("Starting hyperparameter search for the model...")
    # Drop unnecessary columns before training
    data = DatasetDict(
        {
            split: ds.remove_columns(
                [c for c in cfg.dataset.cols_to_drop if c in ds.column_names]
            )
            for split, ds in data.items()
        }
    )

    # Print number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized with %d parameters", num_params)

    # search_results = hyperparameter_tuning(
    #     cfg=OmegaConf.merge(cfg.bert_model, training_cfg.training),
    #     data=data,
    #     model_fn=lambda: BertClassifier(cfg.bert_model, device=DEVICE),
    # )
    # logger.info(
    #     "Hyperparameter search completed. Best trial: %s", search_results["best"]
    # )

    search_results = hyperparameter_tuning(
        cfg=OmegaConf.merge(cfg.lstm_model, training_cfg.training),
        data=data,
        model_fn=lambda: LSTMClassifier(cfg.lstm_model, device=DEVICE),
    )
    logger.info(
        "Hyperparameter search completed. Best trial: %s", search_results["best"]
    )


if __name__ == "__main__":
    main()
