import torch
from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf
from penguinlp.helpers import logger

from models.bert import BertClassifier
from models.lstm import LSTMClassifier
from utils.dataset import load_data, preprocess_data
from utils.training import hyperparameter_tuning

# Set the device for training (GPU if available, otherwise CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE


def dataset_prep(cfg: dict) -> DatasetDict:
    """Load and preprocess the dataset."""
    data = load_data(cfg)
    logger.info("Loaded dataset %s with splits: %s", cfg.hf_dataset, data.keys())
    # Preprocess the datasets using the defined pipeline.
    logger.info("Preprocessing splits...")
    data = preprocess_data(data)
    logger.info(
        "Dataset preprocessing complete. Sample from preprocessed dataset: %s",
        data["train"][0],
    )
    return data


def tokenize_data(data: DatasetDict, tokenization) -> DatasetDict:
    """Tokenize the datasets using the model's tokenizer."""
    # Tokenize the datasets using the model's tokenizer.
    for split_name in ["train", "dev", "test"]:
        logger.info("Tokenizing %s set...", split_name)
        data[split_name] = data[split_name].map(
            tokenization, batched=False, load_from_cache_file=False
        )
        logger.info(
            "Completed tokenization for %s set. Set size: %d",
            split_name,
            len(data[split_name]),
        )

    logger.info("Tokenization complete.")
    return data


def try_load_tokenized_data(tokenized_data_path, data, tokenization):
    try:
        logger.info("Attempting to load tokenized data from %s...", tokenized_data_path)
        data = load_from_disk(tokenized_data_path)
    except Exception as e:
        logger.warning(
            "Failed to load tokenized data from disk: %s. Tokenizing now...", e
        )
        data = tokenize_data(data, tokenization)
        # save to disk for future runs
        data.save_to_disk(tokenized_data_path)
        logger.info("Successfully loaded tokenized data from disk.")
    return data


def main() -> None:
    """Main function to run the training pipeline with OmegaConf YAML config and CLI overrides."""
    # Load config from YAML file (default: configs/bert.yaml)
    bert_model_cfg = OmegaConf.load("configs/bert.yaml")
    lstm_model_cfg = OmegaConf.load("configs/lstm.yaml")

    dataset_cfg = OmegaConf.load("configs/dataset.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(lstm_model_cfg, bert_model_cfg, dataset_cfg, cli_cfg)
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    data = dataset_prep(cfg.dataset)

    logger.info("Initializing model and tokenizer...")
    # model = BertClassifier(cfg.bert_model, device=DEVICE)
    model = LSTMClassifier(cfg.lstm_model, device=DEVICE)

    # load tokenized data from disk if it exists, otherwise tokenize and save to disk for future runs
    tokenized_data_path = f"./{cfg.dataset.hf_dataset}_tokenized"
    data = try_load_tokenized_data(tokenized_data_path, data, model.tokenize)

    logger.info("Starting grid search for the model...")

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

    search_results = hyperparameter_tuning(cfg=cfg.lstm_model, data=data, model=model)
    # search_results = hyperparameter_tuning(cfg=cfg.bert_model, data=data, model=model)
    logger.info("Grid search completed. Best trial: %s", search_results["best"])


if __name__ == "__main__":
    main()
