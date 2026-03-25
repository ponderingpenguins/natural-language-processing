import random
from copy import deepcopy
from itertools import product
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from penguinlp.helpers import logger
from models.bert import BertClassifier


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_bert(model, data, cfg):
    """Fine-tune a transformer-based model on the provided dataset."""
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',  # Metric to monitor
        greater_is_better=False,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_steps=cfg.warmup_steps,
        logging_dir=cfg.logging_dir,
    )
    trainer = Trainer(
        model=model.bert,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]  # Set patience
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    return {
        "best_eval_loss": trainer.state.best_metric,
        "final_eval_loss": eval_metrics.get("eval_loss"),
    }


def run_bert_grid_search(preprocessed_data, cfg, tokenize_data_fn, device):
    """Run a grid search over learning rate, max length, and batch size for BERT."""
    search_space = list(product(cfg.grid_learning_rates, cfg.grid_max_lengths, cfg.grid_batch_sizes))
    logger.info("Starting BERT grid search with %d combinations", len(search_space))

    results = []
    best_result = None

    for run_idx, (learning_rate, max_length, batch_size) in enumerate(search_space, start=1):
        trial_cfg = deepcopy(cfg)
        trial_cfg.learning_rate = learning_rate
        trial_cfg.max_length = max_length
        trial_cfg.per_device_train_batch_size = batch_size
        trial_cfg.per_device_eval_batch_size = batch_size
        trial_cfg.output_dir = f"{cfg.output_dir}/grid_lr{learning_rate}_ml{max_length}_bs{batch_size}"
        trial_cfg.logging_dir = f"{cfg.logging_dir}/grid_lr{learning_rate}_ml{max_length}_bs{batch_size}"

        logger.info(
            "[%d/%d] Trial with lr=%s, max_length=%d, batch_size=%d",
            run_idx,
            len(search_space),
            learning_rate,
            max_length,
            batch_size,
        )

        model = BertClassifier(max_length=max_length)
        model.bert.to(device)

        trial_data = {split_name: dataset for split_name, dataset in preprocessed_data.items()}
        tokenized_data = tokenize_data_fn(trial_data, model)
        metrics = train_bert(model, tokenized_data, trial_cfg)

        result = {
            "learning_rate": learning_rate,
            "max_length": max_length,
            "batch_size": batch_size,
            **metrics,
        }
        results.append(result)

        current_score = result["best_eval_loss"]
        if current_score is None:
            current_score = result["final_eval_loss"]

        if current_score is None:
            continue

        if best_result is None:
            best_result = result
        else:
            best_score = best_result["best_eval_loss"]
            if best_score is None:
                best_score = best_result["final_eval_loss"]

            if best_score is None or current_score < best_score:
                best_result = result

    if best_result is not None:
        logger.info(
            "Best trial: lr=%s, max_length=%d, batch_size=%d, best_eval_loss=%s",
            best_result["learning_rate"],
            best_result["max_length"],
            best_result["batch_size"],
            best_result["best_eval_loss"],
        )

    return {
        "best": best_result,
        "all_results": results,
    }
    
def train_lstm(model, data, cfg):
    """Train an LSTM-based model on the provided dataset."""
    pass