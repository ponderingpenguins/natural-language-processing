import random
import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


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
    
def train_lstm(model, data, cfg):
    """Train an LSTM-based model on the provided dataset."""
    pass