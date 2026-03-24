from transformers import Trainer, TrainingArguments

def train_bert(model, data, cfg):
    """Fine-tune a transformer-based model on the provided dataset."""
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
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
    )
    trainer.train()
    
def train_lstm(model, data, cfg):
    """Train an LSTM-based model on the provided dataset."""
    pass