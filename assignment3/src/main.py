import torch

from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger
from models.base_model import BaseModel
from models.bert import BertClassifier
from models.lstm import LSTM
from utils.dataset import load_data, preprocess_data
from utils.training import train_bert, train_lstm
from utils.config import LSTMConfig, BERTConfig

cfg = TrainingConfig()
bert_cfg = BERTConfig()
lstm_cfg = LSTMConfig()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_prep():
    data = load_data(cfg)
    logger.info("Loaded dataset %s with splits: %s", cfg.hf_dataset, data.keys())
        
    # Preprocess the datasets using the defined pipeline.
    logger.info("Preprocessing splits...")
    data = preprocess_data(data)
    logger.info("Dataset preprocessing complete. Sample from preprocessed dataset: %s", data["train"][0])
    
    return data

def tokenize_data(data, model):
    # Tokenize the datasets using the model's tokenizer.
    tokenizer_function = model.tokenize
    for split_name in ["train", "dev", "test"]:
        logger.info("Tokenizing %s set...", split_name)
        data[split_name] = data[split_name].map(tokenizer_function, batched=False, load_from_cache_file=False)
        logger.info("Completed tokenization for %s set. Set size: %d", split_name, len(data[split_name]))

    logger.info("Tokenization complete.")
    # data = tokenize_data(data, tokenizer_function)
    return data

def main():
    data = dataset_prep()
    
    bert_model = BertClassifier()
    bert_model.bert.to(DEVICE)
    bert_data = tokenize_data(data, bert_model)
    
    # lstm_model = LSTM(
    #     lstm_cfg.vocab_size,
    #     lstm_cfg.embed_dim,
    #     lstm_cfg.hidden_dim,
    #     lstm_cfg.num_classes,
    #     lstm_cfg.num_layers,
    #     lstm_cfg.bidirectional,
    #     lstm_cfg.dropout,
    # )
    # lstm_model.to(DEVICE)
    # lstm_data = tokenize_data(data, lstm_model)
    
    # Train the BERT model.
    logger.info("Starting training for BERT model...")
    train_bert(bert_model, bert_data, bert_cfg)

    # # Train the LSTM model.
    # logger.info("Starting training for LSTM model...")
    # train_lstm(lstm_model, lstm_data, lstm_cfg)

if __name__ == "__main__":
    main()
