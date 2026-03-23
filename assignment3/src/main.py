import torch

from utils.dataset import load_dataset, preprocess_data\
from torch import nn

from penguinlp.config import TrainingConfig
from models.bert import BertClassifier

cfg = TrainingConfig()

def dataset_prep(model: nn.Module):
    data = load_dataset(cfg.hf_dataset)
    
    # Preprocess and tokenize the data
    def process_pipeline(batch):
        return model.tokenize(preprocess_data(batch), batched=True)
    
    train = data["train"].map(process_pipeline)
    dev = data["dev"].map(process_pipeline)
    test = data["test"].map(process_pipeline)

    return train, dev, test

def main():
    device = 0 if torch.cuda.is_available() else -1
    
    bert_model = BertClassifier().bert.to(device)
    dataset_prep(model=bert_model)


if __name__ == "__main__":
    main()
