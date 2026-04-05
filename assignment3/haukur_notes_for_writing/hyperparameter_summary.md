# Hyperparameter Space Summary: LSTM vs BERT Models

## Overview
This document outlines the hyperparameter search spaces for two NLP models tested on the AG News classification task: an LSTM-based model and a DistilBERT-based model.

---

## LSTM Model Hyperparameter Space

### Architecture Parameters
| Parameter | Baseline | Range/Variants |
|-----------|----------|-----------------|
| **embed_dim** | 768 (BERT init) | 256 (reduced variants) |
| **hidden_dim** | 512 | 128 (regularized variants) |
| **num_layers** | 2 | 2 (fixed) |
| **dropout** | 0.0 (baseline) | 0.5 (regularized variants) |
| **init_embeddings_from_bert** | true | false (in regularized/scaled variants) |

### Sequence/Data Parameters
| Parameter | Default | Range/Variants |
|-----------|---------|-----------------|
| **sequence_length** | 256 | 128 (shorter_sequence variant) |
| **max_samples** | Default (2000) | 5000 (more_data variant) |
| **eval_max_samples** | Default | 1000 (more_data variant) |

### Training Parameters
| Parameter | Baseline | Early Stop Variant | Regularized Variant |
|-----------|----------|-------------------|---------------------|
| **learning_rate** | Default | 0.0003 | 0.0003 |
| **weight_decay** | Default | 0.01 | 0.01 |
| **num_train_epochs** | Default | 8 | 8 |
| **eval_strategy** | Disabled | "epoch" | "epoch" |
| **save_strategy** | Default | "epoch" | "epoch" |
| **early_stopping_patience** | N/A | 2 | 2 |
| **metric_for_best_model** | N/A | eval_loss | eval_loss |
| **greater_is_better** | N/A | false | false |

### Ablation Variants Tested
1. **baseline**: Default config (reference point)
2. **early_stopping**: Checkpoint selection + evaluation per epoch
3. **regularized_lstm**: Smaller capacity (256→128 dims, 2 layers, 0.5 dropout)
4. **shorter_sequence**: 256→128 token sequence length
5. **more_data**: Scaled data (2000→5000 samples) + regularization

### Key Observations for LSTM
- **Regularization focus**: Dropout, reduced dimensions, weight decay
- **Early stopping**: Metric is eval_loss (lower is better)
- **Embedding init**: Can use pretrained BERT embeddings or random init
- **Goal**: Understand overfitting trade-offs across architecture, data scale, and regularization

---

## BERT Model Hyperparameter Space

### Architecture Parameters
| Parameter | Value | Type |
|-----------|-------|------|
| **model_name** | distilbert-base-uncased | Fixed |
| **num_labels** | 4 | Fixed (AG News: 4 classes) |
| **max_length** | 128 | Fixed (also in grid search) |
| **head_dims** | [256] | Optional hidden layers in classification head |
| **head_dropout** | 0.1 | Dropout in head hidden layers |
| **freeze_bert** | false | Whether to freeze BERT weights |

### Training Parameters
| Parameter | Value | Type |
|-----------|-------|------|
| **learning_rate** | 2e-5 | Base (also in grid search) |
| **weight_decay** | 0.01 | Fixed |
| **per_device_train_batch_size** | 16 | Fixed (also in grid search) |
| **per_device_eval_batch_size** | 16 | Fixed |
| **warmup_steps** | 500 | Fixed |
| **batch_size** | 16 | Fixed (redundant with per_device) |

### Grid Search Space
| Parameter | Search Grid |
|-----------|-------------|
| **grid_learning_rates** | [1e-5, 2e-5, 3e-5] (3 values) |
| **grid_max_lengths** | [96, 128, 160] (3 values) |
| **grid_batch_sizes** | [16, 32] (2 values) |

**Total Grid Combinations**: 3 × 3 × 2 = **18 hyperparameter combinations**

### Other Parameters
| Parameter | Value |
|-----------|-------|
| **early_stopping_patience** | 2 |
| **output_dir** | ./bert_output |
| **logging_dir** | ./bert_logs |

### Key Observations for BERT
- **Pretrained model**: Uses DistilBERT (distilled BERT for efficiency)
- **Classification head**: Simple (single 256-dim layer) or can be extended with head_dims
- **Grid search scope**: Learning rate, sequence length, and batch size (18 configurations)
- **Frozen weights option**: Can freeze BERT backbone to reduce training cost
- **Goal**: Systematic exploration of fine-tuning hyperparameters within reasonable ranges

---

## Comparative Analysis

### Model Complexity
| Aspect | LSTM | BERT |
|--------|------|------|
| **Architecture** | Recurrent, 2-layer BiLSTM | Transformer-based (distilled) |
| **Embeddings** | Learned or BERT-init | Pretrained + fine-tuned |
| **Baseline hidden size** | 512 | Fixed (pretrained) |
| **Regularization** | Dropout, weight decay, early stopping | Head dropout, weight decay, optional freeze |

### Hyperparameter Tuning Philosophy
- **LSTM**: Ablation-driven (focus on overfitting mechanisms)
  - Tests individual factors sequentially
  - Varies capacity, data, and regularization

- **BERT**: Grid search-driven (systematic exploration)
  - Explores learning rate, sequence length, batch size
  - Fewer total variants but more combinations

### Search Scope
| Model | Discrete Variants | Total Configurations |
|-------|-------------------|----------------------|
| LSTM | 5 named experiments | 5 sequential ablations |
| BERT | 1 base + grid search | 18 grid combinations |

### Regularization Strategies
- **LSTM**: Dropout (0.0 → 0.5), weight decay (0.01), early stopping (patience=2)
- **BERT**: Head dropout (0.1), weight decay (0.01), optional freeze, early stopping (patience=2)

---

## Summary Table: Quick Reference

### LSTM Search Space Dimensions
- Embedding dim: {768, 256}
- Hidden dim: {512, 128}
- Dropout: {0.0, 0.5}
- Sequence length: {256, 128}
- Data scale (train): {~2000, 5000}
- Learning rate: {default, 0.0003}
- Weight decay: {default, 0.01}
- Early stopping: {disabled, enabled}

### BERT Search Space Dimensions
- Learning rate: [1e-5, 2e-5, 3e-5]
- Max sequence length: [96, 128, 160]
- Batch size: [16, 32]
- Head dropout: 0.1 (fixed)
- Freeze BERT: false (fixed)
- Head dims: [256] (fixed)

---

## Notes for Interpretation
1. **LSTM**: Designed to isolate overfitting causes; each experiment adds one or more changes
2. **BERT**: Grid search is more exploratory; independent variation of three key hyperparameters
3. **Early stopping**: Both use eval_loss as metric with patience=2 (in applicable variants)
4. **Data scale**: LSTM varies from ~2000–5000 samples; BERT uses fixed dataset
5. **Reproducibility**: Both use fixed random seeds (implied by controlled bash script)