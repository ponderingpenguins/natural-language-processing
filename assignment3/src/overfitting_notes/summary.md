# LSTM Ablation Summary (Working Notes)

This file is a compact record of the LSTM overfitting experiments.

## Results Table

| Experiment | Main change | Train loss | Eval loss | Gap | Dev macro-F1 | Test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | Debug-scale setup | 0.3327 | 0.6834 | 0.3507 | 0.7299 | 0.7279 |
| early_stopping | Epoch eval + patience 2 | 0.1303 | 0.6642 | 0.5339 | 0.7549 | 0.7198 |
| regularized_lstm | Smaller 2-layer BiLSTM + recurrent dropout | 1.2363 | 1.2495 | 0.0131 | 0.5667 | 0.4898 |
| shorter_sequence | Sequence length 128 | 0.3660 | 0.7210 | 0.3550 | 0.6853 | 0.7057 |
| max_pooling | Max pooling instead of mean | 0.5643 | 0.7704 | 0.2061 | 0.5939 | 0.6362 |
| final_packed | Final-state pooling + packed sequences | 0.5086 | 0.7611 | 0.2526 | 0.6429 | 0.6133 |
| more_data | 5k/1k/1k sample sizes + stronger regularization | 0.1702 | 0.6401 | 0.4699 | 0.7832 | 0.7912 |
| combined_fixes | More data + regularization + final packed | 0.0985 | 0.6599 | 0.5615 | 0.7630 | 0.7838 |
| updated_default_full | Full data + medium 2-layer BiLSTM defaults | 0.0486 | 0.2634 | 0.2148 | 0.9106 | 0.9100 |

## What mattered most

- The biggest jump came from moving off debug sample caps to full data.
- Pooling changes by themselves did not improve held-out macro-F1.
- Heavy regularization removed the loss gap in one run, but underfit badly.
- The full-data `updated_default_full` run is the only LSTM run that is competitive with the A2 CNN baseline.

## Final comparison used in Assignment 3

| Model | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| A2 CNN (`seq64`) | 0.9099 | 0.9096 |
| A3 LSTM (`updated_default_full`) | 0.9099 | 0.9100 |
| A3 DistilBERT (`sanity_distilbert`) | 0.9486 | 0.9486 |

## Artifacts

- LSTM full run metrics: `experiment_outputs/updated_default_full/metrics.json`
- LSTM test evaluation: `lstm_evaluation_updated_full/`
- DistilBERT test evaluation: `bert_evaluation_full/`
- Final table: `comparison_tables/model_comparison.csv`
