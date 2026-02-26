# Assignment 2 — Neural Model Comparison + Ablation


## 1) Overview

Implement two neural text classifiers for AG News (CNN and LSTM), compare them under controlled conditions, and run a small ablation study to test a specific hypothesis about representation or training.

## 2) Learning outcomes

- Implement neural encoders for text classification.
- Run controlled comparisons and interpret learning curves.
- Design and analyze an ablation that isolates one factor.

## 3) Requirements

- Train two neural models (required):
    - CNN text classifier (convolution + pooling)
    - LSTM (or BiLSTM) classifier (sequence encoder + pooling)
- Controlled setup: same splits, same tokenizer, same max length, same metrics.
- Report Accuracy + Macro-F1 + confusion matrix for both models.
- Include learning curves (train loss + dev metric) for both models.

## 4) Ablation (choose ONE, required)

- [ ] Embeddings: pretrained vs random initialization
- [ ] Max length: e.g., 64 vs 128 vs 256 tokens
- [ ] Regularization: dropout 0 vs 0.3 (or another controlled value)
- [ ] Capacity: hidden size small vs medium (keep all else fixed)

## 5) Instructions

1. Reuse the same AG News split from Assignment 1.
2. Implement CNN and LSTM models with documented hyperparameters.
3. Train using dev for early stopping / selection.
4. Run the ablation by changing only the selected factor.
5. Evaluate on test once for final numbers.
6. Update error analysis: ≥10 errors, highlight differences vs Assignment 1 baseline failures.

## 6) Deliverables

- Report (3–4 pages, PDF) including:
    - [ ] Model descriptions (diagrams optional but encouraged)
    - [ ] Hyperparameters + training protocol (optimizer, LR, batch size, epochs, early stopping)
    - [ ] Results table + confusion matrices
    - [ ] Learning curves + interpretation (overfitting/underfitting)
    - [ ] Ablation setup + results + conclusion
    - [ ] Error analysis + comparison to Assignment 1
- Code repository with training + evaluation scripts/notebooks and reproducible runs.

## 7) Student checklist

- [ ] CNN implemented and trained
- [ ] LSTM/BiLSTM implemented and trained
- [ ] Same splits and preprocessing used for both
- [ ] Ablation run (one-factor change) and reported
- [ ] Learning curves included and interpreted
- [ ] >=10 errors analyzed

## 8) Submission (Brightspace)

- Assignment 2 Submission Folder
    - [ ] Report PDF
    - [ ] Repo link + commit hash/tag

