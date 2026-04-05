# Hyperparameter Selection Summary

## Paragraph

**BERT** and **LSTM** employed fundamentally different hyperparameter selection strategies. BERT used a systematic **grid search** across three independent dimensions—learning rate [1e-5, 2e-5, 3e-5], sequence length [96, 128, 160], and batch size [16, 32]—yielding 18 total configurations evaluated in a combinatorial fashion. In contrast, LSTM relied on **targeted ablation** followed by **manual curation**: five hand-designed experiments systematically isolated overfitting mechanisms (early stopping, regularization, sequence length, data scale), and the final production configuration (hidden_dim=256, embed_dim=256, dropout=0.4, batch_size=32) was manually selected as a balanced compromise between the extreme regularized and baseline variants, rather than exhaustively searched.

## Selection Methodology Comparison

| Aspect | BERT | LSTM |
|--------|------|------|
| **Search Type** | Combinatorial grid search | Directed ablation + manual selection |
| **Dimensions Varied** | 3 independent (lr, seq_len, batch) | 4–5 per variant (capacity, dropout, data, training strategy) |
| **Total Configurations** | 18 (exhaustive within grid) | 5 (purposefully designed) + 1 final hand-picked |
| **Rationale** | "What works best?" (optimization) | "What causes overfitting?" (understanding) |
| **Final Config Source** | Best performer from 18 grid trials | Manual selection from 5 ablations based on insights |
| **Hyperparameter Ranges** | Narrow, practical (3e-5 within 1e-5 range) | Wide, pedagogical (128→512 hidden; 0.0→0.5 dropout) |
| **Reproducibility** | Deterministic (all 18 run systematically) | Interpretive (insights drive final choice) |