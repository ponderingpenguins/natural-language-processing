# updated_default_full

Run time: 2026-04-04T19:40:28.179444+00:00 | device: `cuda`
Change tried: Updated default LSTM: full data, medium-size 2-layer BiLSTM, macro-F1-compatible defaults.
Hypothesis: This should improve generalization over the debug-scale setup.

## Key Settings

- Data caps: train=n/a, eval=n/a
- Model: embed=256, hidden=256, layers=2, bidirectional=True, pooling=mean, pack=False, dropout=0.4000, seq_len=128
- Optimizer: lr=0.0003, weight_decay=0.0100
- Training: epochs=8, eval=epoch, save=epoch, patience=2

## Data Sizes

- train=108000, dev=12000, test=7600

## Results

- Loss (best train / best eval): 0.0486 / 0.2634
- Loss (last train / last eval): 0.1037 / 0.2634
- Gap (best): 0.2148
- Gap (last): 0.1597
- Dev acc / macro-F1: 0.9103 / 0.9106
- Test acc / macro-F1: 0.9099 / 0.9100

Takeaway: Gap is 0.2148 with test macro-F1 0.9100. Compare it against baseline in the summary table.
