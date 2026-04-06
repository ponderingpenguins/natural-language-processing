# Model Comparison (Dev + Test)

| Model | Category | Dev Accuracy | Dev Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 CNN (seq64) | baseline | 0.9097 | 0.9096 | 0.9099 | 0.9096 |
| A3 LSTM (updated_full) | neural | 0.9103 | 0.9106 | 0.9099 | 0.9100 |
| A3 DistilBERT (fine-tuned) | transformer | 0.9450 | 0.9450 | 0.9486 | 0.9486 |
