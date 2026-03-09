# Devlog for assignment 2


## Should I increase my vocab size?

This is with vocab size of 1000, which is pretty small. I wonder if increasing it would help. I will try 20000 and see if it improves the results.

```
[13:07:37] TUNING COMPLETE!
[13:07:37] Best F1 score: 0.4694
[13:07:37] Best configuration: {'lr': 0.001, 'num_filters': 100, 'kernel_size': 3, 'embed_dim': 256, 'weight_decay': 1e-05}
[13:07:37] Results saved to: output/hyperparameter_tuning_cnn.json
[13:07:37] ================================================================================
```

Experiment took way too long to run, so I stopped it early

```
[13:38:47] New best F1: 0.4114 with params: {'lr': 0.001, 'num_filters': 100, 'kernel_size': 3, 'embed_dim': 256, 'weight_decay': 0.0}
[13:38:47]
```