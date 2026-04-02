
TODO:
- evaluate with f1, classification report, confusion matrix, and accuracy. report on dev set, then finally on the test set with the best hyperparameters.
- Add a plotting.py, which reads the training history and metrics from disk, and produces nice plots for the report. We can also use it to analyze the training process, e.g., by plotting the learning curves, or the confusion matrix.
- add a tables.py, which reads the metrics from disk and produces nice LaTex tables for the report. We can also use it to analyze the results, e.g., by reporting the misclassifications, or by comparing the performance of the different models.
- report misclassifications (we need this for the appendix of the report, and it can also be useful for the limitations section).