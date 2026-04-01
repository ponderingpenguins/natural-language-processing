
TODO:
- evaluate with f1, classification report, confusion matrix, and accuracy. report on dev set, then finally on the test set with the best hyperparameters.
- save the metrics to disk, so we can easily report them in the final report (per time step, per hyperparameter combination, etc.).
- Add a plotting.py, which reads the training history and metrics from disk, and produces nice plots for the report. We can also use it to analyze the training process, e.g., by plotting the learning curves, or the confusion matrix.
- add a tables.py, which reads the metrics from disk and produces nice LaTex tables for the report. We can also use it to analyze the results, e.g., by reporting the misclassifications, or by comparing the performance of the different models.
- report misclassifications (we need this for the appendix of the report, and it can also be useful for the limitations section).
- data analysis, to find keywords for each class, for later evaluation of the models interpretability.
- **label noise sensitivity**: train the model with different amounts of training data (e.g., 25%, 50%, 100%) and compare the performance. This can reveal how well the model learns from limited data, and how it generalizes to unseen examples. We can also calculate the "scaling law" for our model, which is a very interesting analysis to do.
- Don't build the model for each hyperparameter combination, but only once and then pass the model to the grid search function.