
TODO:
- don't tokenize the dataset for each hyperparameter combination, but only once and then pass the tokenized dataset to the grid search function. Honestly just save to disk and load it again in the grid search function, that way we can also easily inspect the tokenized dataset if needed.
- make the logging more readable, and informative, currently it is very verbose.

- retrain the CNN with the same tokenizer as the bert model, and compare the results.
- evaluate with f1, classification report, confusion matrix, and accuracy. report on dev set, then finally on the test set with the best hyperparameters.
- nice figure for hyperparameter search results, maybe a heatmap or something like that. also report the best hyperparameters and the corresponding metrics in a nice table.
- report misclassifications.
- data analysis, to find keywords for each class, for later evaluation of the models interpretability.

- Don't build the model for each hyperparameter combination, but only once and then pass the model to the grid search function. Honestly just save to disk and load it again in the grid search function, that way we can also easily inspect the model if needed.
- Don't build the tokenizer for each model, we use the SAME tokenizer for both models.