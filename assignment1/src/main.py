from datasets import load_dataset

# Load AG News and create train/dev/test splits (dev from train).
ds = load_dataset("sh0416/ag_news")
breakpoint()

# Use the official train/test split.
# Fix a random seed and report it.
# Create a dev split from train (e.g., 90/10). Keep the test set untouched until final reporting.

# Implement preprocessing (tokenization, normalization) and document it.

# Use word-level TF-IDF features (document the preprocessing choices).
# Train two classical models (required):
#     TF-IDF + Logistic Regression
#     TF-IDF + Linear SVM

# Train both baseline models. Keep the dev split for model selection/tuning.

# Report Accuracy + Macro-F1 + confusion matrix.

# Metrics (required)
# - Primary: Accuracy
# - Secondary: Macro-F1
# - Also include: confusion matrix + 3–5 sentences interpreting it

# Evaluate on test once for the final numbers.

# Collect ≥20 misclassified examples from test and categorize them into 3–5 error types.
