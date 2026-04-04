# AG News DistilBERT Model Card (Assignment 3)

This is the Transformer model used for Assignment 3.

- Base model: `distilbert-base-uncased`
- Labels: `World`, `Sports`, `Business`, `Sci/Tech`
- Final checkpoint used for reporting: `bert_output/sanity_distilbert/checkpoint-20250`

## What we used it for

The model is for course benchmarking on AG News and for comparison against our Assignment 2 models on the same split.  
It is not meant as a production model or for high-stakes decisions.

## Data and split

We used `sh0416/ag_news` from Hugging Face.  
Split procedure: AG News official train/test, then a stratified train/dev split from train with `seed=67`.  
Input text is `title + description` after preprocessing in `utils/dataset.py`.

## Training settings (reported run)

- max length: `128`
- learning rate: `2e-5`
- weight decay: `0.01`
- train/eval batch size: `16/16`
- early stopping enabled

## Main results

From `comparison_tables/model_comparison.csv` and `bert_evaluation_full`:

- dev accuracy: `0.9450`
- dev macro-F1: `0.9450`
- test accuracy: `0.9486`
- test macro-F1: `0.9486`

Confusion matrix (`bert_evaluation_full/confusion_matrix-regular-bert.json`, rows=true class, cols=predicted class):

```text
[[1813, 10, 43, 34],
 [9, 1875, 11, 5],
 [32, 5, 1732, 131],
 [23, 5, 83, 1789]]
```

## Failure modes we observed

Most errors are confusion between `Business` and `Sci/Tech`.  
A smaller pattern is `World` being predicted as `Business` or `Sci/Tech`.

Robustness checks show:

- shortest texts are the weakest slice
- keyword masking drops macro-F1 from `0.9486` to `0.9396` (drop `0.0090`)

## Limitations

- single English dataset only
- no out-of-domain or temporal drift check
- no multilingual evaluation
- robustness probes are lexical, not paraphrase/adversarial paraphrase tests

Error files used in analysis:

- `bert_evaluation_full/misclassified_examples-regular-bert.json`
- `lstm_evaluation_updated_full/misclassified_examples-regular-lstm.json`

## Repro commands

- full pipeline: `python run_assignment3_pipeline.py`
- robustness only: `python robustness_slices.py`
- comparison table only: `python generate_comparison_table.py`
