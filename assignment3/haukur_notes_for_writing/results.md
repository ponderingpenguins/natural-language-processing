## Results

This document summarizes the methods and results of our experiments. I will use this later to write the method/discussion sections of the report, but I wanted to have a single place to write down all the details and numbers first.

Bert
- ./bert_evaluation_full
LSTM
- ./lstm_evaluation_full

make sure to also take a look at

- (model comparison table)[comparison_tables/model_comparison.md](comparison_tables/model_comparison.md)


---

Analysis


### Robustness and evaluation results are saved in

- Length buckets and kayword masking results: (robustness_slices/summary.md)[robustness_slices/summary.md]

- Length buckets: Definer length-based slices using train quartile to define short/medium/long documents. Evaluate accuracy and macro-F1 on each slice. The quartiles are [34.0, 39.0, 45.0] words.
Note: a limitation of this approach is that although we used text data after preprocessing, we defined length as word count, not token count. We suspect this would have an impact on the results, because the tokenizer for the CNN model was trained on the preprocessed text (not the case for LSTM and BERT, which used pretrained tokenizers, vocab size for CNN: 5000, vocab size for others: 30522), and thus the token count may differ more from the word count for CNN than for the other models.

- keyword masking: Mask keywords (min length 2 characters) identified by TF-IDF. Keywords are the 24 words per class with the highest TF-IDF score. As a control, we choose 24 random words per class that are frequency-matched to the keywords. We evaluate the drop in macro-F1 when masking keywords vs. random control words, and report the excess drop (keyword drop - random drop) as a measure of how much the model relies on keywords vs. random words.