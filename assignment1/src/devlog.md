
# Exploring the effect of lowercasing as a preprocessing step 16.02.26

## With lowercase

```bash
[22:09:17] Evaluating Logistic Regression on test set...
[22:09:17] Results for Logistic Regression (Test):
[22:09:17]               precision    recall  f1-score   support

           1       0.93      0.91      0.92      1900
           2       0.95      0.98      0.96      1900
           3       0.88      0.89      0.88      1900
           4       0.89      0.88      0.89      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600

[22:09:17] Confusion Matrix:
[22:09:17] [[1721   57   75   47]
 [  16 1863   12    9]
 [  55   17 1683  145]
 [  57   25  138 1680]]
[22:09:17] [[1721   57   75   47]
 [  16 1863   12    9]
 [  55   17 1683  145]
 [  57   25  138 1680]]
[22:09:17] Found 653 misclassified examples
[22:09:17] Categorizing misclassified examples...
[22:09:17] Misclassified example 3: True=4, Predicted=2
[22:09:17] Misclassified example 15: True=4, Predicted=3
[22:09:17] Misclassified example 23: True=4, Predicted=3
[22:09:17] Misclassified example 24: True=4, Predicted=3
[22:09:17] Misclassified example 36: True=1, Predicted=3
[22:09:17] Misclassified example 47: True=4, Predicted=3
[22:09:17] Misclassified example 56: True=1, Predicted=3
[22:09:17] Misclassified example 79: True=1, Predicted=2
[22:09:17] Misclassified example 83: True=3, Predicted=4
[22:09:17] Misclassified example 88: True=1, Predicted=2
[22:09:17] Misclassified example 106: True=1, Predicted=3
[22:09:17] Misclassified example 110: True=3, Predicted=4
[22:09:17] Misclassified example 111: True=4, Predicted=3
[22:09:17] Misclassified example 120: True=1, Predicted=3
[22:09:17] Misclassified example 125: True=3, Predicted=4
[22:09:17] Misclassified example 126: True=2, Predicted=1
[22:09:17] Misclassified example 154: True=1, Predicted=3
[22:09:17] Misclassified example 155: True=1, Predicted=4
[22:09:17] Misclassified example 172: True=3, Predicted=4
[22:09:17] Misclassified example 183: True=4, Predicted=3

[22:09:17] Evaluating Linear SVM on test set...
[22:09:17] Results for Linear SVM (Test):
[22:09:17]               precision    recall  f1-score   support

           1       0.94      0.90      0.92      1900
           2       0.95      0.98      0.97      1900
           3       0.89      0.88      0.88      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.92      7600
   macro avg       0.91      0.92      0.91      7600
weighted avg       0.91      0.92      0.91      7600

[22:09:17] Confusion Matrix:
[22:09:17] [[1715   59   78   48]
 [   9 1871   11    9]
 [  53   19 1677  151]
 [  56   26  127 1691]]
[22:09:17] [[1715   59   78   48]
 [   9 1871   11    9]
 [  53   19 1677  151]
 [  56   26  127 1691]]
[22:09:17] Found 646 misclassified examples
[22:09:17] Categorizing misclassified examples...
[22:09:17] Misclassified example 3: True=4, Predicted=2
[22:09:17] Misclassified example 15: True=4, Predicted=3
[22:09:17] Misclassified example 23: True=4, Predicted=3
[22:09:17] Misclassified example 24: True=4, Predicted=3
[22:09:17] Misclassified example 36: True=1, Predicted=3
[22:09:17] Misclassified example 47: True=4, Predicted=3
[22:09:17] Misclassified example 56: True=1, Predicted=3
[22:09:17] Misclassified example 79: True=1, Predicted=2
[22:09:17] Misclassified example 83: True=3, Predicted=4
[22:09:17] Misclassified example 88: True=1, Predicted=2
[22:09:17] Misclassified example 106: True=1, Predicted=3
[22:09:17] Misclassified example 110: True=3, Predicted=4
[22:09:17] Misclassified example 111: True=4, Predicted=3
[22:09:17] Misclassified example 120: True=1, Predicted=3
[22:09:17] Misclassified example 125: True=3, Predicted=4
[22:09:17] Misclassified example 154: True=1, Predicted=3
[22:09:17] Misclassified example 183: True=4, Predicted=3
[22:09:17] Misclassified example 196: True=1, Predicted=3
[22:09:17] Misclassified example 207: True=4, Predicted=3
[22:09:17] Misclassified example 215: True=4, Predicted=3
```


## without lowercase

```bash
[22:10:39] Evaluating Logistic Regression on test set...
[22:10:39] Results for Logistic Regression (Test):
[22:10:39]               precision    recall  f1-score   support

           1       0.93      0.91      0.92      1900
           2       0.95      0.98      0.96      1900
           3       0.88      0.88      0.88      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.92      7600
   macro avg       0.91      0.92      0.91      7600
weighted avg       0.91      0.92      0.91      7600

[22:10:39] Confusion Matrix:
[22:10:39] [[1726   57   70   47]
 [  17 1861   13    9]
 [  59   18 1677  146]
 [  52   23  135 1690]]
[22:10:39] [[1726   57   70   47]
 [  17 1861   13    9]
 [  59   18 1677  146]
 [  52   23  135 1690]]
[22:10:39] Found 646 misclassified examples
[22:10:39] Categorizing misclassified examples...
[22:10:39] Misclassified example 3: True=4, Predicted=2
[22:10:39] Misclassified example 20: True=4, Predicted=3
[22:10:39] Misclassified example 23: True=4, Predicted=3
[22:10:39] Misclassified example 36: True=1, Predicted=3
[22:10:39] Misclassified example 47: True=4, Predicted=3
[22:10:39] Misclassified example 79: True=1, Predicted=2
[22:10:39] Misclassified example 83: True=3, Predicted=4
[22:10:39] Misclassified example 88: True=1, Predicted=2
[22:10:39] Misclassified example 106: True=1, Predicted=3
[22:10:39] Misclassified example 110: True=3, Predicted=4
[22:10:39] Misclassified example 120: True=1, Predicted=3
[22:10:39] Misclassified example 125: True=3, Predicted=4
[22:10:39] Misclassified example 126: True=2, Predicted=1
[22:10:39] Misclassified example 154: True=1, Predicted=3
[22:10:39] Misclassified example 155: True=1, Predicted=4
[22:10:39] Misclassified example 172: True=3, Predicted=4
[22:10:39] Misclassified example 183: True=4, Predicted=3
[22:10:39] Misclassified example 196: True=1, Predicted=3
[22:10:39] Misclassified example 207: True=4, Predicted=3
[22:10:39] Misclassified example 215: True=4, Predicted=3
[22:10:39] Evaluating Linear SVM on test set...
[22:10:39] Results for Linear SVM (Test):
[22:10:39]               precision    recall  f1-score   support

           1       0.94      0.90      0.92      1900
           2       0.95      0.99      0.97      1900
           3       0.89      0.88      0.88      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600

[22:10:39] Confusion Matrix:
[22:10:39] [[1718   59   77   46]
 [  10 1872    9    9]
 [  50   19 1679  152]
 [  55   26  131 1688]]
[22:10:39] [[1718   59   77   46]
 [  10 1872    9    9]
 [  50   19 1679  152]
 [  55   26  131 1688]]
[22:10:40] Found 643 misclassified examples
[22:10:40] Categorizing misclassified examples...
[22:10:40] Misclassified example 3: True=4, Predicted=2
[22:10:40] Misclassified example 15: True=4, Predicted=3
[22:10:40] Misclassified example 23: True=4, Predicted=3
[22:10:40] Misclassified example 24: True=4, Predicted=3
[22:10:40] Misclassified example 36: True=1, Predicted=3
[22:10:40] Misclassified example 47: True=4, Predicted=3
[22:10:40] Misclassified example 79: True=1, Predicted=2
[22:10:40] Misclassified example 83: True=3, Predicted=4
[22:10:40] Misclassified example 88: True=1, Predicted=2
[22:10:40] Misclassified example 106: True=1, Predicted=3
[22:10:40] Misclassified example 110: True=3, Predicted=4
[22:10:40] Misclassified example 120: True=1, Predicted=3
[22:10:40] Misclassified example 125: True=3, Predicted=4
[22:10:40] Misclassified example 154: True=1, Predicted=3
[22:10:40] Misclassified example 155: True=1, Predicted=4
[22:10:40] Misclassified example 183: True=4, Predicted=3
[22:10:40] Misclassified example 196: True=1, Predicted=3
[22:10:40] Misclassified example 207: True=4, Predicted=3
[22:10:40] Misclassified example 215: True=4, Predicted=3
[22:10:40] Misclassified example 258: True=4, Predicted=3
```

## Key Findings

Lowercasing has minimal impact on model performance. Both Logistic Regression and Linear SVM show virtually identical accuracy (approx 91-92%) regardless of whether text is lowercased.

# Exploring unigram vs unigram + bigram 16.02.26

This was done with lowercasing enabled (for preprocessing)

# unigram

```
[22:22:39] Results for Logistic Regression (Test):
[22:22:40]               precision    recall  f1-score   support

           1       0.93      0.90      0.92      1900
           2       0.95      0.98      0.97      1900
           3       0.88      0.88      0.88      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600

[22:22:40] Confusion Matrix:
[22:22:40] [[1716   56   73   55]
 [  14 1861   15   10]
 [  68   16 1664  152]
 [  51   22  141 1686]]
[22:22:40] [[1716   56   73   55]
 [  14 1861   15   10]
 [  68   16 1664  152]
 [  51   22  141 1686]]
[22:22:40] Found 673 misclassified examples
[22:22:40] Categorizing misclassified examples...
[22:22:40] Misclassified example 20: True=4, Predicted=3
[22:22:40] Misclassified example 23: True=4, Predicted=3
[22:22:40] Misclassified example 24: True=4, Predicted=3
[22:22:40] Misclassified example 47: True=4, Predicted=3
[22:22:40] Misclassified example 56: True=1, Predicted=3
[22:22:40] Misclassified example 79: True=1, Predicted=2
[22:22:40] Misclassified example 83: True=3, Predicted=4
[22:22:40] Misclassified example 88: True=1, Predicted=2
[22:22:40] Misclassified example 106: True=1, Predicted=3
[22:22:40] Misclassified example 110: True=3, Predicted=4
[22:22:40] Misclassified example 111: True=4, Predicted=3
[22:22:40] Misclassified example 120: True=1, Predicted=3
[22:22:40] Misclassified example 125: True=3, Predicted=4
[22:22:40] Misclassified example 126: True=2, Predicted=1
[22:22:40] Misclassified example 139: True=1, Predicted=3
[22:22:40] Misclassified example 154: True=1, Predicted=3
[22:22:40] Misclassified example 155: True=1, Predicted=4
[22:22:40] Misclassified example 172: True=3, Predicted=4
[22:22:40] Misclassified example 183: True=4, Predicted=3
[22:22:40] Misclassified example 196: True=1, Predicted=3
[22:22:40] Evaluating Linear SVM on test set...
[22:22:40] Results for Linear SVM (Test):
[22:22:40]               precision    recall  f1-score   support

           1       0.94      0.90      0.92      1900
           2       0.95      0.98      0.97      1900
           3       0.88      0.88      0.88      1900
           4       0.89      0.90      0.89      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600

[22:22:40] Confusion Matrix:
[22:22:40] [[1712   59   79   50]
 [  10 1870   13    7]
 [  53   19 1671  157]
 [  45   20  129 1706]]
[22:22:40] [[1712   59   79   50]
 [  10 1870   13    7]
 [  53   19 1671  157]
 [  45   20  129 1706]]
[22:22:40] Found 641 misclassified examples
[22:22:40] Categorizing misclassified examples...
[22:22:40] Misclassified example 9: True=4, Predicted=3
[22:22:40] Misclassified example 20: True=4, Predicted=3
[22:22:40] Misclassified example 23: True=4, Predicted=3
[22:22:40] Misclassified example 24: True=4, Predicted=3
[22:22:40] Misclassified example 36: True=1, Predicted=3
[22:22:40] Misclassified example 56: True=1, Predicted=3
[22:22:40] Misclassified example 79: True=1, Predicted=2
[22:22:40] Misclassified example 83: True=3, Predicted=4
[22:22:40] Misclassified example 88: True=1, Predicted=2
[22:22:40] Misclassified example 106: True=1, Predicted=3
[22:22:40] Misclassified example 110: True=3, Predicted=4
[22:22:40] Misclassified example 111: True=4, Predicted=3
[22:22:40] Misclassified example 120: True=1, Predicted=3
[22:22:40] Misclassified example 139: True=1, Predicted=3
[22:22:40] Misclassified example 154: True=1, Predicted=3
[22:22:40] Misclassified example 155: True=1, Predicted=4
[22:22:40] Misclassified example 183: True=4, Predicted=3
[22:22:40] Misclassified example 196: True=1, Predicted=3
[22:22:40] Misclassified example 207: True=4, Predicted=3
[22:22:40] Misclassified example 215: True=4, Predicted=3
```

# unigram + bigram

```
[22:19:27] Results for Linear SVM:
[22:19:27]               precision    recall  f1-score   support

           1       0.90      0.86      0.88      2976
           2       0.91      0.98      0.94      2789
           3       0.85      0.86      0.86      3039
           4       0.89      0.85      0.87      3196

    accuracy                           0.88     12000
   macro avg       0.89      0.89      0.89     12000
weighted avg       0.88      0.88      0.88     12000

[22:19:27] Confusion Matrix:
[22:19:27] [[2547  190  148   91]
 [  32 2730   12   15]
 [ 159   25 2615  240]
 [ 103   69  297 2727]]
[22:19:27] [[2547  190  148   91]
 [  32 2730   12   15]
 [ 159   25 2615  240]
 [ 103   69  297 2727]]
[22:19:27] Evaluating Logistic Regression on test set...
[22:19:27] Results for Logistic Regression (Test):
[22:19:27]               precision    recall  f1-score   support

           1       0.93      0.91      0.92      1900
           2       0.95      0.98      0.97      1900
           3       0.88      0.88      0.88      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600

[22:19:27] Confusion Matrix:
[22:19:27] [[1720   56   73   51]
 [  14 1861   14   11]
 [  57   15 1680  148]
 [  54   23  135 1688]]
[22:19:27] [[1720   56   73   51]
 [  14 1861   14   11]
 [  57   15 1680  148]
 [  54   23  135 1688]]
[22:19:27] Found 651 misclassified examples
[22:19:27] Categorizing misclassified examples...
[22:19:27] Misclassified example 15: True=4, Predicted=3
[22:19:27] Misclassified example 23: True=4, Predicted=3
[22:19:27] Misclassified example 24: True=4, Predicted=3
[22:19:27] Misclassified example 36: True=1, Predicted=3
[22:19:27] Misclassified example 47: True=4, Predicted=3
[22:19:27] Misclassified example 79: True=1, Predicted=2
[22:19:27] Misclassified example 83: True=3, Predicted=4
[22:19:27] Misclassified example 88: True=1, Predicted=2
[22:19:27] Misclassified example 106: True=1, Predicted=3
[22:19:27] Misclassified example 110: True=3, Predicted=4
[22:19:27] Misclassified example 111: True=4, Predicted=3
[22:19:27] Misclassified example 120: True=1, Predicted=3
[22:19:27] Misclassified example 125: True=3, Predicted=4
[22:19:27] Misclassified example 126: True=2, Predicted=1
[22:19:27] Misclassified example 139: True=1, Predicted=3
[22:19:27] Misclassified example 154: True=1, Predicted=3
[22:19:27] Misclassified example 155: True=1, Predicted=4
[22:19:27] Misclassified example 172: True=3, Predicted=4
[22:19:27] Misclassified example 183: True=4, Predicted=3
[22:19:27] Misclassified example 196: True=1, Predicted=3
[22:19:27] Evaluating Linear SVM on test set...
[22:19:27] Results for Linear SVM (Test):
[22:19:27]               precision    recall  f1-score   support

           1       0.94      0.90      0.92      1900
           2       0.95      0.98      0.97      1900
           3       0.89      0.88      0.88      1900
           4       0.89      0.89      0.89      1900

    accuracy                           0.92      7600
   macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600

[22:19:27] Confusion Matrix:
[22:19:28] [[1717   58   77   48]
 [   9 1869   11   11]
 [  54   20 1679  147]
 [  54   25  129 1692]]
[22:19:28] [[1717   58   77   48]
 [   9 1869   11   11]
 [  54   20 1679  147]
 [  54   25  129 1692]]
[22:19:28] Found 643 misclassified examples
[22:19:28] Categorizing misclassified examples...
[22:19:28] Misclassified example 3: True=4, Predicted=2
[22:19:28] Misclassified example 15: True=4, Predicted=3
[22:19:28] Misclassified example 23: True=4, Predicted=3
[22:19:28] Misclassified example 24: True=4, Predicted=3
[22:19:28] Misclassified example 36: True=1, Predicted=3
[22:19:28] Misclassified example 47: True=4, Predicted=3
[22:19:28] Misclassified example 79: True=1, Predicted=2
[22:19:28] Misclassified example 83: True=3, Predicted=4
[22:19:28] Misclassified example 88: True=1, Predicted=2
[22:19:28] Misclassified example 106: True=1, Predicted=3
[22:19:28] Misclassified example 110: True=3, Predicted=4
[22:19:28] Misclassified example 111: True=4, Predicted=3
[22:19:28] Misclassified example 120: True=1, Predicted=3
[22:19:28] Misclassified example 125: True=3, Predicted=4
[22:19:28] Misclassified example 154: True=1, Predicted=3
[22:19:28] Misclassified example 183: True=4, Predicted=3
[22:19:28] Misclassified example 196: True=1, Predicted=3
[22:19:28] Misclassified example 207: True=4, Predicted=3
[22:19:28] Misclassified example 215: True=4, Predicted=3
[22:19:28] Misclassified example 258: True=4, Predicted=3
```

## Key findings

The bigram features provide minimal improvement - surprisingly negligible actually.

Performance Comparison

Logistic Regression:
- Unigram: 91% accuracy, 673 misclassifications
- Unigram+Bigram: 91% accuracy, 651 misclassifications

Difference: Only 22 fewer errors (3.3% reduction in misclassifications)


Linear SVM:
- Unigram: 92% accuracy, 641 misclassifications
- Unigram+Bigram: 92% accuracy, 643 misclassifications

Difference: Actually 2 MORE errors with bigrams