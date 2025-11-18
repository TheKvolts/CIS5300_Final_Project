# Strong Baseline (strong-baseline.md)
## Overview
This baseline uses the [FinBERT](https://huggingface.co/ProsusAI/finbert) text-classification pipeline from HuggingFace to predict financial sentiment (negative / neutral / positive). It serves as a strong, reproducible baseline for the task and includes inference and evaluation instructions.

- Test set size: 585 examples. (refer to data.md)

Sentence                                                                 | Sentiment | Label |
|--------------------------------------------------------------------------|:---------:|:-----:|
| The inventors are Mukkavilli Krishna Kiran, Sabharwal Ashutosh and Aazhang Behnaam. | neutral   |   1   |
| $IBIO up 10% in premarket ready for lift off                             | positive  |   2   |

## Running the script
```
python strong-baseline.py
```

This creates two files with two columns- Sentence and Label:
- strong_baseline_test_predictions.csv
- test_sentence_label.csv

These files should be used by `scoring.py`.
```
python scoring.py milestone2/strong_baseline_test_predictions.csv milestone2/test_sentence_label.csv
```

## Reported baseline scores (test set: 585 examples)
Overall metrics (weighted):
- Accuracy: 0.7436 (74.36%)
- Precision (weighted): 0.7720 (77.20%)
- Recall (weighted): 0.7436 (74.36%)
- F1-Score (weighted): 0.7508 (75.08%)

Classification report:
```
         precision    recall  f1-score   support

   negative       0.52      0.78      0.62       100
    neutral       0.78      0.73      0.75       284
   positive       0.89      0.75      0.81       201

   accuracy                           0.74       585
  macro avg       0.73      0.75      0.73       585
weighted avg       0.77      0.74      0.75       585
```
