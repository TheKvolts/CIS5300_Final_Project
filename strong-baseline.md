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

## GENERATING PREDICTIONS

**Note:** Unlike the simple baseline, FinBERT is a pre-trained transformer model specifically fine-tuned on financial text. It **analyzes the actual content** of each sentence to predict sentiment, making it a strong baseline for financial sentiment analysis.

Test Set Class Distribution (True Labels):
```
  Negative (0):   100 (17.09%)
   Neutral (1):   284 (48.55%)
  Positive (2):   201 (34.36%)
```

### What FinBERT Gets RIGHT:
- **78 out of 100 Negative** examples correctly identified (78% recall)
- **207 out of 284 Neutral** examples correctly identified (72.89% recall)
- **150 out of 201 Positive** examples correctly identified (74.63% recall)
- **Total: 435 out of 585** examples correctly predicted (74.36% accuracy)

### What FinBERT Struggles With:
- **Negative class precision (52%):** Out of 150 predictions as Negative, only 78 were actually Negative
  - 61 Neutral examples misclassified as Negative
  - 11 Positive examples misclassified as Negative

- **Neutral confusion:**
  - 20 Negative examples misclassified as Neutral
  - 40 Positive examples misclassified as Neutral

- **Positive class strength:** Best performing class with 89.29% precision and 74.63% recall

### Performance Comparison:
Compared to the simple baseline (48.55% accuracy), FinBERT achieves **25.81 percentage points improvement** (74.36% accuracy), demonstrating the value of using a domain-specific pre-trained model that actually analyzes financial text content.

## Raw Output: python scoring.py milestone2/strong_baseline_test_predictions.csv milestone2/test_sentence_label.csv

```
============================================================
SENTIMENT ANALYSIS EVALUATION RESULTS
============================================================

Total samples: 585

Accuracy: 0.7436

------------------------------------------------------------
AGGREGATE METRICS
------------------------------------------------------------
Macro-averaged Precision:  0.7294
Macro-averaged Recall:     0.7517
Macro-averaged F1:         0.7295

Weighted-averaged Precision: 0.7720
Weighted-averaged Recall:    0.7436
Weighted-averaged F1:        0.7508

------------------------------------------------------------
PER-CLASS METRICS
------------------------------------------------------------
Class        Precision    Recall       F1           Support
------------------------------------------------------------
Negative     0.5200       0.7800       0.6240       100
Neutral      0.7753       0.7289       0.7514       284
Positive     0.8929       0.7463       0.8130       201

------------------------------------------------------------
CONFUSION MATRIX
------------------------------------------------------------
         Predicted:
           Neg    Neu    Pos
Actual:
  Neg       78     20      2
  Neu       61    207     16
  Pos       11     40    150
============================================================
```
