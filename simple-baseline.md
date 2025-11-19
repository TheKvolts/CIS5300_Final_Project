# Simple Baseline (simple-baseline.md)
## Overview
This baseline uses a majority class predictor that predicts the most frequent sentiment class from the training data for all test examples. It **completely ignores the input text** and always predicts the same class, regardless of the headline content. This serves as a simple, lower-bound baseline to establish the minimum performance threshold for the task.

- Test set size: 585 examples. (refer to data.md)

Sentence                                                                 | Sentiment | Label |
|--------------------------------------------------------------------------|:---------:|:-----:|
| The inventors are Mukkavilli Krishna Kiran, Sabharwal Ashutosh and Aazhang Behnaam. | neutral   |   1   |
| $IBIO up 10% in premarket ready for lift off                             | positive  |   2   |

## Running the script
```
python simple-baseline.py
```

This creates two files with two columns- Sentence and Label:
- simple_baseline_test_predictions.csv
- test_sentence_label.csv

These files should be used by `scoring.py`.
```
python scoring.py milestone2/simple_baseline_test_predictions.csv milestone2/test_sentence_label.csv
```

## Reported baseline scores (test set: 585 examples)
**Note:** The simple baseline's performance is intentionally low as it predicts the same class (the majority class from training data) for all test examples, regardless of the actual content. This establishes a minimum performance threshold that any meaningful model should exceed.

Overall metrics (weighted):
- Total samples: 585
- Accuracy: 0.4855 (48.55%)
- Precision (weighted): 0.2357 (23.57%)
- Recall (weighted): 0.4855 (48.55%)
- F1-Score (weighted): 0.3173 (31.73%)

Classification report:
```
         precision    recall  f1-score   support

   negative       0.00      0.00      0.00       100
    neutral       0.49      1.00      0.65       284
   positive       0.00      0.00      0.00       201

   accuracy                           0.49       585
  macro avg       0.16      0.33      0.22       585
weighted avg       0.24      0.49      0.32       585
```

**Explanation:** The baseline predicts Neutral (class 1) for all 585 test examples, which is the majority class from the training data (54.29% of training examples). It correctly predicts 284 out of 585 examples (48.55% accuracy), which equals the proportion of Neutral examples in the test set. All Negative and Positive examples are incorrectly predicted as Neutral, resulting in 0.00 precision/recall/F1 for those classes.


## Raw Output (simple-baseline.py)

```
======================================================================
SIMPLE BASELINE: MAJORITY CLASS PREDICTOR
======================================================================

IMPORTANT: This baseline ignores headline content completely!
   It always predicts the same class for everything.

Training Set Class Distribution:
  Negative (0):   679 (14.53%)
   Neutral (1):  2537 (54.29%) ← MAJORITY (will predict this for ALL test examples)
  Positive (2):  1457 (31.18%)

Majority Class: 1 (Neutral)

Strategy: Predict 'Neutral' for EVERY test example
   (even for obviously Negative or Positive headlines!)

======================================================================
GENERATING PREDICTIONS
======================================================================

Test Set Class Distribution (True Labels):
  Negative (0):   100 (17.09%)
   Neutral (1):   284 (48.55%)
  Positive (2):   201 (34.36%)

What This Baseline Gets WRONG:
  • All 100 Negative examples will be predicted as Neutral
  • All 201 Positive examples will be predicted as Neutral

What This Baseline Gets RIGHT:
  • The 284 Neutral examples (by luck!)

Expected Accuracy: 0.4855 (48.55%)
   (= 284 correct / 585 total)

test_sentence_label.csv created successfully.
simple_baseline_test_predictions.csv created with integer labels successfully.
Total predictions: 585 (all class 1)
```


## Raw Output: python scoring.py milestone2/simple_baseline_test_predictions.csv milestone2/test_sentence_label.csv

```
============================================================
SENTIMENT ANALYSIS EVALUATION RESULTS
============================================================

Total samples: 585

Accuracy: 0.4855

------------------------------------------------------------
AGGREGATE METRICS
------------------------------------------------------------
Macro-averaged Precision:  0.1618
Macro-averaged Recall:     0.3333
Macro-averaged F1:         0.2179

Weighted-averaged Precision: 0.2357
Weighted-averaged Recall:    0.4855
Weighted-averaged F1:        0.3173

------------------------------------------------------------
PER-CLASS METRICS
------------------------------------------------------------
Class        Precision    Recall       F1           Support     
------------------------------------------------------------
Negative     0.0000       0.0000       0.0000       100         
Neutral      0.4855       1.0000       0.6536       284         
Positive     0.0000       0.0000       0.0000       201         

------------------------------------------------------------
CONFUSION MATRIX
------------------------------------------------------------
         Predicted:
           Neg    Neu    Pos
Actual:
  Neg        0    100      0
  Neu        0    284      0
  Pos        0    201      0
```