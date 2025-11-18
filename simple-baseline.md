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

