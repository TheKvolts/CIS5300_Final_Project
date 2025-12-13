## Evaluation Metrics

This document describes the evaluation metrics used to assess the performance of our financial sentiment analysis model. The task is a **3-class sentiment classification** problem with classes: negative (0), neutral (1), and positive (2).

We employ several standard metrics for multi-class classification tasks:

### 1. Accuracy
**Definition:** The proportion of correct predictions out of the total predictions.
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$ 
**Interpretation:** Measures overall correctness. Can be misleading with imbalanced datasets.

### 2. Precision
**Definition:** The proportion of true positives among all positive predictions for a specific class.
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$
**Interpretation:** Indicates the reliability of positive predictions for a class (minimizing false positives).

### 3. Recall (Sensitivity)
**Definition:** The proportion of true positives identified among all actual positive instances for a class.
$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
**Interpretation:** Measures the ability to capture all instances of a class (minimizing false negatives).

### 4. F1-Score
**Definition:** The harmonic mean of precision and recall.
$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
**Interpretation:** Balances precision and recall; useful for imbalanced datasets.

### 5. Macro-Averaged Metrics
**Definition:** Unweighted mean of per-class metrics.
$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$
**Interpretation:** Treats all classes as equally important, highlighting performance on minority classes.

### 6. Weighted-Averaged Metrics
**Definition:** Mean of per-class metrics weighted by class support ($n_c$).
$$\text{Weighted-F1} = \sum_{c=1}^{C} \frac{n_c}{n} \cdot F1_c$$
**Interpretation:** Accounts for class imbalance by favoring majority classes.

### 7. Confusion Matrix
**Definition:** A table comparing predicted vs. actual classifications.
**Interpretation:** Visualizes specific misclassifications between classes.

---

## Primary Metric

For this sentiment analysis task, we use **Macro-averaged F1-score** as our primary evaluation metric because:
1. It balances precision and recall
2. It treats all sentiment classes equally (important for financial analysis where all sentiments matter)
3. It is robust to class imbalance

We also report **Accuracy** and **Weighted F1-score** for completeness.

---

## Usage

### Command Line Interface
Run the evaluation script `scoring.py` with the predictions and gold labels.

#### Syntax
```bash
python scoring.py <predictions_file> <gold_labels_file> [options]
```

#### Arguments
- `<predictions_file>`: Path to predictions (CSV with `Predicted_Label` or text file).
- `<gold_labels_file>`: Path to true labels (CSV with `Label` or text file).
- `--format`: Output format, either `text` (default) or `json`.
- `--output`: Optional path to save the output.

### Examples

**1. Basic Evaluation (Text Output)**
```bash
python scoring.py predictions.csv data/test/test.csv
```

**2. JSON Output to File**
```bash
python scoring.py predictions.csv data/test/test.csv --format json --output results.json
```

**3. Using Text Files (One label per line)**
```bash
python scoring.py my_predictions.txt gold_labels.txt
```

---

## Input File Formats

### CSV Format

The script accepts CSV files with the following structure:

**Predictions CSV:**
- Must contain a column named `Predicted_Label` or `Label`
- Each row represents one prediction

**Gold Labels CSV:**
- Must contain a column named `Label`
- Each row represents one true label

Example:
```csv
Sentence,Predicted_Label
"Company reports strong earnings",2
"Market uncertainty continues",1
"Stock price falls sharply",0
```

### Text Format

Simple text files with one integer label per line:
```
0
1
2
1
0
```

### Preprocessing Predictions
To evaluate models with non-standard output formats (e.g., raw finBERT output), use `preprocess_predictions.py`.

#### Features
- **Auto-detection**: Supports finBERT strings ("positive"), standard "Predicted_Label" columns, and simple lists.
- **standardization**: Converts all inputs to the required numeric format (0, 1, 2).

#### Usage
```bash
# Basic (Auto-detect and convert)
python preprocess_predictions.py <input_file>

# Specify output file
python preprocess_predictions.py <input_file> -o <output_file>

# Force specific format
python preprocess_predictions.py <input_file> -f finbert
```

#### Workflow Example
```bash
# 1. Convert finBERT output
python preprocess_predictions.py output/predictions.csv -o predictions_std.csv
```

Output:
```
Detected format: finbert
Converting finBERT string format to standard numeric format...

Conversion successful!
Total predictions: 585
Label distribution:
  Negative (0): 145
  Neutral (1): 240
  Positive (2): 200

Processed predictions saved to: predictions_processed.csv
```

The processed file now looks like:
```csv
sentence,Predicted_Label
"Company reports strong earnings",2
"Market uncertainty continues",1
"Stock price falls sharply",0
```

3. **Evaluate using scoring.py:**
```bash
python scoring.py predictions_processed.csv data/test/test.csv
```

### Complete Pipeline Example

You can chain the preprocessing and evaluation steps:

```bash
# Preprocess
python preprocess_predictions.py finbert_output.csv -o predictions_std.csv

# Evaluate
python scoring.py predictions_std.csv data/test/test.csv --format json --output results.json
```

### Label Mapping Reference

The preprocessing script uses this standard mapping (matching your dataset):

| String Label | Numeric Value | Meaning |
|--------------|---------------|---------|
| `"negative"` | `0` | Negative sentiment |
| `"neutral"` | `1` | Neutral sentiment |
| `"positive"` | `2` | Positive sentiment |

This ensures consistency between finBERT output and your gold standard labels.


---


## Simple Baseline Result
### Overview
The simple baseline predicts the majority class (Neutral) for all test examples, disregarding the input text. This establishes a minimum performance threshold (lower bound) for the task.

### Performance (Weighted)
- **Accuracy:** 48.55%
- **F1-Score:** 31.73%

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

### Error Analysis
- **Systematic Errors:** All Negative (100) and Positive (201) examples are misclassified as Neutral.
- **Correct Predictions:** Only the 284 actual Neutral examples are classified correctly (due to the majority class strategy).

**Expected Accuracy:** 48.55% (284/585)

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

## Strong Baseline Result
### Overview
The strong baseline uses FinBERT, a transformer model pre-trained on financial text. Unlike the simple baseline, it analyzes the semantic content of the input.

### Performance Analysis
- **Overall Accuracy:** 74.36% (435/585 correct).
- **Positive Sentiment:** Highest performance (Precision: 89.29%, Recall: 74.63%).
- **Negative Sentiment:** High recall (78%) but lower precision (52%).
- **Neutral Sentiment:** Solid recall (72.89%) and precision (77.53%).

### Areas for Improvement
- **Precision in Negative Class:** Significant over-prediction of Negative sentiment (low precision: 52%).
- **Class Confusion:** Neutral examples are frequently misclassified as Negative (61 instances), and Positive examples as Neutral (40 instances).

### Comparison
FinBERT achieves a **25.8% improvement** over the simple baseline, confirming the effectiveness of domain-specific pre-training.

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




## References

### Academic Papers

1. **Precision and Recall:**
   - Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." *Journal of Machine Learning Technologies*, 2(1), 37-63.

2. **F1-Score:**
   - Sokolova, M., & Lapalme, G. (2009). "A systematic analysis of performance measures for classification tasks." *Information Processing & Management*, 45(4), 427-437.
   - [Wikipedia: F-score](https://en.wikipedia.org/wiki/F-score)

3. **Multi-class Classification Metrics:**
   - Grandini, M., Bagli, E., & Visani, G. (2020). "Metrics for Multi-Class Classification: an Overview." *arXiv preprint arXiv:2008.05756*.
   - [arXiv:2008.05756](https://arxiv.org/abs/2008.05756)

4. **Confusion Matrix:**
   - Stehman, S. V. (1997). "Selecting and interpreting measures of thematic classification accuracy." *Remote Sensing of Environment*, 62(1), 77-89.
   - [Wikipedia: Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

5. **Sentiment Analysis Evaluation:**
   - Rosenthal, S., Farra, N., & Nakov, P. (2017). "SemEval-2017 Task 4: Sentiment Analysis in Twitter." *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, 502-518.
   - Liu, B. (2012). "Sentiment analysis and opinion mining." *Synthesis Lectures on Human Language Technologies*, 5(1), 1-167.

### Implementation

6. **Scikit-learn Documentation:**
   - [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
   - [sklearn.metrics.accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
   - [sklearn.metrics.precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
   - [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---