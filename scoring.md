# Evaluation Metrics for Financial Sentiment Analysis

## Overview

This document describes the evaluation metrics used to assess the performance of our financial sentiment analysis model. The task is a **3-class sentiment classification** problem with classes: negative (0), neutral (1), and positive (2).

---

## Evaluation Metrics

We employ several standard metrics for multi-class classification tasks:

### 1. Accuracy

**Definition:** The proportion of correctly classified instances out of all instances.

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

For multi-class problems:

$$\text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i)$$

where $n$ is the total number of samples, $y_i$ is the true label, and $\hat{y}_i$ is the predicted label.

**Interpretation:** Accuracy gives an overall measure of how often the model is correct, but can be misleading with imbalanced datasets.

### 2. Precision

**Definition:** The proportion of true positive predictions among all positive predictions for a given class.

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

where $TP_c$ is the number of true positives for class $c$, and $FP_c$ is the number of false positives.

**Interpretation:** Precision measures how accurate the model is when it predicts a specific class. High precision means few false positives.

### 3. Recall (Sensitivity)

**Definition:** The proportion of true positive predictions among all actual positive instances for a given class.

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

where $FN_c$ is the number of false negatives for class $c$.

**Interpretation:** Recall measures how well the model identifies all instances of a specific class. High recall means few false negatives.

### 4. F1-Score

**Definition:** The harmonic mean of precision and recall, providing a single score that balances both metrics.

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**Interpretation:** F1-score is especially useful when classes are imbalanced, as it accounts for both false positives and false negatives.

### 5. Macro-Averaged Metrics

**Definition:** The unweighted mean of per-class metrics. Each class contributes equally regardless of support.

$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

where $C$ is the number of classes (3 in our case: negative, neutral, positive).

**Interpretation:** Macro-averaging treats all classes equally, making it suitable for evaluating performance across imbalanced datasets.

### 6. Weighted-Averaged Metrics

**Definition:** The weighted mean of per-class metrics, where weights are proportional to class support (number of true instances).

$$\text{Weighted-F1} = \sum_{c=1}^{C} \frac{n_c}{n} \cdot F1_c$$

where $n_c$ is the number of instances in class $c$, and $n$ is the total number of instances.

**Interpretation:** Weighted-averaging accounts for class imbalance by giving more weight to classes with more instances.

### 7. Confusion Matrix

**Definition:** A table showing the counts of true vs. predicted classifications for each class.

```
                Predicted
              Neg  Neu  Pos
Actual  Neg  [ ]  [ ]  [ ]
        Neu  [ ]  [ ]  [ ]
        Pos  [ ]  [ ]  [ ]
```

**Interpretation:** The confusion matrix provides detailed insight into which classes are being confused with each other.

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

#### Basic Usage

```bash
python scoring.py <predictions_file> <gold_labels_file>
```

#### Arguments

- `predictions`: Path to predictions file (CSV with "Predicted_Label" column or text file with one prediction per line)
- `gold_labels`: Path to gold standard labels file (CSV with "Label" column or text file with one label per line)
- `--format`: Output format - `text` (default, human-readable) or `json` (structured)
- `--output`: Output file path (optional, prints to stdout if not specified)

### Examples

#### Example 1: Basic Evaluation (Text Output)

```bash
python scoring.py predictions.csv data/test/test.csv
```


#### Example 2: JSON Output to File

```bash
python scoring.py predictions.csv data/test/test.csv --format json --output results.json
```



#### Example 3: Using Text File Inputs

If your predictions and labels are in simple text files (one label per line):

```bash
python scoring.py my_predictions.txt gold_labels.txt
```

**my_predictions.txt:**
```
0
1
2
1
...
```

**gold_labels.txt:**
```
0
1
2
0
...
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

---

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

## Preprocessing Predictions

### Handling Different Output Formats

If your model outputs predictions in a different format (e.g., finBERT), you need to preprocess them first using the `preprocess_predictions.py` script.

### Supported Input Formats

The preprocessing script automatically detects and converts:

1. **finBERT format**: Column `prediction` with string labels ("positive", "negative", "neutral")
2. **Standard string format**: Column `Predicted_Label` with string labels
3. **Label column**: Column `Label` with numeric or string values
4. **Any numeric format**: Automatically standardizes column names

### Usage

#### Basic Usage (Auto-detect format)

```bash
python preprocess_predictions.py <input_file>
```

This creates a new file with `_processed` suffix containing standardized predictions.

#### Specify Output File

```bash
python preprocess_predictions.py <input_file> -o <output_file>
```

#### Force Specific Format

```bash
python preprocess_predictions.py <input_file> -f finbert
```

#### Show Label Mapping

```bash
python preprocess_predictions.py <input_file> --show-mapping
```

### Example Workflow with finBERT

1. **Run finBERT prediction:**
```bash
python finbert/predict.py --text_path test.txt --output_dir output/ --model_path models/finbert-sentiment
```

This creates `output/predictions.csv` with finBERT format:
```csv
sentence,logit,prediction,sentiment_score
"Company reports strong earnings","[...]",positive,0.85
"Market uncertainty continues","[...]",neutral,0.05
"Stock price falls sharply","[...]",negative,-0.78
```

2. **Preprocess finBERT output:**
```bash
python preprocess_predictions.py output/predictions.csv -o predictions_processed.csv
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

