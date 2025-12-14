## Evaluation Metrics

This document describes the evaluation metrics used to assess the performance of our financial sentiment analysis model and our aspect-based sentiment analysis (ABSA) extension. 

---

## Task 1: Sentiment Classification (3-class) - Baselines, Extension 1, Extension 2

The primary task is a **3-class sentiment classification** problem with classes: negative (0), neutral (1), and positive (2).

## Task 2: Aspect Classification (4-class) - ABSA Extension 3

The extension task is a **4-class aspect classification** problem using the FiQA dataset, with classes: Corporate (0), Economy (1), Market (2), and Stock (3).

---

## Metrics Overview

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

For both tasks, we use **Macro-averaged F1-score** as our primary evaluation metric because:
1. It balances precision and recall
2. It treats all classes equally (important for financial analysis where all sentiments/aspects matter)
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

**Sentiment Classification (Task 1):**

| String Label | Numeric Value | Meaning |
|--------------|---------------|---------|
| `"negative"` | `0` | Negative sentiment |
| `"neutral"` | `1` | Neutral sentiment |
| `"positive"` | `2` | Positive sentiment |

**Aspect Classification (Task 2 - ABSA Extension):**

| String Label | Numeric Value | Meaning |
|--------------|---------------|---------|
| `"Corporate"` | `0` | Corporate-related aspects (M&A, strategy, leadership) |
| `"Economy"` | `1` | Economy-related aspects (macro trends, policy) |
| `"Market"` | `2` | Market-related aspects (indices, sectors) |
| `"Stock"` | `3` | Stock-related aspects (price action, volatility) |

---

# Results

## Task 1: Sentiment Classification

### Simple Baseline Result
#### Overview
The simple baseline predicts the majority class (Neutral) for all test examples, disregarding the input text. This establishes a minimum performance threshold (lower bound) for the task.

#### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 48.55% |
| Macro F1 | 0.2179 |
| Weighted F1 | 0.3173 |

Classification report:
```
              precision    recall  f1-score   support

    Negative       0.00      0.00      0.00       100
     Neutral       0.49      1.00      0.65       284
    Positive       0.00      0.00      0.00       201

    accuracy                           0.49       585
   macro avg       0.16      0.33      0.22       585
weighted avg       0.24      0.49      0.32       585
```

**Explanation:** The baseline predicts Neutral (class 1) for all 585 test examples, which is the majority class from the training data (54.29% of training examples). It correctly predicts 284 out of 585 examples (48.55% accuracy), which equals the proportion of Neutral examples in the test set. All Negative and Positive examples are incorrectly predicted as Neutral, resulting in 0.00 precision/recall/F1 for those classes.

---

### Strong Baseline Result (Pre-trained FinBERT)
#### Overview
The strong baseline uses FinBERT, a transformer model pre-trained on financial text. Unlike the simple baseline, it analyzes the semantic content of the input without any task-specific fine-tuning.

#### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 74.36% |
| Macro F1 | 0.7295 |
| Weighted F1 | 0.7508 |

Classification report:
```
              precision    recall  f1-score   support

    Negative       0.52      0.78      0.62       100
     Neutral       0.78      0.73      0.75       284
    Positive       0.89      0.75      0.81       201

    accuracy                           0.74       585
   macro avg       0.73      0.75      0.73       585
weighted avg       0.77      0.74      0.75       585
```

Confusion Matrix:
```
         Predicted:
           Neg    Neu    Pos
Actual:
  Neg       78     20      2
  Neu       61    207     16
  Pos       11     40    150
```

**Analysis:** FinBERT achieves a **25.8% improvement** over the simple baseline, confirming the effectiveness of domain-specific pre-training. Main weakness: over-prediction of Negative sentiment (low precision: 52%).

---

### MS3 Extension 1: Standard Fine-tuned FinBERT
#### Overview
Fine-tuning FinBERT on our training dataset with standard cross-entropy loss.

#### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 77.26% |
| Macro F1 | 0.7456 |
| Weighted F1 | 0.7770 |

**Improvement over Strong Baseline:** +2.90% accuracy, +1.61 percentage points Macro F1.

---

### MS3 Extension 1: Class-Weighted Fine-tuned FinBERT
#### Overview
Fine-tuning FinBERT with inverse frequency class weighting to address class imbalance (Negative: 2.29, Neutral: 0.61, Positive: 1.07).

#### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 80.17% |
| Macro F1 | 0.7868 |
| Weighted F1 | 0.8030 |

Per-class F1 comparison:
| Class | Standard Fine-tuned | Class-Weighted | Change |
|-------|---------------------|----------------|--------|
| Negative | 0.6079 | 0.6996 | +0.0917 |
| Neutral | 0.7993 | 0.7951 | -0.0042 |
| Positive | 0.8296 | 0.8656 | +0.0360 |

**Improvement over Strong Baseline:** +5.81% accuracy, +5.73 percentage points Macro F1.

**Key Finding:** Class weighting dramatically improves minority class performance, with Negative F1 improving by +9.17 percentage points.

---

## Extension 2: Fine-Tuned Llama Model

Accuracy: 0.8256

------------------------------------------------------------
AGGREGATE METRICS
------------------------------------------------------------
Macro-averaged Precision:  0.8113
Macro-averaged Recall:     0.7468
Macro-averaged F1:         0.7633

Weighted-averaged Precision: 0.8213
Weighted-averaged Recall:    0.8256
Weighted-averaged F1:        0.8142

------------------------------------------------------------
PER-CLASS METRICS
------------------------------------------------------------
Class        Precision    Recall       F1           Support     
------------------------------------------------------------
Negative     0.7368       0.4200       0.5350       100         
Neutral      0.7908       0.9049       0.8440       284         
Positive     0.9064       0.9154       0.9109       201         

------------------------------------------------------------
CONFUSION MATRIX
------------------------------------------------------------
         Predicted:
           Neg    Neu    Pos
Actual:
  Neg       42     56      2
  Neu       10    257     17
  Pos        5     12    184
============================================================

---

## Extension 3: Aspect Classification (ABSA Extension)

### Overview
This extension applies our approach to a new task: **Aspect-Based Sentiment Analysis (ABSA)** using the FiQA dataset from WWW'18. Instead of predicting sentiment, we classify financial text into one of four aspect categories.

This extension is motivated by Yang et al. (2018), "Financial Aspect-Based Sentiment Analysis using Deep Representations," which demonstrated that aspect classification enables more granular analysis of financial text.

### Dataset: FiQA
| Split | Examples |
|-------|----------|
| Train | 959 |
| Validation | 102 |
| Test | 149 |

**Class Distribution (Training Set):**
| Aspect | Count | Percentage |
|--------|-------|------------|
| Stock | 562 | 58.5% |
| Corporate | 367 | 38.2% |
| Market | 26 | 2.7% |
| Economy | 4 | 0.4% |

**Note:** Extreme class imbalance—Economy has only 4 training examples.

### Class Weights Applied
| Aspect | Weight |
|--------|--------|
| Corporate | 0.6533 |
| Economy | 59.9375 |
| Market | 9.2212 |
| Stock | 0.4266 |

### Results: Class-Weighted FinBERT for Aspect Classification

#### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 88.59% |
| Macro F1 | 0.5429 |
| Weighted F1 | 0.8688 |

Classification report:
```
              precision    recall  f1-score   support

   Corporate       0.91      0.94      0.92        64
     Economy       0.00      0.00      0.00         3
      Market       0.50      0.25      0.33         8
       Stock       0.89      0.95      0.92        74

    accuracy                           0.89       149
   macro avg       0.57      0.53      0.54       149
weighted avg       0.86      0.89      0.87       149
```

Confusion Matrix:
```
              Predicted:
              Corporate  Economy  Market  Stock
Actual:
  Corporate        60        0       0       4
    Economy         1        0       2       0
     Market         1        0       2       5
      Stock         4        0       0      70
```

### Analysis

**Strengths:**
- High overall accuracy (88.59%) demonstrates successful transfer of FinBERT to aspect classification
- Corporate (F1: 0.92) and Stock (F1: 0.92) classes perform excellently
- These two majority classes comprise 93% of the test set

**Limitations:**
- Economy F1: 0.00 — the model never predicted Economy (only 4 training examples made this class unlearnable)
- Market F1: 0.33 — only 2 of 8 test examples predicted correctly
- Macro F1 (0.54) is dragged down by minority class failures

**Key Insight:** This result validates the motivation from Yang et al. (2018) for using transfer learning on small, domain-specific datasets. With only 959 training examples and extreme class imbalance, traditional approaches would struggle. Our class-weighted FinBERT achieves strong performance on majority classes while revealing the fundamental limitation: no amount of class weighting can overcome having only 4 training examples for a class.

---

## Summary: All Results

| Model | Task | Accuracy | Macro F1 | Weighted F1 |
|-------|------|----------|----------|-------------|
| Simple Baseline (Majority Class) | Sentiment | 48.55% | 0.2179 | 0.3173 |
| Pre-trained FinBERT | Sentiment | 74.36% | 0.7295 | 0.7508 |
| Standard Fine-tuned FinBERT | Sentiment | 77.26% | 0.7456 | 0.7770 |
| **Class-Weighted Fine-tuned FinBERT** | **Sentiment** | **80.17%** | **0.7868** | **0.8030** |
| **Class-Weighted FinBERT** | **Aspect (ABSA)** | **88.59%** | **0.5429** | **0.8688** |

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

6. **Aspect-Based Sentiment Analysis:**
   - Yang, S., Rosenfeld, J., & Makutonin, J. (2018). "Financial Aspect-Based Sentiment Analysis using Deep Representations." *arXiv:1808.07931*.

7. **FinBERT:**
   - Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." *arXiv preprint arXiv:1908.10063*.

8. **FiQA Dataset:**
   - Maia, M., et al. (2018). "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering." *Companion Proceedings of The Web Conference 2018*.

### Implementation

9. **Scikit-learn Documentation:**
   - [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
   - [sklearn.metrics.accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
   - [sklearn.metrics.precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
   - [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

10. **HuggingFace Transformers:**
    - [Transformers Documentation](https://huggingface.co/docs/transformers/)
    - [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

11. **FiQA Dataset on HuggingFace:**
    - [pauri32/fiqa-2018](https://huggingface.co/datasets/pauri32/fiqa-2018)

---