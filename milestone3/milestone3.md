## Extension Description and Empirical Evaluation

### How to Use the Code

The fine-tuning implementation is provided in `milestone3_finetune_extension.ipynb`, a Google Colab notebook which our group ran using Google Colab and Google Drive. To run the code:

1. **Setup**: Open the notebook in Google Colab and mount your Google Drive containing the project directory. The notebook will automatically install required packages (transformers, datasets, torch, scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm, accelerate).

2. **Data Paths**: Ensure your data files are located at:
   - `data/train/train.csv`
   - `data/development/development.csv`
   - `data/test/test.csv`

3. **Execution**: Run all cells sequentially. The notebook will:
   - Load and explore the dataset characteristics
   - Define evaluation functions (accuracy, F1-macro, F1-weighted)
   - Train the model(s) on the training set
   - Evaluate on the development set
   - Generate predictions and evaluate on the test set
   - Save predictions to CSV files in the `output/` directory

4. **Output Files**: After execution, predictions are saved as:
   - `output/ms3_strong_baseline_predictions_standard.csv` (Attempt 1)
   - `output/improved_strong_baseline_predictions.csv` (Attempt 2)

### Extension Description

For Milestone 3, we extended our Milestone 2 strong baseline (pre-trained FinBERT) by fine-tuning the model on our Financial PhraseBank training dataset using a standard transformer classification head optimized with cross-entropy loss. While the Milestone 2 baseline used FinBERT out-of-the-box without any training on our data, this extension adapts all model parameters through supervised training on our specific sentiment classification task.

**Attempt 1: Standard Fine-tuning** applies standard supervised learning with cross-entropy loss, training all model parameters end-to-end on our training data. The model is trained for 5 epochs with a learning rate of 2e-5, batch size of 16, and early stopping based on development set F1-macro score.

**Attempt 2: Class-Weighted Fine-tuning** addresses the class imbalance in our dataset (Negative: 14.53%, Neutral: 54.29%, Positive: 31.18% in training set) by applying inverse frequency class weighting to the loss function. This approach assigns higher weights to underrepresented classes (especially Negative sentiment) during training, forcing the model to pay more attention to minority class examples. Additionally, we increased the learning rate to 3e-5, extended training to 10 epochs, added gradient accumulation (effective batch size 32), warmup steps, gradient clipping, and mixed precision training for improved performance.

Refer to milestone 2, simple-baseline.md and strong-baseline.md for reference.


## Attempt 1: Standard Fine-tuned FinBERT
```
======================================================================
EVALUATING FINBERT ON TEST SET
======================================================================

======================================================================
EVALUATION RESULTS
======================================================================
Accuracy:          0.7726 (77.26%)
F1-Score (Macro):  0.7456
F1-Score (Weighted): 0.7770

======================================================================
PER-CLASS METRICS
======================================================================
              precision    recall  f1-score   support

    Negative     0.5433    0.6900    0.6079       100
     Neutral     0.8465    0.7570    0.7993       284
    Positive     0.8235    0.8358    0.8296       201

    accuracy                         0.7726       585
   macro avg     0.7378    0.7610    0.7456       585
weighted avg     0.7868    0.7726    0.7770       585

======================================================================
CONFUSION MATRIX
======================================================================
Rows: True Labels | Columns: Predicted Labels

           Negative  Neutral  Positive
Negative      69       21       10
 Neutral      43      215       26
Positive      15       18      168
```
## Attempt 2: Class-Weighted Fine-tuned FinBERT

```
======================================================================
EVALUATING IMPROVED MODEL ON TEST SET
======================================================================

======================================================================
EVALUATION RESULTS
======================================================================
Accuracy:          0.8017 (80.17%)
F1-Score (Macro):  0.7868
F1-Score (Weighted): 0.8030

======================================================================
PER-CLASS METRICS
======================================================================
              precision    recall  f1-score   support

    Negative     0.5944    0.8500    0.6996       100
     Neutral     0.9510    0.6831    0.7951       284
    Positive     0.7983    0.9453    0.8656       201

    accuracy                         0.8017       585
   macro avg     0.7812    0.8261    0.7868       585
weighted avg     0.8376    0.8017    0.8030       585

======================================================================
CONFUSION MATRIX
======================================================================
Rows: True Labels | Columns: Predicted Labels

           Negative  Neutral  Positive
Negative      85        5       10
 Neutral      52      194       38
Positive       6        5      190
```

## Comparison Summary
```
======================================================================
FINAL COMPARISON: STANDARD FINETUNED vs IMPROVED FINETUNED FINBERT
======================================================================

OVERALL METRICS:
----------------------------------------------------------------------
Metric               Standard Finetuned Improved Finetuned Change         
----------------------------------------------------------------------
Accuracy             0.7726 (77.26%)  0.8017 (80.17%)  +2.91%
F1-Macro             0.7456          0.7868          +0.0412
F1-Weighted          0.7770          0.8030          +0.0260

PER-CLASS F1 SCORES:
----------------------------------------------------------------------
Class                Standard Finetuned Improved Finetuned Change         
----------------------------------------------------------------------
Negative             0.6079          0.6996          +0.0917
Neutral              0.7993          0.7951          -0.0042
Positive             0.8296          0.8656          +0.0360

KEY IMPROVEMENTS:
----------------------------------------------------------------------
Negative class F1 improved by 0.0917 (9.17%)
(This addresses the class imbalance issue)
Macro F1 improved by 0.0412 (4.12%)
Better balanced performance across all classes!
```

The class-weighted fine-tuning approach (Attempt 2) shows significant improvements over standard fine-tuning (Attempt 1):

**Overall Improvements:**
- **Accuracy**: +2.91% (77.26% → 80.17%)
- **F1-Macro**: +0.0412 (0.7456 → 0.7868) - better balanced performance across all classes
- **F1-Weighted**: +0.0260 (0.7770 → 0.8030)

**Per-Class Improvements:**
- **Negative**: F1 improved by +0.0917 (0.6079 → 0.6996) - addresses class imbalance
- **Neutral**: F1 slightly decreased by -0.0042 (0.7993 → 0.7951) - minimal change
- **Positive**: F1 improved by +0.0360 (0.8296 → 0.8656)

**Key Observations:**
1. The class weighting strategy successfully addresses the class imbalance, with Negative sentiment F1 improving by 9.17 percentage points.
2. Macro F1 improved by 4.12 percentage points, indicating better balanced performance across all three classes.
3. The improved model shows higher recall for Negative (85.00% vs 69.00%) and Positive (94.53% vs 83.58%) classes, while maintaining strong precision.
4. The confusion matrix shows the improved model makes fewer errors, especially reducing false negatives for Negative sentiment (from 21 to 5) and false positives for Positive sentiment (from 18 to 5).
