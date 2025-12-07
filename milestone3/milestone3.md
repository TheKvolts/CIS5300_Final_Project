## Extension Description and Empirical Evaluation

To extend the pretrained FinBERT sentiment analysis model, we fine-tuned the model to the FIQA + Financial PhraseBank Dataset. This fine-tuning process resulted in strong empirical performance, achieving **77.26% accuracy**, **0.7456 macro-F1**, and robust class-wise metrics—particularly for the Neutral and Positive categories. The overall results and confusion matrix demonstrate that the adapted FinBERT model generalizes effectively and improves upon the capabilities of the original pretrained version for this task. Refer to milestone 2, simple-baseline.md and strong-baseline.md for reference.


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