# CIS5300_Final_Project

## Setup

To set up the project environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Simple Baseline
This baseline uses a majority class predictor that predicts the most frequent sentiment class from the training data for all test examples. It **completely ignores the input text** and always predicts the same class, regardless of the headline content. This serves as a simple, lower-bound baseline to establish the minimum performance threshold for the task.

- Test set size: 585 examples. (refer to data.md)

| Sentence                                                                 | Sentiment | Label |
|--------------------------------------------------------------------------|:---------:|:-----:|
| The inventors are Mukkavilli Krishna Kiran, Sabharwal Ashutosh and Aazhang Behnaam. | neutral   |   1   |
| $IBIO up 10% in premarket ready for lift off                             | positive  |   2   |

### Running the script
```bash
python simple-baseline.py
```

This creates two files with two columns- Sentence and Label:
- simple_baseline_test_predictions.csv
- test_sentence_label.csv

These files should be used by `scoring.py`.
```bash
python scoring.py milestone2/simple_baseline_test_predictions.csv milestone2/test_sentence_label.csv
```

---

## Strong Baseline
This baseline uses the [FinBERT](https://huggingface.co/ProsusAI/finbert) text-classification pipeline from HuggingFace to predict financial sentiment (negative / neutral / positive). It serves as a strong, reproducible baseline for the task and includes inference and evaluation instructions.

- Test set size: 585 examples. (refer to data.md)

| Sentence                                                                 | Sentiment | Label |
|--------------------------------------------------------------------------|:---------:|:-----:|
| The inventors are Mukkavilli Krishna Kiran, Sabharwal Ashutosh and Aazhang Behnaam. | neutral   |   1   |
| $IBIO up 10% in premarket ready for lift off                             | positive  |   2   |

### Running the script
```bash
python strong-baseline.py
```

This creates two files with two columns- Sentence and Label:
- strong_baseline_test_predictions.csv
- test_sentence_label.csv

These files should be used by `scoring.py`.
```bash
python scoring.py milestone2/strong_baseline_test_predictions.csv milestone2/test_sentence_label.csv
```

---

## Extension 1: Fine-Tuned FinBERT (Sentiment Classification)

### Extension Description

For Milestone 3, we extended our Milestone 2 strong baseline (pre-trained FinBERT) by fine-tuning the model on our Financial PhraseBank training dataset using a standard transformer classification head optimized with cross-entropy loss. While the Milestone 2 baseline used FinBERT out-of-the-box without any training on our data, this extension adapts all model parameters through supervised training on our specific sentiment classification task.

We implemented 4 different fine-tuning strategies:
- **Strategy 1a: Standard Fine-tuning** - Standard supervised learning with cross-entropy loss, training all model parameters end-to-end. Trained for 5 epochs with learning rate 2e-5, batch size 16, and early stopping based on development set F1-macro score.
- **Strategy 1b: Class-Balanced Fine-tuning** - Addresses class imbalance (Negative: 14.53%, Neutral: 54.29%, Positive: 31.18%) by applying inverse frequency class weighting to the loss function. Uses learning rate 3e-5, 10 epochs, gradient accumulation (effective batch size 32), warmup steps, gradient clipping, and mixed precision training.
- **Strategy 1c: Focal Loss** - Uses focal loss (gamma=2.0) with class weights to focus on hard-to-classify examples. Same hyperparameters as Strategy 1b.
- **Strategy 1d: Discriminative Fine-tuning** - Applies layer-wise learning rate decay (95% per layer) with class-weighted loss. Different learning rates for different model layers, with classifier head having the highest learning rate.

### .ipynb Version for Colab (Used for Actual Milestone Submissions)

The fine-tuning implementation is provided in `code/originalnotebooks/milestone3_finetune_extension_additions.ipynb`, a Google Colab notebook which our group ran using Google Colab and Google Drive. To run the code:

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
   - `output/ms3_strong_baseline_predictions_standard.csv` (Strategy 1a)
   - `output/improved_strong_baseline_predictions.csv` (Strategy 1b)

### Script Version: Reformatted for Final Submission

A command-line script version is available in `code/extension1/train_extension1.py` for easier reproduction and final submission.

#### Prerequisites

```bash
pip install transformers datasets torch scikit-learn pandas numpy
```

#### Running the Script

**Train all strategies:**
```bash
python code/extension1/train_extension1.py --all
```

**Train specific strategy:**
```bash
# Strategy 1a: Standard fine-tuning
python code/extension1/train_extension1.py --strategy 1a

# Strategy 1b: Class-balanced fine-tuning
python code/extension1/train_extension1.py --strategy 1b

# Strategy 1c: Focal loss
python code/extension1/train_extension1.py --strategy 1c

# Strategy 1d: Discriminative fine-tuning
python code/extension1/train_extension1.py --strategy 1d
```

#### Output Files

Predictions are saved to `output/extension1(milestone3)/`:
- `strategy1a_standard_finetuned_predictions.csv`
- `strategy1b_class_balanced_finetuned_predictions.csv`
- `strategy1c_focal_loss_predictions.csv`
- `strategy1d_discriminative_finetuning_predictions.csv`

Each CSV contains: `Sentence`, `Predicted` (0/1/2), `Gold` (true label).

#### Evaluation

Evaluate predictions using `scoring.py`:

```bash
# Evaluate Strategy 1a
python scoring.py output/extension1\(milestone3\)/strategy1a_standard_finetuned_predictions.csv data/test/test.csv

# Evaluate Strategy 1b
python scoring.py output/extension1\(milestone3\)/strategy1b_class_balanced_finetuned_predictions.csv data/test/test.csv

# Evaluate Strategy 1c
python scoring.py output/extension1\(milestone3\)/strategy1c_focal_loss_predictions.csv data/test/test.csv

# Evaluate Strategy 1d
python scoring.py output/extension1\(milestone3\)/strategy1d_discriminative_finetuning_predictions.csv data/test/test.csv
```

For more detailed instructions, see `code/README.md`.

Refer to milestone 2, simple-baseline.md and strong-baseline.md for reference.

---

## Extension 2: Fine-Tuned Llama Model

We have implemented a fine-tuned **Llama 3.1 8B** model for classifying the sentiment of news headlines using QLoRA.

### Directory Structure
The fine-tuning logic is located in `extension3-finetune-llama/`:
- `data/`: Contains converted JSONL datasets.
- `scripts/`: Training and inference scripts.
- `prepare_data.py`: Converts raw CSVs to JSONL format.

### How to Run

1.  **Prepare Data**:
    ```bash
    python extension3-finetune-llama/prepare_data.py
    ```

2.  **Train Model**:
    ```bash
    python extension3-finetune-llama/scripts/train.py --hf_token "YOUR_TOKEN" --epochs 3
    ```
    This saves the adapter to `./final_adapter`.

3.  **Run Inference**:
    Evaluate on test data:
    ```bash
    python extension3-finetune-llama/scripts/inference.py --test_file "extension3-finetune-llama/data/test.jsonl"
    ```
    or test a single headline:
    ```bash
    python extension3-finetune-llama/scripts/inference.py --headline "Example news headline..."
    ```

---

## Extension 3: Aspect-Based Sentiment Analysis (ABSA)

### Overview

This extension applies our FinBERT fine-tuning approach to a new task: **Aspect Classification** using the FiQA dataset from WWW'18. Instead of predicting sentiment (positive/neutral/negative), we classify financial text into one of four aspect categories:

| Aspect | Description | Training Examples |
|--------|-------------|-------------------|
| Corporate | M&A, strategy, leadership, legal | 367 (38.2%) |
| Economy | Macro trends, policy | 4 (0.4%) |
| Market | Indices, sectors, broad market movements | 26 (2.7%) |
| Stock | Price action, volatility, individual stocks | 562 (58.5%) |

This extension is motivated by Yang et al. (2018), "Financial Aspect-Based Sentiment Analysis using Deep Representations," which demonstrated that aspect classification enables more granular analysis of financial text.

### How to Use the Code

The implementation is provided in `milestone4_absa_extension.ipynb`, a Google Colab notebook.

1. **Setup**: Open the notebook in Google Colab. The notebook will automatically install required packages and load the FiQA dataset from HuggingFace (`pauri32/fiqa-2018`).

2. **No Local Data Required**: Unlike the sentiment task, the FiQA dataset is loaded directly from HuggingFace:
   ```python
   from datasets import load_dataset
   fiqa_dataset = load_dataset("pauri32/fiqa-2018")
   ```

3. **Execution**: Run all cells sequentially. The notebook will:
   - Load and preprocess the FiQA dataset
   - Extract Level-1 aspect labels from hierarchical aspect annotations
   - Compute class weights to handle extreme imbalance
   - Fine-tune FinBERT with class-weighted loss
   - Evaluate on the test set
   - Save predictions and confusion matrix to `output/`

4. **Output Files**: After execution:
   - `output/fiqa_aspect_distribution.png` - Class distribution visualization
   - `output/fiqa_aspect_confusion_matrix.png` - Test set confusion matrix
   - `output/fiqa_aspect_predictions.csv` - Model predictions

### Results

| Metric | Score |
|--------|-------|
| Accuracy | 88.59% |
| Macro F1 | 0.5429 |
| Weighted F1 | 0.8688 |

Per-class performance:
| Aspect | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Corporate | 0.91 | 0.94 | 0.92 | 64 |
| Economy | 0.00 | 0.00 | 0.00 | 3 |
| Market | 0.50 | 0.25 | 0.33 | 8 |
| Stock | 0.89 | 0.95 | 0.92 | 74 |

**Key Findings**: The model achieves strong performance on majority classes (Corporate: 0.92 F1, Stock: 0.92 F1) but struggles with minority classes due to extreme data imbalance. Economy (only 4 training examples) was never predicted correctly, demonstrating the fundamental limitation that class weighting cannot overcome severe data scarcity.

---

## Results Summary

| Model | Task | Accuracy | Macro F1 | Weighted F1 |
|-------|------|----------|----------|-------------|
| Simple Baseline (Majority Class) | Sentiment | 48.55% | 0.2179 | 0.3173 |
| Pre-trained FinBERT | Sentiment | 74.36% | 0.7295 | 0.7508 |
| Standard Fine-tuned FinBERT | Sentiment | 77.26% | 0.7456 | 0.7770 |
| Class-Weighted Fine-tuned FinBERT | Sentiment | 80.17% | **0.7868** | 0.8030 |
| **Fine-tuned LLM** | **Sentiment** | **82.56%** | 0.7633 | **0.8142** |
| **Class-Weighted FinBERT** | **Aspect (ABSA)** | **88.59%** | **0.5429** | **0.8688** |

---

## References

- Yang, S., Rosenfeld, J., & Makutonin, J. (2018). "Financial Aspect-Based Sentiment Analysis using Deep Representations." arXiv:1808.07931.
- Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." arXiv:1908.10063.
- Maia, M., et al. (2018). "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering." Companion Proceedings of The Web Conference 2018.
- FiQA Dataset: [pauri32/fiqa-2018](https://huggingface.co/datasets/pauri32/fiqa-2018)
- FinBERT Model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)