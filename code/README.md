# Code Directory

This directory contains all code developed for the project, including baselines, extensions, and evaluation scripts.

## Directory Structure

```
code/
├── evaluation scripts
├── baselines/          # Baseline implementations
├── extension1/         # Extension 1: Fine-Tuned FinBERT
├── extension2/        # Extension 2: Fine-Tuned Llama
├── extension3/        # Extension 3: Aspect-Based Sentiment Analysis
└── originalnotebooks/ # Original Jupyter notebooks (for reference)
```

---

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

These files should be used by `code/scoring.py`.
```bash
python code/scoring.py milestone2/simple_baseline_test_predictions.csv milestone2/test_sentence_label.csv
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

These files should be used by `code/scoring.py`.
```bash
python code/scoring.py milestone2/strong_baseline_test_predictions.csv milestone2/test_sentence_label.csv
```

---

## Extension 1: Fine-Tuned FinBERT

### Overview

Fine-tunes FinBERT for financial sentiment classification using 4 different strategies:
- **Strategy 1a**: Standard fine-tuning
- **Strategy 1b**: Class-balanced fine-tuning
- **Strategy 1c**: Focal loss
- **Strategy 1d**: Discriminative fine-tuning

### Prerequisites

Install required packages:
```bash
pip install transformers datasets torch scikit-learn pandas numpy
```

### Running the Script

#### Train All Strategies

```bash
python code/extension1/train_extension1.py --all
```

This will train all 4 strategies sequentially and save predictions to `output/extension1(milestone3)/`.

#### Train Specific Strategy

```bash
# Train only Strategy 1a (Standard fine-tuning)
python code/extension1/train_extension1.py --strategy 1a

# Train only Strategy 1b (Class-balanced)
python code/extension1/train_extension1.py --strategy 1b

# Train only Strategy 1c (Focal loss)
python code/extension1/train_extension1.py --strategy 1c

# Train only Strategy 1d (Discriminative fine-tuning)
python code/extension1/train_extension1.py --strategy 1d
```

#### Custom Data Paths

```bash
# Use custom data directory
python code/extension1/train_extension1.py --all --data_dir data

# Use custom file paths
python code/extension1/train_extension1.py --all \
    --train_file data/train/train.csv \
    --dev_file data/development/development.csv \
    --test_file data/test/test.csv
```

#### Custom Output Directory

```bash
python code/extension1/train_extension1.py --all \
    --output_dir output/extension1(milestone3)
```

### Output Files

After training, predictions are saved as:
- `output/extension1(milestone3)/strategy1a_standard_finetuned_predictions.csv`
- `output/extension1(milestone3)/strategy1b_class_balanced_finetuned_predictions.csv`
- `output/extension1(milestone3)/strategy1c_focal_loss_predictions.csv`
- `output/extension1(milestone3)/strategy1d_discriminative_finetuning_predictions.csv`

Each CSV file contains:
- `Sentence`: The input text
- `Predicted`: Predicted label (0=Negative, 1=Neutral, 2=Positive)
- `Gold`: True label

### Evaluation

Use `code/scoring.py` to evaluate the predictions:

```bash
# Evaluate Strategy 1a
python code/scoring.py output/extension1\(milestone3\)/strategy1a_standard_finetuned_predictions.csv data/test/test.csv

# Evaluate Strategy 1b
python code/scoring.py output/extension1\(milestone3\)/strategy1b_class_balanced_finetuned_predictions.csv data/test/test.csv

# Evaluate Strategy 1c
python code/scoring.py output/extension1\(milestone3\)/strategy1c_focal_loss_predictions.csv data/test/test.csv

# Evaluate Strategy 1d
python code/scoring.py output/extension1\(milestone3\)/strategy1d_discriminative_finetuning_predictions.csv data/test/test.csv
```

### Strategy Details

#### Strategy 1a: Standard Fine-tuning
- Learning rate: 2e-5
- Epochs: 5
- Batch size: 16
- Loss: Standard cross-entropy
- Early stopping: 2 epochs patience

#### Strategy 1b: Class-Balanced Fine-tuning
- Learning rate: 3e-5
- Epochs: 10
- Batch size: 16 (effective 32 with gradient accumulation)
- Loss: Weighted cross-entropy (inverse frequency class weights)
- Features: Warmup, gradient clipping, mixed precision training
- Early stopping: 3 epochs patience

#### Strategy 1c: Focal Loss
- Same hyperparameters as Strategy 1b
- Loss: Focal loss (gamma=2.0) with class weights
- Focuses on hard-to-classify examples

#### Strategy 1d: Discriminative Fine-tuning
- Same hyperparameters as Strategy 1b
- Loss: Weighted cross-entropy
- Learning rate: Layer-wise decay (95% per layer)
- Different learning rates for different model layers

### Command-Line Arguments

```
--strategy {1a,1b,1c,1d,all}
    Which strategy to train (default: all)

--data_dir PATH
    Directory containing train/, development/, test/ subdirectories (default: data)

--output_dir PATH
    Output directory for predictions (default: output/extension1(milestone3))

--train_file PATH
    Path to training CSV (overrides data_dir)

--dev_file PATH
    Path to development CSV (overrides data_dir)

--test_file PATH
    Path to test CSV (overrides data_dir)
```

---

## Extension 2: Fine-Tuned Llama

We have implemented a fine-tuned **Llama 3.1 8B** model for classifying the sentiment of news headlines using QLoRA.

### Directory Structure
The fine-tuning logic is located in `extension2-finetune-llama/`:
- `data/`: Contains converted JSONL datasets.
- `scripts/`: Training and inference scripts.
- `prepare_data.py`: Converts raw CSVs to JSONL format.

### How to Run

1.  **Prepare Data**:
    ```bash
    python extension2-finetune-llama/prepare_data.py
    ```

2.  **Train Model**:
    ```bash
    python extension2-finetune-llama/scripts/train.py --hf_token "YOUR_TOKEN" --epochs 3
    ```
    This saves the adapter to `./final_adapter`.

3.  **Run Inference**:
    Evaluate on test data:
    ```bash
    python extension2-finetune-llama/scripts/inference.py --test_file "extension2-finetune-llama/data/test.jsonl"
    ```
    or test a single headline:
    ```bash
    python extension2-finetune-llama/scripts/inference.py --headline "Example news headline..."
    ```

---

## Extension 3: Aspect-Based Sentiment Analysis

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

---
