# Code Directory

This directory contains all code developed for the project, including baselines, extensions, and evaluation scripts.

## Directory Structure

```
code/
├── baselines/          # Baseline implementations
├── extension1/         # Extension 1: Fine-Tuned FinBERT
├── extension2/        # Extension 2: Fine-Tuned Llama
├── extension3/        # Extension 3: Aspect-Based Sentiment Analysis
└── originalnotebooks/ # Original Jupyter notebooks (for reference)
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

Use `scoring.py` to evaluate the predictions:

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

(Add instructions when available)

---

## Extension 3: Aspect-Based Sentiment Analysis

(Add instructions when available)

---

## Baselines

(Add instructions when available)

---

## Evaluation Script

The `scoring.py` script in the project root evaluates predictions:

```bash
python scoring.py <predictions_file> <gold_labels_file>
```

The predictions file should be a CSV with a `Predicted` column (or `Predicted_Label` or `Label`).
The gold labels file should be a CSV with a `Label` or `label` column.

Example:
```bash
python scoring.py output/extension1\(milestone3\)/strategy1a_standard_finetuned_predictions.csv data/test/test.csv
```

