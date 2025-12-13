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

## Simple Baseline
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

## Strong Baseline
This baseline uses the [FinBERT](https://huggingface.co/ProsusAI/finbert) text-classification pipeline from HuggingFace to predict financial sentiment (negative / neutral / positive). It serves as a strong, reproducible baseline for the task and includes inference and evaluation instructions.

- Test set size: 585 examples. (refer to data.md)

Sentence                                                                 | Sentiment | Label |
|--------------------------------------------------------------------------|:---------:|:-----:|
| The inventors are Mukkavilli Krishna Kiran, Sabharwal Ashutosh and Aazhang Behnaam. | neutral   |   1   |
| $IBIO up 10% in premarket ready for lift off                             | positive  |   2   |

## Running the script
```
python strong-baseline.py
```

This creates two files with two columns- Sentence and Label:
- strong_baseline_test_predictions.csv
- test_sentence_label.csv

These files should be used by `scoring.py`.
```
python scoring.py milestone2/strong_baseline_test_predictions.csv milestone2/test_sentence_label.csv
```

## Extension1: Fine-Tuned FinBert
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



## Extension2: Fine-Tuned Llama Model
We have implemented a fine-tuned **Llama 3.1 8B** model for classifying news sources (Fox News vs. NBC News) using QLoRA. For a high-level overview of our approach, see [APPROACH.md](finetune-llm-ec2/APPROACH.md).

### Directory Structure
The fine-tuning logic is located in `finetune-llm-ec2/`:
- `data/`: Contains converted JSONL datasets.
- `scripts/`: Training and inference scripts.
- `prepare_data.py`: Converts raw CSVs to JSONL format.

### How to Run

1.  **Prepare Data**:
    ```bash
    python finetune-llm-ec2/prepare_data.py
    ```

2.  **Train Model**:
    ```bash
    python finetune-llm-ec2/scripts/train.py --hf_token "YOUR_TOKEN" --epochs 3
    ```
    This saves the adapter to `./final_adapter`.

3.  **Run Inquiry**:
    Evaluate on test data:
    ```bash
    python finetune-llm-ec2/scripts/inference.py --test_file "finetune-llm-ec2/data/test.jsonl"
    ```
    or test a single headline:
    ```bash
    python finetune-llm-ec2/scripts/inference.py --headline "Example news headline..."
    ```

