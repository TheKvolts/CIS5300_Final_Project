#!/usr/bin/env python3
"""
Extension 1: Fine-Tuned FinBERT Training Script

This script trains 4 different fine-tuning strategies for financial sentiment analysis:
- Strategy 1a: Standard fine-tuning
- Strategy 1b: Class-balanced fine-tuning  
- Strategy 1c: Focal loss
- Strategy 1d: Discriminative fine-tuning

Outputs predictions in format compatible with scoring.py
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
MODEL_NAME = "ProsusAI/finbert"
NUM_LABELS = 3
MAX_LENGTH = 128
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']
LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


def load_data(train_file, dev_file, test_file):
    """Load and prepare datasets"""
    print("Loading datasets...")
    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)
    
    print(f"  Training: {len(train_df)} examples")
    print(f"  Development: {len(dev_df)} examples")
    print(f"  Test: {len(test_df)} examples")
    
    return train_df, dev_df, test_df


def prepare_dataset_for_training(df):
    """Convert pandas DataFrame to HuggingFace Dataset"""
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={'Sentence': 'text', 'label': 'label'})
    dataset = Dataset.from_pandas(df_copy[['text', 'label']])
    return dataset


def tokenize_function(examples, tokenizer):
    """Tokenize text inputs"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


def compute_metrics(eval_pred):
    """Compute metrics during training/evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def compute_class_weights(labels):
    """Compute inverse frequency class weights"""
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


# ============================================================================
# Strategy 1a: Standard Fine-tuning
# ============================================================================

def train_strategy1a(train_tokenized, dev_tokenized, test_tokenized, test_df, output_dir):
    """Strategy 1a: Standard fine-tuning"""
    print("\n" + "=" * 70)
    print("STRATEGY 1A: STANDARD FINE-TUNING")
    print("=" * 70)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/models/strategy1a',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Evaluate on dev
    dev_results = trainer.evaluate()
    print(f"Dev Set - Accuracy: {dev_results['eval_accuracy']:.4f}, F1-Macro: {dev_results['eval_f1_macro']:.4f}")
    
    # Generate predictions
    predictions_output = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    
    # Save predictions
    output_df = pd.DataFrame({
        'Sentence': test_df['Sentence'],
        'Predicted': predictions,
        'Gold': test_df['label']
    })
    output_path = f'{output_dir}/strategy1a_standard_finetuned_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    
    return predictions, dev_results


# ============================================================================
# Strategy 1b: Class-Balanced Fine-tuning
# ============================================================================

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def train_strategy1b(train_tokenized, dev_tokenized, test_tokenized, test_df, train_df, output_dir):
    """Strategy 1b: Class-balanced fine-tuning"""
    print("\n" + "=" * 70)
    print("STRATEGY 1B: CLASS-BALANCED FINE-TUNING")
    print("=" * 70)
    
    # Compute class weights
    class_weights = compute_class_weights(train_df['label'].values)
    print("Class weights:")
    for i, weight in enumerate(class_weights):
        print(f"  {LABEL_MAP[i]}: {weight:.4f}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/models/strategy1b',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available()
    )
    
    # Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Evaluate on dev
    dev_results = trainer.evaluate()
    print(f"Dev Set - Accuracy: {dev_results['eval_accuracy']:.4f}, F1-Macro: {dev_results['eval_f1_macro']:.4f}")
    
    # Generate predictions
    predictions_output = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    
    # Save predictions
    output_df = pd.DataFrame({
        'Sentence': test_df['Sentence'],
        'Predicted': predictions,
        'Gold': test_df['label']
    })
    output_path = f'{output_dir}/strategy1b_class_balanced_finetuned_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    
    return predictions, dev_results, class_weights


# ============================================================================
# Strategy 1c: Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss focuses on hard examples and downweights easy ones"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class FocalLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = FocalLoss(
            alpha=torch.tensor(self.class_weights).to(logits.device),
            gamma=2.0
        )
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def train_strategy1c(train_tokenized, dev_tokenized, test_tokenized, test_df, class_weights, output_dir):
    """Strategy 1c: Focal loss"""
    print("\n" + "=" * 70)
    print("STRATEGY 1C: FOCAL LOSS")
    print("=" * 70)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/models/strategy1c',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available()
    )
    
    # Trainer
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=list(class_weights.numpy())
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Evaluate on dev
    dev_results = trainer.evaluate()
    print(f"Dev Set - Accuracy: {dev_results['eval_accuracy']:.4f}, F1-Macro: {dev_results['eval_f1_macro']:.4f}")
    
    # Generate predictions
    predictions_output = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    
    # Save predictions
    output_df = pd.DataFrame({
        'Sentence': test_df['Sentence'],
        'Predicted': predictions,
        'Gold': test_df['label']
    })
    output_path = f'{output_dir}/strategy1c_focal_loss_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    
    return predictions, dev_results


# ============================================================================
# Strategy 1d: Discriminative Fine-tuning
# ============================================================================

def get_parameter_groups_with_decay(model, learning_rate, weight_decay):
    """Apply layer-wise learning rate decay"""
    no_decay = ["bias", "LayerNorm.weight"]
    num_layers = model.config.num_hidden_layers
    
    optimizer_grouped_parameters = []
    
    # Classifier head (highest LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.classifier.named_parameters()],
        "lr": learning_rate,
        "weight_decay": weight_decay
    })
    
    # BERT layers (decreasing LR as we go down)
    for layer_num in range(num_layers - 1, -1, -1):
        decay_factor = 0.95 ** (num_layers - layer_num - 1)
        layer_lr = learning_rate * decay_factor
        
        layer = model.bert.encoder.layer[layer_num]
        
        # With weight decay
        optimizer_grouped_parameters.append({
            "params": [p for n, p in layer.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            "lr": layer_lr,
            "weight_decay": weight_decay
        })
        
        # Without weight decay
        optimizer_grouped_parameters.append({
            "params": [p for n, p in layer.named_parameters()
                      if any(nd in n for nd in no_decay)],
            "lr": layer_lr,
            "weight_decay": 0.0
        })
    
    # Embeddings (lowest LR)
    embedding_lr = learning_rate * (0.95 ** num_layers)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.bert.embeddings.named_parameters()],
        "lr": embedding_lr,
        "weight_decay": weight_decay
    })
    
    return optimizer_grouped_parameters


class DiscriminativeFineTuningTrainer(WeightedLossTrainer):
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = get_parameter_groups_with_decay(
                self.model,
                self.args.learning_rate,
                self.args.weight_decay
            )
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        return self.optimizer


def train_strategy1d(train_tokenized, dev_tokenized, test_tokenized, test_df, class_weights, output_dir):
    """Strategy 1d: Discriminative fine-tuning"""
    print("\n" + "=" * 70)
    print("STRATEGY 1D: DISCRIMINATIVE FINE-TUNING")
    print("=" * 70)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/models/strategy1d',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available()
    )
    
    # Trainer
    trainer = DiscriminativeFineTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Evaluate on dev
    dev_results = trainer.evaluate()
    print(f"Dev Set - Accuracy: {dev_results['eval_accuracy']:.4f}, F1-Macro: {dev_results['eval_f1_macro']:.4f}")
    
    # Generate predictions
    predictions_output = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    
    # Save predictions
    output_df = pd.DataFrame({
        'Sentence': test_df['Sentence'],
        'Predicted': predictions,
        'Gold': test_df['label']
    })
    output_path = f'{output_dir}/strategy1d_discriminative_finetuning_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    
    return predictions, dev_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Extension 1: Fine-Tuned FinBERT')
    parser.add_argument('--strategy', choices=['1a', '1b', '1c', '1d', 'all'], default='all',
                        help='Which strategy to train (default: all)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing train/, development/, test/ subdirectories')
    parser.add_argument('--output_dir', type=str, default='output/extension1(milestone3)',
                        help='Output directory for predictions')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Path to training CSV (overrides data_dir)')
    parser.add_argument('--dev_file', type=str, default=None,
                        help='Path to development CSV (overrides data_dir)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to test CSV (overrides data_dir)')
    
    args = parser.parse_args()
    
    # Set up file paths
    if args.train_file:
        train_file = args.train_file
        dev_file = args.dev_file or f'{args.data_dir}/development/development.csv'
        test_file = args.test_file or f'{args.data_dir}/test/test.csv'
    else:
        train_file = f'{args.data_dir}/train/train.csv'
        dev_file = f'{args.data_dir}/development/development.csv'
        test_file = f'{args.data_dir}/test/test.csv'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/models', exist_ok=True)
    os.makedirs(f'{args.output_dir}/logs', exist_ok=True)
    
    print("=" * 70)
    print("EXTENSION 1: FINE-TUNED FINBERT TRAINING")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load data
    train_df, dev_df, test_df = load_data(train_file, dev_file, test_file)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset_for_training(train_df)
    dev_dataset = prepare_dataset_for_training(dev_df)
    test_dataset = prepare_dataset_for_training(test_df)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    dev_tokenized = dev_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_tokenized = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    print("✓ Tokenization complete")
    
    # Train strategies
    class_weights = None
    
    if args.strategy in ['1a', 'all']:
        train_strategy1a(train_tokenized, dev_tokenized, test_tokenized, test_df, args.output_dir)
    
    if args.strategy in ['1b', 'all']:
        _, _, class_weights = train_strategy1b(
            train_tokenized, dev_tokenized, test_tokenized, test_df, train_df, args.output_dir
        )
    
    if args.strategy in ['1c', 'all']:
        if class_weights is None:
            class_weights = compute_class_weights(train_df['label'].values)
        train_strategy1c(train_tokenized, dev_tokenized, test_tokenized, test_df, class_weights, args.output_dir)
    
    if args.strategy in ['1d', 'all']:
        if class_weights is None:
            class_weights = compute_class_weights(train_df['label'].values)
        train_strategy1d(train_tokenized, dev_tokenized, test_tokenized, test_df, class_weights, args.output_dir)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nPredictions saved to: {args.output_dir}/")
    print("\nTo evaluate with scoring.py, run:")
    print(f"  python scoring.py {args.output_dir}/strategy1a_standard_finetuned_predictions.csv {test_file}")
    print(f"  python scoring.py {args.output_dir}/strategy1b_class_balanced_finetuned_predictions.csv {test_file}")
    print(f"  python scoring.py {args.output_dir}/strategy1c_focal_loss_predictions.csv {test_file}")
    print(f"  python scoring.py {args.output_dir}/strategy1d_discriminative_finetuning_predictions.csv {test_file}")


if __name__ == '__main__':
    main()

