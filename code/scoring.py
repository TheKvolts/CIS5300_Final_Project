"""
Evaluation script for Financial Sentiment Analysis
Computes accuracy, F1-score, precision, recall, and confusion matrix
for 3-class sentiment classification (negative, neutral, positive).
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import json


def load_predictions(pred_file):
    """
    Load predictions from a file.
    Expected format: CSV with columns 'Sentence' and 'Predicted_Label'
    or a simple text file with one prediction per line.
    """
    if pred_file.endswith('.csv'):
        df = pd.read_csv(pred_file)
        if 'Predicted_Label' in df.columns:
            return df['Predicted_Label'].values
        elif 'Predicted' in df.columns:
            return df['Predicted'].values
        elif 'Label' in df.columns:
            return df['Label'].values
        else:
            raise ValueError("CSV must contain 'Predicted_Label' or 'Label' column")
    else:
        # Assume one prediction per line
        with open(pred_file, 'r') as f:
            predictions = [int(line.strip()) for line in f if line.strip()]
        return np.array(predictions)


def load_gold_labels(gold_file):
    """
    Load gold standard labels from a file.
    Expected format: CSV with 'Label' column or text file with one label per line.
    """
    if gold_file.endswith('.csv'):
        df = pd.read_csv(gold_file)
        if 'Label' in df.columns:
            return df['Label'].values
        elif 'label' in df.columns:
            return df['label'].values
        else:
            raise ValueError("Gold standard CSV must contain 'Label' column")
    else:
        # Assume one label per line
        with open(gold_file, 'r') as f:
            labels = [int(line.strip()) for line in f if line.strip()]
        return np.array(labels)


def evaluate(predictions, gold_labels, output_format='text'):
    """
    Evaluate predictions against gold standard labels.

    Args:
        predictions: numpy array of predicted labels (0, 1, 2)
        gold_labels: numpy array of gold standard labels (0, 1, 2)
        output_format: 'text' for human-readable or 'json' for structured output

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Validate inputs
    if len(predictions) != len(gold_labels):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold_labels)} gold labels")

    # Calculate metrics
    accuracy = accuracy_score(gold_labels, predictions)

    # Precision, Recall, F1 (macro, micro, weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_labels, predictions, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        gold_labels, predictions, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gold_labels, predictions, average='weighted', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        gold_labels, predictions, average=None, zero_division=0, labels=[0, 1, 2]
    )

    # Confusion matrix
    cm = confusion_matrix(gold_labels, predictions, labels=[0, 1, 2])

    # Store results
    results = {
        'accuracy': float(accuracy),
        'macro': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro)
        },
        'micro': {
            'precision': float(precision_micro),
            'recall': float(recall_micro),
            'f1': float(f1_micro)
        },
        'weighted': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1': float(f1_weighted)
        },
        'per_class': {
            'negative': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1': float(f1_per_class[0]),
                'support': int(support_per_class[0])
            },
            'neutral': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1': float(f1_per_class[1]),
                'support': int(support_per_class[1])
            },
            'positive': {
                'precision': float(precision_per_class[2]),
                'recall': float(recall_per_class[2]),
                'f1': float(f1_per_class[2]),
                'support': int(support_per_class[2])
            }
        },
        'confusion_matrix': cm.tolist(),
        'num_samples': len(predictions)
    }

    return results


def print_results(results):
    """Print evaluation results in human-readable format."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal samples: {results['num_samples']}")
    print(f"\nAccuracy: {results['accuracy']:.4f}")

    print("\n" + "-" * 60)
    print("AGGREGATE METRICS")
    print("-" * 60)
    print(f"Macro-averaged Precision:  {results['macro']['precision']:.4f}")
    print(f"Macro-averaged Recall:     {results['macro']['recall']:.4f}")
    print(f"Macro-averaged F1:         {results['macro']['f1']:.4f}")
    print()
    print(f"Weighted-averaged Precision: {results['weighted']['precision']:.4f}")
    print(f"Weighted-averaged Recall:    {results['weighted']['recall']:.4f}")
    print(f"Weighted-averaged F1:        {results['weighted']['f1']:.4f}")

    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 60)
    for class_name in ['negative', 'neutral', 'positive']:
        metrics = results['per_class'][class_name]
        print(f"{class_name.capitalize():<12} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['support']:<12}")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print("         Predicted:")
    print("           Neg    Neu    Pos")
    print("Actual:")
    cm = results['confusion_matrix']
    labels = ['Neg', 'Neu', 'Pos']
    for i, label in enumerate(labels):
        print(f"  {label}    {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate sentiment analysis predictions against gold standard labels'
    )
    parser.add_argument(
        'predictions',
        type=str,
        help='Path to predictions file (CSV with "Predicted_Label" column or text file with one prediction per line)'
    )
    parser.add_argument(
        'gold_labels',
        type=str,
        help='Path to gold standard labels file (CSV with "Label" column or text file with one label per line)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format: text (human-readable) or json (structured)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (if not specified, prints to stdout)'
    )

    args = parser.parse_args()

    # Load data
    try:
        predictions = load_predictions(args.predictions)
        gold_labels = load_gold_labels(args.gold_labels)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Evaluate
    try:
        results = evaluate(predictions, gold_labels, output_format=args.format)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

    # Output results
    if args.format == 'json':
        output_str = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_str)
            print(f"Results written to {args.output}")
        else:
            print(output_str)
    else:
        if args.output:
            import sys
            original_stdout = sys.stdout
            with open(args.output, 'w') as f:
                sys.stdout = f
                print_results(results)
            sys.stdout = original_stdout
            print(f"Results written to {args.output}")
        else:
            print_results(results)

    return 0


if __name__ == '__main__':
    exit(main())