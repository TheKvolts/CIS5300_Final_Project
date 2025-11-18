"""
Simple Baseline: Majority Class Predictor
Predicts the most frequent sentiment class from training data for all test examples

Usage:
    python simple-baseline.py

This baseline IGNORES the input text completely. 
It always predicts the majority class from training data, regardless of what the headline says.
This is the expected behavior and it establishes the minimum performance threshold.
"""

import pandas as pd
from collections import Counter
import os

# File paths
data_dir = "data"
train_file = f"{data_dir}/train/train.csv"
test_file = f"{data_dir}/test/test.csv"
prefix = "milestone2"

# Label mapping
LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def train_majority_baseline(train_file):
    """
    Find the majority class from training data.
    
    This function does NOT use the text content at all - it only counts
    how many examples of each class exist in the training data.
    
    Args:
        train_file: Path to training CSV
    
    Returns:
        majority_class: Integer label (0, 1, or 2) of the most common class
    """
    train_df = pd.read_csv(train_file)
    class_counts = Counter(train_df['Label'])
    majority_class = class_counts.most_common(1)[0][0]
    
    print("=" * 70)
    print("SIMPLE BASELINE: MAJORITY CLASS PREDICTOR")
    print("=" * 70)
    print("\nIMPORTANT: This baseline ignores headline content completely!")
    print("   It always predicts the same class for everything.\n")
    
    print("Training Set Class Distribution:")
    for label, count in sorted(class_counts.items()):
        percentage = (count / len(train_df)) * 100
        marker = " ← MAJORITY (will predict this for ALL test examples)" if label == majority_class else ""
        print(f"  {LABEL_MAP[label]:>8} ({label}): {count:>5} ({percentage:>5.2f}%){marker}")
    
    print(f"\nMajority Class: {majority_class} ({LABEL_MAP[majority_class]})")
    print(f"\nStrategy: Predict '{LABEL_MAP[majority_class]}' for EVERY test example")
    print(f"   (even for obviously {LABEL_MAP[0]} or {LABEL_MAP[2]} headlines!)")
    
    return majority_class

def predict_majority_baseline(test_file, majority_class, prefix):
    """
    Generate predictions using majority class baseline.
    
    This creates predictions that are ALL THE SAME - every single test
    example gets predicted as the majority class, regardless of content.
    
    Args:
        test_file: Path to test CSV
        majority_class: Integer label to predict for all examples
        prefix: Output directory prefix
    """
    test_df = pd.read_csv(test_file)
    
    # Predict majority class for EVERYTHING (this is intentional!)
    predictions = [majority_class] * len(test_df)
    
    # Count how many of each true label exist in test set
    test_class_counts = Counter(test_df['Label'])
    
    print(f"\nTest Set Class Distribution (True Labels):")
    for label, count in sorted(test_class_counts.items()):
        percentage = (count / len(test_df)) * 100
        print(f"  {LABEL_MAP[label]:>8} ({label}): {count:>5} ({percentage:>5.2f}%)")
    
    print(f"\nWhat This Baseline Gets WRONG:")
    for label, count in sorted(test_class_counts.items()):
        if label != majority_class:
            print(f"  • All {count} {LABEL_MAP[label]} examples will be predicted as {LABEL_MAP[majority_class]}")
    
    print(f"\nWhat This Baseline Gets RIGHT:")
    correct = test_class_counts[majority_class]
    print(f"  • The {correct} {LABEL_MAP[majority_class]} examples (by luck!)")
    
    expected_accuracy = correct / len(test_df)
    print(f"\nExpected Accuracy: {expected_accuracy:.4f} ({expected_accuracy*100:.2f}%)")
    print(f"   (= {correct} correct / {len(test_df)} total)")
    
    # Create output directory if it doesn't exist
    os.makedirs(prefix, exist_ok=True)
    
    # Create 'test_sentence_label.csv' (gold labels) - same format as strong-baseline
    test_sentence_label_df = test_df[['Sentence', 'Label']].copy()
    test_sentence_label_df.to_csv(f'{prefix}/test_sentence_label.csv', index=False)
    print(f"\ntest_sentence_label.csv created successfully.")
    
    # Create 'simple_baseline_test_predictions.csv' (predictions) - same format as strong-baseline
    simple_baseline_predictions_df = pd.DataFrame({
        'Sentence': test_df['Sentence'],
        'Label': predictions
    })
    simple_baseline_predictions_df.to_csv(f'{prefix}/simple_baseline_test_predictions.csv', index=False)
    print(f"simple_baseline_test_predictions.csv created with integer labels successfully.")
    print(f"Total predictions: {len(predictions)} (all class {majority_class})")

if __name__ == "__main__":
    # Train and predict
    print("\n")
    majority_class = train_majority_baseline(train_file)
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    predict_majority_baseline(test_file, majority_class, prefix)
    
    print("\nTo evaluate, run:")
    print(f"python scoring.py {prefix}/simple_baseline_test_predictions.csv {prefix}/test_sentence_label.csv")
