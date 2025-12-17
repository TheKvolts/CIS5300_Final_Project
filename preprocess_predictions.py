"""
Preprocessing script to convert various prediction formats to the standard format
expected by code/scoring.py

This script handles:
- finBERT output format (string labels: "positive", "negative", "neutral")
- Numeric format with different column names
- Label mapping conversions

Standard output format:
- Column: "Predicted_Label" with integer values (0=negative, 1=neutral, 2=positive)
"""

import argparse
import pandas as pd
import sys


# Standard label mapping for the evaluation script
# This matches your dataset: 0=negative, 1=neutral, 2=positive
STANDARD_LABEL_MAP = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}


def detect_format(df):
    """
    Detect the format of the input predictions.

    Returns:
        str: Format type ('finbert', 'standard_numeric', 'standard_string', 'unknown')
    """
    columns = df.columns.tolist()

    # Check for finBERT format
    if 'prediction' in columns:
        # Check if predictions are strings
        sample_value = df['prediction'].iloc[0]
        if isinstance(sample_value, str):
            return 'finbert'
        else:
            return 'finbert_numeric'

    # Check for standard format with Predicted_Label
    if 'Predicted_Label' in columns:
        sample_value = df['Predicted_Label'].iloc[0]
        if isinstance(sample_value, str):
            return 'standard_string'
        else:
            return 'standard_numeric'

    # Check for Label column
    if 'Label' in columns:
        sample_value = df['Label'].iloc[0]
        if isinstance(sample_value, str):
            return 'label_string'
        else:
            return 'label_numeric'

    return 'unknown'


def convert_finbert_format(df):
    """
    Convert finBERT output format to standard format.

    finBERT uses:
    - Column: 'prediction'
    - Values: string labels ("positive", "negative", "neutral")

    Converts to:
    - Column: 'Predicted_Label'
    - Values: integers (0=negative, 1=neutral, 2=positive)
    """
    result_df = df.copy()

    # Map string labels to numeric
    result_df['Predicted_Label'] = result_df['prediction'].map(STANDARD_LABEL_MAP)

    # Check for any unmapped values
    if result_df['Predicted_Label'].isna().any():
        unmapped_values = df.loc[result_df['Predicted_Label'].isna(), 'prediction'].unique()
        raise ValueError(f"Found unmapped prediction values: {unmapped_values}")

    # Convert to integer
    result_df['Predicted_Label'] = result_df['Predicted_Label'].astype(int)

    # Keep the sentence column if it exists
    if 'sentence' in result_df.columns:
        return result_df[['sentence', 'Predicted_Label']]
    elif 'Sentence' in result_df.columns:
        return result_df[['Sentence', 'Predicted_Label']]
    else:
        return result_df[['Predicted_Label']]


def convert_string_labels(df, column_name):
    """
    Convert string labels to numeric format.
    """
    result_df = df.copy()

    # Map string labels to numeric
    result_df['Predicted_Label'] = result_df[column_name].map(STANDARD_LABEL_MAP)

    # Check for any unmapped values
    if result_df['Predicted_Label'].isna().any():
        unmapped_values = df.loc[result_df['Predicted_Label'].isna(), column_name].unique()
        raise ValueError(f"Found unmapped prediction values: {unmapped_values}")

    # Convert to integer
    result_df['Predicted_Label'] = result_df['Predicted_Label'].astype(int)

    return result_df


def standardize_predictions(input_file, output_file=None, format_type=None):
    """
    Convert predictions to standard format.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional, defaults to input_file with _processed suffix)
        format_type: Format type to force (optional, will auto-detect if None)

    Returns:
        pd.DataFrame: Standardized predictions dataframe
    """
    # Read input file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")

    if df.empty:
        raise ValueError("Input file is empty")

    # Detect or use specified format
    if format_type is None:
        format_type = detect_format(df)

    print(f"Detected format: {format_type}")

    # Convert based on format
    if format_type == 'finbert':
        print("Converting finBERT string format to standard numeric format...")
        result_df = convert_finbert_format(df)

    elif format_type == 'finbert_numeric':
        print("Renaming column from 'prediction' to 'Predicted_Label'...")
        result_df = df.copy()
        result_df['Predicted_Label'] = result_df['prediction'].astype(int)
        if 'sentence' in result_df.columns:
            result_df = result_df[['sentence', 'Predicted_Label']]
        else:
            result_df = result_df[['Predicted_Label']]

    elif format_type == 'standard_string':
        print("Converting string labels to numeric...")
        result_df = convert_string_labels(df, 'Predicted_Label')

    elif format_type == 'label_string':
        print("Converting string labels to numeric and renaming column...")
        result_df = convert_string_labels(df, 'Label')

    elif format_type == 'standard_numeric':
        print("Format is already standard, no conversion needed.")
        result_df = df.copy()

    elif format_type == 'label_numeric':
        print("Renaming 'Label' to 'Predicted_Label'...")
        result_df = df.copy()
        result_df['Predicted_Label'] = result_df['Label'].astype(int)

    else:
        raise ValueError(f"Unknown format: {format_type}. Cannot process this file.")

    # Validate output
    if 'Predicted_Label' not in result_df.columns:
        raise ValueError("Conversion failed: 'Predicted_Label' column not created")

    # Check that all labels are valid (0, 1, or 2)
    valid_labels = {0, 1, 2}
    unique_labels = set(result_df['Predicted_Label'].unique())
    invalid_labels = unique_labels - valid_labels

    if invalid_labels:
        raise ValueError(f"Invalid label values found: {invalid_labels}. Expected only 0, 1, or 2.")

    print(f"\nConversion successful!")
    print(f"Total predictions: {len(result_df)}")
    print(f"Label distribution:")
    print(f"  Negative (0): {(result_df['Predicted_Label'] == 0).sum()}")
    print(f"  Neutral (1): {(result_df['Predicted_Label'] == 1).sum()}")
    print(f"  Positive (2): {(result_df['Predicted_Label'] == 2).sum()}")

    # Save output file
    if output_file is None:
        # Generate default output filename
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_processed.csv')
        else:
            output_file = input_file + '_processed.csv'

    result_df.to_csv(output_file, index=False)
    print(f"\nProcessed predictions saved to: {output_file}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess prediction files to standard format for evaluation'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input predictions CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output file (default: input_file with _processed suffix)'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['finbert', 'finbert_numeric', 'standard_string', 'standard_numeric',
                 'label_string', 'label_numeric'],
        default=None,
        help='Force specific input format (default: auto-detect)'
    )
    parser.add_argument(
        '--show-mapping',
        action='store_true',
        help='Display the label mapping used for conversion'
    )

    args = parser.parse_args()

    if args.show_mapping:
        print("Standard label mapping:")
        print("  'negative' -> 0")
        print("  'neutral'  -> 1")
        print("  'positive' -> 2")
        print()

    try:
        standardize_predictions(args.input_file, args.output, args.format)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    exit(main())
