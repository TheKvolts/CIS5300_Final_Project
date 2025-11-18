from transformers import pipeline
import pandas as pd

pipe = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)

data_dir = "data"
test_data = f"{data_dir}/test/test.csv"
prefix = "milestone2"

test_df = pd.read_csv(test_data)

# Run inference
predictions_raw = pipe(test_df['Sentence'].tolist())

# Process predictions
predicted_labels = []
for result_list_for_sentence in predictions_raw:
    best_item = max(result_list_for_sentence, key=lambda x: x['score'])
    predicted_labels.append(best_item['label'])

test_df['predicted_sentiment'] = predicted_labels

print("Predictions completed and stored in 'predicted_sentiment' column.")
print(test_df[['Sentence', 'Sentiment', 'predicted_sentiment']].head())

# Storing outputs for scoring.py
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

# Create 'test_sentence_label.csv'
test_sentence_label_df = test_df[['Sentence', 'label']].copy()
test_sentence_label_df.rename(columns={'label': 'Label'}, inplace=True)
test_sentence_label_df.to_csv(f'{prefix}/test_sentence_label.csv', index=False)
print("test_sentence_label.csv created successfully.")

# Create 'strong_baseline_predictions.csv'
strong_baseline_predictions_df = test_df[['Sentence', 'predicted_sentiment']].copy()
strong_baseline_predictions_df['Label'] = strong_baseline_predictions_df['predicted_sentiment'].map(label_mapping)
strong_baseline_predictions_df = strong_baseline_predictions_df[['Sentence', 'Label']]
strong_baseline_predictions_df.to_csv(f'{prefix}/strong_baseline_test_predictions.csv', index=False)
print("strong_baseline_predictions.csv created with integer labels successfully.")


# Metrics
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# true_labels = test_df['Sentiment']
# predicted_labels_for_metrics = test_df['predicted_sentiment']

# accuracy = accuracy_score(true_labels, predicted_labels_for_metrics)
# precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels_for_metrics, average='weighted') # Use 'weighted' for imbalanced classes

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")

# from sklearn.metrics import classification_report
# print(classification_report(true_labels, predicted_labels_for_metrics))