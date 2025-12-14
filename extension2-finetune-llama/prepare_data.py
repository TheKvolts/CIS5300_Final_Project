import pandas as pd
import json
import os

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# 2. Define the System Prompt
# We keep this strict and consistent for fine-tuning.
SYSTEM_PROMPT = (
    "You are a financial sentiment analysis expert. "
    "Analyze the provided financial news headline and classify the sentiment "
    "as 'positive', 'neutral', or 'negative'. "
    "Output ONLY the sentiment label."
)

def convert_to_jsonl(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Skipping {input_path} (file not found)")
        return

    df = pd.read_csv(input_path)
    
    with open(output_path, 'w') as f:
        for index, row in df.iterrows():
            label_val = row.get('label', row.get('Label')) 
            sentence = row.get('Sentence', row.get('sentence'))

            # Safety check: Ensure label exists and is in our map
            if label_val not in label_map:
                print(f"Skipping row {index}: Label '{label_val}' not found in map.")
                continue

            # Construct the fine-tuning entry
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Headline: {sentence}"},
                    {"role": "assistant", "content": label_map[label_val]}
                ]
            }
            
            # Write to JSONL
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Converted {input_path} -> {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Convert all three splits
    convert_to_jsonl("/home/ubuntu/CIS5300_Final_Project/data/train/train.csv", "data/train.jsonl")
    convert_to_jsonl("/home/ubuntu/CIS5300_Final_Project/data/development/development.csv", "data/validation.jsonl")
    convert_to_jsonl("/home/ubuntu/CIS5300_Final_Project/data/test/test.csv", "data/test.jsonl")