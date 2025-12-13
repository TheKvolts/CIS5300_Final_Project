import torch
import argparse
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
# --- NEW IMPORT ---
from sklearn.metrics import accuracy_score, classification_report
# ------------------

# Define label mapping (Must match your training logic)
LABEL_MAP = {
    "NBC News": 0,
    "Fox News": 1
}

def load_model(base_model_id, adapter_path):
    print(f"Loading base model: {base_model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def normalize_label(label):
    """
    Normalize the label to handle case variations and extra whitespace.
    """
    if not isinstance(label, str):
        return str(label)
    
    label = label.strip().lower()
    
    # Map to canonical class names if needed
    if "fox" in label:
        return "Fox News"
    if "nbc" in label:
        return "NBC News"
        
    return label.title() # Default fallback

def predict(model, tokenizer, headline):
    # Concise system prompt aligned with data preparation style
    system_prompt = "You are a media bias analyst. Analyze the provided news headline and classify the source as 'Fox News' or 'NBC News'. Output ONLY the source label."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Headline: {headline}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def evaluate_file(model, tokenizer, test_file, output_file):
    print(f"Reading test data from {test_file}...")
    
    results = []
    y_true = []
    y_pred = []
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
        
    print(f"Running inference on {len(lines)} examples...")
    
    for line in tqdm(lines):
        data = json.loads(line)
        
        # 1. Parse Data
        user_content = data['messages'][1]['content'] 
        raw_true_label = data['messages'][2]['content']
        headline_text = user_content.replace("Headline: ", "").strip()
        
        # 2. Predict
        raw_prediction = predict(model, tokenizer, headline_text)
        
        # 3. Clean Prediction (Normalize)
        predicted_label = normalize_label(raw_prediction)
        # Handle the case where the file has 'negative'/'positive' sentiment labels
        # We try to normalize them, but if they are sentiment, they won't match "Fox News"
        true_label = normalize_label(raw_true_label)
        
        # 4. Collect for Metrics
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
        # Check strict correctness for the CSV
        is_correct = (predicted_label == true_label)
        
        results.append({
            "headline": headline_text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "raw_prediction": raw_prediction,
            "correct": is_correct
        })

    # --- NEW: REPORTING BLOCK ---
    print(f"\n📊 Evaluation Results")
    print("-" * 60)
    
    # Calculate simple accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Global Accuracy: {acc:.2%}")
    print("-" * 60)
    
    # Calculate Precision, Recall, F1, Support
    # This handles the breakdown by class (NBC vs Fox)
    # Using 'macro' avg for unweighted mean, or rely on default report
    try:
        report = classification_report(y_true, y_pred, digits=4)
        print("Detailed Classification Report:")
        print(report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")
        
    print("-" * 60)
    # ----------------------------

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Detailed CSV saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="./final_adapter")
    parser.add_argument("--test_file", type=str, help="Path to test.jsonl file")
    parser.add_argument("--headline", type=str, help="Single headline to test")
    parser.add_argument("--output_file", type=str, default="test_results.csv")
    
    args = parser.parse_args()

    BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    if not args.test_file and not args.headline:
        print("Error: Please provide either --test_file or --headline")
        exit(1)

    model, tokenizer = load_model(BASE_MODEL_ID, args.adapter_path)
    
    if args.headline:
        result = predict(model, tokenizer, args.headline)
        print(f"\nHeadline: {args.headline}")
        print(f"Prediction: {result}")
        
    elif args.test_file:
        evaluate_file(model, tokenizer, args.test_file, args.output_file)