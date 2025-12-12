import os
import torch
import sys
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

# --- SAFETY CHECK ---
if torch.__version__ != "2.9.1":
    print(f"⚠️  Running on PyTorch {torch.__version__}, but backend expects 2.9.1")

def train(args):
    login(token=args.hf_token)

    # 1. QLoRA Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # 3. Load Tokenizer & Fix Padding
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 4. Load Data
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
    )

    # 6. Training Args
    args_train = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",       
        save_strategy="epoch",        
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",
        report_to="none"
    )

    # This converts the "messages" list into a single string using Llama 3 template
    def formatting_prompts_func(example):
        output_texts = []
        for message in example['messages']:
            # Apply the chat template (this turns the list of dicts into a string)
            text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts
    # -----------------------------------------

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=args_train,
        max_seq_length=512,
        packing=False,
        formatting_func=formatting_prompts_func,
        # --------------------------------------------------------------
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving adapter...")
    trainer.save_model("./final_adapter")
    tokenizer.save_pretrained("./final_adapter")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/validation.jsonl")
    
    args = parser.parse_args()
    train(args)