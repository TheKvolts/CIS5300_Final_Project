# High-Level Approach: Fine-Tuned Llama 3.1 for News Source Classification

## 1. Problem Statement
The goal is to classify financial news headlines by their source (**Fox News** vs. **NBC News**) based on linguistic patterns, tone, and framing. We utilize a Large Language Model (LLM) fine-tuned on a labeled dataset of headlines.

## 2. Model Selection
We selected **Meta-Llama-3.1-8B-Instruct** as our base model.
- **Reasoning**: It offers strong reasoning capabilities and instruction-following behavior while being small enough to fine-tune on a single GPU using quantization.

## 3. Data Preparation
Data is converted from CSV to JSONL format suitable for the Hugging Face `SFTTrainer`.
- **Script**: `prepare_data.py`
- **Format**: Chat-style conversation.
    - **System**: "You are a media bias analyst..."
    - **User**: "Headline: {text}"
    - **Assistant**: "{Label}" (e.g., "Fox News")

## 4. Training Technique: QLoRA
Full fine-tuning of an 8B parameter model is computationally expensive. We employ **QLoRA (Quantized Low-Rank Adaptation)**:
- **4-bit Quantization**: The base model is loaded in 4-bit precision (NF4) to minimize memory usage.
- **LoRA Adapters**: We freeze the base model and train only a small set of low-rank adapter matrices attached to specific layers (e.g., `q_proj`, `v_proj`).
- **Efficiency**: This drastically reduces the number of trainable parameters while maintaining performance comparable to full fine-tuning.

## 5. Training Configuration
- **Library**: `trl` (SFTTrainer), `peft` (LoRA), `bitsandbytes` (Quantization).
- **Hyperparameters**:
    - Epochs: 3
    - Learning Rate: 2e-4
    - Batch Size: 4 (with gradient accumulation)
    - Optimizer: Paged AdamW (handles memory spikes)

## 6. Inference & Evaluation
At inference time, we load the base model (quantized) and merge the trained LoRA adapters.
- **Prompting**: We use a strict system prompt to force the model to output *only* the classification label.
- **Normalization**: Model outputs are normalized (case-insensitivity, whitespace trimming) to ensure robust evaluation against ground truth labels.
- **Metrics**: Accuracy, Precision, Recall, and F1-Score.
