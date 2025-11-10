"""
DEPRECATED: This file has been refactored into modular components.

This is the original monolithic implementation kept for reference.
Please use the new modular structure instead:

- qwen_model.py: Model initialization
- qwen_data.py: Data loading and preprocessing  
- qwen_trainer.py: Training loop
- qwen_evaluator.py: Evaluation
- qwen_utils.py: Utilities
- train_qwen_coconut.py: Main training script
- inference_qwen_coconut.py: Inference script

See QWEN_INTEGRATION.md for complete documentation.
"""

# Installation commands (for reference):
# !git clone https://github.com/facebookresearch/coconut.git
# # %cd coconut
# !pip install -r requirements.txt
# !pip install -r requirements_qwen.txt
# !pip install transformers accelerate peft bitsandbytes datasets torchvision tqdm --upgrade --extra-index-url https://download.pytorch.org/whl/cu121

# Core PyTorch imports
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm

# Coconut framework
from coconut import Coconut

# =============================================
#  DEVICE SETUP
# =============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =============================================
#  TOKENIZER INITIALIZATION
# =============================================
# Model: Qwen 2.5-3B-Instruct (3 billion parameter instruction-tuned model)
model_name = "Qwen/Qwen2.5-3B-Instruct"

# Load tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add special latent token for Coconut's continuous thoughts
# This token will be replaced with continuous representations during forward passes
latent_token = "<|latent|>"
tokenizer.add_special_tokens({"additional_special_tokens": [latent_token]})
latent_token_id = tokenizer.convert_tokens_to_ids(latent_token)

# =============================================
#  BASE MODEL LOADING
# =============================================
# Load Qwen model in FP16 (half precision) for memory efficiency
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",         # Automatically distributes model across available GPUs
    torch_dtype=torch.float16  # Use FP16 for ~50% memory savings and faster training
)

# Resize embeddings to account for the new latent token
base_model.resize_token_embeddings(len(tokenizer))

# =============================================
#  LORA CONFIGURATION
# =============================================
# LoRA (Low-Rank Adaptation): Parameter-efficient fine-tuning
# Only trains ~0.1-1% of parameters, much faster than full fine-tuning
lora_config = LoraConfig(
    r=8,                              # Rank: controls number of trainable parameters (higher = more capacity)
    lora_alpha=32,                    # Scaling factor for LoRA weights
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention query and value projections
    lora_dropout=0.05,                # Dropout for regularization
    bias="none",                      # Don't train bias terms
    task_type="CAUSAL_LM"             # Causal language modeling task
)

# Apply LoRA adapters to the base model
base_model = get_peft_model(base_model, lora_config)

# =============================================
#  COCONUT WRAPPER
# =============================================
# Wrap the LoRA model with Coconut for continuous latent reasoning
# Coconut enables the model to "think" in continuous space before generating tokens
coconut_model = Coconut(
    base_causallm=base_model,
    latent_token_id=latent_token_id,      # Token that triggers continuous thought
    start_latent_id=latent_token_id,      # Start marker for latent reasoning
    end_latent_id=latent_token_id,        # End marker for latent reasoning
    eos_token_id=tokenizer.eos_token_id   # End-of-sequence token
).to(device)

# Set model to training mode
coconut_model.train()

# =============================================
#  DATASET & PREPROCESSING
# =============================================
# Load GSM8K: Grade School Math 8K - mathematical reasoning dataset
# Contains 8.5K grade school math word problems
dataset = load_dataset("gsm8k", "main", split="train")

def preprocess(example):
    """
    Preprocess a single example from GSM8K dataset.
    
    Args:
        example: Dict with 'question' and 'answer' keys
        
    Returns:
        Dict with tokenized inputs, attention mask, and labels
    """
    # Format: "Question: [question]\nAnswer: [answer]"
    prompt = f"Question: {example['question']}\nAnswer:"
    label = example['answer']
    
    # Tokenize the full sequence (prompt + answer)
    encoding = tokenizer(
        prompt + " " + label,
        truncation=True,          # Truncate if longer than max_length
        padding="max_length",     # Pad to max_length for batching
        max_length=128,           # Maximum sequence length
        return_tensors="pt"       # Return PyTorch tensors
    )
    
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),  # For causal LM, labels = input_ids
        "question": example['question'],
        "answer": example['answer']
    }

# Apply preprocessing to all examples
dataset = dataset.map(preprocess)

# Create DataLoader for batching
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# =============================================
#  TRAINING SETUP
# =============================================
# Optimizer: AdamW (Adam with weight decay for better regularization)
optimizer = AdamW(coconut_model.parameters(), lr=1e-4)

# Training hyperparameters
num_epochs = 1
num_training_steps = num_epochs * len(dataloader)

# Learning rate scheduler: Linear decay from initial LR to 0
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,              # No warmup steps
    num_training_steps=num_training_steps
)

# Gradient scaler for automatic mixed precision (AMP)
# Prevents underflow when using FP16
scaler = GradScaler()

# =============================================
#  TRAINING LOOP
# =============================================
for epoch in range(num_epochs):
    print(f"\n==== Epoch {epoch+1} ====")
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Prepare batch: move tensors to GPU
        tensor_keys = ['input_ids', 'attention_mask', 'labels']
        batch_tensors = {k: torch.stack(batch[k]).to(device) for k in tensor_keys}

        input_ids = batch_tensors["input_ids"]
        attention_mask = batch_tensors["attention_mask"]
        labels = batch_tensors["labels"]

        # Create position IDs for positional encoding
        # Shape: (batch_size, sequence_length)
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # Zero out gradients from previous step
        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        # autocast() automatically uses FP16 for compatible operations
        with autocast():
            outputs = coconut_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids
            )
            loss = outputs.loss

        # Backward pass with gradient scaling
        # Scaling prevents gradient underflow in FP16
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        scheduler.step()

        # Track loss
        epoch_loss += loss.item()

        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Calculate and print average loss for epoch
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

# =============================================
#  EVALUATION (Quick accuracy check)
# =============================================
# Switch to evaluation mode (disables dropout, etc.)
coconut_model.eval()

correct = 0
total = 0

# Evaluate on first 50 examples from training set
# Note: In production, use a separate validation/test set
for example in tqdm(dataset.select(range(50)), desc="Evaluating"):
    # Prepare input
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
    
    # Generate answer (no gradient computation needed)
    with torch.no_grad():
        generated_ids = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=30  # Generate up to 30 new tokens
        )
    
    # Decode generated tokens to text
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Simple substring matching to check correctness
    # Note: This is a basic evaluation metric
    if example["answer"].strip() in gen_text.strip():
        correct += 1
    total += 1

# Calculate and print accuracy
accuracy = correct / total
print(f"\nEvaluation Accuracy: {accuracy:.4f}")

# =============================================
#  END OF ORIGINAL IMPLEMENTATION
# =============================================
# This file has been refactored into modular components.
# See train_qwen_coconut.py for the new implementation.

