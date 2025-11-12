"""
Test script to see actual model generation outputs
"""
import torch
import gc
from qwen_model import initialize_qwen_coconut_model
from transformers import AutoTokenizer
from datasets import load_dataset

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

device = 'cuda'
model_name = "Qwen/Qwen2.5-3B-Instruct"
checkpoint_path = './checkpoints/qwen_coconut_10epochs_full_20251110_164134/final_checkpoint.pt'

print("=" * 80)
print("MODEL GENERATION TEST")
print("=" * 80)

# Initialize model
print("\n1. Loading model...")
lora_config = {
    'r': 8, 
    'alpha': 32, 
    'target_modules': ['q_proj', 'v_proj'], 
    'dropout': 0.05, 
    'bias': 'none', 
    'task_type': 'CAUSAL_LM'
}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
latent_token = "<|latent|>"
tokenizer.add_special_tokens({"additional_special_tokens": [latent_token]})

# Load checkpoint
print(f"2. Loading checkpoint from {checkpoint_path}...")
import os
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    print("Please update the checkpoint_path variable with the correct path.")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize model
model, tokenizer, _ = initialize_qwen_coconut_model(
    model_name, 
    latent_token, 
    lora_config, 
    torch.float16, 
    device
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

# Load test examples
print("\n3. Loading test examples...")
test_dataset = load_dataset("gsm8k", "main", split="test")

# Test generation with different max_new_tokens
print("\n" + "=" * 80)
print("TESTING GENERATION WITH DIFFERENT MAX_NEW_TOKENS")
print("=" * 80)

for idx in range(3):
    example = test_dataset[idx]
    prompt = f"Question: {example['question']}\nAnswer:"
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {idx + 1}")
    print(f"{'='*80}")
    print(f"Question: {example['question']}")
    print(f"\nGround Truth Answer:\n{example['answer']}")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Test with different max_new_tokens
    for max_tokens in [30, 50, 100, 200]:
        print(f"\n{'-'*80}")
        print(f"Generation with max_new_tokens={max_tokens}:")
        print(f"{'-'*80}")
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract just the answer part (after "Answer:")
        if "Answer:" in generated_text:
            answer_part = generated_text.split("Answer:", 1)[1].strip()
        else:
            answer_part = generated_text
        
        print(f"Generated Answer:\n{answer_part}")
        print(f"Length: {len(tokenizer.encode(answer_part))} tokens")
        
        # Check if answer contains the final number
        import re
        gt_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', example['answer'])
        if gt_match:
            gt_number = gt_match.group(1).replace(',', '')
            if gt_number in generated_text.replace(',', ''):
                print(f"✓ Contains correct answer: {gt_number}")
            else:
                print(f"✗ Does NOT contain correct answer: {gt_number}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("""
Key Observations:
1. Check if the model generates coherent reasoning
2. Check if answers are being cut off at 30 tokens
3. Check if longer generation (200 tokens) produces better results
4. If model outputs are nonsensical, training may have failed

Next Steps:
- If generation looks good but is cut off: Increase max_new_tokens
- If generation is nonsensical: Retrain with better hyperparameters
- If generation is reasonable: Increase max_length during training
""")


