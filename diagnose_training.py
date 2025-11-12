"""
Diagnostic script to analyze training issues and test model generation
"""
import torch
import gc
from qwen_model import initialize_qwen_coconut_model
from qwen_data import load_and_prepare_data
from transformers import AutoTokenizer

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

device = 'cuda'
model_name = "Qwen/Qwen2.5-3B-Instruct"

print("=" * 80)
print("DIAGNOSTIC SCRIPT FOR QWEN-COCONUT TRAINING")
print("=" * 80)

# Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
latent_token = "<|latent|>"
tokenizer.add_special_tokens({"additional_special_tokens": [latent_token]})

# Load a few examples from GSM8K
print("\n2. Loading GSM8K test examples...")
from datasets import load_dataset
test_dataset = load_dataset("gsm8k", "main", split="test")

print(f"\n3. Analyzing sequence lengths in GSM8K...")
print("-" * 80)

lengths_128 = []
lengths_256 = []
lengths_512 = []

for i in range(min(100, len(test_dataset))):
    example = test_dataset[i]
    full_text = f"Question: {example['question']}\nAnswer: {example['answer']}"
    tokens = tokenizer.encode(full_text)
    
    lengths_128.append(len(tokens) <= 128)
    lengths_256.append(len(tokens) <= 256)
    lengths_512.append(len(tokens) <= 512)
    
    if i < 3:
        print(f"\nExample {i+1}:")
        print(f"  Question: {example['question'][:80]}...")
        print(f"  Answer length: {len(example['answer'])} chars")
        print(f"  Token length: {len(tokens)} tokens")
        print(f"  Fits in 128 tokens: {len(tokens) <= 128}")

print(f"\n" + "=" * 80)
print("SEQUENCE LENGTH ANALYSIS (first 100 examples):")
print("=" * 80)
print(f"Fit in 128 tokens:  {sum(lengths_128)}/100 ({sum(lengths_128)}%)")
print(f"Fit in 256 tokens:  {sum(lengths_256)}/100 ({sum(lengths_256)}%)")
print(f"Fit in 512 tokens:  {sum(lengths_512)}/100 ({sum(lengths_512)}%)")

# Analyze answer lengths
print(f"\n" + "=" * 80)
print("ANSWER LENGTH ANALYSIS:")
print("=" * 80)

answer_lengths = []
for i in range(min(100, len(test_dataset))):
    example = test_dataset[i]
    answer_tokens = tokenizer.encode(example['answer'])
    answer_lengths.append(len(answer_tokens))
    
    if i < 3:
        print(f"\nExample {i+1} answer:")
        print(f"  Tokens: {len(answer_tokens)}")
        print(f"  Text: {example['answer'][:100]}...")

import statistics
print(f"\nAnswer token statistics:")
print(f"  Mean: {statistics.mean(answer_lengths):.1f}")
print(f"  Median: {statistics.median(answer_lengths):.1f}")
print(f"  Min: {min(answer_lengths)}")
print(f"  Max: {max(answer_lengths)}")
print(f"  Can generate in 30 tokens: {sum([l <= 30 for l in answer_lengths])}/100")
print(f"  Can generate in 50 tokens: {sum([l <= 50 for l in answer_lengths])}/100")
print(f"  Can generate in 100 tokens: {sum([l <= 100 for l in answer_lengths])}/100")
print(f"  Can generate in 200 tokens: {sum([l <= 200 for l in answer_lengths])}/100")

# Test if checkpoint exists
print(f"\n" + "=" * 80)
print("CHECKPOINT ANALYSIS:")
print("=" * 80)

import os
checkpoint_dir = "./checkpoints/qwen_coconut_10epochs_full_20251110_164134"
if os.path.exists(checkpoint_dir):
    print(f"✓ Checkpoint directory exists: {checkpoint_dir}")
    checkpoint_file = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    if os.path.exists(checkpoint_file):
        print(f"✓ Checkpoint file exists: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    else:
        print(f"✗ Checkpoint file not found: {checkpoint_file}")
        print(f"  Available files:")
        for f in os.listdir(checkpoint_dir):
            print(f"    - {f}")
else:
    print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
    print(f"  Looking for checkpoint directories...")
    for root, dirs, files in os.walk("."):
        if "checkpoint" in root.lower():
            print(f"    Found: {root}")

print(f"\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
Based on the analysis above, here are the recommended changes:

1. INCREASE MAX_LENGTH:
   - Current: 128 tokens
   - Recommended: 512 tokens (to fit most examples)
   - This allows the model to see full reasoning chains

2. INCREASE MAX_NEW_TOKENS:
   - Current: 30 tokens
   - Recommended: 200 tokens (to generate complete answers)
   - Most answers need 50-150 tokens

3. TRAINING CONFIGURATION:
   - Reduce epochs from 10 to 3-5 (prevent overfitting)
   - Consider increasing batch size if memory allows
   - Add gradient accumulation if needed

4. EVALUATION:
   - Test on a few examples with verbose output
   - Check if model is generating reasonable text
   - Verify the answer extraction logic works

5. NEXT STEPS:
   - Run test_generation.py to see actual model outputs
   - Retrain with corrected hyperparameters
   - Monitor training loss to ensure it's decreasing
""")

print("=" * 80)
print("Diagnostic complete!")
print("=" * 80)


