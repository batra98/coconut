"""Quick evaluation test script"""
import torch
import gc
from qwen_model import initialize_qwen_coconut_model
from qwen_data import load_and_prepare_data
from qwen_evaluator import evaluate_and_report
from qwen_utils import load_checkpoint

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

device = 'cuda'
checkpoint_path = './checkpoints/qwen_coconut_5epochs_20251109_205836/final_checkpoint.pt'

print("Loading model...")
lora_config = {
    'r': 8, 
    'alpha': 32, 
    'target_modules': ['q_proj', 'v_proj'], 
    'dropout': 0.05, 
    'bias': 'none', 
    'task_type': 'CAUSAL_LM'
}

# Load checkpoint state first (without loading full model twice)
print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU first

# Initialize model
model, tokenizer, _ = initialize_qwen_coconut_model(
    'Qwen/Qwen2.5-3B-Instruct', 
    '<|latent|>', 
    lora_config, 
    torch.float16, 
    device
)

# Load weights
print("Loading checkpoint weights...")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

print("Loading test set...")
dataset, _ = load_and_prepare_data('gsm8k', 'test', tokenizer, 64, 2, False)

print("Evaluating on full test set (1319 examples)...")
results = evaluate_and_report(model, dataset, tokenizer, 1319, 100, device)
print(f'\nFinal GSM8K Test Accuracy: {results["accuracy"]:.4f} ({results["correct"]}/{results["total"]})')

