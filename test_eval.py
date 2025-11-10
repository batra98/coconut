"""Quick evaluation test script"""
import torch
from qwen_model import initialize_qwen_coconut_model
from qwen_data import load_and_prepare_data
from qwen_evaluator import evaluate_and_report
from qwen_utils import load_checkpoint

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

model, tokenizer, _ = initialize_qwen_coconut_model(
    'Qwen/Qwen2.5-3B-Instruct', 
    '<|latent|>', 
    lora_config, 
    torch.float16, 
    device
)

print(f"Loading checkpoint from {checkpoint_path}...")
model, _, _ = load_checkpoint(model, checkpoint_path, device)

print("Loading dataset...")
dataset, _ = load_and_prepare_data('gsm8k', 'train', tokenizer, 64, 2, False)

print("Evaluating...")
results = evaluate_and_report(model, dataset, tokenizer, 10, 50, device)
print(f'\nFinal Accuracy: {results["accuracy"]:.4f}')

