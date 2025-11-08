"""
Qwen Coconut Fine-tuning - Main Training Script
Orchestrates model initialization, data loading, training, and evaluation.
"""

import torch
from qwen_model import initialize_qwen_coconut_model
from qwen_data import load_and_prepare_data
from qwen_trainer import train_model
from qwen_evaluator import evaluate_and_report


def main():
    """
    Main training and evaluation pipeline.
    """
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # =============================================
    #  MODEL INITIALIZATION
    # =============================================
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    latent_token = "<|latent|>"
    
    lora_config = {
        "r": 8,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    coconut_model, tokenizer, latent_token_id = initialize_qwen_coconut_model(
        model_name=model_name,
        latent_token=latent_token,
        lora_config=lora_config,
        torch_dtype=torch.float16,
        device=device
    )
    
    # =============================================
    #  DATA LOADING
    # =============================================
    dataset, dataloader = load_and_prepare_data(
        dataset_name="gsm8k",
        split="train",
        tokenizer=tokenizer,
        max_length=128,
        batch_size=8,
        shuffle=True,
        prompt_template="Question: {question}\nAnswer:"
    )
    
    # =============================================
    #  TRAINING
    # =============================================
    coconut_model = train_model(
        model=coconut_model,
        dataloader=dataloader,
        num_epochs=1,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=0,
        scheduler_type="linear",
        device=device,
        log_interval=10
    )
    
    # =============================================
    #  EVALUATION
    # =============================================
    results = evaluate_and_report(
        model=coconut_model,
        dataset=dataset,
        tokenizer=tokenizer,
        num_samples=50,
        max_new_tokens=30,
        device=device
    )
    
    print(f"\nFinal Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()

