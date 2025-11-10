"""
Qwen Coconut Fine-tuning - Main Training Script
Orchestrates model initialization, data loading, training, and evaluation.
"""

import argparse
import torch
from qwen_model import initialize_qwen_coconut_model
from qwen_data import load_and_prepare_data
from qwen_trainer import train_model
from qwen_evaluator import evaluate_and_report
from qwen_utils import (
    create_checkpoint_dir,
    save_checkpoint,
    save_model,
    save_training_config,
    print_model_info
)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with Coconut")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--latent_token", type=str, default="<|latent|>",
                        help="Latent token for continuous thoughts")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="Dataset name")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N batches")
    
    # Evaluation arguments
    parser.add_argument("--eval_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum tokens to generate")
    
    # Checkpoint arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints/qwen_coconut",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_model_path", type=str, default=None,
                        help="Path to save final model")
    
    return parser.parse_args()


def main():
    """
    Main training and evaluation pipeline.
    """
    # Parse arguments
    args = parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args.save_dir)
    
    # =============================================
    #  MODEL INITIALIZATION
    # =============================================
    lora_config = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj"],
        "dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    coconut_model, tokenizer, latent_token_id = initialize_qwen_coconut_model(
        model_name=args.model_name,
        latent_token=args.latent_token,
        lora_config=lora_config,
        torch_dtype=torch.float16,
        device=device
    )
    
    # Print model information
    print_model_info(coconut_model)
    
    # =============================================
    #  DATA LOADING
    # =============================================
    dataset, dataloader = load_and_prepare_data(
        dataset_name=args.dataset,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        prompt_template="Question: {question}\nAnswer:"
    )
    
    # Save training configuration
    config = vars(args)
    config['device'] = device
    config['latent_token_id'] = latent_token_id
    save_training_config(config, checkpoint_dir)
    
    # =============================================
    #  TRAINING
    # =============================================
    coconut_model = train_model(
        model=coconut_model,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        scheduler_type="linear",
        device=device,
        log_interval=args.log_interval
    )
    
    # Save checkpoint after training
    save_checkpoint(
        model=coconut_model,
        tokenizer=tokenizer,
        epoch=args.num_epochs - 1,
        loss=0.0,  # Final loss would be tracked in enhanced version
        checkpoint_dir=checkpoint_dir,
        filename="final_checkpoint.pt"
    )
    
    # =============================================
    #  EVALUATION
    # =============================================
    results = evaluate_and_report(
        model=coconut_model,
        dataset=dataset,
        tokenizer=tokenizer,
        num_samples=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    
    print(f"\nFinal Accuracy: {results['accuracy']:.4f}")
    
    # Save final model if path specified
    if args.save_model_path:
        save_model(coconut_model, tokenizer, args.save_model_path)
    
    print(f"\nTraining complete! Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

