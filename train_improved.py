"""
Improved Qwen Coconut Training Script
Addresses the low accuracy issues with better hyperparameters
"""

import argparse
import torch
import yaml
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


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """Convert config dict to argparse-like namespace"""
    class Args:
        pass
    
    args = Args()
    
    # Model args
    args.model_name = config['model']['name']
    args.latent_token = config['latent']['token']
    
    # LoRA args
    args.lora_r = config['lora']['r']
    args.lora_alpha = config['lora']['alpha']
    args.lora_dropout = config['lora']['dropout']
    args.lora_target_modules = config['lora']['target_modules']
    
    # Data args
    args.dataset = config['dataset']['name']
    args.max_length = config['dataset']['max_length']
    args.batch_size = config['training']['batch_size']
    args.prompt_template = config['dataset']['prompt_template']
    
    # Training args
    args.num_epochs = config['training']['num_epochs']
    args.learning_rate = config['training']['learning_rate']
    args.weight_decay = config['training']['weight_decay']
    args.warmup_steps = config['training']['warmup_steps']
    args.scheduler_type = config['training']['scheduler_type']
    args.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    args.log_interval = config['logging']['log_interval']
    args.max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    
    # Evaluation args
    args.eval_samples = config['evaluation']['num_samples']
    args.max_new_tokens = config['evaluation']['max_new_tokens']
    
    # Checkpoint args
    args.save_dir = config['logging']['save_path']
    args.save_every_epoch = config['logging'].get('save_every_epoch', True)
    
    # Seed
    args.seed = config['seed']
    
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training and evaluation pipeline with improved configuration"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Improved Qwen Coconut Training")
    parser.add_argument("--config", type=str, default="args/qwen_coconut_improved.yaml",
                        help="Path to configuration file")
    cmd_args = parser.parse_args()
    
    # Load configuration
    print("=" * 80)
    print("IMPROVED QWEN COCONUT TRAINING")
    print("=" * 80)
    print(f"\nLoading configuration from: {cmd_args.config}")
    config = load_config(cmd_args.config)
    args = config_to_args(config)
    
    # Print key configuration
    print("\nKey Configuration:")
    print(f"  Max Length: {args.max_length} tokens")
    print(f"  Max New Tokens: {args.max_new_tokens} tokens")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  LoRA Rank: {args.lora_r}")
    
    # Set seed
    set_seed(args.seed)
    print(f"\n✓ Random seed set to {args.seed}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args.save_dir)
    print(f"✓ Checkpoint directory: {checkpoint_dir}")
    
    # =============================================
    #  MODEL INITIALIZATION
    # =============================================
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)
    
    lora_config = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "target_modules": args.lora_target_modules,
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
    print("\n" + "=" * 80)
    print("DATA LOADING")
    print("=" * 80)
    
    dataset, dataloader = load_and_prepare_data(
        dataset_name=args.dataset,
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        prompt_template=args.prompt_template
    )
    
    # Save training configuration
    config_dict = vars(args)
    config_dict['device'] = device
    config_dict['latent_token_id'] = latent_token_id
    save_training_config(config_dict, checkpoint_dir)
    
    # =============================================
    #  TRAINING
    # =============================================
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Total steps: {len(dataloader) * args.num_epochs}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Enhanced training with gradient accumulation
    from torch.optim import AdamW
    from torch.cuda.amp import autocast, GradScaler
    from transformers import get_scheduler
    from tqdm import tqdm
    
    optimizer = AdamW(
        coconut_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = args.num_epochs * len(dataloader)
    scheduler = get_scheduler(
        args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler()
    coconut_model.train()
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            from qwen_data import prepare_batch, create_position_ids
            batch_tensors = prepare_batch(batch, device)
            
            input_ids = batch_tensors["input_ids"]
            attention_mask = batch_tensors["attention_mask"]
            labels = batch_tensors["labels"]
            position_ids = create_position_ids(input_ids, device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = coconut_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids
                )
                loss = outputs.loss / args.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    coconut_model.parameters(),
                    args.max_grad_norm
                )
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Track loss
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log periodically
            if batch_idx % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"\n  Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        if args.save_every_epoch:
            save_checkpoint(
                model=coconut_model,
                tokenizer=tokenizer,
                epoch=epoch,
                loss=avg_epoch_loss,
                checkpoint_dir=checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pt"
            )
            print(f"  ✓ Checkpoint saved")
        
        # Track best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(
                model=coconut_model,
                tokenizer=tokenizer,
                epoch=epoch,
                loss=avg_epoch_loss,
                checkpoint_dir=checkpoint_dir,
                filename="best_checkpoint.pt"
            )
            print(f"  ✓ Best model saved (loss: {best_loss:.4f})")
    
    # Save final checkpoint
    save_checkpoint(
        model=coconut_model,
        tokenizer=tokenizer,
        epoch=args.num_epochs - 1,
        loss=avg_epoch_loss,
        checkpoint_dir=checkpoint_dir,
        filename="final_checkpoint.pt"
    )
    
    # =============================================
    #  EVALUATION
    # =============================================
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    results = evaluate_and_report(
        model=coconut_model,
        dataset=dataset,
        tokenizer=tokenizer,
        num_samples=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    
    print(f"\nFinal Training Set Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")
    
    # Test on test set (small sample)
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION (Sample)")
    print("=" * 80)
    
    test_dataset, _ = load_and_prepare_data(
        dataset_name=args.dataset,
        split="test",
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=1,
        shuffle=False,
        prompt_template=args.prompt_template
    )
    
    test_results = evaluate_and_report(
        model=coconut_model,
        dataset=test_dataset,
        tokenizer=tokenizer,
        num_samples=min(100, len(test_dataset)),
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    
    print(f"\nTest Set Accuracy (100 samples): {test_results['accuracy']:.4f}")
    print(f"Correct: {test_results['correct']}/{test_results['total']}")
    
    # =============================================
    #  SUMMARY
    # =============================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final training accuracy: {results['accuracy']:.4f}")
    print(f"Test accuracy (sample): {test_results['accuracy']:.4f}")
    print("\nNext steps:")
    print("1. Run full test evaluation: python test_eval.py")
    print("2. Test generation quality: python test_generation.py")
    print("3. If accuracy is still low, consider:")
    print("   - Training for more epochs")
    print("   - Adjusting learning rate")
    print("   - Using different LoRA configuration")
    print("=" * 80)


if __name__ == "__main__":
    main()


