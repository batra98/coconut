"""
Qwen Utilities Module
Handles checkpointing, model saving, and logging utilities.
"""

import os
import torch
from datetime import datetime


def create_checkpoint_dir(base_path="./checkpoints/qwen_coconut"):
    """
    Create checkpoint directory with timestamp.
    
    Args:
        base_path (str): Base path for checkpoints
        
    Returns:
        str: Path to checkpoint directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{base_path}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"✓ Checkpoint directory created: {checkpoint_dir}")
    
    return checkpoint_dir


def save_checkpoint(model, tokenizer, epoch, loss, checkpoint_dir, filename=None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        epoch (int): Current epoch number
        loss (float): Current loss value
        checkpoint_dir (str): Directory to save checkpoint
        filename (str): Optional custom filename
        
    Returns:
        str: Path to saved checkpoint
    """
    if filename is None:
        filename = f"checkpoint_epoch_{epoch+1}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Save model state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def save_model(model, tokenizer, save_dir):
    """
    Save final trained model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        save_dir (str): Directory to save model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save base model (LoRA adapters)
    model_path = os.path.join(save_dir, "model")
    if hasattr(model, 'base_causallm'):
        # Save the base model with LoRA adapters
        model.base_causallm.save_pretrained(model_path)
    else:
        # Fallback: save full model state
        torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
    
    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"✓ Model saved to: {save_dir}")


def load_checkpoint(model, checkpoint_path, device="cuda"):
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on
        
    Returns:
        tuple: (model, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint loaded from: {checkpoint_path}")
    print(f"  - Epoch: {epoch}")
    print(f"  - Loss: {loss:.4f}")
    
    return model, epoch, loss


def log_training_info(epoch, batch_idx, total_batches, loss, learning_rate=None):
    """
    Log training information in a formatted way.
    
    Args:
        epoch (int): Current epoch
        batch_idx (int): Current batch index
        total_batches (int): Total number of batches
        loss (float): Current loss
        learning_rate (float): Current learning rate (optional)
    """
    log_msg = f"[Epoch {epoch+1}] Batch {batch_idx}/{total_batches} | Loss: {loss:.4f}"
    
    if learning_rate is not None:
        log_msg += f" | LR: {learning_rate:.6f}"
    
    print(log_msg)


def log_epoch_summary(epoch, avg_loss, elapsed_time=None):
    """
    Log epoch summary.
    
    Args:
        epoch (int): Current epoch
        avg_loss (float): Average loss for the epoch
        elapsed_time (float): Time taken for epoch in seconds (optional)
    """
    print("\n" + "-" * 60)
    print(f"Epoch {epoch+1} Summary:")
    print(f"  - Average Loss: {avg_loss:.4f}")
    
    if elapsed_time is not None:
        print(f"  - Time: {elapsed_time:.2f}s")
    
    print("-" * 60 + "\n")


def save_training_config(config, checkpoint_dir):
    """
    Save training configuration to file.
    
    Args:
        config (dict): Configuration dictionary
        checkpoint_dir (str): Directory to save config
    """
    import json
    
    config_path = os.path.join(checkpoint_dir, "training_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Training config saved: {config_path}")


def get_model_size(model):
    """
    Calculate total number of parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model):
    """
    Print model information including parameter counts.
    
    Args:
        model: PyTorch model
    """
    total_params, trainable_params = get_model_size(model)
    
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print("=" * 60 + "\n")

