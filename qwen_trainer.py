"""
Qwen Training Module
Handles training loop, optimizer setup, and gradient scaling for Coconut fine-tuning.
"""

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from tqdm import tqdm

from qwen_data import prepare_batch, create_position_ids


def setup_optimizer(model, learning_rate=1e-4, weight_decay=0.01):
    """
    Setup AdamW optimizer for training.
    
    Args:
        model: Model to optimize
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print(f"✓ Optimizer configured:")
    print(f"  - Type: AdamW")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    
    return optimizer


def setup_scheduler(optimizer, num_training_steps, warmup_steps=0, scheduler_type="linear"):
    """
    Setup learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        num_training_steps (int): Total number of training steps
        warmup_steps (int): Number of warmup steps
        scheduler_type (str): Type of scheduler ('linear', 'cosine', etc.)
        
    Returns:
        LRScheduler: Configured scheduler
    """
    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"✓ Scheduler configured:")
    print(f"  - Type: {scheduler_type}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Total steps: {num_training_steps}")
    
    return scheduler


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, log_interval=10):
    """
    Train for one epoch.
    
    Args:
        model: Coconut model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device (str): Device to train on
        log_interval (int): Log every N batches
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Prepare batch
        batch_tensors = prepare_batch(batch, device)
        input_ids = batch_tensors["input_ids"]
        attention_mask = batch_tensors["attention_mask"]
        labels = batch_tensors["labels"]
        
        # Create position IDs
        position_ids = create_position_ids(input_ids, device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids
            )
            loss = outputs.loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Track loss
        epoch_loss += loss.item()
        
        # Log progress
        if batch_idx % log_interval == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss


def train_model(
    model,
    dataloader,
    num_epochs=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=0,
    scheduler_type="linear",
    device="cuda",
    log_interval=10
):
    """
    Complete training pipeline.
    
    Args:
        model: Coconut model to train
        dataloader: Training data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        warmup_steps (int): Number of warmup steps
        scheduler_type (str): Type of scheduler
        device (str): Device to train on
        log_interval (int): Log every N batches
        
    Returns:
        model: Trained model
    """
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Setup optimizer
    optimizer = setup_optimizer(model, learning_rate, weight_decay)
    
    # Setup scheduler
    num_training_steps = num_epochs * len(dataloader)
    scheduler = setup_scheduler(optimizer, num_training_steps, warmup_steps, scheduler_type)
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler()
    print("✓ Gradient scaler initialized for mixed precision training")
    
    print("=" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n==== Epoch {epoch+1}/{num_epochs} ====")
        
        avg_loss = train_epoch(
            model, 
            dataloader, 
            optimizer, 
            scheduler, 
            scaler, 
            device, 
            log_interval
        )
        
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model

