# Qwen 2.5-3B Coconut Integration

This document describes the integration of **Qwen 2.5-3B-Instruct** with the **Coconut** framework for continuous latent reasoning. This implementation uses LoRA (Low-Rank Adaptation) for efficient fine-tuning on the GSM8K mathematical reasoning dataset.

## 📋 Overview

This integration provides a modular, production-ready implementation for fine-tuning Qwen models with Coconut's continuous latent thought mechanism. The codebase is organized into separate modules for maintainability and reusability.

## 🏗️ Architecture

### Module Structure

```
qwen_model.py          # Model initialization (Qwen + LoRA + Coconut)
qwen_data.py           # Data loading and preprocessing
qwen_trainer.py        # Training loop and optimization
qwen_evaluator.py      # Model evaluation and metrics
qwen_utils.py          # Checkpointing, logging, and utilities
train_qwen_coconut.py  # Main training script
inference_qwen_coconut.py  # Inference script for trained models
args/qwen_coconut.yaml # Configuration file
requirements_qwen.txt  # Qwen-specific dependencies
```

### Key Features

- ✅ **Modular Design**: Separate modules for different concerns
- ✅ **LoRA Fine-tuning**: Efficient parameter-efficient training
- ✅ **Mixed Precision**: FP16 training with automatic mixed precision
- ✅ **Checkpointing**: Automatic model checkpointing and recovery
- ✅ **Command-line Interface**: Flexible configuration via arguments
- ✅ **Inference Support**: Dedicated script for model inference
- ✅ **Comprehensive Logging**: Training progress and model statistics

## 🚀 Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_qwen.txt
```

For CUDA 12.1 support:
```bash
pip install -r requirements_qwen.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Training

#### Basic Training (with defaults)

```bash
python train_qwen_coconut.py
```

#### Custom Configuration

```bash
python train_qwen_coconut.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 64 \
  --max_length 256 \
  --save_model_path ./models/qwen_coconut_final
```

### 3. Inference

#### Single Question

```bash
python inference_qwen_coconut.py \
  --checkpoint_path ./checkpoints/qwen_coconut_*/final_checkpoint.pt \
  --question "What is 15 * 23?"
```

#### Interactive Mode

```bash
python inference_qwen_coconut.py \
  --checkpoint_path ./checkpoints/qwen_coconut_*/final_checkpoint.pt
```

## ⚙️ Configuration

### Command-line Arguments

#### Model Arguments
- `--model_name`: HuggingFace model identifier (default: `Qwen/Qwen2.5-3B-Instruct`)
- `--latent_token`: Special token for latent thoughts (default: `<|latent|>`)

#### LoRA Arguments
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha scaling (default: 32)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)

#### Data Arguments
- `--dataset`: Dataset name (default: `gsm8k`)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Training batch size (default: 8)

#### Training Arguments
- `--num_epochs`: Number of training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for regularization (default: 0.01)
- `--warmup_steps`: Number of warmup steps (default: 0)
- `--log_interval`: Log every N batches (default: 10)

#### Evaluation Arguments
- `--eval_samples`: Number of samples to evaluate (default: 50)
- `--max_new_tokens`: Maximum tokens to generate (default: 30)

#### Checkpoint Arguments
- `--save_dir`: Directory for checkpoints (default: `./checkpoints/qwen_coconut`)
- `--save_model_path`: Path to save final model (optional)

### YAML Configuration

You can also use the YAML configuration file:

```yaml
# args/qwen_coconut.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  torch_dtype: "float16"

lora:
  r: 8
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
  dropout: 0.05

training:
  num_epochs: 1
  batch_size: 8
  learning_rate: 1e-4
```

## 📊 Results

### GSM8K Performance

Training on GSM8K with default settings:
- **Model**: Qwen 2.5-3B-Instruct
- **Training**: 1 epoch, batch size 8
- **LoRA**: r=8, alpha=32
- **Evaluation**: 50 samples

Expected performance metrics will vary based on training duration and hyperparameters.

## 🔧 Technical Details

### LoRA Configuration

This implementation uses LoRA (Low-Rank Adaptation) for efficient fine-tuning:

- **Rank (r)**: 8 (controls the number of trainable parameters)
- **Alpha**: 32 (scaling factor for LoRA weights)
- **Target Modules**: `q_proj`, `v_proj` (attention query and value projections)
- **Dropout**: 0.05 (regularization)

**Parameter Efficiency**: With LoRA, only ~0.1-1% of parameters are trainable, making training much faster and memory-efficient.

### Mixed Precision Training

The implementation uses PyTorch's Automatic Mixed Precision (AMP):
- Model weights: FP16
- Gradient scaling: Automatic
- Memory savings: ~50% compared to FP32

### Coconut Integration

The Coconut wrapper adds continuous latent reasoning:
- **Latent Token**: `<|latent|>` inserted in sequences
- **Continuous Thoughts**: Hidden states fed back as continuous representations
- **Multi-pass Forward**: Multiple forward passes for latent reasoning steps

## 📁 Output Structure

After training, the following structure is created:

```
checkpoints/qwen_coconut_YYYYMMDD_HHMMSS/
├── final_checkpoint.pt          # Final model checkpoint
├── training_config.json         # Training configuration
└── tokenizer/                   # Saved tokenizer
    ├── tokenizer_config.json
    ├── vocab.json
    └── ...
```

## 🐛 Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce batch size: `--batch_size 4`
2. Reduce sequence length: `--max_length 64`
3. Reduce LoRA rank: `--lora_r 4`

### Slow Training

To speed up training:
1. Increase batch size: `--batch_size 16`
2. Use gradient accumulation (modify `qwen_trainer.py`)
3. Ensure CUDA is available and being used

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install -r requirements_qwen.txt
```

## 📚 Module Documentation

### qwen_model.py

Functions for model initialization:
- `initialize_tokenizer()`: Setup tokenizer with latent token
- `load_base_model()`: Load Qwen base model
- `apply_lora()`: Apply LoRA adapters
- `create_coconut_model()`: Wrap with Coconut
- `initialize_qwen_coconut_model()`: Complete pipeline

### qwen_data.py

Functions for data handling:
- `load_gsm8k_dataset()`: Load and preprocess GSM8K
- `create_dataloader()`: Create PyTorch DataLoader
- `prepare_batch()`: Prepare batch for training
- `create_position_ids()`: Generate position IDs

### qwen_trainer.py

Functions for training:
- `setup_optimizer()`: Configure AdamW optimizer
- `setup_scheduler()`: Setup learning rate scheduler
- `train_epoch()`: Train for one epoch
- `train_model()`: Complete training pipeline

### qwen_evaluator.py

Functions for evaluation:
- `generate_answer()`: Generate answer for question
- `evaluate_model()`: Evaluate on dataset
- `evaluate_and_report()`: Complete evaluation pipeline

### qwen_utils.py

Utility functions:
- `create_checkpoint_dir()`: Create checkpoint directory
- `save_checkpoint()`: Save model checkpoint
- `save_model()`: Save final model
- `load_checkpoint()`: Load from checkpoint
- `print_model_info()`: Display model statistics

## 🔬 Extending the Code

### Adding New Datasets

To add support for a new dataset, modify `qwen_data.py`:

```python
def load_custom_dataset(split="train", tokenizer=None, max_length=128):
    dataset = load_dataset("your_dataset", split=split)
    # Add preprocessing logic
    return dataset
```

### Custom LoRA Configuration

Modify the LoRA config in `train_qwen_coconut.py`:

```python
lora_config = {
    "r": 16,  # Higher rank for more capacity
    "alpha": 64,
    "target_modules": ["q_proj", "v_proj", "k_proj"],  # Add more modules
    "dropout": 0.1,
}
```

### Adding Logging (WandB, TensorBoard)

Add logging in `qwen_trainer.py`:

```python
import wandb

# In train_epoch()
wandb.log({"loss": loss.item(), "epoch": epoch})
```

## 📖 References

- [Coconut Paper](https://arxiv.org/abs/2412.06769): Training Large Language Models to Reason in a Continuous Latent Space
- [Qwen Model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct): Qwen 2.5-3B-Instruct on HuggingFace
- [LoRA Paper](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation of Large Language Models
- [GSM8K Dataset](https://arxiv.org/abs/2110.14168): Training Verifiers to Solve Math Word Problems
