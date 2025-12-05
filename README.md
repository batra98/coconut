# Coconut with Qwen 2.5 Support

This branch (`feature/qwen-support`) adapts the Coconut architecture to work with **Qwen 2.5** models (specifically tested with 0.5B and 3B).

## Overview

Coconut (Continuous Latent Thoughts) was originally designed for GPT-2. This branch extends it to support modern, more capable base models like Qwen 2.5.

## Key Changes

- **Model Loading**: Modified to support Qwen's architecture.
- **Tokenizer**: Adjusted to handle Qwen's vocabulary and special tokens.
- **Position Embeddings**: Updated to work with Qwen's rotary embeddings (RoPE).
- **LoRA Support**: Added Low-Rank Adaptation (LoRA) for efficient fine-tuning of larger models.

## Usage

### Training Qwen 2.5-0.5B

To train a Coconut model initialized with Qwen 2.5-0.5B:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_qwen.yaml
```

### Configuration

See `args/gsm_coconut_qwen.yaml` for the configuration details. Key parameters include:

- `model_id`: `Qwen/Qwen2.5-0.5B`
- `lora`: `True`
- `lora_r`: 8
- `lora_target_modules`: `["q_proj", "v_proj"]`

## Other Experiments

For other experiments (GRPO, Soft Thinking, etc.), please check the respective branches:
- `sra/moun`: GRPO Training
- `soft-thinking2`: Soft Thinking Experiments
