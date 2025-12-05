# Coconut with GRPO Training

This branch (`sra/moun`) implements **Group Relative Policy Optimization (GRPO)** to fine-tune Coconut models using reinforcement learning with custom reward functions.

## Overview

GRPO is a reinforcement learning technique that optimizes the model's policy based on group-relative rewards. This allows us to fine-tune the model for specific objectives, such as answer correctness and reasoning format adherence, without needing a separate value network.

## Key Features

- **Custom Reward Function**: Combines answer correctness (weight: 1.0) and format adherence (weight: 0.1).
- **Group-based Normalization**: Rewards are normalized within each batch for stable training.
- **Robust Policy Updates**: Generates multiple completions (e.g., 8) per prompt to estimate the baseline.
- **Integrated with TRL**: Uses `GRPOTrainer` from the Transformer Reinforcement Learning (TRL) library.

## Usage

### Running GRPO Training

To start GRPO training on GSM8K:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run_grpo.py args/gsm_grpo.yaml
```

### Configuration

See `args/gsm_grpo.yaml` for configuration details. Key parameters:

- `grpo`: `True`
- `num_generations`: 8 (Number of completions per prompt)
- `beta`: 0.01 (KL penalty)
- `reward_weight_final_answer`: 1.0
- `reward_weight_format`: 0.1

## Other Experiments

For other experiments, please check the respective branches:
- `feature/qwen-support`: Qwen 2.5 Support
- `soft-thinking2`: Soft Thinking Experiments
