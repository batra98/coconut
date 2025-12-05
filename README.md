# Coconut with Soft Thinking

This branch (`soft-thinking2`) explores **Soft Thinking** - a technique to dynamically control the length of latent reasoning chains during inference based on entropy.

## Overview

Standard Coconut uses a fixed number of latent thoughts per reasoning step. Soft Thinking allows the model to decide when it has "thought enough" by monitoring the entropy of its latent state distribution.

## Key Features

- **Dynamic Halting**: Stops generating latent thoughts when the model is "confident" (low entropy).
- **Entropy-based Control**: Monitors the entropy of the latent state distribution.
- **Configurable Parameters**:
  - `soft_thinking_cold_stop_threshold`: Entropy threshold for stopping (e.g., 0.1).
  - `soft_thinking_cold_stop_patience`: Number of consecutive low-entropy steps required.

## Usage

### Running Soft Thinking Evaluation

To evaluate a trained Coconut model with Soft Thinking enabled:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_soft_thinking_eval.yaml
```

### Configuration

See `args/gsm_coconut_soft_thinking_eval.yaml` for configuration details.

```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1
soft_thinking_cold_stop_patience: 2
soft_thinking_temperature: 1.0
```

## Other Experiments

For other experiments, please check the respective branches:
- `feature/qwen-support`: Qwen 2.5 Support
- `sra/moun`: GRPO Training
