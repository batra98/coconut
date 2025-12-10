# Coconut: Training LLMs to Reason in Continuous Latent Space

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

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/batra98/coconut.git
cd coconut

# Create virtual environment with uv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies using uv (much faster than pip)
uv pip install -r requirements.txt

# Login to wandb for experiment tracking
wandb login
```

### Data Preparation

#### GSM8K Dataset
```bash
bash preprocessing/gsm_icot.bash
```

This downloads and processes the [GSM8K](https://arxiv.org/abs/2110.14168) dataset with augmented training data.

#### Data Format
All datasets should be JSON files with the following structure:
```json
[
  {
    "question": "Janet's ducks lay 16 eggs per day...",
    "answer": "18",
    "steps": [
      "Janet sells 3 eggs at the farmers market...",
      "She uses 4+16=20 eggs to bake muffins..."
    ]
  }
]
```

---

## Experiments

**Hardware Setup**: All experiments were conducted on UW-Madison's instgpu cluster (instgpu-01 through instgpu-4), each equipped with 8x NVIDIA 2080Ti GPUs.

### 1. Coconut Replication on GSM8K

Our core contribution - successfully replicating the Coconut paper's results.

#### Step 1: Train CoT Baseline (Stage 0)

On each node (instgpu-01 through instgpu-4):

```bash
# Set up distributed training environment
export MASTER_ADDR=instgpu-01.cs.wisc.edu
export MASTER_PORT=29500

# On master node (instgpu-01)
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_cot.yaml

# On worker nodes (instgpu-1, instgpu-2, instgpu-3, instgpu-4)
# Update node_rank to 1, 2, 3, 4 respectively
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 1 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_cot.yaml
```

This trains GPT-2 with explicit chain-of-thought reasoning, achieving ~40% validation accuracy.

**Pre-trained checkpoint**: [gsm-cot-checkpoint-22](https://huggingface.co/batra98/gsm-cot-checkpoint-22) on HuggingFace

#### Step 2: Train Coconut with Continuous Thoughts

Update `args/gsm_coconut.yaml` with your CoT checkpoint path (or use our pre-trained checkpoint), then run on each node:

```bash
# On master node (instgpu-01)
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_coconut.yaml

# On worker nodes (instgpu-1, instgpu-2, instgpu-3, instgpu-4)
# Update node_rank accordingly
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 1 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_coconut.yaml
```

Coconut progressively replaces text reasoning steps with continuous latent thoughts through staged curriculum training.

**Pre-trained checkpoints**: [gsm-coconut](https://huggingface.co/batra98/gsm-coconut) on HuggingFace (checkpoints 4-25)

#### Step 3: Evaluate

Update `args/gsm_coconut_eval.yaml` with the best checkpoint path, then run:

```bash
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_coconut_eval.yaml
```

**Results**: [Wandb links to be added]

#### Using Pre-trained Checkpoints

To use our pre-trained checkpoints from HuggingFace:

```bash
# Download CoT checkpoint
git clone https://huggingface.co/batra98/gsm-cot-checkpoint-22

# Download Coconut checkpoints
git clone https://huggingface.co/batra98/gsm-coconut

# Update your config files with the local paths
# For example, in args/gsm_coconut.yaml:
# load_model_path: gsm-cot-checkpoint-22/checkpoint_22
```

### 2. Other Experiments

We have explored several extensions to the base Coconut model. Please check the following branches for details:

- **GRPO Training**: `sra/moun`
- **Pass@K Evaluation**: `sra/pass_k` and `sra/pass_k_coconut`
- **Qwen 2.5 Support**: `feature/qwen-support` and `qwen2.5-3B`
- **Soft Thinking**: `soft-thinking2`

### 3. Qwen 2.5 Support

This branch (`feature/qwen-support`) adapts the Coconut architecture to work with **Qwen 2.5** models (specifically tested with 0.5B and 3B).

#### Key Changes

- **Model Loading**: Modified to support Qwen's architecture.
- **Tokenizer**: Adjusted to handle Qwen's vocabulary and special tokens.
- **Position Embeddings**: Updated to work with Qwen's rotary embeddings (RoPE).
- **LoRA Support**: Added Low-Rank Adaptation (LoRA) for efficient fine-tuning of larger models.

#### Usage

To train a Coconut model initialized with Qwen 2.5-0.5B:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_qwen.yaml
```

#### Configuration

See `args/gsm_coconut_qwen.yaml` for the configuration details. Key parameters include:

- `model_id`: `Qwen/Qwen2.5-0.5B`
- `lora`: `True`
- `lora_r`: 8
- `lora_target_modules`: `["q_proj", "v_proj"]`

---

---

## Repository Structure

```
coconut/
├── coconut.py              # Core Coconut model implementation
├── run.py                  # Main training/evaluation script
├── run_grpo.py            # GRPO training script
├── grpo.py                # GRPO reward functions and trainer
├── dataset.py             # Dataset loading and processing
├── utils.py               # Utility functions
├── args/                  # Configuration files
│   ├── gsm_cot.yaml       # CoT baseline config
│   ├── gsm_coconut.yaml   # Coconut training config
│   ├── gsm_grpo.yaml      # GRPO training config
│   └── ...
├── preprocessing/         # Data preprocessing scripts
│   ├── gsm_icot.bash      # GSM8K download script
│   ├── gsm_icot.py        # GSM8K processing
│   └── prontoqa.py        # ProntoQA processing
└── data/                  # Dataset directory
```

## Configuration

All experiments are controlled via YAML configuration files in the `args/` directory.

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `coconut` | Enable Coconut continuous thoughts | `True` |
| `cot` | Enable Chain-of-Thought baseline | `False` |
| `grpo` | Enable GRPO training | `False` |
| `c_thought` | Number of continuous thoughts per step | `2` |
| `max_latent_stage` | Maximum curriculum training stages | `3` |
| `batch_size_training` | Batch size per GPU | `8-32` |
| `gradient_accumulation_steps` | Gradient accumulation | `2-4` |
| `lr` | Learning rate | `1e-4` |
| `model_id` | Base model | `openai-community/gpt2` |

### Training Stages

Coconut uses curriculum learning with progressive stages:
- **Stage 0**: Train with full CoT (text reasoning)
- **Stage 1**: Replace 1st reasoning step with continuous thoughts
- **Stage 2**: Replace 1st and 2nd steps with continuous thoughts
- **Stage 3**: Replace 1st, 2nd, and 3rd steps with continuous thoughts

Each stage trains for `epochs_per_stage` epochs (typically 3).

---

## Advanced Usage

### Multi-Node Training on instgpu Cluster

For large-scale experiments across multiple instgpu nodes (each with 8x 2080Ti GPUs):

**Important**: NCCL does not work reliably on the instgpu cluster. Use Gloo backend instead.

```bash
# On master node (e.g., instgpu-01)
export MASTER_ADDR=instgpu-01.cs.wisc.edu
export MASTER_PORT=29500

# Run on master node
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_coconut.yaml

# On worker nodes (instgpu-1, instgpu-2, instgpu-3, instgpu-4)
# Set node_rank appropriately (1, 2, 3, 4)
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 1 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_coconut.yaml
```

**Disabling NCCL**: To use Gloo backend instead of NCCL, set before running:

```bash
export GLOO_SOCKET_IFNAME=eth0  # or appropriate network interface
# Then modify distributed initialization in your code to use 'gloo' instead of 'nccl'
```

Alternatively, run with single node but all 8 GPUs:

```bash
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml
```

### Debugging Mode

Set `debug: True` in your config file to:
- Disable wandb logging
- Skip model checkpointing
- Use subset of data for quick iteration

### Resuming Training

Training automatically resumes from the latest checkpoint if interrupted. Manual resume:
=======
See `args/gsm_coconut_soft_thinking_eval.yaml` for configuration details.
>>>>>>> soft-thinking2

```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1
soft_thinking_cold_stop_patience: 2
soft_thinking_temperature: 1.0
```

## Other Experiments

<<<<<<< HEAD
## Evaluation Metrics

We track multiple metrics to comprehensively evaluate reasoning capabilities:

| Metric | Description | Branch |
|--------|-------------|--------|
| **Accuracy** | Exact match of final answer (greedy decoding) | All |
| **CoT EM** | Exact match of entire reasoning chain | All |
| **Pass@20** | Success rate with 20 samples per problem (temperature sampling) | `sra/pass_k`, `sra/pass_k_coconut` |
| **Format Score** | Adherence to reasoning format with `<<computation=result>>` | `sra/grpo` |

**Why Pass@20 Matters:**
- Standard accuracy only measures if the model **always** produces the correct answer
- Pass@20 measures if the model **can** produce the correct answer with diverse sampling
- More robust indicator of model capabilities and reasoning understanding
- Better reflects real-world usage where multiple attempts or diverse outputs are acceptable

---

## Key Implementation Details

### Coconut Architecture

The core innovation is in `coconut.py`:

1. **Latent Token Injection**: Special `<LATENT>` tokens are inserted in the input
2. **Multi-Pass Forward**: Model processes input in multiple passes, feeding hidden states as continuous thoughts
3. **KV Cache Optimization**: Efficiently reuses computations between passes
4. **Position-Aware Processing**: Maintains proper position embeddings despite multi-pass architecture

### GRPO Training

Our GRPO implementation (`grpo.py`) features:

1. **Dual Reward Components**:
   - Answer correctness: Binary reward for correct final answer
   - Format adherence: Continuous reward for well-structured reasoning

2. **Group Normalization**: Rewards are normalized within each batch for stable training

3. **Policy Regularization**: KL penalty (β=0.01) prevents catastrophic forgetting

### Pass@K Sampling

**Branch-specific implementation:**

**`sra/pass_k`** (vanilla models):
- Modified `run.py` to generate multiple samples during evaluation
- Uses HuggingFace's built-in sampling parameters

**`sra/pass_k_coconut`** (Coconut models):
- Implemented custom `_sample_token()` method in `coconut.py`
- Supports temperature and top-p (nucleus) sampling
- Maintains continuous thought processing during sampling

**Sampling parameters:**
- Temperature: 0.7 (controls randomness)
- Top-p (nucleus): 0.95 (filters low-probability tokens)
- Samples per problem: 20

---

## Technical Stack

- **PyTorch 2.5.1** - Deep learning framework
- **Transformers 4.51.1** - HuggingFace models and tokenizers
- **TRL 0.19.0** - Transformer Reinforcement Learning
- **FSDP** - Fully Sharded Data Parallel for efficient multi-GPU training
- **Wandb** - Experiment tracking and visualization

---

## Datasets

### GSM8K (Grade School Math 8K)
- **Training**: 7,473 problems
- **Validation**: 1,319 problems  
- **Task**: Multi-step arithmetic reasoning
- **Format**: Natural language questions with numerical answers

### ProntoQA (Optional)
- **Task**: Logical reasoning with fictional entities
- **Hops**: 5-hop reasoning chains
- **Format**: True/False questions

### ProsQA (Optional)
- **Task**: Procedural reasoning
- **Format**: Step-by-step procedural questions

---

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `batch_size_training`
- Increase `gradient_accumulation_steps`
- Enable `bf16: true` for memory-efficient training

**Training Instability**
- Lower learning rate (try `5e-5` or `1e-5`)
- Enable `reset_optimizer: True` when switching stages
- Check gradient clipping in GRPO config

**Slow Training**
- Verify GPU utilization with `nvidia-smi`
- Ensure NCCL backend is properly configured for multi-GPU
- Use `num_workers=1` in dataloader for stability

---

## Citation

If you use this codebase in your research, please cite the original Coconut paper:

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

---

## Contributing

This is a course project repository, but we welcome:
- Bug reports and fixes
- Documentation improvements
- Suggestions for experiments

Please open an issue or submit a pull request!

---

## License

This project is released under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Original Coconut paper authors at Meta AI Research
- CS 739 course staff at UW-Madison
- HuggingFace for Transformers library
- TRL team for reinforcement learning tools

---

## Contact

For questions about this project:
- Open a GitHub issue
- Email: gbatra3@wisc.edu

**Note**: Wandb experiment links and final results will be added upon completion of all experiments.
### Qwen 2.5-3B Specifics

#### Training

To train Coconut with Qwen 2.5-3B on GSM8K:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut.yaml
```

**Configuration (`args/qwen_coconut.yaml`):**
- `lora.r`: 8 (Rank for LoRA adapters)
- `lora.target_modules`: `["q_proj", "v_proj"]`
- `training.batch_size`: 8
- `training.learning_rate`: 1e-4

#### Improved Training Configuration

For better stability and performance, we recommend using the improved configuration:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut_improved.yaml
```

**Key Improvements:**
- Increased context length (512 tokens)
- Optimized learning rate schedule (Cosine)
- Enhanced LoRA configuration (Rank 16, more target modules)
- Gradient accumulation for larger effective batch sizes

#### Evaluation

To evaluate the trained model:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml \
    --load_model_path ./checkpoints/qwen_coconut/checkpoint_best
```

#### Model Architecture

The implementation wraps Qwen 2.5 with Coconut's continuous thought mechanism:
1. **Latent Tokens**: `<|latent|>` tokens are injected to represent continuous thoughts.
2. **Continuous Reasoning**: The model processes these latent tokens to generate hidden states that guide subsequent generation.
3. **LoRA Adapters**: Fine-tuning is applied only to LoRA adapters and the new latent token embeddings, preserving the pre-trained knowledge of Qwen 2.5.
=======
For other experiments, please check the respective branches:
- `feature/qwen-support`: Qwen 2.5 Support
- `sra/moun`: GRPO Training

### Soft Thinking Specifics

This branch (`soft-thinking2`) explores **Soft Thinking** - a technique to dynamically control the length of latent reasoning chains during inference based on entropy.

#### Overview

Standard Coconut uses a fixed number of latent thoughts per reasoning step. Soft Thinking allows the model to decide when it has "thought enough" by monitoring the entropy of its latent state distribution.

#### Key Features

- **Dynamic Halting**: Stops generating latent thoughts when the model is "confident" (low entropy).
- **Entropy-based Control**: Monitors the entropy of the latent state distribution.
- **Configurable Parameters**:
  - `soft_thinking_cold_stop_threshold`: Entropy threshold for stopping (e.g., 0.1).
  - `soft_thinking_cold_stop_patience`: Number of consecutive low-entropy steps required.

#### Usage

To evaluate a trained Coconut model with Soft Thinking enabled:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_soft_thinking_eval.yaml
```

**Configuration (`args/gsm_coconut_soft_thinking_eval.yaml`):**

```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1
soft_thinking_cold_stop_patience: 2
soft_thinking_temperature: 1.0
```
