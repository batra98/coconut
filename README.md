# Coconut: Training LLMs to Reason in Continuous Latent Space

**CS 739: Advanced NLP Course Project**  
*University of Wisconsin-Madison*

This repository contains our implementation and extensions of [Coconut](https://arxiv.org/abs/2412.06769) - a novel approach to training large language models to reason using continuous latent thoughts instead of discrete chain-of-thought tokens.

![Coconut Architecture](assets/coconut.png)

## Team Members

- **Gaurav Batra** ([batra98](https://github.com/batra98))
- **Sujan Reddy Ale** ([Sujan242](https://github.com/Sujan242))
- **Aayush Gupta** ([AayGup](https://github.com/AayGup))
- **Srishti Lodha** ([Srish-tii](https://github.com/Srish-tii))

## Project Overview

Coconut introduces a paradigm shift in how language models perform reasoning. Instead of generating explicit reasoning steps as text tokens (like Chain-of-Thought), Coconut learns to reason in a continuous latent space, leading to:

- **More efficient reasoning**: Continuous thoughts are more compact than text
- **Better generalization**: Latent representations can capture abstract reasoning patterns
- **Improved performance**: Achieves competitive results with fewer tokens

### Our Contributions

This project goes beyond replicating the original Coconut paper. We have:

1. **Successfully replicated Coconut on GSM8K** - Validated the original paper's claims with GPT-2
2. **Extended to Qwen 2.5 models** - Tested Coconut with modern, more capable base models (0.5B and 3B)
3. **Implemented GRPO training** - Applied Group Relative Policy Optimization for reinforcement learning fine-tuning
4. **Explored Soft Thinking** - Developed training-free inference techniques for dynamic reasoning control

---

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/batra98/coconut.git
cd coconut

# Create conda environment
conda create --name coconut python=3.12
conda activate coconut

# Install dependencies
pip install -r requirements.txt

# Login to wandb for experiment tracking
wandb login
```

### Data Preparation

Download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset:

```bash
bash preprocessing/gsm_icot.bash
```

This creates `data/gsm_train.json` and `data/gsm_valid.json` with the required format.

---

## Experiments

**Hardware Setup**: All experiments were conducted on UW-Madison's instgpu cluster, each node equipped with 8x NVIDIA 2080Ti GPUs.

### 1. Baseline Replication: Coconut on GSM8K with GPT-2

Our first contribution - successfully replicating the original Coconut paper's results.

#### Step 1: Train CoT Baseline (Stage 0)

Train GPT-2 with explicit chain-of-thought reasoning:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

This achieves ~40% validation accuracy and serves as the initialization for Coconut training.

**Pre-trained checkpoint**: [gsm-cot-checkpoint-22](https://huggingface.co/batra98/gsm-cot-checkpoint-22) on HuggingFace

#### Step 2: Train Coconut with Continuous Thoughts

Update `args/gsm_coconut.yaml` with your CoT checkpoint path, then run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Coconut progressively replaces text reasoning steps with continuous latent thoughts through staged curriculum training:
- **Stage 0**: Full CoT (text reasoning)
- **Stage 1**: Replace 1st reasoning step with continuous thoughts
- **Stage 2**: Replace 1st and 2nd steps
- **Stage 3**: Replace 1st, 2nd, and 3rd steps

Each stage trains for 3 epochs.

**Pre-trained checkpoints**: [gsm-coconut](https://huggingface.co/batra98/gsm-coconut) on HuggingFace (checkpoints 4-25)

#### Step 3: Evaluate

Update `args/gsm_coconut_eval.yaml` with the best checkpoint path:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

---

### 2. Qwen 2.5 Experiments

We extended Coconut to work with modern, more capable base models: **Qwen 2.5-0.5B** and **Qwen 2.5-3B**.

#### Key Modifications

- **Model Architecture**: Adapted for Qwen's architecture and rotary embeddings (RoPE)
- **LoRA Integration**: Added Low-Rank Adaptation for efficient fine-tuning of larger models
- **Tokenizer Updates**: Handled Qwen's vocabulary and special tokens

#### Training Qwen 2.5-0.5B with Coconut

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut.yaml
```

**Configuration highlights** (`args/qwen_coconut.yaml`):
- `model_id`: `Qwen/Qwen2.5-0.5B`
- `lora`: `True`
- `lora_r`: 8
- `lora_target_modules`: `["q_proj", "v_proj"]`

#### Training Qwen 2.5-3B with Coconut

For the 3B model, we use more aggressive memory optimizations:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_3b_coconut.yaml
```

**Key differences**:
- Smaller batch size (8 vs 16)
- Higher gradient accumulation (4 vs 2)
- FP16 training enabled by default

---

### 3. GRPO Training

We implemented **Group Relative Policy Optimization (GRPO)** to fine-tune Coconut models using reinforcement learning with custom reward functions.

#### Overview

GRPO optimizes the model's policy based on group-relative rewards, allowing us to fine-tune for specific objectives without a separate value network.

#### Key Features

- **Dual Reward Function**: Combines answer correctness (weight: 1.0) and format adherence (weight: 0.1)
- **Group Normalization**: Rewards normalized within each batch for stable training
- **Multiple Completions**: Generates 8 completions per prompt to estimate baseline
- **TRL Integration**: Uses `GRPOTrainer` from Transformer Reinforcement Learning library

#### Running GRPO Training

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_grpo.yaml
```

**Configuration** (`args/gsm_grpo.yaml`):
- `grpo`: `True`
- `num_generations`: 8
- `beta`: 0.01 (KL penalty)
- `reward_weight_final_answer`: 1.0
- `reward_weight_format`: 0.1

---

### 4. Soft Thinking

**Soft Thinking** is a training-free inference technique that dynamically controls the length of latent reasoning chains based on entropy.

#### Overview

Unlike Coconut which requires special training, Soft Thinking is a plug-and-play wrapper that:
- Replaces discrete token sampling with probability-weighted concept tokens
- Monitors entropy of latent state distribution
- Implements dynamic halting when model is "confident" (low entropy)

#### Key Parameters

- `soft_thinking_cold_stop_threshold`: Entropy threshold for stopping (default: 0.1)
- `soft_thinking_cold_stop_patience`: Consecutive low-entropy steps required (default: 2)
- `soft_thinking_temperature`: Softmax temperature (default: 1.0)

#### Running Soft Thinking Evaluation

Evaluate a trained Coconut model with Soft Thinking:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_soft_thinking_eval.yaml
```

**Configuration** (`args/gsm_coconut_soft_thinking_eval.yaml`):
```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1
soft_thinking_cold_stop_patience: 2
soft_thinking_temperature: 1.0
```

Soft Thinking can also be applied to standard CoT models for comparison.

---

## Repository Structure

```
coconut/
├── coconut.py              # Core Coconut model implementation
├── soft_thinking.py        # Soft Thinking inference wrapper
├── run.py                  # Main training/evaluation script
├── grpo.py                # GRPO reward functions and trainer
├── dataset.py             # Dataset loading and processing
├── utils.py               # Utility functions
├── args/                  # Configuration files
│   ├── gsm_cot.yaml       # CoT baseline config
│   ├── gsm_coconut.yaml   # Coconut training config
│   ├── qwen_coconut.yaml  # Qwen 0.5B config
│   ├── qwen_3b_coconut.yaml  # Qwen 3B config
│   ├── gsm_grpo.yaml      # GRPO training config
│   └── gsm_coconut_soft_thinking_eval.yaml  # Soft Thinking eval
├── preprocessing/         # Data preprocessing scripts
│   ├── gsm_icot.bash      # GSM8K download script
│   └── gsm_icot.py        # GSM8K processing
└── data/                  # Dataset directory
```

---

## Configuration

All experiments are controlled via YAML configuration files in the `args/` directory.

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `coconut` | Enable Coconut continuous thoughts | `True` |
| `cot` | Enable Chain-of-Thought baseline | `False` |
| `grpo` | Enable GRPO training | `False` |
| `soft_thinking` | Enable Soft Thinking inference | `False` |
| `c_thought` | Number of continuous thoughts per step | `2` |
| `max_latent_stage` | Maximum curriculum training stages | `3` |
| `batch_size_training` | Batch size per GPU | `8-32` |
| `gradient_accumulation_steps` | Gradient accumulation | `2-4` |
| `lr` | Learning rate | `1e-4` |
| `model_id` | Base model | `openai-community/gpt2` or `Qwen/Qwen2.5-0.5B` |
| `lora` | Enable LoRA for Qwen models | `True` |
| `lora_r` | LoRA rank | `8` |

---

## Advanced Usage

### Multi-Node Training

For large-scale experiments across multiple nodes:

```bash
# On master node (e.g., instgpu-01)
export MASTER_ADDR=instgpu-01.cs.wisc.edu
export MASTER_PORT=29500

torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run.py args/gsm_coconut.yaml

# On worker nodes, set node_rank to 1, 2, 3, 4
```

### Debugging Mode

Set `debug: True` in your config to:
- Disable wandb logging
- Skip model checkpointing
- Use subset of data for quick iteration

### Resuming Training

Training automatically resumes from the latest checkpoint if interrupted. For manual resume:

```yaml
resume: 5  # Resume from epoch 5
load_model_path: /path/to/checkpoint_5
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Exact match of final answer (greedy decoding) |
| **CoT EM** | Exact match of entire reasoning chain |
| **Format Score** | Adherence to `<<computation=result>>` format (GRPO) |

---

## Key Implementation Details

### Coconut Architecture

The core innovation in `coconut.py`:

1. **Latent Token Injection**: Special `<|latent|>` tokens inserted in input
2. **Multi-Pass Forward**: Model processes input in multiple passes, feeding hidden states as continuous thoughts
3. **KV Cache Optimization**: Efficiently reuses computations between passes
4. **Position-Aware Processing**: Maintains proper position embeddings

### LoRA for Qwen Models

For efficient fine-tuning of larger models:
- **Rank (r)**: 8 (controls trainable parameters)
- **Alpha**: 32 (scaling factor)
- **Target Modules**: `q_proj`, `v_proj` (attention projections)
- **Parameter Efficiency**: Only ~0.1-1% of parameters are trainable

### GRPO Reward Function

Dual reward components in `grpo.py`:
- **Answer Correctness**: Binary reward (1.0 for correct, 0.0 otherwise)
- **Format Adherence**: Continuous reward for well-structured reasoning
- **Group Normalization**: Rewards normalized within batch
- **KL Penalty**: β=0.01 prevents catastrophic forgetting

### Soft Thinking Algorithm

Training-free inference modification in `soft_thinking.py`:
- **Soft Embeddings**: Probability-weighted concept tokens instead of discrete sampling
- **Entropy Monitoring**: Tracks distribution entropy at each step
- **Cold Stop**: Halts generation when entropy falls below threshold for consecutive steps

---

## Technical Stack

- **PyTorch 2.5.1** - Deep learning framework
- **Transformers 4.51.1** - HuggingFace models and tokenizers
- **TRL 0.19.0** - Transformer Reinforcement Learning
- **PEFT 0.13.2** - Parameter-Efficient Fine-Tuning (LoRA)
- **FSDP** - Fully Sharded Data Parallel for multi-GPU training
- **Wandb** - Experiment tracking and visualization

---

## Troubleshooting

### Out of Memory
- Reduce `batch_size_training`
- Increase `gradient_accumulation_steps`
- Enable `bf16: true` for memory-efficient training
- For Qwen 3B, use provided config with optimized settings

### Training Instability
- Lower learning rate (try `5e-5` or `1e-5`)
- Enable `reset_optimizer: True` when switching stages
- Check gradient clipping (max_norm=1.0)

### Soft Thinking Issues
- **Generation stops too early**: Increase `cold_stop_threshold` or `cold_stop_patience`
- **Generation too long**: Decrease `cold_stop_threshold` or `cold_stop_patience`
- **Repetitive output**: Decrease `temperature`

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
