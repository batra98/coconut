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

1. **Successfully replicated Coconut on GSM8K** - Validated the original paper's claims
2. **Implemented GRPO training** - Applied Group Relative Policy Optimization for reinforcement learning fine-tuning
3. **Added Pass@K evaluation** - Implemented Pass@20 metric for both vanilla CoT and Coconut models with temperature and nucleus sampling
4. **Experimented with Qwen 2.5-3B** - Tested Coconut architecture with different base models
5. **Soft Thinking experiments** - Explored alternative training strategies and smoothing techniques

---

## Quick Start

### Environment Setup

**Option 1: Using Conda**

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

**Option 2: Using uv (Faster Alternative)**

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

### 2. GRPO Training (Reinforcement Learning)

Branch: `sra/grpo`

We implemented Group Relative Policy Optimization to fine-tune Coconut using reinforcement learning with custom reward functions.

```bash
torchrun --nnodes 1 --nproc_per_node 4 run_grpo.py args/gsm_grpo.yaml
```

**Key Features:**
- Custom reward function combining answer correctness (weight: 1.0) and format adherence (weight: 0.1)
- Group-based reward scaling for stable training
- Generates 8 completions per prompt for robust policy updates
- Integrated with TRL's `GRPOTrainer`

**Reward Function:**
```python
reward = 1.0 * correct_answer_reward + 0.1 * format_reward
```

The format reward encourages the model to maintain proper reasoning structure with `<<computation=result>>` format.

### 3. Pass@K Evaluation

**Branch: `sra/pass_k`** - For vanilla CoT models  
**Branch: `sra/pass_k_coconut`** - For Coconut models with continuous thoughts

Standard greedy decoding only generates one answer. Pass@K evaluation generates K diverse samples and measures if any of them are correct - a more robust metric for reasoning tasks.

**Implementation:**
- Generates 20 samples per problem with temperature sampling (temperature=0.7, top_p=0.95)
- Checks if at least one sample produces the correct answer
- Reports Pass@1 (greedy), Pass@20 (sampling), and CoT match metrics

**Key Features:**
- `sra/pass_k`: Implements Pass@20 evaluation for vanilla models
- `sra/pass_k_coconut`: Extends Coconut's `generate()` method with sampling support
  - Adds `_sample_token()` method with temperature and nucleus (top-p) sampling
  - Enables diverse output generation while maintaining quality
  - Backward compatible with greedy decoding

**Usage:**
```bash
# For Coconut models - checkout sra/pass_k_coconut branch
git checkout sra/pass_k_coconut

# Run evaluation with Pass@20
torchrun \
  --nnodes 5 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr instgpu-01.cs.wisc.edu \
  --master_port 29500 \
  run.py args/gsm_coconut_eval.yaml
```

**Metrics Reported:**
- `eval/acc` - Standard greedy accuracy
- `eval/cot_em` - Chain-of-thought exact match
- `eval/pass@20` - Success rate with 20 samples per problem

### 4. Qwen 2.5-3B Experiments

Branch: `qwen2.5-3B`

We adapted Coconut to work with Qwen 2.5-3B, a more capable base model than GPT-2.

**Changes:**
- Modified model loading to support Qwen architecture
- Adjusted tokenizer handling for Qwen's vocabulary
- Updated attention mechanisms and position embeddings

### 5. Soft Thinking Experiments

Branch: `soft-thinking2`

We explored "Soft Thinking" - a technique to dynamically control the length of latent reasoning chains during inference based on entropy.

**To run Soft Thinking evaluation:**

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_soft_thinking_eval.yaml
```

**Key Features:**
- **Dynamic Halting**: Stops generating latent thoughts when the model is "confident" (low entropy).
- **Entropy-based Control**: Monitors the entropy of the latent state distribution.
- **Configurable Parameters**:
  - `soft_thinking_cold_stop_threshold`: Entropy threshold for stopping (e.g., 0.1).
  - `soft_thinking_cold_stop_patience`: Number of consecutive low-entropy steps required.

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

```yaml
resume: 5  # Resume from epoch 5
load_model_path: /path/to/checkpoint_5
```

---

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
