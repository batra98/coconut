# Coconut with Qwen 2.5-3B

This branch implements **Coconut (Continuous Latent Thoughts)** on top of the **Qwen 2.5-3B** model. It leverages the stronger reasoning capabilities of Qwen 2.5 compared to the original GPT-2 baseline.

## Key Features

- **Base Model**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Efficient Fine-tuning**: Integrated **LoRA (Low-Rank Adaptation)** support for memory-efficient training.
- **Optimized Architecture**: Adapted Coconut's multi-pass latent reasoning for Qwen's architecture.
- **FP16 Training**: Enabled by default for faster throughput.

## Installation

```bash
# Clone the repository
git clone https://github.com/batra98/coconut.git
cd coconut
git checkout qwen2.5-3B

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train Coconut with Qwen 2.5-3B on GSM8K:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut.yaml
```

**Configuration (`args/qwen_coconut.yaml`):**
- `lora.r`: 8 (Rank for LoRA adapters)
- `lora.target_modules`: `["q_proj", "v_proj"]`
- `training.batch_size`: 8
- `training.learning_rate`: 1e-4

### Improved Training Configuration

For better stability and performance, we recommend using the improved configuration:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut_improved.yaml
```

**Key Improvements:**
- Increased context length (512 tokens)
- Optimized learning rate schedule (Cosine)
- Enhanced LoRA configuration (Rank 16, more target modules)
- Gradient accumulation for larger effective batch sizes

## Evaluation

To evaluate the trained model:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml \
    --load_model_path ./checkpoints/qwen_coconut/checkpoint_best
```

## Model Architecture

The implementation wraps Qwen 2.5 with Coconut's continuous thought mechanism:
1. **Latent Tokens**: `<|latent|>` tokens are injected to represent continuous thoughts.
2. **Continuous Reasoning**: The model processes these latent tokens to generate hidden states that guide subsequent generation.
3. **LoRA Adapters**: Fine-tuning is applied only to LoRA adapters and the new latent token embeddings, preserving the pre-trained knowledge of Qwen 2.5.
