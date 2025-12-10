# Soft Thinking: Inference Guide

This guide explains how to run inference with the **Soft Thinking** method, which is a training-free approach to enhance reasoning in both standard Chain-of-Thought (CoT) and Coconut models.

## What is Soft Thinking?

Soft Thinking is a purely **inference-time modification** that:
- Replaces discrete token sampling with **probability-weighted concept tokens**
- Uses the model's output probability distribution to create soft embeddings
- Implements a **Cold Stop mechanism** to prevent model collapse when entropy becomes too low
- Works with any pre-trained causal language model (no retraining needed)

Unlike Coconut, which requires special training, Soft Thinking is a plug-and-play wrapper that works on existing models.

## Quick Start

### Prerequisites

1. **Environment setup** (if not already done):
```bash
cd coconut
conda activate coconut  # or your venv
pip install -r requirements.txt
```

2. **Pre-trained model**: You'll need either:
   - A standard CoT-trained model (e.g., GPT-2 fine-tuned on GSM8K)
   - A Coconut-trained model with continuous latent thoughts
   - Any pre-trained causal language model from HuggingFace

## Running Inference with Soft Thinking

### Option 1: Evaluate Coconut Model with Soft Thinking

**Create a config file** `args/gsm_coconut_soft_thinking_eval.yaml`:
```yaml
# Soft Thinking Coconut Evaluation Config
project: coconut
save_path: ./results
name: gsm-coconut-soft-thinking

only_eval: True

# Model architecture
coconut: True
cot: False
no_thoughts: False
no_cot: False

# Soft Thinking settings (inference-time only)
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1    # Stop if entropy falls below this
soft_thinking_cold_stop_patience: 2       # Consecutive low-entropy steps before stopping
soft_thinking_temperature: 1.0            # Softmax temperature (1.0 = standard)

# Model loading
model_id: openai-community/gpt2
load_model_path: /path/to/your/coconut/checkpoint
seed: 0
bf16: false

# Data paths
train_path: data/gsm_train.json
val_path: data/gsm_valid.json
batch_size_training: 1
debug: False
num_epochs: 1
lr: !!float "1e-4"
weight_decay: 0.01
```

**Run evaluation**:
```bash
# Single GPU
python run.py args/gsm_coconut_soft_thinking_eval.yaml

# Multiple GPUs (distributed)
torchrun --nproc_per_node 4 run.py args/gsm_coconut_soft_thinking_eval.yaml
```

### Option 2: Evaluate CoT Model with Soft Thinking

**Create a config file** `args/gsm_cot_soft_thinking_eval.yaml`:
```yaml
# Soft Thinking CoT Evaluation Config
project: coconut
save_path: ./results
name: gsm-cot-soft-thinking

only_eval: True

# Model architecture
coconut: False
cot: True
no_thoughts: False
no_cot: False

# Soft Thinking settings
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1
soft_thinking_cold_stop_patience: 2
soft_thinking_temperature: 1.0

# Model loading
model_id: openai-community/gpt2
load_model_path: /path/to/your/cot/checkpoint
seed: 0
bf16: false

# Data paths
train_path: data/gsm_train.json
val_path: data/gsm_valid.json
batch_size_training: 1
debug: False
num_epochs: 1
lr: !!float "1e-4"
weight_decay: 0.01
```

**Run evaluation**:
```bash
python run.py args/gsm_cot_soft_thinking_eval.yaml
```

### Option 3: Disable Soft Thinking (Baseline Comparison)

To run the **same model without Soft Thinking** for comparison:
```yaml
soft_thinking: False
```

## Understanding Soft Thinking Hyperparameters

### 1. `soft_thinking_cold_stop_threshold` (default: 0.1)

**Range**: 0.0 to 1.0 (normalized entropy)

**Effect**:
- **Lower values** (0.05): Very strict cold stop - stops generation when model becomes slightly uncertain
- **Higher values** (0.2+): Lenient cold stop - allows model to generate longer sequences
- **Default 0.1**: Balanced approach

**When to adjust**:
- Increase if generation is stopping too early
- Decrease if generation is too long or repetitive

### 2. `soft_thinking_cold_stop_patience` (default: 2)

**Range**: 1 or higher (number of consecutive steps)

**Effect**:
- **Lower values** (1): Stop immediately after one low-entropy step
- **Higher values** (3+): Require multiple consecutive low-entropy steps before stopping
- **Default 2**: Allows brief moments of low entropy without stopping

**When to adjust**:
- Increase if stopping too early despite high thresholds
- Decrease if generation continues too long

### 3. `soft_thinking_temperature` (default: 1.0)

**Range**: > 0.0 (typically 0.5 to 2.0)

**Effect**:
- **< 1.0** (e.g., 0.7): Sharper probability distribution - model commits more to top tokens
- **= 1.0**: Standard softmax (no modification)
- **> 1.0** (e.g., 1.5): Softer distribution - more uniform across vocabulary

**When to adjust**:
- Use < 1.0 for more focused/confident reasoning
- Use > 1.0 for more diverse reasoning paths

## Example Experiments

### Experiment 1: Conservative Soft Thinking
```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.05    # Very strict
soft_thinking_cold_stop_patience: 1        # Stop immediately
soft_thinking_temperature: 0.7             # Sharp distribution
```
**Expected**: Shorter, more focused answers

### Experiment 2: Relaxed Soft Thinking
```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.2     # Lenient
soft_thinking_cold_stop_patience: 4        # Allow longer reasoning
soft_thinking_temperature: 1.5             # Soft distribution
```
**Expected**: Longer, more exploratory reasoning

### Experiment 3: Standard Soft Thinking (Recommended)
```yaml
soft_thinking: True
soft_thinking_cold_stop_threshold: 0.1     # Default
soft_thinking_cold_stop_patience: 2        # Default
soft_thinking_temperature: 1.0             # Default
```
**Expected**: Balanced reasoning with natural stopping

## Understanding the Output

When running evaluation, you'll see output like:

```
Question 0: Answer = '18' CoT = 'Janet sells 3 eggs...'
Full output: 'Question: Janet's ducks... Answer: 18'
Extracted Output: '18'
...
Test accuracy: 0.45
Test CoT match: 0.38
Accuracy on validation set: 223 / 500 = 0.446
CoT match on validation set: 190 / 500 = 0.38
```

**Key metrics**:
- **Accuracy**: Whether extracted final answer matches ground truth
- **CoT match**: Whether generated reasoning exactly matches reference steps
- **For Soft Thinking**: May have different accuracy/CoT scores due to different reasoning paths

## Troubleshooting

### Issue: Generation stops too early
**Solution**: 
- Increase `soft_thinking_cold_stop_threshold` (e.g., 0.15)
- Increase `soft_thinking_cold_stop_patience` (e.g., 3)
- Increase `soft_thinking_temperature` (e.g., 1.2)

### Issue: Generation is too long or repetitive
**Solution**:
- Decrease `soft_thinking_cold_stop_threshold` (e.g., 0.08)
- Decrease `soft_thinking_cold_stop_patience` (e.g., 1)
- Decrease `soft_thinking_temperature` (e.g., 0.8)

### Issue: Out of memory
**Solution**:
- Make sure `only_eval: True` (no training)
- Use smaller batch size (already 1 in eval configs)
- Check if model checkpoint is loading correctly

### Issue: Model not found
**Solution**:
- Verify `load_model_path` points to correct checkpoint
- Or use `model_id: openai-community/gpt2` to use pre-trained HF model
- Check file permissions

## Advanced Usage: Custom Inference Script

You can also use Soft Thinking directly in your own code:

```python
from soft_thinking import SoftThinking
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Prepare input
prompt = "Janet's ducks lay 16 eggs per day..."
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate with Soft Thinking
outputs = SoftThinking.generate(
    model,
    tokenizer,
    input_ids,
    attention_mask=torch.ones_like(input_ids),
    max_new_tokens=64,
    cold_stop_threshold=0.1,
    cold_stop_patience=2,
    temperature=1.0,
)

# Decode and print
print(tokenizer.decode(outputs[0]))
```

## Comparing Methods

To compare CoT, Coconut, and Soft Thinking:

```bash
# 1. Baseline CoT
python run.py args/gsm_cot_eval.yaml > results_cot.txt

# 2. Coconut (standard)
python run.py args/gsm_coconut_eval.yaml > results_coconut.txt

# 3. Coconut + Soft Thinking
python run.py args/gsm_coconut_soft_thinking_eval.yaml > results_soft_thinking.txt

# 4. CoT + Soft Thinking
python run.py args/gsm_cot_soft_thinking_eval.yaml > results_cot_soft_thinking.txt
```

Then compare the accuracy scores from each output.

## References

- **Soft Thinking Paper**: [arXiv:2505.15778](https://arxiv.org/abs/2505.15778)
- **Coconut Paper**: [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- **Implementation**: See `soft_thinking.py` for core algorithm details

## Questions?

Refer to the main [README.md](README.md) for general project information or check the issue tracker for common problems.
