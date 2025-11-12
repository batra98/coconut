# Troubleshooting Low Accuracy in Qwen-CoCoNut Training

## 🔴 Problem: Low Accuracy (9.78%)

You trained a Qwen-CoCoNut model for 10 epochs and achieved only **9.78% accuracy** (129/1319) on the GSM8K test set.

## 🔍 Root Causes

### 1. **Sequence Length Too Short (128 tokens)**

**Problem:**
- GSM8K problems require long reasoning chains (typically 200-400 tokens)
- Training with `max_length=128` truncates most examples
- The model never sees complete reasoning chains during training

**Evidence:**
```python
# Typical GSM8K example:
Question: "Janet's ducks lay 16 eggs per day..." (50 tokens)
Answer: "Janet sells 16 - 3 - 4 = 9 duck eggs a day..." (150+ tokens)
Total: 200+ tokens (TRUNCATED at 128!)
```

**Impact:**
- Model learns incomplete patterns
- Cannot learn full reasoning process
- Training loss may decrease but model doesn't learn the task

---

### 2. **Generation Length Too Short (30 tokens)**

**Problem:**
- During evaluation, model can only generate 30 new tokens
- GSM8K answers typically need 50-200 tokens
- Model gets cut off before producing the final answer

**Evidence:**
```python
# What happens during evaluation:
Prompt: "Question: ... Answer:" (50 tokens)
Model generates: "Let's solve this step by step..." (30 tokens - CUT OFF!)
Expected: "... Therefore, the answer is #### 42" (150 tokens needed)
```

**Impact:**
- Even if model has learned correctly, it can't finish the answer
- Evaluation metric looks for "#### [number]" at the end
- Truncated answers are marked as incorrect

---

### 3. **Too Many Epochs (10 epochs)**

**Problem:**
- 10 epochs on GSM8K training set (7,473 examples) may cause overfitting
- Model memorizes training examples instead of learning reasoning
- Doesn't generalize to test set

**Impact:**
- Training loss decreases but test accuracy stays low
- Model overfits to training data patterns

---

### 4. **Evaluation Metric Issues**

**Problem:**
- GSM8K answers have format: `"reasoning text #### 42"`
- Evaluator extracts the number after `####`
- If generation is truncated, the number never appears

**Code:**
```python
# In qwen_evaluator.py
gt_match = re.search(r'####\s*(-?\d+)', ground_truth)
# If model output is cut off, this never matches!
```

---

## ✅ Solutions

### Solution 1: Increase max_length to 512

**Change:**
```yaml
# OLD: args/qwen_coconut.yaml
dataset:
  max_length: 128

# NEW: args/qwen_coconut_improved.yaml
dataset:
  max_length: 512  # Fits 90%+ of examples
```

**Why:**
- Allows model to see complete reasoning chains
- Model learns full problem-solving process
- No information loss during training

---

### Solution 2: Increase max_new_tokens to 200

**Change:**
```yaml
# OLD: args/qwen_coconut.yaml
evaluation:
  max_new_tokens: 30

# NEW: args/qwen_coconut_improved.yaml
evaluation:
  max_new_tokens: 200  # Allows full answer generation
```

**Why:**
- Model can generate complete answers
- Evaluation metric can find the final number
- Proper assessment of model capability

---

### Solution 3: Reduce epochs to 3-5

**Change:**
```yaml
# OLD: args/qwen_coconut.yaml
training:
  num_epochs: 10

# NEW: args/qwen_coconut_improved.yaml
training:
  num_epochs: 3  # Prevent overfitting
```

**Why:**
- Reduces overfitting risk
- Faster training iterations
- Better generalization

---

### Solution 4: Adjust batch size and add gradient accumulation

**Change:**
```yaml
# OLD: args/qwen_coconut.yaml
training:
  batch_size: 8

# NEW: args/qwen_coconut_improved.yaml
training:
  batch_size: 4  # Due to longer sequences
  gradient_accumulation_steps: 2  # Effective batch size = 8
```

**Why:**
- Longer sequences (512 tokens) use more memory
- Gradient accumulation maintains effective batch size
- Stable training with limited memory

---

### Solution 5: Improve LoRA configuration

**Change:**
```yaml
# OLD: args/qwen_coconut.yaml
lora:
  r: 8
  target_modules: ["q_proj", "v_proj"]

# NEW: args/qwen_coconut_improved.yaml
lora:
  r: 16  # More capacity
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # More modules
```

**Why:**
- Higher rank = more trainable parameters
- More target modules = better adaptation
- Better performance on complex reasoning

---

## 🚀 How to Fix Your Training

### Step 1: Diagnose Current Model

Run the diagnostic script to understand the issues:

```bash
python diagnose_training.py
```

This will show:
- How many examples are truncated at 128 tokens
- Average answer length in tokens
- Whether your checkpoint exists
- Specific recommendations

### Step 2: Test Current Model Generation

See what your trained model actually generates:

```bash
python test_generation.py
```

This will:
- Show actual model outputs
- Test with different max_new_tokens (30, 50, 100, 200)
- Help you understand if the model learned anything

### Step 3: Retrain with Improved Configuration

Use the improved training script:

```bash
# Activate your environment
source venv/bin/activate

# Run improved training
python train_improved.py --config args/qwen_coconut_improved.yaml
```

This will:
- Use max_length=512 (full examples)
- Use max_new_tokens=200 (complete answers)
- Train for 3 epochs (prevent overfitting)
- Save checkpoints after each epoch
- Evaluate on both train and test sets

### Step 4: Monitor Training

Watch for these signs of healthy training:

**Good Signs:**
```
Epoch 1 Average Loss: 2.5
Epoch 2 Average Loss: 1.8
Epoch 3 Average Loss: 1.5
Training Set Accuracy: 0.65 (65%)
Test Set Accuracy: 0.45 (45%)
```

**Bad Signs:**
```
Epoch 1 Average Loss: 2.5
Epoch 2 Average Loss: 2.4
Epoch 3 Average Loss: 2.4  # Not decreasing!
Training Set Accuracy: 0.10 (10%)  # Too low
```

### Step 5: Full Evaluation

After training, run full test set evaluation:

```bash
python test_eval.py
```

Expected results with improved configuration:
- **Before:** 9.78% accuracy
- **After:** 30-50% accuracy (realistic for 3 epochs)
- **With more training:** 50-70% accuracy possible

---

## 📊 Expected Performance

### With Original Configuration (max_length=128, max_new_tokens=30):
- **Training:** Loss decreases, but model learns truncated patterns
- **Evaluation:** 5-15% accuracy (most answers cut off)
- **Issue:** Configuration prevents proper learning

### With Improved Configuration (max_length=512, max_new_tokens=200):
- **After 3 epochs:** 30-50% accuracy
- **After 5 epochs:** 40-60% accuracy
- **After 10 epochs:** 50-70% accuracy (with careful monitoring)

### Comparison to Baselines:
- **Random guessing:** ~0% (numerical answers)
- **Untrained Qwen-2.5-3B:** 10-20% (base reasoning ability)
- **Fine-tuned (proper config):** 40-70%
- **State-of-the-art:** 80-90% (larger models, more training)

---

## 🔧 Additional Improvements

### If accuracy is still low after retraining:

1. **Increase training epochs:**
   ```yaml
   training:
     num_epochs: 5  # or 10 with careful monitoring
   ```

2. **Adjust learning rate:**
   ```yaml
   training:
     learning_rate: 3e-5  # Lower for stability
     warmup_steps: 200    # More warmup
   ```

3. **Use better LoRA configuration:**
   ```yaml
   lora:
     r: 32  # Even more capacity
     alpha: 64
     target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
   ```

4. **Train on more data:**
   - Use full GSM8K training set (7,473 examples)
   - Consider data augmentation
   - Add related datasets (MATH, SVAMP)

5. **Improve evaluation:**
   - Use more sophisticated answer extraction
   - Check for numerical equivalence (42 = 42.0)
   - Handle different answer formats

---

## 📝 Quick Reference

### Files Created:
- `diagnose_training.py` - Analyze training issues
- `test_generation.py` - Test model generation quality
- `train_improved.py` - Improved training script
- `args/qwen_coconut_improved.yaml` - Fixed configuration
- `TROUBLESHOOTING.md` - This guide

### Commands:
```bash
# 1. Diagnose issues
python diagnose_training.py

# 2. Test current model
python test_generation.py

# 3. Retrain with fixes
python train_improved.py

# 4. Full evaluation
python test_eval.py
```

### Key Configuration Changes:
| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| max_length | 128 | 512 | Fit full examples |
| max_new_tokens | 30 | 200 | Complete answers |
| num_epochs | 10 | 3 | Prevent overfitting |
| batch_size | 8 | 4 | Memory for longer sequences |
| lora_r | 8 | 16 | More capacity |
| target_modules | 2 | 4 | Better adaptation |

---

## ❓ FAQ

**Q: Why is my accuracy so low (9.78%)?**
A: Your max_length (128) and max_new_tokens (30) are too short for GSM8K. The model never sees or generates complete reasoning chains.

**Q: Should I train for more epochs?**
A: Not with the current configuration. First fix max_length and max_new_tokens, then train for 3-5 epochs.

**Q: How long will retraining take?**
A: With max_length=512 and batch_size=4, expect ~2-4 hours per epoch on a single GPU (depends on GPU model).

**Q: What accuracy should I expect?**
A: With improved config and 3 epochs: 30-50%. With 5-10 epochs: 40-70%.

**Q: Can I resume from my current checkpoint?**
A: Not recommended. The model was trained on truncated data. Better to start fresh with correct configuration.

**Q: My GPU runs out of memory with max_length=512**
A: Reduce batch_size to 2 or 1, and increase gradient_accumulation_steps to maintain effective batch size.

---

## 🎯 Summary

**The Problem:**
Your model achieved 9.78% accuracy because:
1. Training sequences were truncated (128 tokens)
2. Generated answers were cut off (30 tokens)
3. Model couldn't learn or demonstrate full reasoning

**The Solution:**
1. Increase max_length to 512
2. Increase max_new_tokens to 200
3. Reduce epochs to 3-5
4. Retrain from scratch

**Expected Outcome:**
- Accuracy should improve to 30-50% (3 epochs)
- Model will generate complete, coherent answers
- Further training can reach 50-70% accuracy

**Next Steps:**
1. Run `diagnose_training.py` to confirm issues
2. Run `test_generation.py` to see current model behavior
3. Run `train_improved.py` to retrain with fixes
4. Run `test_eval.py` for full evaluation

Good luck! 🚀


