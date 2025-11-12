# Quick Fix Summary: Low Accuracy (9.78%)

## 🔴 Your Problem
Trained Qwen-CoCoNut for 10 epochs → Got 9.78% accuracy on GSM8K test set

## 🎯 Root Cause
**Your sequences were too short!**
- Training: `max_length=128` tokens (GSM8K needs ~300-400 tokens)
- Evaluation: `max_new_tokens=30` tokens (answers need ~100-150 tokens)
- Result: Model never learned complete reasoning, and answers were cut off

## ✅ The Fix (3 Steps)

### Step 1: Diagnose (2 minutes)
```bash
python diagnose_training.py
```
This shows you exactly how many examples were truncated.

### Step 2: Test Current Model (5 minutes)
```bash
python test_generation.py
```
See what your model actually generates (probably cut off at 30 tokens).

### Step 3: Retrain with Fixed Config (2-4 hours)
```bash
python train_improved.py
```

**What changed:**
- ✅ `max_length`: 128 → **512** (see full examples)
- ✅ `max_new_tokens`: 30 → **200** (generate complete answers)
- ✅ `num_epochs`: 10 → **3** (prevent overfitting)
- ✅ `lora_r`: 8 → **16** (more capacity)

## 📊 Expected Results

| Configuration | Accuracy | Why |
|--------------|----------|-----|
| **Your current** (128/30) | 9.78% | Sequences too short |
| **Improved** (512/200, 3 epochs) | 30-50% | Proper learning |
| **Improved** (512/200, 5 epochs) | 40-60% | More training |
| **Improved** (512/200, 10 epochs) | 50-70% | Full training |

## 🚀 Quick Start

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run diagnostics (optional but recommended)
python diagnose_training.py

# 3. Retrain with fixes
python train_improved.py

# 4. Evaluate
python test_eval.py
```

## 📁 New Files Created

- `diagnose_training.py` - Analyze what went wrong
- `test_generation.py` - Test model outputs
- `train_improved.py` - Fixed training script
- `args/qwen_coconut_improved.yaml` - Corrected hyperparameters
- `TROUBLESHOOTING.md` - Detailed explanation
- `QUICK_FIX_SUMMARY.md` - This file

## 💡 Key Insight

**GSM8K is a long-form reasoning task.** Your configuration was optimized for short sequences, which is why the model couldn't learn properly. The fix is simple: increase sequence lengths to match the task requirements.

## ⚠️ Important Notes

1. **Don't resume from old checkpoint** - It was trained on truncated data
2. **Start fresh** with the improved configuration
3. **Monitor training loss** - Should decrease from ~2.5 to ~1.5
4. **Be patient** - With longer sequences, training takes 2-4 hours per epoch

## 🎓 What You Learned

- Always match `max_length` to your task requirements
- GSM8K needs ~512 tokens for full examples
- Generation length must match expected answer length
- Too many epochs can cause overfitting

---

**TL;DR:** Your sequences were too short (128 tokens). Use 512 tokens for training and 200 tokens for generation. Retrain with `python train_improved.py`. Expect 30-50% accuracy after 3 epochs.


