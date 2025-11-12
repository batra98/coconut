"""
Compare training configurations to understand what changed
"""
import yaml

def load_yaml(path):
    """Load YAML configuration"""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

def print_comparison():
    """Print side-by-side comparison of configurations"""
    
    old_config = load_yaml('args/qwen_coconut.yaml')
    new_config = load_yaml('args/qwen_coconut_improved.yaml')
    
    if not old_config or not new_config:
        print("Error: Could not load configuration files")
        return
    
    print("=" * 100)
    print("CONFIGURATION COMPARISON: OLD vs NEW")
    print("=" * 100)
    
    # Key parameters to compare
    comparisons = [
        ("Max Sequence Length", 
         old_config['dataset']['max_length'], 
         new_config['dataset']['max_length'],
         "Allows model to see complete reasoning chains"),
        
        ("Max Generation Tokens", 
         old_config['evaluation']['max_new_tokens'], 
         new_config['evaluation']['max_new_tokens'],
         "Allows model to generate complete answers"),
        
        ("Number of Epochs", 
         old_config['training']['num_epochs'], 
         new_config['training']['num_epochs'],
         "Prevents overfitting"),
        
        ("Batch Size", 
         old_config['training']['batch_size'], 
         new_config['training']['batch_size'],
         "Adjusted for longer sequences"),
        
        ("Learning Rate", 
         old_config['training']['learning_rate'], 
         new_config['training']['learning_rate'],
         "More stable training"),
        
        ("LoRA Rank (r)", 
         old_config['lora']['r'], 
         new_config['lora']['r'],
         "More trainable parameters"),
        
        ("LoRA Target Modules", 
         len(old_config['lora']['target_modules']), 
         len(new_config['lora']['target_modules']),
         "Better model adaptation"),
        
        ("Warmup Steps", 
         old_config['training']['warmup_steps'], 
         new_config['training']['warmup_steps'],
         "Smoother training start"),
        
        ("Scheduler Type", 
         old_config['training']['scheduler_type'], 
         new_config['training']['scheduler_type'],
         "Better convergence"),
        
        ("Gradient Accumulation", 
         old_config['training'].get('gradient_accumulation_steps', 1), 
         new_config['training']['gradient_accumulation_steps'],
         "Maintains effective batch size"),
    ]
    
    print(f"\n{'Parameter':<30} {'Old Value':<20} {'New Value':<20} {'Impact':<30}")
    print("-" * 100)
    
    for param, old_val, new_val, reason in comparisons:
        change_marker = "[!]" if old_val != new_val else "   "
        print(f"{change_marker} {param:<28} {str(old_val):<20} {str(new_val):<20} {reason:<30}")
    
    print("\n" + "=" * 100)
    print("KEY CHANGES SUMMARY")
    print("=" * 100)
    
    print("""
[CRITICAL] CRITICAL CHANGES (Fix the 9.78% accuracy issue):

1. Max Sequence Length: 128 -> 512 tokens
   - WHY: GSM8K examples are typically 200-400 tokens
   - IMPACT: Model can now see complete reasoning chains
   - RESULT: Proper learning instead of truncated patterns

2. Max Generation Tokens: 30 -> 200 tokens
   - WHY: GSM8K answers need 50-150 tokens
   - IMPACT: Model can generate complete answers
   - RESULT: Evaluation can find the final answer

3. Number of Epochs: 10 -> 3 epochs
   - WHY: Prevent overfitting on training set
   - IMPACT: Better generalization to test set
   - RESULT: Higher test accuracy

[IMPORTANT] IMPORTANT CHANGES (Improve training quality):

4. LoRA Rank: 8 -> 16
   - More trainable parameters for complex reasoning

5. LoRA Target Modules: 2 -> 4 modules
   - Better adaptation of the model

6. Gradient Accumulation: 1 -> 2 steps
   - Maintains effective batch size with longer sequences

7. Scheduler: linear -> cosine
   - Better learning rate schedule

8. Warmup Steps: 0 -> 100
   - Smoother training start

[MINOR] MINOR CHANGES (Fine-tuning):

9. Batch Size: 8 -> 4
   - Necessary due to longer sequences (memory)

10. Learning Rate: 1e-4 -> 5e-5
    - More stable training
""")
    
    print("=" * 100)
    print("EXPECTED RESULTS")
    print("=" * 100)
    
    print("""
With OLD configuration (qwen_coconut.yaml):
  Training Loss: Decreases (model learns truncated patterns)
  Training Accuracy: ~20-30% (on truncated data)
  Test Accuracy: ~10% (9.78% in your case)
  Problem: Sequences too short for proper learning

With NEW configuration (qwen_coconut_improved.yaml):
  Training Loss: Decreases properly (model learns full reasoning)
  Training Accuracy: ~60-80% (on complete data)
  Test Accuracy: ~30-50% after 3 epochs
  Test Accuracy: ~40-60% after 5 epochs
  Test Accuracy: ~50-70% after 10 epochs
  Result: Proper learning and evaluation
""")
    
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    
    print("""
1. Run diagnostics to confirm the issues:
   python diagnose_training.py

2. Test your current model's generation:
   python test_generation.py

3. Retrain with improved configuration:
   python train_improved.py

4. Evaluate on full test set:
   python test_eval.py

5. Compare results:
   - Old: 9.78% accuracy (129/1319)
   - New: Expected 30-50% accuracy (400-650/1319)
""")
    
    print("=" * 100)

if __name__ == "__main__":
    print_comparison()

