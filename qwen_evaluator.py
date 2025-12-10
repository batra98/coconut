"""
Qwen Evaluation Module
Handles model evaluation, generation, and accuracy calculation.
"""

import torch
from tqdm import tqdm


def generate_answer(model, input_ids, attention_mask, max_new_tokens=30, device="cuda"):
    """
    Generate answer for a single example.
    
    Args:
        model: Coconut model
        input_ids (torch.Tensor): Input token IDs
        attention_mask (torch.Tensor): Attention mask
        max_new_tokens (int): Maximum number of tokens to generate
        device (str): Device to run on
        
    Returns:
        torch.Tensor: Generated token IDs
    """
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
    
    return generated_ids


def check_answer_match(generated_text, ground_truth):
    """
    Check if the generated answer matches the ground truth.
    
    Args:
        generated_text (str): Generated answer text
        ground_truth (str): Ground truth answer
        
    Returns:
        bool: True if answer is correct
    """
    import re
    
    # Extract final numerical answer from ground truth (format: "#### 123")
    gt_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', ground_truth)
    if gt_match:
        gt_answer = gt_match.group(1).replace(',', '')
    else:
        # Fallback: just use the ground truth as-is
        gt_answer = ground_truth.strip()
    
    # Check if the numerical answer appears in generated text
    # Look for the number with or without commas
    gen_text_clean = generated_text.replace(',', '')
    return gt_answer in gen_text_clean or gt_answer in generated_text


def evaluate_model(
    model, 
    dataset, 
    tokenizer, 
    num_samples=50, 
    max_new_tokens=30, 
    device="cuda",
    verbose=True
):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Coconut model to evaluate
        dataset: Dataset with preprocessed examples
        tokenizer: Tokenizer for decoding
        num_samples (int): Number of samples to evaluate
        max_new_tokens (int): Maximum tokens to generate per example
        device (str): Device to run on
        verbose (bool): Whether to show progress bar
        
    Returns:
        dict: Evaluation results with accuracy and metrics
    """
    model.eval()
    correct = 0
    total = 0
    
    # Select subset of dataset
    eval_dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Iterate through examples
    iterator = tqdm(eval_dataset, desc="Evaluating") if verbose else eval_dataset
    
    for idx, example in enumerate(iterator):
        # Prepare input
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
        
        # Generate answer
        generated_ids = generate_answer(
            model, 
            input_ids, 
            attention_mask, 
            max_new_tokens, 
            device
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Debug: print first 3 examples
        if verbose and idx < 3:
            print(f"\n--- Example {idx+1} ---")
            print(f"Question: {example['question'][:100]}...")
            print(f"Generated: {generated_text[-100:]}")
            print(f"Ground truth: {example['answer'][:100]}...")
        
        # Check if correct
        if check_answer_match(generated_text, example["answer"]):
            correct += 1
        
        total += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
    
    return results


def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results (dict): Evaluation results
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print("=" * 60)


def evaluate_and_report(
    model,
    dataset,
    tokenizer,
    num_samples=50,
    max_new_tokens=30,
    device="cuda"
):
    """
    Complete evaluation pipeline with reporting.
    
    Args:
        model: Coconut model to evaluate
        dataset: Dataset with preprocessed examples
        tokenizer: Tokenizer for decoding
        num_samples (int): Number of samples to evaluate
        max_new_tokens (int): Maximum tokens to generate per example
        device (str): Device to run on
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print(f"Evaluating on {num_samples} samples...")
    
    results = evaluate_model(
        model,
        dataset,
        tokenizer,
        num_samples,
        max_new_tokens,
        device,
        verbose=True
    )
    
    print_evaluation_results(results)
    
    return results

