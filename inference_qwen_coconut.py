"""
Qwen Coconut Inference Script
Load a trained model and run inference on custom questions.
"""

import argparse
import torch
from transformers import AutoTokenizer
from qwen_utils import load_checkpoint
from qwen_model import initialize_qwen_coconut_model


def parse_args():
    """
    Parse command-line arguments for inference.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run inference with trained Qwen Coconut model")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--question", type=str, default=None,
                        help="Question to answer (if not provided, enters interactive mode)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--latent_token", type=str, default="<|latent|>",
                        help="Latent token used during training")
    
    return parser.parse_args()


def load_trained_model(checkpoint_path, model_name, latent_token, device):
    """
    Load a trained Coconut model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint
        model_name (str): Base model name
        latent_token (str): Latent token
        device (str): Device to load on
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading trained model...")
    
    # Initialize model architecture
    lora_config = {
        "r": 8,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    model, tokenizer, _ = initialize_qwen_coconut_model(
        model_name=model_name,
        latent_token=latent_token,
        lora_config=lora_config,
        torch_dtype=torch.float16,
        device=device
    )
    
    # Load checkpoint weights
    model, epoch, loss = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens, device):
    """
    Generate answer for a given question.
    
    Args:
        model: Trained Coconut model
        tokenizer: Tokenizer
        question (str): Input question
        max_new_tokens (int): Maximum tokens to generate
        device (str): Device
        
    Returns:
        str: Generated answer
    """
    # Format prompt
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer, max_new_tokens, device):
    """
    Run interactive question-answering mode.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        max_new_tokens (int): Max tokens to generate
        device (str): Device
    """
    print("\n" + "=" * 60)
    print("Interactive Mode - Enter questions (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not question:
            continue
        
        print("\nGenerating answer...")
        answer = generate_answer(model, tokenizer, question, max_new_tokens, device)
        print(f"\n{answer}")


def main():
    """
    Main inference pipeline.
    """
    args = parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_trained_model(
        args.checkpoint_path,
        args.model_name,
        args.latent_token,
        device
    )
    
    # Single question or interactive mode
    if args.question:
        print(f"\nQuestion: {args.question}")
        answer = generate_answer(model, tokenizer, args.question, args.max_new_tokens, device)
        print(f"\nAnswer:\n{answer}")
    else:
        interactive_mode(model, tokenizer, args.max_new_tokens, device)


if __name__ == "__main__":
    main()

