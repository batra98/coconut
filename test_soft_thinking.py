#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Simple test script to verify Soft Thinking implementation.

This script tests Soft Thinking on a small prompt without requiring
the full training/evaluation pipeline. It's useful for quick verification
that the method works correctly.

Usage:
    python test_soft_thinking.py [--model_id MODEL] [--temperature TEMP]
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from soft_thinking import SoftThinking


def test_soft_thinking_basic():
    """Test Soft Thinking with default GPT-2."""
    print("=" * 80)
    print("Testing Soft Thinking with GPT-2")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load model and tokenizer
    print("Loading GPT-2...")
    model_id = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    prompts = [
        "Q: What is 2 + 2? A:",
        "Q: Janet's ducks lay 16 eggs per day. She sells 3 eggs at the farmers market. She uses 4+16=20 eggs to bake muffins. How many eggs does she have left? A:",
    ]

    # Test configurations
    configs = [
        {
            "name": "Standard (default)",
            "temperature": 1.0,
            "cold_stop_threshold": 0.1,
            "cold_stop_patience": 2,
        },
        {
            "name": "Conservative (sharp, short)",
            "temperature": 0.7,
            "cold_stop_threshold": 0.05,
            "cold_stop_patience": 1,
        },
        {
            "name": "Exploratory (soft, long)",
            "temperature": 1.3,
            "cold_stop_threshold": 0.2,
            "cold_stop_patience": 3,
        },
    ]

    model.eval()
    with torch.no_grad():
        for config in configs:
            print(f"\n{'=' * 80}")
            print(f"Configuration: {config['name']}")
            print(f"  Temperature: {config['temperature']}")
            print(f"  Cold Stop Threshold: {config['cold_stop_threshold']}")
            print(f"  Cold Stop Patience: {config['cold_stop_patience']}")
            print("=" * 80)

            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                print("-" * 80)

                # Tokenize prompt
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                attention_mask = torch.ones_like(input_ids).to(device)

                # Generate with Soft Thinking
                outputs = SoftThinking.generate(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    max_new_tokens=32,
                    device=device,
                    cold_stop_threshold=config["cold_stop_threshold"],
                    cold_stop_patience=config["cold_stop_patience"],
                    temperature=config["temperature"],
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode and display
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated_text.split("A:")[-1].strip()

                print(f"Full output: {generated_text}")
                print(f"Extracted answer: {answer}")
                print(f"Num tokens generated: {outputs.shape[1] - input_ids.shape[1]}")


def test_soft_thinking_vs_standard():
    """Compare Soft Thinking vs standard generation."""
    print("\n" + "=" * 80)
    print("Comparing Soft Thinking vs Standard Generation")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load model
    print("Loading GPT-2...")
    model_id = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Q: What is the capital of France? A:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    print(f"Prompt: {prompt}\n")

    model.eval()
    with torch.no_grad():
        # Standard generation (greedy)
        print("Standard Generation (greedy decoding):")
        print("-" * 80)
        standard_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
        )
        standard_text = tokenizer.decode(standard_outputs[0], skip_special_tokens=True)
        print(f"Output: {standard_text}")
        print(f"Num tokens: {standard_outputs.shape[1]}")

        # Soft Thinking generation
        print("\nSoft Thinking Generation:")
        print("-" * 80)
        soft_outputs = SoftThinking.generate(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            max_new_tokens=32,
            device=device,
            cold_stop_threshold=0.1,
            cold_stop_patience=2,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        soft_text = tokenizer.decode(soft_outputs[0], skip_special_tokens=True)
        print(f"Output: {soft_text}")
        print(f"Num tokens: {soft_outputs.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Test Soft Thinking implementation")
    parser.add_argument(
        "--model_id",
        default="openai-community/gpt2",
        help="Model ID from HuggingFace (default: openai-community/gpt2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax (default: 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Cold stop threshold (default: 0.1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Cold stop patience (default: 2)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Max new tokens to generate (default: 32)",
    )

    args = parser.parse_args()

    # Check for CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Generation will be slow on CPU.")

    # Run tests
    test_soft_thinking_basic()
    test_soft_thinking_vs_standard()

    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
