"""
Qwen Model Initialization Module
Handles tokenizer setup, base model loading, LoRA configuration, and Coconut wrapper.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from coconut import Coconut


def initialize_tokenizer(model_name, latent_token="<|latent|>"):
    """
    Initialize tokenizer and add latent token.
    
    Args:
        model_name (str): HuggingFace model identifier
        latent_token (str): Special token for latent thoughts
        
    Returns:
        tuple: (tokenizer, latent_token_id)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add latent token as special token
    tokenizer.add_special_tokens({"additional_special_tokens": [latent_token]})
    latent_token_id = tokenizer.convert_tokens_to_ids(latent_token)
    
    print(f"✓ Tokenizer initialized with latent token: {latent_token} (ID: {latent_token_id})")
    
    return tokenizer, latent_token_id


def load_base_model(model_name, torch_dtype=torch.float16, device_map="auto"):
    """
    Load base causal language model.
    
    Args:
        model_name (str): HuggingFace model identifier
        torch_dtype: Torch data type for model weights
        device_map (str): Device mapping strategy
        
    Returns:
        AutoModelForCausalLM: Loaded base model
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    
    print(f"✓ Base model loaded: {model_name}")
    print(f"  - Data type: {torch_dtype}")
    print(f"  - Device map: {device_map}")
    
    return base_model


def apply_lora(base_model, lora_config_dict):
    """
    Apply LoRA adapters to the base model.
    
    Args:
        base_model: Base causal language model
        lora_config_dict (dict): LoRA configuration parameters
        
    Returns:
        PeftModel: Model with LoRA adapters
    """
    lora_config = LoraConfig(
        r=lora_config_dict.get("r", 8),
        lora_alpha=lora_config_dict.get("alpha", 32),
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_config_dict.get("dropout", 0.05),
        bias=lora_config_dict.get("bias", "none"),
        task_type=lora_config_dict.get("task_type", "CAUSAL_LM")
    )
    
    model_with_lora = get_peft_model(base_model, lora_config)
    
    print(f"✓ LoRA adapters applied:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Target modules: {lora_config.target_modules}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    
    return model_with_lora


def create_coconut_model(base_model, latent_token_id, eos_token_id, device):
    """
    Wrap the base model with Coconut for continuous latent reasoning.
    
    Args:
        base_model: Base model (with or without LoRA)
        latent_token_id (int): Token ID for latent thoughts
        eos_token_id (int): End-of-sequence token ID
        device (str): Device to place model on
        
    Returns:
        Coconut: Coconut-wrapped model
    """
    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=latent_token_id,
        end_latent_id=latent_token_id,
        eos_token_id=eos_token_id
    ).to(device)
    
    print(f"✓ Coconut wrapper applied")
    print(f"  - Latent token ID: {latent_token_id}")
    print(f"  - EOS token ID: {eos_token_id}")
    print(f"  - Device: {device}")
    
    return coconut_model


def initialize_qwen_coconut_model(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    latent_token="<|latent|>",
    lora_config=None,
    torch_dtype=torch.float16,
    device="cuda"
):
    """
    Complete initialization pipeline for Qwen + LoRA + Coconut.
    
    Args:
        model_name (str): HuggingFace model identifier
        latent_token (str): Special token for latent thoughts
        lora_config (dict): LoRA configuration parameters
        torch_dtype: Torch data type for model weights
        device (str): Device to place model on
        
    Returns:
        tuple: (coconut_model, tokenizer, latent_token_id)
    """
    print("=" * 60)
    print("Initializing Qwen Coconut Model")
    print("=" * 60)
    
    # Step 1: Initialize tokenizer
    tokenizer, latent_token_id = initialize_tokenizer(model_name, latent_token)
    
    # Step 2: Load base model
    base_model = load_base_model(model_name, torch_dtype, device_map="auto")
    
    # Step 3: Resize embeddings for new token
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"✓ Token embeddings resized to {len(tokenizer)}")
    
    # Step 4: Apply LoRA if config provided
    if lora_config:
        base_model = apply_lora(base_model, lora_config)
    
    # Step 5: Wrap with Coconut
    coconut_model = create_coconut_model(
        base_model, 
        latent_token_id, 
        tokenizer.eos_token_id, 
        device
    )
    
    print("=" * 60)
    print("Model initialization complete!")
    print("=" * 60)
    
    return coconut_model, tokenizer, latent_token_id

