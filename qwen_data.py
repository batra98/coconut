"""
Qwen Data Preprocessing Module
Handles dataset loading, preprocessing, and DataLoader creation for GSM8K and other datasets.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def create_prompt(question, prompt_template="Question: {question}\nAnswer:"):
    """
    Create a formatted prompt from a question.
    
    Args:
        question (str): The input question
        prompt_template (str): Template string with {question} placeholder
        
    Returns:
        str: Formatted prompt
    """
    return prompt_template.format(question=question)


def preprocess_example(example, tokenizer, max_length=128, prompt_template="Question: {question}\nAnswer:"):
    """
    Preprocess a single example from the dataset.
    
    Args:
        example (dict): Dataset example with 'question' and 'answer' keys
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        prompt_template (str): Template for formatting the prompt
        
    Returns:
        dict: Preprocessed example with input_ids, attention_mask, labels
    """
    # Create prompt and combine with answer
    prompt = create_prompt(example['question'], prompt_template)
    full_text = prompt + " " + example['answer']
    
    # Tokenize
    encoding = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),  # For causal LM, labels are same as input_ids
        "question": example['question'],
        "answer": example['answer']
    }


def load_gsm8k_dataset(split="train", tokenizer=None, max_length=128, prompt_template="Question: {question}\nAnswer:"):
    """
    Load and preprocess GSM8K dataset.
    
    Args:
        split (str): Dataset split ('train', 'test')
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        prompt_template (str): Template for formatting the prompt
        
    Returns:
        Dataset: Preprocessed dataset
    """
    print(f"Loading GSM8K dataset (split: {split})...")
    dataset = load_dataset("gsm8k", "main", split=split)
    print(f"✓ Loaded {len(dataset)} examples")
    
    if tokenizer is not None:
        print("Preprocessing dataset...")
        dataset = dataset.map(
            lambda example: preprocess_example(
                example, 
                tokenizer, 
                max_length, 
                prompt_template
            )
        )
        print("✓ Preprocessing complete")
    
    return dataset


def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    """
    Create a DataLoader from a preprocessed dataset.
    
    Args:
        dataset: Preprocessed dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    print(f"✓ DataLoader created:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Shuffle: {shuffle}")
    print(f"  - Total batches: {len(dataloader)}")
    
    return dataloader


def prepare_batch(batch, device):
    """
    Prepare a batch for training by moving tensors to device.
    
    Args:
        batch (dict): Batch from DataLoader
        device (str): Device to move tensors to
        
    Returns:
        dict: Batch with tensors on specified device
    """
    tensor_keys = ['input_ids', 'attention_mask', 'labels']
    batch_tensors = {
        k: torch.stack(batch[k]).to(device) 
        for k in tensor_keys
    }
    
    return batch_tensors


def create_position_ids(input_ids, device):
    """
    Create position IDs for the input sequence.
    
    Args:
        input_ids (torch.Tensor): Input token IDs (batch_size, seq_len)
        device (str): Device to create tensor on
        
    Returns:
        torch.Tensor: Position IDs (batch_size, seq_len)
    """
    position_ids = torch.arange(
        input_ids.shape[1], 
        dtype=torch.long, 
        device=device
    ).unsqueeze(0).expand(input_ids.shape[0], -1)
    
    return position_ids


def load_and_prepare_data(
    dataset_name="gsm8k",
    split="train",
    tokenizer=None,
    max_length=128,
    batch_size=8,
    shuffle=True,
    prompt_template="Question: {question}\nAnswer:"
):
    """
    Complete data loading and preparation pipeline.
    
    Args:
        dataset_name (str): Name of the dataset
        split (str): Dataset split
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        prompt_template (str): Template for formatting the prompt
        
    Returns:
        tuple: (dataset, dataloader)
    """
    print("=" * 60)
    print("Loading and Preparing Data")
    print("=" * 60)
    
    # Load and preprocess dataset
    if dataset_name == "gsm8k":
        dataset = load_gsm8k_dataset(split, tokenizer, max_length, prompt_template)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size, shuffle)
    
    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    
    return dataset, dataloader

