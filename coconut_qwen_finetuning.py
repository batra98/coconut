!git clone https://github.com/facebookresearch/coconut.git
# %cd coconut
!pip install -r requirements.txt
!pip install transformers accelerate peft bitsandbytes datasets torchvision tqdm --upgrade --extra-index-url https://download.pytorch.org/whl/cu121

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm

from coconut import Coconut

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add latent token
latent_token = "<|latent|>"
tokenizer.add_special_tokens({"additional_special_tokens": [latent_token]})
latent_token_id = tokenizer.convert_tokens_to_ids(latent_token)

# Load model (16-bit for performance)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",         # automatically uses GPU if available
    torch_dtype=torch.float16  # use FP16 for faster training
)
base_model.resize_token_embeddings(len(tokenizer))

# LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
base_model = get_peft_model(base_model, lora_config)

# Wrap with Coconut
coconut_model = Coconut(
    base_causallm=base_model,
    latent_token_id=latent_token_id,
    start_latent_id=latent_token_id,
    end_latent_id=latent_token_id,
    eos_token_id=tokenizer.eos_token_id
).to(device)
coconut_model.train()

# =============================================
#  DATASET & PREPROCESSING
# =============================================
dataset = load_dataset("gsm8k", "main", split="train")

def preprocess(example):
    prompt = f"Question: {example['question']}\nAnswer:"
    label = example['answer']
    encoding = tokenizer(
        prompt + " " + label,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
        "question": example['question'],
        "answer": example['answer']
    }

dataset = dataset.map(preprocess)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# =============================================
#  TRAINING SETUP
# =============================================
optimizer = AdamW(coconut_model.parameters(), lr=1e-4)
num_epochs = 1
num_training_steps = num_epochs * len(dataloader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
scaler = GradScaler()

# =============================================
#  TRAINING LOOP
# =============================================
for epoch in range(num_epochs):
    print(f"\n==== Epoch {epoch+1} ====")
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        tensor_keys = ['input_ids', 'attention_mask', 'labels']
        batch_tensors = {k: torch.stack(batch[k]).to(device) for k in tensor_keys}

        input_ids = batch_tensors["input_ids"]
        attention_mask = batch_tensors["attention_mask"]
        labels = batch_tensors["labels"]

        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        optimizer.zero_grad()

        with autocast():
            outputs = coconut_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

# =============================================
#  EVALUATION (Quick accuracy check)
# =============================================
coconut_model.eval()
correct = 0
total = 0

for example in tqdm(dataset.select(range(50)), desc="Evaluating"):
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_ids = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=30
        )
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if example["answer"].strip() in gen_text.strip():
        correct += 1
    total += 1

accuracy = correct / total
print(f"\nEvaluation Accuracy: {accuracy:.4f}")

