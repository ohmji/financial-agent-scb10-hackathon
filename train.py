import os
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets
import torch

# Configuration
model_name = "scb10x/typhoon2.1-gemma3-2b"
output_dir = "./finetuned-typhoon2b"
set_seed(42)

# STEP 1: Load and concatenate datasets from Hugging Face
ds1 = load_dataset("airesearch/WangchanThaiInstruct", split="train")
ds2 = load_dataset("PowerInfer/LONGCOT-Refine-500K", split="train")
ds3 = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")

# STEP 2: Prompt template
def format_prompt(example):
    prompt = f"""Below is an instruction. Write a helpful response in Thai.
{example['instruction']}
"""
    if example.get("input"):
        prompt += f"\n### Input:\n{example['input']}\n"
    prompt += f"\n### Response:\n{example['output']}"
    return {"text": prompt}

# Format the datasets with the prompt template before concatenation
ds1 = ds1.map(format_prompt)
ds2 = ds2.map(format_prompt)
ds3 = ds3.map(format_prompt)
dataset = concatenate_datasets([ds1, ds2, ds3])

# STEP 3: Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# STEP 4: Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# STEP 5: Tokenize dataset and remove raw columns
raw_columns = ["instruction", "input", "output"]
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048, padding="max_length"),
    batched=True,
    remove_columns=raw_columns,
    num_proc=4,
)

# STEP 6: Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    learning_rate=2e-5,
    bf16=False,
    save_total_limit=2,
    report_to="none",
    logging_dir="./logs",
)

# STEP 7: Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# STEP 8: Train model
trainer.train()

# STEP 9: Save final model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)