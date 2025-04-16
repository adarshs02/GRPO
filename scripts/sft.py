from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


model_name = "mistralai/Mistral-7B-v0.1"

# Load dataset
dataset = load_dataset("gsm8k", "main", split="train[:5000]")  # subset to speed up
def format_example(example):
    return {"text": f"Q: {example['question']}\nA: {example['answer']}"}
dataset = dataset.map(format_example)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Training args
args = TrainingArguments(
    output_dir="./sft-lora-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Train!
trainer.train()

# Save the LoRA adapter
model.save_pretrained("./sft-lora-checkpoints")
tokenizer.save_pretrained("./sft-lora-checkpoints")
