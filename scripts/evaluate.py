import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import re
from tqdm import tqdm

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "./grpo-lora-checkpoints"

def load_model():
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer

def extract_answer(text):
    match = re.search(r"####\s*(\d+)", text)
    return match.group(1) if match else None

def evaluate(model, tokenizer, n=250):
    dataset = load_dataset("gsm8k", "main", split=f"test[:{n}]")
    correct = 0

    for sample in tqdm(dataset):
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = extract_answer(decoded)
        gt = extract_answer(sample['answer'])

        if pred == gt:
            correct += 1

    print(f"\nAccuracy on {n} samples: {correct}/{n} = {correct / n:.2%}")

if __name__ == "__main__":
    model, tokenizer = load_model()
    evaluate(model, tokenizer, n=100)