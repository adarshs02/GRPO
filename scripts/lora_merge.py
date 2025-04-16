import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, merge_and_unload

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "./sft-lora-checkpoints"
MERGED_OUTPUT = "./sft-merged-model"

def main():
    # Load base model + LoRA adapter
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")
    model = PeftModel.from_pretrained(base, LORA_PATH)

    # Merge LoRA weights into the base model
    print("🔧 Merging LoRA adapter into base model...")
    merged_model = merge_and_unload(model)

    # Save merged model and tokenizer
    print(f"💾 Saving merged model to: {MERGED_OUTPUT}")
    merged_model.save_pretrained(MERGED_OUTPUT)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    tokenizer.save_pretrained(MERGED_OUTPUT)

    print("✅ Done. You can now load the full merged model directly.")

if __name__ == "__main__":
    main()
