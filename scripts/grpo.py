import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from accelerate import Accelerator
import re
import os

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
SFT_PATH = "./sft-lora-checkpoints"
GRPO_PATH = "./grpo-lora-checkpoints"
NUM_QUESTIONS = 150
NUM_SAMPLES = 5
LEARNING_RATE = 1e-5

def extract_answer(text):
    match = re.search(r"####\s*(\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None

def reward_fn(pred, gt):
    try:
        if pred is None or gt is None:
            return 0.0
        pred = float(pred.strip())
        gt = float(gt.strip())
        if pred == gt:
            return 1.0
        # Give partial credit if the answer is within a small margin
        relative_error = abs(pred - gt) / max(abs(gt), 1e-8)
        if relative_error <= 0.01:  # within 1%
            return 0.7
        elif relative_error <= 0.05:  # within 5%
            return 0.4
        elif relative_error <= 0.1:  # within 10%
            return 0.2
        else:
            return 0.0
    except Exception:
        return 0.0

def compute_advantages(rewards):
    r = torch.tensor(rewards)
    return ((r - r.mean()) / (r.std() + 1e-8)).tolist()

def load_models():
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    base1 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
    base2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(SFT_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    old_policy = PeftModel.from_pretrained(base1, SFT_PATH).eval()

    policy = PeftModel.from_pretrained(base2, SFT_PATH)  # Load pretrained LoRA (same as old_policy)
    policy.enable_adapter_layers()
    policy.train()
    policy.print_trainable_parameters()

    return old_policy, policy, tokenizer

def main():
    old_policy, policy, tokenizer = load_models()
    accelerator = Accelerator()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    policy, optimizer = accelerator.prepare(policy, optimizer)

    dataset = load_dataset("gsm8k", "main", split=f"train[:{NUM_QUESTIONS}]")

    for idx, sample in enumerate(dataset):
        question = sample['question']
        gt = extract_answer(sample['answer'])
        prompt = f"You are a helpful math tutor. Solve the following question step-by-step. Finish with '#### <number>'.\n\nQ: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(policy.device)

        sampled_outputs = []
        with torch.no_grad():
            for _ in range(NUM_SAMPLES):
                out = old_policy.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # reduce spam
                )
                generated = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
                sampled_outputs.append(generated)

        rewards = [reward_fn(extract_answer(o), gt) for o in sampled_outputs]
        advantages = compute_advantages(rewards)

        print(f"\n[Sample {idx}] GT Answer: {gt}")
        for i in range(NUM_SAMPLES):
            pred = extract_answer(sampled_outputs[i])
            print(f"  ▸ Sample {i+1} Reward = {rewards[i]} | Pred = {pred}")
            print("---")
            print(sampled_outputs[i])
            print("---")

        if sum(rewards) == 0:
            print("⚠️ All rewards are zero, skipping update.\n")
            continue

        for i in range(NUM_SAMPLES):
            optimizer.zero_grad()
            response_start = generated.find("A:")
            response_only = generated[response_start + 2:] if response_start != -1 else generated
            full_text = prompt + response_only
            full_input = tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(policy.device)
            output = policy(**full_input, labels=full_input["input_ids"])

            with torch.no_grad():
                ref_logits = old_policy(**full_input).logits

            kl_div = torch.nn.functional.kl_div(
                torch.log_softmax(output.logits, dim=-1),
                torch.softmax(ref_logits, dim=-1),
                reduction='batchmean'
            )

            # Proper scalar tensor for advantage
            adv = torch.tanh(torch.tensor([advantages[i]], dtype=torch.float32, device=policy.device))

            scaled_loss = -adv.squeeze(0) * output.loss
            loss_with_kl = scaled_loss + 0.01 * kl_div

            accelerator.backward(loss_with_kl)
            optimizer.step()
            
    if accelerator.is_main_process:
        os.makedirs(GRPO_PATH, exist_ok=True)
        policy.save_pretrained(GRPO_PATH)
        tokenizer.save_pretrained(GRPO_PATH)
        print(f"\n✅ GRPO training complete. Weights saved to {GRPO_PATH}")

if __name__ == "__main__":
    main()
