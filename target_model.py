from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import LlamaForCausalLM


# Load the target generation model (LLaMA-2-7B assumed to be local or HF)
model_path = "/data1/SANCHAYANghosh01/llama-2-7b"  # change to your path if local

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

def generate_response(prompt: str, max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
