import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple

class LLMaMGuardFilter:
    """
    Output filtering using LlamaGuard: classifies model-generated responses as SAFE or UNSAFE.
    Optimized to prevent CUDA OOM using device_map and float16.
    """

    def __init__(self, guard_model_name: str = "llamas-community/LlamaGuard-7b"):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(guard_model_name)

        # Load model with float16 precision and automatic device placement
        self.model = AutoModelForCausalLM.from_pretrained(
            guard_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def classify_response(self, response: str, taxonomy_instructions: str = None) -> Tuple[bool, str]:
        """
        Classify a given response as SAFE or UNSAFE using LlamaGuard.

        Returns:
            (is_safe, raw_output)
        """
        prompt = (taxonomy_instructions or "Classify the following response as SAFE or UNSAFE:\n") + response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)

        verdict = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
        is_safe = verdict.startswith("SAFE")
        return is_safe, verdict

    def filter_and_refuse(self, user_prompt: str, response: str, taxonomy_instructions: str = None) -> str:
        """
        Applies classification; returns response or refusal message if unsafe.
        """
        safe, verdict = self.classify_response(response, taxonomy_instructions)
        if safe:
            return response
        else:
            return f"⚠️ Response blocked by guard (verdict: {verdict})"
