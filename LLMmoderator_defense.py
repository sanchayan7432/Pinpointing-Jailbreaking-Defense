import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple

class LLMModerator:
    """
    Implements LLM-based moderation via a Supervisor LLM.
    The generator LLM produces content, and the moderator LLM reviews it for policy violations.
    """

    def __init__(
        self,
        generator_model_name: str,
        moderator_model_name: str,
        policy_prompt_template: Optional[str] = None,
        threshold: float = 0.5,
        gen_model=None,
        gen_tokenizer=None
    ):
        """
        Parameters
        ----------
        generator_model_name : str
            Model used to generate outputs (if not preloaded).
        moderator_model_name : str
            Model used to assess safety and compliance.
        policy_prompt_template : Optional[str]
            Template to guide moderator LLM's classification.
        threshold : float
            Minimum confidence required to consider a response safe.
        gen_model : Optional
            Preloaded generator model to avoid reloading.
        gen_tokenizer : Optional
            Preloaded tokenizer to avoid reloading.
        """
        self.gen_tokenizer = gen_tokenizer or AutoTokenizer.from_pretrained(generator_model_name)
        self.gen_model = gen_model or AutoModelForCausalLM.from_pretrained(generator_model_name, torch_dtype=torch.float16).cuda()

        self.mod_tokenizer = AutoTokenizer.from_pretrained(moderator_model_name)
        self.mod_model = AutoModelForCausalLM.from_pretrained(moderator_model_name, torch_dtype=torch.float16).cuda()

        self.policy_template = policy_prompt_template or """You are a content moderation assistant.
Given the generated response and the policy below, classify whether the response is COMPLIANT or VIOLATIVE, and provide a confidence score.

Policy: {policy}
Response: {response}

Answer format:
LABEL: COMPLIANT or VIOLATIVE
CONFIDENCE: <float between 0 and 1>""".strip()

        self.threshold = threshold

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """Generate a response from the generator model."""
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.gen_model.generate(**inputs, **gen_kwargs)
        return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def moderate(self, response: str, policy: str) -> Tuple[bool, float]:
        """
        Moderate a generated response using the policy.

        Returns
        -------
        Tuple[bool, float]
            (is_safe, confidence_score)
        """
        mod_prompt = self.policy_template.format(policy=policy, response=response)
        inputs = self.mod_tokenizer(mod_prompt, return_tensors="pt").to("cuda")
        outputs = self.mod_model.generate(**inputs, max_new_tokens=64)
        reply = self.mod_tokenizer.decode(outputs[0], skip_special_tokens=True)

        label = "VIOLATIVE"
        confidence = 0.0
        for line in reply.splitlines():
            if "LABEL:" in line:
                label = line.split(":", 1)[1].strip().upper()
            if "CONFIDENCE:" in line:
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.0

        is_safe = (label == "COMPLIANT" and confidence >= self.threshold)
        return is_safe, confidence

    def generate_with_moderation(self, prompt: str, policy: str, **gen_kwargs) -> str:
        """Generate a response and pass it through moderation. If not safe, refuse output."""
        response = self.generate(prompt, **gen_kwargs)
        is_safe, confidence = self.moderate(response, policy)
        if is_safe:
            return response
        else:
            return f"⚠️ Response blocked by moderator (confidence={confidence:.2f})"
