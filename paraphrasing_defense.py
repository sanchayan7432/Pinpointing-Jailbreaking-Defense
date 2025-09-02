import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer as CausalTokenizer
)
from typing import List

class ParaphrasingDefense:
    """
    Defense strategy that paraphrases user prompts before feeding them into the model,
    as described in Baseline Defenses for Adversarial Attacks Against Aligned Language Models :contentReference[oaicite:3]{index=3}.
    """

    def __init__(
        self,
        paraphraser_name: str = "t5-base",
        target_model_name: str = "gpt2",
        paraphrase_beams: int = 5,
        paraphrase_max_length: int = 128,
        paraphrase_min_length: int = 20
    ):
        # Paraphraser (seq2seq) model
        self.parap_tokenizer = AutoTokenizer.from_pretrained(paraphraser_name)
        self.parap_model = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_name).cuda()

        # Target language model
        self.target_tokenizer = CausalTokenizer.from_pretrained(target_model_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).cuda()

        self.paraphrase_beams = paraphrase_beams
        self.paraphrase_max_length = paraphrase_max_length
        self.paraphrase_min_length = paraphrase_min_length

    def paraphrase(self, prompt: str) -> str:
        """Generate a paraphrased version of the prompt."""
        text = "paraphrase: " + prompt.strip()
        inputs = self.parap_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.paraphrase_max_length
        ).to("cuda")
        outputs = self.parap_model.generate(
            **inputs,
            max_length=self.paraphrase_max_length,
            min_length=self.paraphrase_min_length,
            num_beams=self.paraphrase_beams,
            early_stopping=True,
            do_sample=False
        )
        paraphrased = self.parap_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """
        Paraphrases the input prompt, then generates using the target model.
        Returns: paraphrased input, and generated response.
        """
        paraphrased = self.paraphrase(prompt)
        # Generate response using the target model
        inputs = self.target_tokenizer(paraphrased, return_tensors="pt").to("cuda")
        out = self.target_model.generate(**inputs, **gen_kwargs)
        generated = self.target_tokenizer.decode(out[0], skip_special_tokens=True)
        return paraphrased, generated

    def evaluate(self, original_prompt: str, **gen_kwargs):
        """
        For analysis, returns:
        - original response
        - paraphrased prompt
        - paraphrased response
        """
        # Original
        inputs = self.target_tokenizer(original_prompt, return_tensors="pt").to("cuda")
        orig_out = self.target_model.generate(**inputs, **gen_kwargs)
        orig_resp = self.target_tokenizer.decode(orig_out[0], skip_special_tokens=True)

        # Paraphrased
        paraphrased, paraphrased_resp = self.generate(original_prompt, **gen_kwargs)

        return {
            "original_prompt": original_prompt,
            "original_response": orig_resp,
            "paraphrased_prompt": paraphrased,
            "paraphrased_response": paraphrased_resp,
        }