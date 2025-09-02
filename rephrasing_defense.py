import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

class RephrasingDefense:
    """
    Defend against adversarial prompts via Round-Trip Translation (RTT).
    Translates the prompt into an intermediate language and back.
    """

    def __init__(
        self,
        translater_ab: str = "Helsinki-NLP/opus-mt-en-fr",
        translater_ba: str = "Helsinki-NLP/opus-mt-fr-en",
        target_model_path: str = "/data1/SANCHAYANghosh01/llama-2-7b"
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # English → French
        self.tok_ab = AutoTokenizer.from_pretrained(translater_ab)
        self.mod_ab = AutoModelForSeq2SeqLM.from_pretrained(
            translater_ab,
            trust_remote_code=True,
            use_safetensors=True
        ).to(device)

        # French → English
        self.tok_ba = AutoTokenizer.from_pretrained(translater_ba)
        self.mod_ba = AutoModelForSeq2SeqLM.from_pretrained(
            translater_ba,
            trust_remote_code=True,
            use_safetensors=True
        ).to(device)

        # Target model: LLaMA-2
        self.tok_tgt = AutoTokenizer.from_pretrained(target_model_path, use_fast=False)
        self.mod_tgt = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            device_map="auto",           # ensures automatic GPU usage
            torch_dtype=torch.float16    # reduce memory use
        )
        self.device = device

    def _translate(self, text: str, tokenizer, model) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def roundtrip(self, prompt: str) -> str:
        """
        Perform English → French → English translation.
        """
        mid = self._translate(prompt, self.tok_ab, self.mod_ab)
        back = self._translate(mid, self.tok_ba, self.mod_ba)
        return back

    def generate(self, prompt: str, **gen_kwargs) -> tuple[str, str]:
        """
        Generate a response to the round-tripped prompt.
        Returns (cleaned_prompt, response).
        """
        cleaned = self.roundtrip(prompt)
        inputs = self.tok_tgt(cleaned, return_tensors="pt").to(self.device)
        out = self.mod_tgt.generate(**inputs, max_new_tokens=256, **gen_kwargs)
        resp = self.tok_tgt.decode(out[0], skip_special_tokens=True)
        return cleaned, resp

    def evaluate(self, prompt: str, **gen_kwargs):
        """
        Compare generation before and after rephrasing.
        Returns dict with original vs rephrased.
        """
        # Without defense
        orig_in = self.tok_tgt(prompt, return_tensors="pt").to(self.device)
        orig_out = self.mod_tgt.generate(**orig_in, max_new_tokens=256, **gen_kwargs)
        orig_resp = self.tok_tgt.decode(orig_out[0], skip_special_tokens=True)

        # With RTT
        cleaned, rtt_resp = self.generate(prompt, **gen_kwargs)

        return {
            "original_prompt": prompt,
            "original_response": orig_resp,
            "cleaned_prompt": cleaned,
            "rtt_response": rtt_resp
        }
