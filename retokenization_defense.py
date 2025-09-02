import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RetokenizationDefense:
    """
    Input preprocessing defense via retokenization.
    Uses BPE-dropout if supported, else falls back to manual token shuffle.
    """

    def __init__(
        self,
        model_name: str = "/data1/SANCHAYANghosh01/llama-2-7b",  # Local LLaMA path
        dropout_prob: float = 0.4,
        num_retries: int = 3
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.dropout_prob = dropout_prob
        self.num_retries = num_retries

        if hasattr(self.tokenizer, "enable_dropout"):
            print("[Info] Fast tokenizer detected. Using BPE-dropout.")
            self.use_dropout = True
        else:
            print("[Warning] Fast tokenizer not available. Falling back to token shuffle.")
            self.use_dropout = False

    def retokenize(self, text: str) -> str:
        """
        Randomly retokenizes text:
        - If BPE-dropout is supported, applies dropout.
        - Else, shuffles or drops tokens to introduce perturbation.
        """
        text = str(text).strip()
        if not text:
            return ""

        # Optional: Debug log
        # print(f"[DEBUG] Retokenizing prompt: {repr(text)}")

        try:
            if self.use_dropout:
                self.tokenizer.enable_dropout(self.dropout_prob)
                tokens = self.tokenizer.tokenize(text)
                self.tokenizer.disable_dropout()
            else:
                tokens = self.tokenizer.tokenize(text)
                tokens = [tok for i, tok in enumerate(tokens) if i % 3 != 0]  # simple perturbation
            return self.tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            print(f"[Retokenization Error] {e} for input: {repr(text)}")
            return text  # fallback to original text if tokenization fails

    def generate(self, prompt: str, **gen_kwargs) -> str:
        rt = self.retokenize(prompt)
        inputs = self.tokenizer(rt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_best(self, prompt: str, **gen_kwargs) -> str:
        best_resp = None
        best_score = float("inf")

        for _ in range(self.num_retries):
            rt = self.retokenize(prompt)
            inputs = self.tokenizer(rt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                loss = self.model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss.item()

            output = self.model.generate(**inputs, **gen_kwargs)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

            if loss < best_score:
                best_score = loss
                best_resp = decoded

        return best_resp

    def evaluate(self, prompt: str, **gen_kwargs):
        inputs_orig = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            orig_loss = self.model(input_ids=inputs_orig["input_ids"], labels=inputs_orig["input_ids"]).loss.item()
        output_orig = self.model.generate(**inputs_orig, **gen_kwargs)
        resp_orig = self.tokenizer.decode(output_orig[0], skip_special_tokens=True)

        rt = self.retokenize(prompt)
        inputs_rt = self.tokenizer(rt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            rt_loss = self.model(input_ids=inputs_rt["input_ids"], labels=inputs_rt["input_ids"]).loss.item()
        output_rt = self.model.generate(**inputs_rt, **gen_kwargs)
        resp_rt = self.tokenizer.decode(output_rt[0], skip_special_tokens=True)

        return {
            "original_prompt": prompt,
            "retokenized_prompt": rt,
            "original_response": resp_orig,
            "retokenized_response": resp_rt,
            "original_loss": orig_loss,
            "retokenized_loss": rt_loss
        }
