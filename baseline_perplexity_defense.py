import torch
import numpy as np

class BaselinePerplexityDefense:
    """
    Implements the baseline self-perplexity defense from the 'Baseline Defenses' paper.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The language model used to compute perplexity.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the model.
    threshold : float
        Log-perplexity threshold. Prompts exceeding this will be flagged.
    window_size : int
        Number of tokens in each window for windowed filtering.
    """
    def __init__(self, model, tokenizer, threshold=3.5, window_size=10):
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.window_size = window_size
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_log_probs(self, text):
        """Return negative log-likelihoods per token."""
        inputs = self.tokenizer(text, return_tensors="pt").to('cuda')
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        logits = outputs.logits[:, :-1]
        targets = input_ids[:, 1:]
        losses = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return losses.cpu()

    def is_adversarial(self, text):
        """Apply the basic perplexity threshold."""
        log_probs = self.compute_log_probs(text)
        return log_probs.mean().item() > self.threshold

    def is_adversarial_windowed(self, text):
        """Apply the windowed perplexity defense."""
        log_probs = self.compute_log_probs(text)
        for i in range(0, len(log_probs), self.window_size):
            window = log_probs[i:i + self.window_size]
            if window.mean().item() > self.threshold:
                return True
        return False

    def filter_batch(self, prompts):
        """Apply both normal and windowed perplexity checks on a batch of prompts."""
        normal_flags = []
        window_flags = []
        for text in prompts:
            normal_flags.append(self.is_adversarial(text))
            window_flags.append(self.is_adversarial_windowed(text))
        return normal_flags, window_flags
