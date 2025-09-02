import torch
import numpy as np

class SecAlignULLFilter:
    """
    Filter sequences based on Universal Logit Likelihood (ULL) as proposed in SecAlign.
    
    Parameters
    ----------
    model : transformers.PreTrainedModel
        Language model used to compute log-probabilities.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the LM.
    threshold : float
        Threshold for mean negative log-likelihood (NLL). Lower is better (more fluent).
    system_prompt : str
        The defensive system prompt used as a secure prefix.
    """
    def __init__(self, model, tokenizer, threshold=3.5, system_prompt="You are a helpful and harmless assistant."):
        self.model = model.cuda()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.system_prompt = system_prompt
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_ull(self, user_prompt):
        """
        Compute ULL score of a user prompt, conditioned on the system prompt.

        Parameters
        ----------
        user_prompt : str

        Returns
        -------
        float : mean log-likelihood (NLL) of the user prompt continuation.
        """
        full_input = self.system_prompt + "\n\n" + user_prompt
        enc = self.tokenizer(full_input, return_tensors='pt').to('cuda')
        input_ids = enc['input_ids']

        # Identify the token positions that correspond to user_prompt
        sys_prompt_ids = self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
        prefix_len = len(sys_prompt_ids) + 2  # +2 for \n\n
        labels = input_ids.clone()
        labels[:, :prefix_len] = -100  # mask out system prompt in loss

        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

        return loss.item()

    def filter(self, prompts):
        """
        Filter a list of prompts based on ULL.

        Parameters
        ----------
        prompts : list of str

        Returns
        -------
        scores : list of float
            ULL scores for each prompt.
        passed_filter : list of bool
            Whether each prompt passed the ULL threshold.
        """
        scores = []
        passed_filter = []

        for prompt in prompts:
            score = self.compute_ull(prompt)
            scores.append(score)
            passed_filter.append(score <= self.threshold)

        return scores, passed_filter
