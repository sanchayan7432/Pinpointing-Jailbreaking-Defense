import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptModerator(nn.Module):
    """
    Lightweight MLP-based classifier for Legilimens conceptual feature vectors.
    Input: [m * d_model] vector
    Output: Binary classification: 0 (safe), 1 (unsafe)
    """
    def __init__(self, input_dim, hidden_dim=512, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits


class LegilimensFilter:
    """
    A Legilimens-style content moderation filter using conceptual features from decoder layers.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Host LLM (should already be device-managed, e.g., via Accelerate).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the LLM.
    moderator : ConceptModerator
        Pre-trained or fine-tuned MLP classifier on conceptual features.
    m : int
        Number of final decoder layers to use for concept extraction.
    moderation_type : str
        One of ['I', 'O', 'IO'] for I-Moderation, O-Moderation, or IO-Moderation.
    """

    def __init__(self, model, tokenizer, moderator, m=1, moderation_type='IO'):             # IO, I
        self.model = model  # Do NOT call .cuda() here
        self.tokenizer = tokenizer
        self.moderator = moderator  # Do NOT call .cuda() here
        self.moderator.eval()
        self.m = m
        assert moderation_type in ['I', 'O', 'IO']
        self.mod_type = moderation_type

    def _get_conceptual_features(self, input_text, output_text=None):
        """
        Extracts conceptual features from last m decoder layers.
        """
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""

        if self.mod_type == 'I':
            prompt = input_text
        elif self.mod_type in ['O', 'IO']:
            prompt = input_text + self.tokenizer.eos_token + output_text

        if not isinstance(prompt, str):
            raise ValueError(f"[Tokenizer Error] Expected str, got {type(prompt)}: {prompt}")

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)

        # Send input tensors to same device as model's first parameter
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        selected_layers = hidden_states[-self.m:]
        final_token_index = -1
        concept_vector = torch.cat([layer[:, final_token_index, :] for layer in selected_layers], dim=-1)

        return concept_vector.float()

    def classify(self, prompt, response=None):
        """
        Classify prompt (and optionally response) as safe/unsafe.

        Returns
        -------
        score : float
            Confidence score for unsafe class.
        prediction : bool
            True if unsafe, False if safe.
        """
        vec = self._get_conceptual_features(prompt, response)

        # Move vector to same device as moderator
        vec = vec.to(next(self.moderator.parameters()).device)

        logits = self.moderator(vec)
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, 1].item()
        prediction = score > 0.5
        return score, prediction

    def batch_classify(self, prompts, responses=None):
        results = []
        for i, prompt in enumerate(prompts):
            response = responses[i] if responses else None
            score, pred = self.classify(prompt, response)
            results.append((score, pred))
        return results

    def sanitize(self, text: str) -> str:
        """
        Input filtering: if prompt is unsafe, return a placeholder or empty string.

        Returns
        -------
        str: sanitized or original text
        """
        score, is_unsafe = self.classify(prompt=text)
        if is_unsafe:
            print(f"[LegilimensFilter] ❌ Blocked unsafe prompt (score={score:.4f})")
            return "[REDACTED PROMPT]"
        print(f"[LegilimensFilter] ✅ Safe prompt (score={score:.4f})")
        return text
