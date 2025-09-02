# Pinpointing-Jailbreaking-Defense
Large Language Models (LLMs) are increasingly deployed in critical domains such as healthcare, law, and education. However, they remain vulnerable to jailbreak attacks, where adversaries craft prompts to override alignment safeguards and force unsafe outputs.
```bash
This project provides a comprehensive framework for evaluating jailbreak attacks and defenses, combining:

(i) 16 distinct jailbreak attacks across multiple categories.
(ii) 7 defense systems including perplexity filters, rephrasing, semantic moderation, and policy guardrails.
(iii) Systematic evaluation protocol to pinpoint missing defense layers.
(iv) Extensive dataset-level success/failure analysis across thousands of adversarial queries.
```
#⚔️ Attack Categories
Jailbreak attacks are grouped into four main categories:

(i) Adversarial Suffix Appendages
  AutoPrompt, AutoDAN, GBDA, GCG, GCG-M, GCG-T, UAT
  Use optimization or suffix tokens to bypass safety filters.
(ii) Prompt Rewriting
  DirectRequest, HumanJailbreaks, PAIR, PAP, PEZ, TAP, TAP-T
  Reformulate queries to obscure intent and evade detection.
(iii) Optimized Many-Shot Prompting
  FewShot, TAP
  Overwhelm the model with malicious examples.
(iv) Malicious Content Obfuscation
  Encoding tricks, homograph substitution, syntactic distortions.

#🛡️ Defense Categories
Defenses are grouped into four primary classes:

(i) Log-Likelihood & Perplexity Filters
  Detect anomalous/unreadable adversarial suffixes.
  Example: Perplexity defense.
(ii) Prompt Sanitization & Rephrasing Modules
  Retokenization and rephrasing to break brittle adversarial tokens.
  Example: RephrasingDefense, SmoothLLM.
(iii) Content Moderation & Semantic Filtering
  Concept-level probes and classification of unsafe content.
  Example: Legilimens, LLMModerator.
(iv) Policy-Enforcing Guardrails
  Embedding safety alignment into system prompts and preference fine-tuning.
  Example: LlamaGuard, SecAlign.

  
