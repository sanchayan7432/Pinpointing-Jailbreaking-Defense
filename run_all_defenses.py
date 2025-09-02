import csv
import os
import pandas as pd
from typing import Callable, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import LlamaForCausalLM


# ========== CONFIGURABLE ENTRYPOINTS ==========
def load_defense_modules():
    """
    Load and return all necessary defense modules.
    Modify this function to load your actual defense classes.
    """
    from legilimens_defense import LegilimensFilter
    from retokenization_defense import RetokenizationDefense
    from rephrasing_defense import RephrasingDefense
    from paraphrasing_defense import ParaphrasingDefense
    from perplexity_filter import PerplexityFilter
    from LLMmoderator_defense import LLMModerator
    from llamaGuard_filter import LLMaMGuardFilter
    from wildGuard_defense import WildGuardFilter

    return {
        'legilimens': LegilimensFilter(...),
        'retokenize': RetokenizationDefense(...),
        'rephrase': RephrasingDefense(...),
        'paraphrase': ParaphrasingDefense(...),
        'perplexity': PerplexityFilter(...),
        'llm_moderator': LLMModerator(...),
        'llmaguard': LLMaMGuardFilter(...),
        'wildguard': WildGuardFilter(...)
    }

# ========== UPDATED FUNCTION WITH BOOLEAN ATTACK SUCCESS ==========
def run_defense_pipeline_with_stats(
    dataset_path: str,
    model_name: str,
    dataset_name: str,
    llm_generate_fn: Callable[[str], str],
    detect_fn: Callable[[str, str], bool],  # now returns boolean
    sanitize_input_fn: Callable[[str], str],
    defense_type: str = 'input',
    max_iter: int = 100
) -> tuple[str, int, int]:
    """
    Runs the defense pipeline and returns output file name + success/failure stats.
    """
    df = pd.read_csv(dataset_path)
    assert 'attack_prompt' in df.columns

    output_records: List[Dict[str, Any]] = []
    total_successes = 0
    total_failures = 0

    for idx in range(min(max_iter, len(df))):
        raw_prompt = df.loc[idx, 'attack_prompt']
        sanitized_prompt = sanitize_input_fn(raw_prompt) if defense_type == 'input' else raw_prompt
        llm_response = llm_generate_fn(sanitized_prompt)

        success = detect_fn(raw_prompt, llm_response)
        total_successes += int(success)
        total_failures += int(not success)

        print(f"\n[Iteration {idx+1}] Attack Prompt: {raw_prompt}")
        if defense_type == 'input':
            print(f"Sanitized Prompt: {sanitized_prompt}")
        print(f"LLM Response: {llm_response}")
        print(f"Attack Success: {'✅' if success else '❌'}")

        output_records.append({
            'attack_no': idx + 1,
            'attack_prompt': raw_prompt,
            'sanitized_prompt': sanitized_prompt if defense_type == 'input' else '',
            'llm_response': llm_response,
            'attack_success': success
        })

    # Determine file name
    defense_name = sanitize_input_fn.__name__ if defense_type == 'input' else llm_generate_fn.__name__
    output_file = f"{defense_name}_{dataset_name}.csv"

    # Write results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_records[0].keys())
        writer.writeheader()
        writer.writerows(output_records)

    print(f"\n✅ Completed. Results saved to: {output_file}")
    return output_file, total_successes, total_failures
