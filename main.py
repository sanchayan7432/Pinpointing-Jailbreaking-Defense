import pandas as pd
import csv
import tempfile
import torch

from run_all_defenses import run_defense_pipeline_with_stats
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from legilimens_defense import LegilimensFilter, ConceptModerator
from target_model import generate_response
from evaluator import detect_attack_success

# ==== CONFIGURATION ====
original_dataset_path = "/data1/SANCHAYANghosh01/attackXdefense_score_jayesh/Defense_evaluation/datasets/HumanJailbreaks.csv"
model_name = "/data1/SANCHAYANghosh01/llama-2-7b"
moderator_path = "/data1/SANCHAYANghosh01/attackXdefense_score_jayesh/legilimens_trainer/legilimens_moderator.pt"
dataset_name = "HumanJailbreaks"
defense_type = "input"
max_iterations = 60
m_layers = 1  # Number of decoder layers for conceptual features

# ==== Step 1: Load Dataset ====
try:
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        raw_lines = f.read().splitlines()
    df = pd.DataFrame({'attack_prompt': raw_lines})
    df = df[df['attack_prompt'].str.strip().astype(bool)]  # Remove empty lines
except Exception as e:
    raise RuntimeError(f"Failed to read CSV file: {e}")

df_fixed = df.copy()

# ==== Step 2: Save Cleaned Dataset ====
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as temp_file:
    cleaned_dataset_path = temp_file.name
    df_fixed.to_csv(cleaned_dataset_path, index=False)

# ==== Step 3: Load Tokenizer ====
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ==== Step 4: Accelerate-Based Model Loading ====
print("🧠 Loading model with Accelerate for memory efficiency...")
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model.tie_weights()
model = load_checkpoint_and_dispatch(
    model,
    model_name,
    device_map="auto",
    no_split_module_classes=["LlamaDecoderLayer"]
).eval()

# ==== Step 5: Load ConceptModerator ====
print("🔎 Initializing ConceptModerator...")
with torch.no_grad():
    dummy_input = tokenizer("test", return_tensors="pt").to(model.device)
    dummy_output = model(**dummy_input, output_hidden_states=True)
    hidden_states = dummy_output.hidden_states[-m_layers:]
    feature_dim = sum(layer.shape[-1] for layer in hidden_states)

moderator = ConceptModerator(input_dim=feature_dim)
moderator.load_state_dict(torch.load(moderator_path, map_location=model.device))
moderator.eval().to(model.device)

# ==== Step 6: Initialize LegilimensFilter ====
legilimens_filter = LegilimensFilter(
    model=model,
    tokenizer=tokenizer,
    moderator=moderator,
    m=m_layers,
    moderation_type='I'  # Change to 'IO' or 'O' as needed
)
sanitize_input_fn = legilimens_filter.sanitize

# ==== Step 7: Run Defense Pipeline ====
output_file, total_successes, total_failures = run_defense_pipeline_with_stats(
    dataset_path=cleaned_dataset_path,
    model_name=model_name,
    dataset_name=dataset_name,
    llm_generate_fn=generate_response,
    detect_fn=detect_attack_success,
    sanitize_input_fn=sanitize_input_fn,
    defense_type=defense_type,
    max_iter=max_iterations
)

# ==== Step 8: Append Summary to Output CSV ====
try:
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Total Successes", total_successes])
        writer.writerow(["Total Failures", total_failures])
        writer.writerow(["Attack Success Boolean", "✔️" if total_successes > 0 else "❌"])
        writer.writerow(["Attack Failure Boolean", "✔️" if total_failures > 0 else "❌"])
    print(f"\n📊 Summary:\n✔️ Successes: {total_successes}\n❌ Failures: {total_failures}")
except Exception as e:
    print(f"[ERROR saving summary to CSV] {e}")
