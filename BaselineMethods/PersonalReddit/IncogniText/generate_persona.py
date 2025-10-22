import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# --- Client Initialization ---
try:
    client = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.deepseek.com/v1" # Your DeepSeek V3 endpoint
    )
    client.models.list()
except Exception as e:
    print(f"Error: Failed to initialize API client: {e}")
    sys.exit(1)

API_LLM_MODEL = "deepseek-chat" # Your DeepSeek V3 model
GLOBAL_CLIENT = client

# --- PROMPT: Target Persona Generator (Enhanced for DeepSeek V3) ---
PROMPT_PERSONA_GENERATOR_SYSTEM = """
You are a creative privacy strategist. Your goal is to generate alternative, plausible personas based on a real user profile, ensuring the generated persona is significantly different but internally consistent and adheres to specified constraints.
"""
PROMPT_PERSONA_GENERATOR_USER = """
Analyze the [Real Profile]. Create a significantly different but plausible [Target Persona] JSON object.

[Real Profile]:
{true_profile_str}

[Attributes to Change]:
{attributes_to_change_list_str}

**Instructions & Constraints:**
1.  The [Target Persona] MUST have different values compared to the [Real Profile] for ALL attributes listed in [Attributes to Change].
2.  The [Target Persona] MUST be internally consistent (e.g., age aligns with occupation, education level aligns with occupation, location makes sense).
3.  **Specific Attribute Constraints (Apply ONLY if the attribute is in [Attributes to Change]):**
    * **age**: Must differ by at least 10 years from the real age AND be between 18 and 75.
    * **sex**: Must be the alternative binary gender (if real is male, target is female, and vice versa). Use "male" or "female".
    * **income_level**: Must be at least one level different. Use only values: "low", "middle", "high", "very high".
    * **relationship_status**: Must be different. Use only values: "single", "in a relationship", "married", "divorced", "widowed".
    * **city_country / birth_city_country**: Must be a different country if possible, or a significantly different major city within the same country if changing country is not plausible. Format as "City, Country" or "Country".
    * **occupation**: Must be in a different industry or field. Be specific.
    * **education**: Must be different and align with target age and occupation.
4.  For attributes NOT listed in [Attributes to Change], you SHOULD ideally keep the original value from the [Real Profile] unless changing them is required for internal consistency.
5.  Output ONLY a single, valid JSON object for the [Target Persona], including ALL original keys from the [Real Profile]. Do not add explanation or preamble.
"""

# --- Helper Function: Extract JSON ---
def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first complete JSON object from text."""
    start = text.find("{")
    if start == -1: return None
    depth = 0; end = -1
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end != -1:
        candidate = text[start:end+1]
        try: return json.loads(candidate)
        except Exception: pass
    try: # Fallback
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{"); end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start: return json.loads(cleaned[start:end+1])
    except Exception: return None
    return None

# --- Function: Call API to Generate Target Persona (Enhanced Validation) ---
def generate_target_persona_api(
    client: OpenAI, model: str, true_personality: Dict[str, Any],
    attributes_to_anonymize: List[str], record_id: int = -1, max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """Uses the API client (DeepSeek V3) to generate a constrained misleading Target Persona."""
    record_log_prefix = f"[Record-{record_id}]"
    logging.info(f"{record_log_prefix} Generating Target Persona via API ({model}) constrained on: {attributes_to_anonymize}")
    true_profile_str = json.dumps(true_personality, indent=2, ensure_ascii=False)
    attributes_to_change_list_str = json.dumps(attributes_to_anonymize)
    user_content = PROMPT_PERSONA_GENERATOR_USER.format(
        true_profile_str=true_profile_str,
        attributes_to_change_list_str=attributes_to_change_list_str
    )
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": PROMPT_PERSONA_GENERATOR_SYSTEM.strip()}, {"role": "user", "content": user_content}], temperature=0.7, max_tokens=512)
            out = completion.choices[0].message.content or ""
            logging.debug(f"{record_log_prefix} [Attempt {attempt+1}] --- TARGET PERSONA GENERATOR RAW OUTPUT ---\n{out}\n--- End ---")
            target_persona = extract_first_json_object(out)
            if not target_persona or not isinstance(target_persona, dict):
                logging.warning(f"{record_log_prefix} [Attempt {attempt+1}] Failed to parse valid JSON persona.")
                continue
            all_keys_present = all(key in target_persona for key in true_personality.keys())
            if not all_keys_present:
                logging.warning(f"{record_log_prefix} [Attempt {attempt+1}] Generated persona missing keys. Retrying...")
                continue
            all_different = True; mismatched_attributes = []
            for attr in attributes_to_anonymize:
                if attr not in target_persona: all_different = False; mismatched_attributes.append(f"{attr} (missing)"); break
                true_val = true_personality.get(attr); target_val = target_persona.get(attr)
                try:
                    if str(target_val).lower() == str(true_val).lower(): all_different = False; mismatched_attributes.append(attr)
                except Exception: all_different = False; mismatched_attributes.append(f"{attr} (compare error)")
            if all_different:
                 logging.info(f"{record_log_prefix} Successfully generated valid and different target persona via API after {attempt+1} attempts.")
                 return target_persona
            else:
                 logging.warning(f"{record_log_prefix} [Attempt {attempt+1}] API failed on difference constraint. Same attrs: {mismatched_attributes}. Retrying...")
                 continue
        except Exception as e:
            logging.error(f"{record_log_prefix} [Attempt {attempt+1}] API Error: {e}", exc_info=True)
    logging.error(f"{record_log_prefix} Failed after {max_retries + 1} attempts.")
    return None

# --- Main Function (Stage 1 - API) ---
def main_generate_api():
    global GLOBAL_CLIENT, API_LLM_MODEL
    parser = argparse.ArgumentParser(description="Stage 1: Generate Target Personas using API.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output_persona_file", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--model", type=str, default=API_LLM_MODEL, help=f"API model name (default: {API_LLM_MODEL})")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key")
    parser.add_argument("--base_url", type=str, default=None, help="Override API base URL.")
    parser.add_argument("--max_workers", type=int, default=10, help="Max parallel API calls.")
    parser.add_argument("--attributes_to_change", nargs='+', default=None, help="Attributes to change (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Process first N records")
    parser.add_argument("--log_file", type=str, default="generate_personas_api.log", help="Log file path")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.log_file, 'w', 'utf-8')]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.getLogger("openai").setLevel(logging.WARNING); logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}.")

    # --- Client/Model Re-initialization ---
    if args.model: API_LLM_MODEL = args.model
    if args.api_key or args.base_url:
        try:
            GLOBAL_CLIENT = OpenAI(api_key=args.api_key or os.environ.get("API_KEY"), base_url=args.base_url or "https://api.deepseek.com/v1")
            GLOBAL_CLIENT.models.list(); logging.info("API Client re-initialized.")
        except Exception as e: logging.error(f"Failed to reinitialize API client: {e}", exc_info=True); sys.exit(1)

    # --- Data Loading ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError: logging.error(f"Input file not found: '{args.input_file}'"); sys.exit(1)
    records_with_ids = [(i, json.loads(line)) for i, line in enumerate(lines) if line.strip()]
    if args.limit: records_with_ids = records_with_ids[:args.limit]

    # --- Persona Generation via API (Parallel) ---
    logging.info(f"Starting Target Persona generation via API for {len(records_with_ids)} records...")
    results = []; skipped_count = 0

    def process_persona_generation(record_tuple):
        record_id, data = record_tuple
        try:
            true_personality = data.get("personality")
            if not isinstance(true_personality, dict): data["persona_generation_status"] = "skipped_invalid_personality"; return data
            attributes_to_change = args.attributes_to_change if args.attributes_to_change else list(true_personality.keys())
            attributes_to_change = [attr for attr in attributes_to_change if attr in true_personality]
            if not attributes_to_change: data["persona_generation_status"] = "skipped_no_attributes_to_change"; return data
            target_persona = generate_target_persona_api(GLOBAL_CLIENT, API_LLM_MODEL, true_personality, attributes_to_change, record_id)
            if target_persona: data["target_persona"] = target_persona; data["persona_generation_status"] = "success"
            else: data["persona_generation_status"] = "failed"
            return data
        except Exception as exc: logging.error(f"[Record {record_id}] Error: {exc}", exc_info=True); data["persona_generation_status"] = "error"; return data

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_record = {executor.submit(process_persona_generation, record): record for record in records_with_ids}
        for future in tqdm(as_completed(future_to_record), total=len(records_with_ids), desc="Generating Personas via API"):
            try:
                processed_record = future.result()
                results.append(processed_record)
                if processed_record.get("persona_generation_status") != "success": skipped_count += 1
            except Exception as exc:
                logging.error(f"A task raised an exception: {exc}", exc_info=True)
                skipped_count += 1 # Count as skipped/failed

    # --- Write results ---
    logging.info(f"Persona generation complete. Writing {len(results)} records ({skipped_count} skipped/failed) to '{args.output_persona_file}'.")
    results.sort(key=lambda x: records_with_ids[[r[1] for r in records_with_ids].index(x)][0] if x in [r[1] for r in records_with_ids] else float('inf'))
    with open(args.output_persona_file, 'w', encoding='utf-8') as outfile:
        for result in results: outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    logging.info("Stage 1 (API Persona Generation) finished.")

if __name__ == "__main__":
    main_generate_api()