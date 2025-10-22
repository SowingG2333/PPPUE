import os
import json
import argparse
import sys
import logging
import re
import random
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# --- PROMPT 1: Anonymizer ---
PROMPT_ANONYMIZER_SYSTEM = """
You are a privacy assistant designed to rewrite any given text to hide the author’s real identity while keeping the relevant and non-private information in the text unchanged.
"""
PROMPT_ANONYMIZER_USER = """
Rewrite the [Original Comments] below. The goal is to make it seem like the author matches the [Target Persona], NOT the [Real Profile]. Use the [Clues to Fix] to identify parts of the original comments that revealed the real profile. Change ONLY relevant parts to match the [Target Persona]. Keep the rewritten text natural and coherent.

[Target Persona]:
{target_attributes_dict_str}

[Original Comments]:
{user_response}

[Real Profile]: (Attributes to Hide)
{true_attributes_dict_str}

[Clues to Fix]: (Reasoning why the original comments revealed the real profile)
{combined_feedback}

First, briefly explain your planned changes.
Then, on a new line, write ONLY a single '#'.
Then, provide ONLY the rewritten anonymized comment.

Output MUST be in this exact JSON format:
{{
    "explanation": "<Your brief explanation>",
    "anonymized_comment": "<Your rewritten comment>"
}}
"""

# --- PROMPT 2: Attacker ---
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER = """
Analyze the [Comments] below to guess the author's {attribute_name}.

[Comments]:
{user_response}

Provide your response in this exact format:

Reasoning: [Your step-by-step reasoning based ONLY on the Comments]
Guess: [Your top 3 guesses for {attribute_name}, separated by a semicolon (;). Format EACH guess according to the Value Rules. Use "unknown" if unable to infer.]
Certainty: [A number from 1 (low certainty/bias) to 5 (high certainty/evidence)]

**JSON Value Rules (CRITICAL):**
1.  **age**: MUST be an integer (e.g., 51, not "51") or string "unknown".
2.  **sex**: MUST be one of the following exact strings: "male", "female", or "unknown".
3.  **relationship_status**: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced", "widowed", or "unknown".
4.  **income_level**: MUST be one of the following exact strings: "low", "middle", "high", "very high", or "unknown".
5.  **city_country** & **birth_city_country**: Provide your best guess as a **string** in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China") if the city is unknown, or "unknown".
6.  **education** & **occupation**: Provide your best guess as a **string** (e.g., "Masters in Computer Science", "shop owner") or "unknown".
7.  If you cannot reasonably infer an attribute, you MUST use the string value "unknown".

Output ONLY the Reasoning, Guess, and Certainty lines in the specified format.
"""

# --- 辅助函数 (复制 generate_personas.py 中的 build_pipeline, parse_attacker_output, compare_profiles, call_attacker_local, call_anonymizer_local, extract_first_json_object) ---
def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。(无需修改)"""
    # ... (代码同上一个脚本) ...
    if dtype is None:
        if torch.cuda.is_available(): torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available(): torch_dtype = torch.float16
        else: torch_dtype = torch.float32
    else:
        dtype = dtype.lower()
        if dtype == "fp16": torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"): torch_dtype = torch.bfloat16
        else: torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = {"device_map": "auto"} if device is None else {}
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa", **kwargs)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device if device else None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))
    return gen_pipe, tokenizer

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取第一个完整 JSON 对象。(无需修改)"""
    # ... (代码同上一个脚本) ...
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
    try: # Fallback: simple find
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{"); end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start: return json.loads(cleaned[start:end+1])
    except Exception: return None
    return None


def parse_attacker_output(response_text: str, attribute_name: str) -> Dict[str, Any]:
    """从 IncogniText 攻击者的自由格式输出中解析。(无需修改)"""
    # ... (代码同上一个脚本) ...
    reasoning = "Parsing failed."; first_guess = None; certainty = 0
    try:
        reasoning_match = re.search(r"Reasoning:(.*?)(Guess:|Certainty:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match: reasoning = reasoning_match.group(1).strip()
        guess_match = re.search(r"Guess:(.*?)(Certainty:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if guess_match:
            guesses = [g.strip() for g in guess_match.group(1).strip().split(';') if g.strip()]
            if guesses:
                first_guess_str = guesses[0]
                if attribute_name == 'age':
                    try: first_guess = int(first_guess_str)
                    except ValueError: logging.warning(f"Failed age parse: '{first_guess_str}'"); first_guess = None
                elif attribute_name in ['sex', 'relationship_status', 'income_level']: first_guess = first_guess_str.lower()
                else: first_guess = first_guess_str
        else: logging.warning("No 'Guess:' line found.")
        certainty_match = re.search(r"Certainty:.*?(\d)", response_text, re.IGNORECASE)
        if certainty_match:
            try: certainty = int(certainty_match.group(1))
            except ValueError: certainty = 0
        else: logging.warning("No 'Certainty:' line found.")
    except Exception as e: logging.error(f"Error parsing attacker output: {e}\nOutput:\n{response_text[:500]}...", exc_info=True)
    return {"reasoning": reasoning, "first_guess": first_guess, "certainty": certainty}


def call_attacker_local(pipe, user_response: str, attribute_name: str, terminator_ids: List[int], record_id: int) -> Dict[str, Any]:
    """为单个属性调用本地 Attacker LLM 并解析结果。(无需修改)"""
    # ... (代码同上一个脚本) ...
    iteration_log_prefix = f"[Record-{record_id}] Attacking '{attribute_name}'"
    logging.debug(f"{iteration_log_prefix}...")
    user_content = PROMPT_ATTACKER_USER.format(
        attribute_name=attribute_name,
        user_response=user_response
    )
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Attacker PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.1; do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=384, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Attacker FULL OUTPUT ---\n{response_text}\n--- End ---")
        return parse_attacker_output(response_text, attribute_name)
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local attack.", exc_info=True)
        return {"reasoning": f"Local LLM Error: {e}", "first_guess": None, "certainty": 0, "error": True}


def call_anonymizer_local(pipe, user_response: str, target_attributes: Dict[str, Any], combined_feedback: str, true_attributes: Dict[str, Any], terminator_ids: List[int], record_id: int, use_true_value: bool) -> str:
    """调用多属性本地 Anonymizer LLM。(无需修改)"""
    # ... (代码同上一个脚本) ...
    iteration_log_prefix = f"[Record-{record_id}] Anonymizing..."
    logging.debug(f"{iteration_log_prefix} with target: {target_attributes}")
    attribute_names_list = ", ".join(target_attributes.keys())
    target_attributes_dict_str = json.dumps(target_attributes, indent=2, ensure_ascii=False)
    true_attributes_dict_str = json.dumps(true_attributes, indent=2, ensure_ascii=False) if use_true_value else "[REDACTED]"
    user_content = PROMPT_ANONYMIZER_USER.format(
        attribute_names_list=attribute_names_list,
        target_attributes_dict_str=target_attributes_dict_str,
        user_response=user_response,
        true_attributes_dict_str=true_attributes_dict_str,
        combined_feedback=combined_feedback if combined_feedback else "No specific cues identified. Focus on reflecting the target persona."
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Anonymizer PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.5; do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Anonymizer FULL OUTPUT ---\n{response_text}\n--- End ---")
        anon_output_json = extract_first_json_object(response_text)
        if anon_output_json and isinstance(anon_output_json, dict) and "anonymized_comment" in anon_output_json:
             explanation = anon_output_json.get("explanation", "No explanation parsed.")
             logging.debug(f"{iteration_log_prefix} Anonymizer explanation: {explanation}")
             return anon_output_json["anonymized_comment"]
        else:
            logging.warning(f"{iteration_log_prefix} Anonymizer failed to produce valid JSON or missing 'anonymized_comment'. Using raw output.")
            parts = response_text.split('#', 1)
            return parts[1].strip().strip('"').strip() if len(parts) == 2 else response_text.strip().strip('"').strip()
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local anonymization.", exc_info=True)
        return user_response


def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """使用确定性的 Python 逻辑比较 profiles。(无需修改)"""
    # ... (代码同上一个脚本) ...
    leaked_attributes: List[str] = []; _log = lambda k, g, t: f"{k} (guessed: '{g}', true: '{t}')"
    true_age = true_profile.get("age"); guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int) and abs(true_age - guessed_age) <= 3: leaked_attributes.append(_log("age", guessed_age, true_age))
    for key in ["sex", "relationship_status"]:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val and true_val.strip().lower() == guess_val.strip().lower(): leaked_attributes.append(_log(key, guess_val, true_val))
    for key in ["city_country", "birth_city_country", "education", "occupation", "income_level"]:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
            if len(guess_s) >= 3 and ((guess_s in true_s) or (true_s in guess_s)): leaked_attributes.append(_log(key, guess_val, true_val))
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- 修改后的核心函数：接收 target_persona ---
def adversarial_anonymization_multi_local(
    pipe, # transformers pipeline
    original_response: str,
    true_personality: Dict[str, Any],
    target_persona_dict: Dict[str, Any], # <--- 接收 Target Persona
    attributes_to_anonymize: List[str], # 需要知道哪些属性是目标
    terminator_ids: List[int],
    max_iterations: int = 3,
    record_id: int = -1,
    use_true_value_in_anonymizer: bool = True
) -> Tuple[str, Dict[str, Any]]:
    '''执行多属性 IncogniText (本地版)，使用预先生成的 Target Persona。'''
    record_log_prefix = f"[Record-{record_id}]"
    current_anonymized_response = original_response
    # 从 target_persona_dict 确定实际处理的属性
    valid_attributes_to_process = [attr for attr in attributes_to_anonymize if attr in target_persona_dict]

    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guesses": {}, "target_attributes": {k: target_persona_dict[k] for k in valid_attributes_to_process}, "attributes_processed": valid_attributes_to_process}

    if not valid_attributes_to_process:
        meta["status"] = "skipped_no_valid_targets_in_input"
        logging.error(f"{record_log_prefix} No valid attributes found in the provided target_persona. Skipping.")
        return original_response, meta

    logging.info(f"{record_log_prefix} Starting IncogniText for {valid_attributes_to_process}. Targets: {meta['target_attributes']}")

    # --- IncogniText 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"{record_log_prefix} [Round {i+1}/{max_iterations}]"

        # 1) 攻击所有目标属性
        attack_results = {}
        any_attack_failed = False
        for attr in valid_attributes_to_process:
            result = call_attacker_local(pipe, current_anonymized_response, attr, terminator_ids, record_id)
            if result.get("error"): any_attack_failed = True; logging.error(f"{iteration_log_prefix} Attack failed for '{attr}'. Stopping round."); meta["status"] = "model_error"; meta["error"] = f"Attacker error on '{attr}'"; break
            attack_results[attr] = result
        if any_attack_failed: break

        # 2) 检查停止条件
        combined_feedback = ""; correctly_guessed_attributes = []; keep_looping = False; final_guesses_this_round = {}
        for attr in valid_attributes_to_process:
            true_value = true_personality.get(attr); true_value_norm = str(true_value).lower() if isinstance(true_value, str) else true_value
            adversary_inference = attack_results[attr].get("first_guess"); final_guesses_this_round[attr] = adversary_inference
            logging.debug(f"{iteration_log_prefix} Comparing for '{attr}': Guess ({type(adversary_inference)}) '{adversary_inference}' vs True ({type(true_value_norm)}) '{true_value_norm}'")
            match = False
            if adversary_inference is not None:
                if isinstance(adversary_inference, str) and isinstance(true_value_norm, str): match = adversary_inference.lower() == true_value_norm.lower()
                else: match = adversary_inference == true_value_norm
            if match:
                keep_looping = True; reasoning = attack_results[attr].get("reasoning", f"No reasoning for {attr}."); combined_feedback += f"Reasoning for correctly guessing {attr} as {true_value_norm}:\n{reasoning}\n\n"; correctly_guessed_attributes.append(attr)
        meta["final_attacker_guesses"] = final_guesses_this_round
        if not keep_looping:
            logging.info(f"{iteration_log_prefix} Early Stopping: Attacker failed to guess any true attribute correctly.")
            meta["status"] = "success"; meta["final_leaked_attributes"] = compare_profiles(true_personality, final_guesses_this_round)
            return current_anonymized_response, meta
        logging.info(f"{iteration_log_prefix} Attack successful for: {correctly_guessed_attributes}. Proceeding to anonymization.")
        if not combined_feedback: combined_feedback = "Adversary guessed correctly but provided no reasoning."

        # 3) 多目标匿名化
        current_anonymized_response = call_anonymizer_local(
            pipe, current_anonymized_response, meta["target_attributes"], combined_feedback,
            true_personality, terminator_ids, record_id, use_true_value_in_anonymizer
        )

    # --- 达到最大迭代次数 ---
    if meta["status"] != "model_error":
        logging.warning(f"{record_log_prefix} Max iterations reached. Attack was still successful on the last iteration.")
        meta["status"] = "max_iterations_reached"
        meta["final_leaked_attributes"] = compare_profiles(true_personality, meta["final_attacker_guesses"])

    return current_anonymized_response, meta

# --- 修改后的 process_record (本地版) ---
def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录 (IncogniText Local version - 读取 Persona)"""
    record_log_prefix = f"[Record-{record_id}]"
    logging.debug(f"{record_log_prefix} Starting processing.")
    try:
        personality = data.get("personality")
        question = str(data.get("question_asked")) # 保留读取
        response = str(data.get("response"))
        # --- 读取预生成的目标人设 ---
        target_persona = data.get("target_persona")
        if not isinstance(target_persona, dict):
            raise ValueError("'target_persona' key is missing or not a dictionary.")
        # 确定哪些属性是目标（即 target_persona 中存在的键）
        attributes_to_anonymize = list(target_persona.keys())
        if not attributes_to_anonymize:
            raise ValueError("'target_persona' dictionary is empty.")
        # ---
    except Exception as e:
        status = "skipped_data_read_error"; logging.warning(f"{record_log_prefix} Skipped: Data read error or missing/invalid 'target_persona': {e}")
        data["anonymization_meta"] = {"status": status, "error": str(e)}; return data
    if not all([personality, question, response]): # question 仍然需要存在
        status = "skipped_incomplete_data"; logging.warning(f"{record_log_prefix} Skipped: Incomplete data.")
        data["anonymization_meta"] = {"status": status}; return data
    if not isinstance(personality, dict):
        status = "skipped_invalid_personality"; logging.warning(f"{record_log_prefix} Skipped: Invalid personality format.")
        data["anonymization_meta"] = {"status": status}; return data

    # --- 调用多属性 IncogniText (本地版)，传入 target_persona ---
    anonymized_response, meta = adversarial_anonymization_multi_local(
        pipe=pipe,
        original_response=response,
        true_personality=personality,
        target_persona_dict=target_persona, # <--- 传入 Persona
        attributes_to_anonymize=attributes_to_anonymize, # <--- 传入目标属性列表
        terminator_ids=terminator_ids,
        max_iterations=max_iterations,
        record_id=record_id,
        use_true_value_in_anonymizer=True
    )
    # --- 结束调用 ---
    data["anonymized_response"] = anonymized_response
    # 将 meta 信息合并或替换
    data["anonymization_meta"] = meta
    # 保留 target_persona 在数据中，但 meta 中也有一份
    # data["anonymization_meta"]["target_attributes"] = meta["target_attributes"]
    # data["anonymization_meta"]["attributes_processed"] = meta["attributes_processed"]

    return data

def main_run():
    parser = argparse.ArgumentParser(description="Stage 2: Run IncogniText Anonymization using pre-generated personas.")
    # --- 输入/输出参数 ---
    parser.add_argument("--input_persona_file", type=str, required=True, help="Input JSONL path (data + target_persona)")
    parser.add_argument("--output_anonymized_file", type=str, required=True, help="Output JSONL path (final anonymized data)")
    # --- 模型参数 ---
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Hugging Face model name")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu/mps/cuda:0. Default auto")
    parser.add_argument("--dtype", type=str, default=None, help="Tensor dtype: fp32/fp16/bf16. Default auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory hint")
    # --- IncogniText 参数 ---
    parser.add_argument("--max_iterations", type=int, default=3, help="Max IncogniText iterations per record")
    # --- 控制参数 ---
    parser.add_argument("--limit", type=int, default=None, help="Only process first N records")
    parser.add_argument("--success_file", type=str, default=None, help="Optional path for successful records")
    parser.add_argument("--failed_file", type=str, default=None, help="Optional path for failed/skipped records")
    # --- 日志参数 ---
    parser.add_argument("--log_file", type=str, default="run_incognitext.log", help="Log file path")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.log_file, 'w', 'utf-8')]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.getLogger("transformers").setLevel(logging.ERROR); logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}.")

    # --- 数据加载 ---
    try:
        with open(args.input_persona_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError: logging.error(f"Input persona file not found: '{args.input_persona_file}'"); sys.exit(1)
    records_to_process = [(i, json.loads(line)) for i, line in enumerate(lines) if line.strip()]
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- 模型加载 ---
    try:
        gen_pipe, tokenizer = build_pipeline(model_name=args.model_name, device=args.device, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
        # 注意：这里没有设置全局 max_new_tokens，因为 attacker 和 anonymizer 可能需要不同的值
    except Exception as e: logging.error(f"Failed to load local model '{args.model_name}': {e}", exc_info=True); sys.exit(1)
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")

    # --- IncogniText 处理 ---
    logging.info(f"Starting IncogniText anonymization for {len(records_to_process)} records...")
    results = []; counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "model_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "skipped_missing_true_value": 0, "skipped_cannot_choose_target": 0, "skipped_no_target_options": 0, "skipped_no_valid_targets": 0, "skipped_persona_generation_failed": 0, "skipped_no_valid_targets_in_persona": 0, "skipped_no_valid_targets_in_input":0, "unknown_fail": 0}
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids)
    for i, rec in enumerate(tqdm(records_to_process, desc="Running IncogniText")):
        try:
            processed_record = _task(i, rec)
            results.append(processed_record)
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            counters[status] = counters.get(status, 0) + 1
        except Exception as exc:
            logging.error(f"[Record {i}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1

    # --- 结果写入 ---
    logging.info(f"\nAnonymization complete. Writing {len(results)} results...")
    with open(args.output_anonymized_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            status = result.get("anonymization_meta", {}).get("status")
            if status in ("success", "success_on_original"): success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    # --- 记录最终摘要 ---
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in sorted(counters.items()):
        if count > 0: logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Output saved to '{args.output_anonymized_file}'.")
    if args.success_file: logging.info(f"Successful records saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped records saved to '{args.failed_file}'.")
    logging.info(f"Logs saved to '{args.log_file}'.")

if __name__ == "__main__":
    # 根据需要选择运行哪个 main 函数
    # main_generate()
    main_run()
    # 注意：实际使用时，您需要将这两个 main 函数分别放在
    # generate_personas.py 和 run_incognitext_local.py 脚本中
    # 或者添加一个命令行参数来选择运行哪个阶段。
    # 为了方便，这里暂时放在一起，需要手动取消注释来运行。