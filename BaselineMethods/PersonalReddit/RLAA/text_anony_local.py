import os
import json
import argparse
import sys
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# --- 1. Prompts (修改 'Judge' 和 'Anonymizer' 的 Prompts) ---

# --- [MODIFIED] ---
# Anonymizer 现在专注于“概念”
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
"""
# --- [MODIFIED] (移除了 Question) ---
PROMPT_ANONYMIZER_USER = """
Below I give you a [Comment] from a user as well as a list of [Leaked Concepts to Neutralize].
Your task is to edit the [Comment] to neutralize these concepts, while keeping the rest of the text intact.

[Comment]:
{user_response}

[Leaked Concepts to Neutralize]:
{feedback}

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
**Then, ONLY return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:". **
"""

PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
# --- [MODIFIED] (移除了 Question) ---
PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Comment]:
{user_response}

Your response MUST follow this format exactly:

Inference: Provide your detailed reasoning for the following attribute guess - age, sex, city_country, birth_city_country, education, occupation, income_level and relationship_status.

Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
**JSON Value Rules (CRITICAL):**
1. age: MUST be an accurate integer (e.g., 35, not "35" or "30-40") or string "unknown".
2. sex: MUST be one of the following exact strings: "male", "female", or "unknown".
3. relationship_status: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced", "widowed", or "unknown".
4. income_level: MUST be one of the following exact strings: "low", "middle", "high", "very high", or "unknown".
5. city_country & birth_city_country: Provide your best guess as a string in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China") if the city is unknown, or "unknown".
6. education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner") or "unknown".
7. **MOST IMPORTANT: If you cannot reasonably infer an attribute with high confidence, you MUST use the string value "unknown" instead of randomly guessing an answer.**
"""

# --- [MODIFIED] ---
# Judge/Filter 被重写为 "Arbitrator" (仲裁者)
# 它现在需要验证推理的有效性
PROMPT_ARBITRATOR_SYSTEM = """
You are an expert Arbitrator. Your job is to validate an Attacker's reasoning and extract the core concepts behind valid leaks.
You MUST output a single, valid JSON list.
"""
# --- [MODIFIED] (移除了 [Question] 引用) ---
PROMPT_ARBITRATOR_USER = """
Below is an [Attacker Reasoning] block, a [List of Leaked Attributes] and the [Comment] text.

Your task is to analyze EACH leaked attribute and determine if the Attacker's reasoning for it is VALID.
A reasoning is INVALID if:
1.  It is based on information not found in the [Comment].
2.  It is based on a vague stereotype (e.g., "polite tone implies female", "using 'ahaha' implies young").

For EACH leaked attribute, provide a JSON object with the following fields:
- "attribute": The name of the leaked attribute (e.g., "age").
- "is_valid": true (boolean) if the reasoning is based on specific evidence in the [Comment], false (boolean) otherwise.
- "reasoning_evidence": The specific phrase(s) from the [Comment] that the Attacker used.
- "leaked_concept": If valid, describe the core concept that needs to be neutralized (e.g., "Author works in fashion industry" or "Author lives in Paris").
- "validation_notes": Explain WHY the reasoning is valid or invalid (e.g., "Invalid: Subjective guess based on tone." or "Valid: Based on [Comment] text 'fabrics and designs'.").

Return ONLY a single JSON list containing these objects.

[Comment]:
{current_comment}

[Attacker Reasoning]:
{attacker_inference_block}

[List of Leaked Attributes]:
{leaked_attributes_list}

[Arbitrator's JSON Output]:
"""


# --- 2. 辅助函数 (构建, 解析, 比较) ---

def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。"""
    if dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype = dtype.lower()
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = {"device_map": "auto"} if device is None else {}
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa", **kwargs)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device if device else None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))
    return gen_pipe, tokenizer

def extract_first_json(text: str) -> Optional[Any]:
    """
    从可能包含额外文本的字符串中提取第一个有效的 JSON 对象 ( { } ) 或列表 ( [ ] )。
    """
    first_brace = text.find("{")
    first_bracket = text.find("[")
    
    start_index = -1
    start_char = ''
    end_char = ''

    if first_brace == -1 and first_bracket == -1:
        logging.debug("No JSON object or list found in text.")
        return None 
        
    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        start_index = first_brace
        start_char = '{'
        end_char = '}'
    else:
        start_index = first_bracket
        start_char = '['
        end_char = ']'

    depth = 0
    for i in range(start_index, len(text)):
        if text[i] == start_char:
            depth += 1
        elif text[i] == end_char:
            depth -= 1
            if depth == 0:
                candidate = text[start_index : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    logging.warning(f"Found matching brackets but invalid JSON: {e}. Text: {candidate[:100]}...")
                    return None 
    
    logging.warning("Found JSON start but no matching end bracket/brace.")
    return None

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """使用确定性的 Python 逻辑比较 profiles。"""
    # (此函数保持不变)
    leaked_attributes: List[str] = []
    _log = lambda k, g, t: f"{k} (guessed: '{g}', true: '{t}')"
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int) and abs(true_age - guessed_age) <= 3:
        leaked_attributes.append(_log("age", guessed_age, true_age))
    for key in ["sex", "relationship_status"]:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val and guess_val != "unknown" and true_val.strip().lower() == guess_val.strip().lower():
            leaked_attributes.append(_log(key, guess_val, true_val))
    for key in ["city_country", "birth_city_country", "education", "occupation", "income_level"]:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val and guess_val != "unknown":
            true_s = true_val.strip().lower()
            guess_s = guess_val.strip().lower()
            if len(guess_s) >= 3 and ((guess_s in true_s) or (true_s in guess_s)):
                leaked_attributes.append(_log(key, guess_val, true_val))
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """从攻击者的自由格式输出中解析 *非结构化的推理* 和 猜测(JSON)。"""
    # (此函数保持不变)
    inference_block = ""
    guess_json = {}
    try:
        inference_block_match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if inference_block_match:
            inference_block = inference_block_match.group(1).strip()
        else:
            logging.warning("Could not find 'Inference:' block in attacker output.")
            inference_block = "No Inference block found."
        guess_json = extract_first_json(response_text)
        if guess_json is None:
             guess_json = {"error": "No JSON object found in attacker output"}
    except Exception as e:
        logging.error(f"Error parsing attacker output: {e}\nOutput:\n{response_text[:500]}...", exc_info=True)
        return {"inference": str(e), "guess_json": {}}
    return {"inference": inference_block, "guess_json": guess_json}

# --- [MODIFIED] (移除了 question_asked) ---
def call_attacker_local(pipe, user_response: str, terminator_ids: List[int], record_id: int) -> str:
    """调用本地 Attacker LLM，返回 *原始文本输出*"""
    iteration_log_prefix = f"[Record-{record_id}] Attacking"
    logging.debug(f"{iteration_log_prefix}...")
    user_content = PROMPT_ATTACKER_USER.format(
        # question_asked=str(question_asked), # REMOVED
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Attacker PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.1
    do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Attacker FULL OUTPUT ---\n{response_text}\n--- End ---")
        return response_text
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local attack.", exc_info=True)
        return f"Inference: Error during generation: {e}\nGuess: {{}}"

# --- [MODIFIED] (移除了 question_asked) ---
def call_arbitrator_local(pipe, 
                          # question_asked: str, # REMOVED
                          current_comment: str, 
                          attacker_inference_block: str, 
                          leaked_attributes_list: List[str], 
                          terminator_ids: List[int], 
                          record_id: int) -> str:
    """使用 LLM 仲裁者验证推理并提取有效的泄露概念。"""
    iteration_log_prefix = f"[Record-{record_id}] Arbitrating Feedback"
    logging.debug(f"{iteration_log_prefix} for attributes: {leaked_attributes_list}")
    
    if not leaked_attributes_list:
        return "[]" # 返回一个空的 JSON 列表

    user_content = PROMPT_ARBITRATOR_USER.format(
        # question_asked=question_asked, # REMOVED
        current_comment=current_comment,
        attacker_inference_block=attacker_inference_block,
        leaked_attributes_list=json.dumps(leaked_attributes_list)
    )
    messages = [{"role": "system", "content": PROMPT_ARBITRATOR_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- ARBITRATOR PROMPT ---\n{prompt}\n--- End ---")
    
    temperature = 0.0 # 零温，使其具有确定性
    do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        # 尝试从 LLM 的（可能混乱的）输出中提取 JSON 列表
        response_text = outputs[0]["generated_text"].strip()
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start != -1 and end != -1:
            arbitrator_json_output = response_text[start:end+1]
            logging.debug(f"{iteration_log_prefix} --- ARBITRATOR FULL OUTPUT (Raw) ---\n{response_text}\n--- End ---")
            logging.debug(f"{iteration_log_prefix} --- ARBITRATOR PARSED JSON ---\n{arbitrator_json_output}\n--- End ---")
            return arbitrator_json_output
        else:
             logging.warning(f"{iteration_log_prefix} Could not find JSON list '[]' in Arbitrator output. Returning empty list.")
             return "[]"
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local feedback arbitration.", exc_info=True)
        return f"[]" # 失败时返回空列表

def parse_arbitrator_output(response_text: str, record_id: int) -> Optional[List[Dict[str, Any]]]:
    """
    解析 Arbitrator 的输出，使用 extract_first_json 来处理额外的文本。
    只返回一个值：解析后的列表，如果失败则返回 None。
    """
    record_log_prefix = f"[Record-{record_id}]"
    
    try:
        # 使用新的辅助函数来安全地提取 JSON，忽略任何后续文本
        parsed_json = extract_first_json(response_text)
        
        if isinstance(parsed_json, list):
            logging.debug(f"{record_log_prefix} Arbitrator output parsed successfully.")
            return parsed_json
            
        elif parsed_json:
            logging.warning(f"{record_log_prefix} Arbitrator returned a JSON object, but a LIST was expected. Output: {str(parsed_json)[:200]}")
            return None
        else:
            logging.warning(f"{record_log_prefix} No valid JSON list found in Arbitrator output. Raw: {response_text[:200]}")
            return None
            
    except Exception as e:
        logging.error(f"{record_log_prefix} Unexpected error in parse_arbitrator_output: {e}", exc_info=True)
        return None

# --- [MODIFIED] (移除了 question) ---
def is_sanity_check_failed(new_comment: str, old_comment: str, min_len: int = 20) -> Optional[str]:
    """
    对 Anonymizer 的输出执行健全性检查，以捕获灾难性故障（如记录 2）。
    返回一个“失败原因”字符串，如果没有失败则返回 None。
    """
    if not new_comment or len(new_comment) < min_len:
        return f"catastrophic_anonymizer_failure: Output is too short (len: {len(new_comment)})."
    
    # --- [REMOVED] ---
    # 移除了检查评论是否复制了问题的逻辑
    # --- [END REMOVED] ---
    
    # 检查它是否与旧评论完全相同
    if new_comment.strip() == old_comment.strip():
        return f"anonymizer_stuck: Output is identical to the previous iteration."
    
    return None # 检查通过

# --- [MODIFIED] (移除了 question_asked) ---
def call_anonymizer_local(pipe, user_response: str, feedback: str, terminator_ids: List[int], record_id: int) -> str:
    """调用本地 Anonymizer LLM，解析 # 并返回匿名化文本"""
    iteration_log_prefix = f"[Record-{record_id}] Anonymizing..."
    logging.debug(f"{iteration_log_prefix} with feedback: {feedback[:100]}...")
    user_content = PROMPT_ANONYMIZER_USER.format(
        # question_asked=str(question_asked), # REMOVED
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Anonymizer PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.5
    do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Anonymizer FULL OUTPUT ---\n{response_text}\n--- End ---")
        
        parts = response_text.split('#', 1)
        if len(parts) == 2:
            explanation = parts[0].strip()
            final_response = parts[1].strip().strip('"').strip()
            logging.debug(f"{iteration_log_prefix} Anonymizer explanation: {explanation}")
            # 清理
            final_response = final_response.split("<|eot_id|>")[0].strip()
            final_response = final_response.split("<|end_of_text|>")[0].strip()
            return final_response
        else:
             logging.warning(f"{iteration_log_prefix} Anonymizer failed to produce '#'. Using raw output.")
             return response_text.strip().strip('"').strip()
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local anonymization.", exc_info=True)
        return user_response # 失败时返回*旧的*响应，而不是抛出异常

# ... (所有其他函数保持不变) ...

# --- 3. 核心逻辑 (实现 "LLM Judge" 架构) ---

def adversarial_anonymization(
    pipe,
    # question_asked: str, # REMOVED
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int = 3,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """
    执行 "Attacker -> Python Judge -> LLM Filter -> Anonymizer" 架构
    """
    
    CORE_PII_ATTRIBUTES = {"age", "city_country", "birth_city_country", "education", "occupation", "relationship_status", "sex", "income_level"}

    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting 'LLM Judge' process (Max {max_iterations} iterations, Core PII: {CORE_PII_ATTRIBUTES}).")

    # --- 步骤 0: 初始攻击 (Attacker) ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        raw_attack_output = call_attacker_local(pipe, original_response, terminator_ids, record_id)
        parsed_attack = parse_attacker_output(raw_attack_output)
        attacker_guess = parsed_attack.get("guess_json", {})
        full_inference_block = parsed_attack.get("inference", "No reasoning provided.")
        meta["final_attacker_guess"] = attacker_guess
        if "error" in attacker_guess:
            raise ValueError(f"Failed to parse attacker output: {attacker_guess.get('error')}")
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker/Parser failed: {e}", exc_info=True)
        meta["status"] = "model_error"; meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta

    # --- 步骤 0.5: 初始评判 (Python Judge) ---
    leaked = compare_profiles(true_personality, attacker_guess)
    meta["final_leaked_attributes"] = leaked
    leaked_attribute_names = [detail.split(" ")[0] for detail in leaked]
    core_leaked_names = set(leaked_attribute_names).intersection(CORE_PII_ATTRIBUTES)

    if not core_leaked_names:
        non_core_leaked = set(leaked_attribute_names).difference(CORE_PII_ATTRIBUTES)
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No Core PII leaks found. (Non-core leaks: {non_core_leaked or 'None'})")
        meta["status"] = "success_on_original"; meta["iterations_used"] = 0
        return original_response, meta
    
    logging.info(f"[Record {record_id}] Initial attack leaked Core PII: {list(core_leaked_names)}. (All leaks: {list(leaked_attribute_names)})")

    # --- 步骤 0.75: 初始反馈过滤 (LLM Judge) ---
    logging.info(f"[Record {record_id}][Round 0] Filtering feedback...")
    
    arbitrator_raw_output = call_arbitrator_local(
        pipe, 
        original_response, 
        full_inference_block, 
        leaked_attribute_names, 
        terminator_ids, 
        record_id
    )
    
    try:
        arbitration_results = parse_arbitrator_output(arbitrator_raw_output, record_id)
        
        if arbitration_results is None:
            logging.error(f"[Record {record_id}][Round 0] LLM Judge (Filter) failed to produce valid JSON list. Using raw (noisy) feedback.")
            filtered_feedback = full_inference_block # Fallback
        else:
            filtered_feedback = "The attacker's reasoning for the correctly guessed attributes (which the Judge validated) is as follows:\n\n"
            found_valid_reasoning = False
            for item in arbitration_results:
                if item.get("is_valid") == True: # 只反馈 Judge 认为有效的
                    found_valid_reasoning = True
                    filtered_feedback += f"**Attribute:** {item.get('attribute')}\n"
                    filtered_feedback += f"**Leaked Concept:** {item.get('leaked_concept')}\n"
                    filtered_feedback += f"**Reasoning Evidence:** {json.dumps(item.get('reasoning_evidence'))}\n\n"
            
            # --- [NEW STOP CONDITION] ---
            # 如果猜对了，但所有推理都是无效的 (例如偏见)，则停止。
            if not found_valid_reasoning:
                logging.info(f"[Record {record_id}][Round 0] Success: Attacker core leaks ({list(core_leaked_names)}) were based on invalid reasoning (bias). Stopping.")
                meta["status"] = "success_arbitrator_invalidated"
                meta["iterations_used"] = 0 
                return original_response, meta
            # --- [END NEW STOP CONDITION] ---
    
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Arbitrator/Parser failed: {e}", exc_info=True)
        meta["status"] = "model_error"; meta["error"] = f"arbitrator_parse_error: {e}"
        return current_anonymized_response, meta

    # --- 循环开始 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (filtered by Judge LLM): {filtered_feedback[:150]}...")

        previous_anonymized_response = current_anonymized_response

        # 1) 匿名化 (Anonymizer)
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer_local(pipe, current_anonymized_response, filtered_feedback, terminator_ids, record_id)
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "model_error"; meta["error"] = f"anonymizer_error: {e}"
            return previous_anonymized_response, meta 

        # 2) 健全性检查 (Sanity Check)
        sanity_fail_reason = is_sanity_check_failed(
            current_anonymized_response, 
            previous_anonymized_response
        )
        if sanity_fail_reason:
            logging.error(f"{iteration_log_prefix} Anonymizer output failed sanity check: {sanity_fail_reason}. Aborting record.")
            meta["status"] = "model_error"
            meta["error"] = sanity_fail_reason
            return previous_anonymized_response, meta 


        # 3) 攻击者推断 (Attacker)
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            raw_attack_output = call_attacker_local(pipe, current_anonymized_response, terminator_ids, record_id)
            parsed_attack = parse_attacker_output(raw_attack_output)
            attacker_guess = parsed_attack.get("guess_json", {})
            full_inference_block = parsed_attack.get("inference", "No reasoning provided.")
            meta["final_attacker_guess"] = attacker_guess
            if "error" in attacker_guess:
                raise ValueError(f"Failed to parse attacker output: {attacker_guess.get('error')}")
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker/Parser failed: {e}. Skipping judge for this round.", exc_info=True)
            continue

        # 4) 评判泄露 (Python Judge) & 检查停止条件
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        leaked = compare_profiles(true_personality, attacker_guess)
        meta["final_leaked_attributes"] = leaked
        leaked_attribute_names = [detail.split(" ")[0] for detail in leaked]
        core_leaked_names = set(leaked_attribute_names).intersection(CORE_PII_ATTRIBUTES)
        
        if not core_leaked_names:
            non_core_leaked = set(leaked_attribute_names).difference(CORE_PII_ATTRIBUTES)
            logging.info(f"{iteration_log_prefix} Success! No CORE PII attributes leaked. (Non-core leaks: {non_core_leaked or 'None'})")
            meta["status"] = "success"
            return current_anonymized_response, meta
        
        # 5) 反馈过滤 (LLM Judge)
        logging.info(f"{iteration_log_prefix} Failed. Leaked Core PII: {list(core_leaked_names)}. (All leaks: {list(leaked_attribute_names)})")
        logging.info(f"{iteration_log_prefix} Filtering new feedback...")
        
        arbitrator_raw_output = call_arbitrator_local(
            pipe,
            current_anonymized_response, 
            full_inference_block, 
            leaked_attribute_names, 
            terminator_ids, 
            record_id
        )
        
        try:
            arbitration_results = parse_arbitrator_output(arbitrator_raw_output, record_id)
            
            if arbitration_results is None:
                logging.error(f"{iteration_log_prefix} LLM Judge failed. Using raw (noisy) feedback.")
                filtered_feedback = full_inference_block # Fallback
            else:
                filtered_feedback = "The attacker's reasoning for the correctly guessed attributes (which the Judge validated) is as follows:\n\n"
                found_valid_reasoning = False
                for item in arbitration_results:
                    if item.get("is_valid") == True: # 只反馈 Judge 认为有效的
                        found_valid_reasoning = True
                        filtered_feedback += f"**Attribute:** {item.get('attribute')}\n"
                        filtered_feedback += f"**Leaked Concept:** {item.get('leaked_concept')}\n"
                        filtered_feedback += f"**Reasoning Evidence:** {json.dumps(item.get('reasoning_evidence'))}\n\n"
                
                # --- [NEW STOP CONDITION] ---
                # 如果在循环中猜对了，但所有推理都是无效的 (例如偏见)，则停止。
                if not found_valid_reasoning:
                    logging.info(f"{iteration_log_prefix} Success: Attacker core leaks ({list(core_leaked_names)}) were based on invalid reasoning (bias). Stopping.")
                    meta["status"] = "success_arbitrator_invalidated"
                    return current_anonymized_response, meta
                # --- [END NEW STOP CONDITION] ---

        except Exception as e:
             logging.error(f"{iteration_log_prefix} Feedback filtering failed: {e}", exc_info=True)
             filtered_feedback = full_inference_block # Fallback

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

# --- 4. Wrapper 和 Main ---
# --- [MODIFIED] (移除了 question) ---
def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    try:
        personality = data.get("personality")
        # question = str(data.get("question_asked")) # REMOVED
        response = str(data.get("response"))
    except Exception as e:
        logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data
    
    # --- [MODIFIED] (移除了 question) ---
    if not all([personality, response]):
        logging.warning(f"[Record {record_id}] Skipped due to incomplete data (missing personality or response).")
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data
    if not isinstance(personality, dict):
        logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary.")
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data
    
    # --- [MODIFIED] (移除了 question) ---
    anonymized_response, meta = adversarial_anonymization(
        pipe, 
        # question, # REMOVED
        response, 
        personality, 
        terminator_ids, 
        max_iterations, 
        record_id
    )
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用本地模型运行 'LLM Arbitrator' 匿名化") 
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=3, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大新 token 数（Attacker/Arbitrator 需要更大空间）")
    parser.add_argument("--log_file", type=str, default="anonymizer_local_arbitrator.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)
    records_to_process = [(i, json.loads(line)) for i, line in enumerate(lines) if line.strip()]
    if args.limit:
        records_to_process = records_to_process[:args.limit]
    try:
        gen_pipe, tokenizer = build_pipeline(model_name=args.model_name, device=args.device, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}", exc_info=True)
        sys.exit(1)
    
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with model {args.model_name} ...")
    
    results = []
    counters = {
        "success": 0, 
        "success_on_original": 0, 
        "success_arbitrator_invalidated": 0,
        "max_iterations_reached": 0, 
        "model_error": 0, 
        "skipped_data_read_error": 0, 
        "skipped_incomplete_data": 0, 
        "skipped_invalid_personality": 0, 
        "unknown_fail": 0
    }
    
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids)
    
    for i, rec_tuple in enumerate(tqdm(records_to_process, desc="Anonymizing profiles (Arbitrator)")): 
        rec_idx, rec_data = rec_tuple
        try:
            processed_record = _task(rec_idx, rec_data)
            results.append(processed_record)
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            counters[status] = counters.get(status, 0) + 1
        except Exception as exc:
            logging.error(f"[Record {rec_idx}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1
            
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
            open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
            open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            meta = result.get("anonymization_meta", {})
            status = meta.get("status")
            
            if status in ("success", "success_on_original", "success_arbitrator_invalidated"): 
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: 
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in sorted(counters.items()):
        if count > 0:
            logging.info(f"    {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file:
        logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file:
        logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()