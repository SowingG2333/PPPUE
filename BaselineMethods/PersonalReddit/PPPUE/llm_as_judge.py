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

# --- 1. Prompts (使用上面的 Prompts) ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
"""
PROMPT_ANONYMIZER_USER = """
Below I give you a [Comment] from a user to a [Question] as well as [Personal Inferences] made about the user that can be made from the [Comment]. Your task is to anonymize the [Comment] to prevent these [Personal Inferences], while keeping the rest of the text intact and still answering the [Question].

[Question]:
{question_asked}

[Comment]:
{user_response}

[Personal Inferences]:
{feedback}

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
Then, return the anonymized text without any extra preamble like "Here is the anonymized text:" or suffix like "Note:". 
"""
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following is a public [Comment] made on Reddit by one author, in response to a [Question]. Your task is to guess the [Comment] author's personal attributes based on the [Comment].

[Question]:
{question_asked}

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
PROMPT_JUDGE_FILTER_SYSTEM = """
You are an expert text extractor. Your job is to analyze a reasoning text and extract ONLY the parts relevant to a specific list of topics. Be concise and precise.
"""
PROMPT_JUDGE_FILTER_USER = """
Below is a [Reasoning Text] from an attacker and a [List of Leaked Attributes].
Your task is to extract ONLY the sentences or phrases from the [Reasoning Text] that directly explain why the attacker guessed the attributes in the [List of Leaked Attributes].

If the reasoning for an attribute is missing or vague, state that.
Combine the extracted reasoning for all listed attributes into a single, clean text block. Do not add any preamble or your own opinions.

[Reasoning Text]:
{attacker_inference_block}

[List of Leaked Attributes]:
{leaked_attributes_list}

Extracted Feedback:
"""

# --- 2. 辅助函数 (构建, 解析, 比较) ---

def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。"""
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
    """从文本中提取第一个完整 JSON 对象。"""
    guess_match = re.search(r"Guess:\s*({.*})", text, re.DOTALL | re.IGNORECASE)
    if guess_match:
        json_str = guess_match.group(1)
        try: return json.loads(json_str)
        except Exception as e: logging.warning(f"Failed to parse JSON in Guess block: {e}. Trying fallback.")
    start = text.find("{")
    if start == -1: return None
    depth = 0; end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        candidate = text[start:end+1]
        try: return json.loads(candidate)
        except Exception: pass
    try:
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{"); end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start: return json.loads(cleaned[start:end+1])
    except Exception: return None
    return None

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """使用确定性的 Python 逻辑比较 profiles。"""
    leaked_attributes: List[str] = []; _log = lambda k, g, t: f"{k} (guessed: '{g}', true: '{t}')"
    true_age = true_profile.get("age"); guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int) and abs(true_age - guessed_age) <= 3: leaked_attributes.append(_log("age", guessed_age, true_age))
    for key in ["sex", "relationship_status"]:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val and guess_val != "unknown" and true_val.strip().lower() == guess_val.strip().lower(): leaked_attributes.append(_log(key, guess_val, true_val))
    for key in ["city_country", "birth_city_country", "education", "occupation", "income_level"]:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val and guess_val != "unknown":
            true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
            if len(guess_s) >= 3 and ((guess_s in true_s) or (true_s in guess_s)): leaked_attributes.append(_log(key, guess_val, true_val))
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """从攻击者的自由格式输出中解析 *非结构化的推理* 和 猜测(JSON)。"""
    inference_block = ""
    guess_json = {}
    try:
        inference_block_match = re.search(r"Inference:(.*?)(Guess:|$)", response_text, re.DOTALL | re.IGNORECASE)
        if inference_block_match:
            inference_block = inference_block_match.group(1).strip()
        else:
            logging.warning("Could not find 'Inference:' block in attacker output.")
            inference_block = "No Inference block found."
        guess_json = extract_first_json_object(response_text)
        if guess_json is None:
             guess_json = {"error": "No JSON object found in attacker output"}
    except Exception as e:
        logging.error(f"Error parsing attacker output: {e}\nOutput:\n{response_text[:500]}...", exc_info=True)
        return {"inference": str(e), "guess_json": {}}
    return {"inference": inference_block, "guess_json": guess_json}

def call_attacker_local(pipe, question_asked: str, user_response: str, terminator_ids: List[int], record_id: int) -> str:
    """调用本地 Attacker LLM，返回 *原始文本输出*"""
    iteration_log_prefix = f"[Record-{record_id}] Attacking"
    logging.debug(f"{iteration_log_prefix}...")
    user_content = PROMPT_ATTACKER_USER.format(
        question_asked=str(question_asked),
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Attacker PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.1; do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=1024, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        response_text = outputs[0]["generated_text"]
        logging.debug(f"{iteration_log_prefix} --- Attacker FULL OUTPUT ---\n{response_text}\n--- End ---")
        return response_text
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local attack.", exc_info=True)
        return f"Inference: Error during generation: {e}\nGuess: {{}}"

# --- *** 新函数：LLM Judge/Filter *** ---
def call_feedback_filter_local(pipe, attacker_inference_block: str, leaked_attributes_list: List[str], terminator_ids: List[int], record_id: int) -> str:
    """使用 LLM 从完整推理中提取与泄露属性相关的部分"""
    iteration_log_prefix = f"[Record-{record_id}] Judging/Filtering Feedback"
    logging.debug(f"{iteration_log_prefix} for attributes: {leaked_attributes_list}")
    
    # 如果没有泄露，则无需调用 LLM
    if not leaked_attributes_list:
        return "No attributes were leaked in the last round."

    user_content = PROMPT_JUDGE_FILTER_USER.format(
        attacker_inference_block=attacker_inference_block,
        leaked_attributes_list=json.dumps(leaked_attributes_list)
    )
    messages = [{"role": "system", "content": PROMPT_JUDGE_FILTER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- JUDGE/FILTER PROMPT ---\n{prompt}\n--- End ---")
    
    temperature = 0.0 # 零温，使其具有确定性
    do_sample = temperature > 0.0
    try:
        outputs = pipe(prompt, max_new_tokens=512, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
        filtered_feedback = outputs[0]["generated_text"].strip()
        logging.debug(f"{iteration_log_prefix} --- JUDGE/FILTER FULL OUTPUT ---\n{filtered_feedback}\n--- End ---")
        return filtered_feedback
    except Exception as e:
        logging.error(f"{iteration_log_prefix} Error during local feedback filtering.", exc_info=True)
        # Fallback: 如果 LLM 过滤器失败，返回完整的（嘈杂的）推理
        return attacker_inference_block

def call_anonymizer_local(pipe, question_asked: str, user_response: str, feedback: str, terminator_ids: List[int], record_id: int) -> str:
    """调用本地 Anonymizer LLM，解析 # 并返回匿名化文本"""
    iteration_log_prefix = f"[Record-{record_id}] Anonymizing..."
    logging.debug(f"{iteration_log_prefix} with feedback: {feedback[:100]}...")
    user_content = PROMPT_ANONYMIZER_USER.format(
        question_asked=str(question_asked),
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"{iteration_log_prefix} --- Anonymizer PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.5; do_sample = temperature > 0.0
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
        return user_response

# --- 3. 核心逻辑 (实现 "LLM Judge" 架构) ---

def adversarial_anonymization(
    pipe,
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int = 3, # 默认 3 次
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """
    执行 "Attacker -> Python Judge -> LLM Filter -> Anonymizer" 架构的匿名化过程
    """
    
    # 核心 PII（用于选择性迭代）
    CORE_PII_ATTRIBUTES = {"age", "sex", "city_country", "birth_city_country", "education", "occupation", "relationship_status", "income_level"}
    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting 'LLM Judge' process (Max {max_iterations} iterations, Core PII: {CORE_PII_ATTRIBUTES}).")

    # --- 步骤 0: 初始攻击 (Attacker) ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        raw_attack_output = call_attacker_local(pipe, question_asked, original_response, terminator_ids, record_id)
        parsed_attack = parse_attacker_output(raw_attack_output) # 使用非结构化解析器
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
    leaked_attribute_names = [detail.split(" ")[0] for detail in leaked] # 获取 ["age", "income_level"]
    core_leaked_names = set(leaked_attribute_names).intersection(CORE_PII_ATTRIBUTES)

    if not core_leaked_names:
        non_core_leaked = set(leaked_attribute_names).difference(CORE_PII_ATTRIBUTES)
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No Core PII leaks found. (Non-core leaks: {non_core_leaked or 'None'})")
        meta["status"] = "success_on_original"; meta["iterations_used"] = 0
        return original_response, meta
    
    logging.info(f"[Record {record_id}] Initial attack leaked Core PII: {list(core_leaked_names)}. (All leaks: {list(leaked_attribute_names)})")

    # --- 步骤 0.75: 初始反馈过滤 (LLM Judge) ---
    logging.info(f"[Record {record_id}][Round 0] Filtering feedback...")
    filtered_feedback = call_feedback_filter_local(pipe, full_inference_block, leaked_attribute_names, terminator_ids, record_id)

    # --- 循环开始 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (filtered by LLM): {filtered_feedback[:150]}...")

        # 1) 匿名化 (Anonymizer) - 接收过滤后的反馈
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer_local(pipe, question_asked, current_anonymized_response, filtered_feedback, terminator_ids, record_id)
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "model_error"; meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta

        # 2) 攻击者推断 (Attacker)
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            raw_attack_output = call_attacker_local(pipe, question_asked, current_anonymized_response, terminator_ids, record_id)
            parsed_attack = parse_attacker_output(raw_attack_output) # <--- 使用非结构化解析器
            attacker_guess = parsed_attack.get("guess_json", {})
            full_inference_block = parsed_attack.get("inference", "No reasoning provided.")
            meta["final_attacker_guess"] = attacker_guess
            if "error" in attacker_guess:
                raise ValueError(f"Failed to parse attacker output: {attacker_guess.get('error')}")
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker/Parser failed: {e}. Skipping judge for this round.", exc_info=True)
            continue

        # 3) 评判泄露 (Python Judge) & 检查停止条件
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
        
        # 4) 反馈过滤 (LLM Judge) - 仅当需要继续时
        logging.info(f"{iteration_log_prefix} Failed. Leaked Core PII: {list(core_leaked_names)}. (All leaks: {list(leaked_attribute_names)})")
        logging.info(f"{iteration_log_prefix} Filtering new feedback...")
        filtered_feedback = call_feedback_filter_local(pipe, full_inference_block, leaked_attribute_names, terminator_ids, record_id)

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

# --- 4. Wrapper 和 Main (与上一版相同) ---
def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    try:
        personality = data.get("personality"); question = str(data.get("question_asked")); response = str(data.get("response"))
    except Exception as e:
        logging.error(f"[Record {record_id}] Failed to read data fields: {e}"); data["anonymization_meta"] = {"status": "skipped_data_read_error"}; return data
    if not all([personality, question, response]):
        logging.warning(f"[Record {record_id}] Skipped due to incomplete data."); data["anonymization_meta"] = {"status": "skipped_incomplete_data"}; return data
    if not isinstance(personality, dict):
        logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary."); data["anonymization_meta"] = {"status": "skipped_invalid_personality"}; return data
    anonymized_response, meta = adversarial_anonymization(pipe, question, response, personality, terminator_ids, max_iterations, record_id)
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用本地模型运行 'LLM Judge' FgAA 匿名化")
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
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大新 token 数（Attacker 需要更大空间）")
    parser.add_argument("--log_file", type=str, default="anonymizer_local_llm_judge.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.getLogger("transformers").setLevel(logging.ERROR); logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError: logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    records_to_process = [(i, json.loads(line)) for i, line in enumerate(lines) if line.strip()]
    if args.limit: records_to_process = records_to_process[:args.limit]
    try:
        gen_pipe, tokenizer = build_pipeline(model_name=args.model_name, device=args.device, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception as e: logging.error(f"Error: failed to load local model '{args.model_name}': {e}", exc_info=True); sys.exit(1)
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []; counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "model_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "unknown_fail": 0}
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids)
    for i, rec_tuple in enumerate(tqdm(records_to_process, desc="Anonymizing profiles (LLM Judge)")):
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
            if status in ("success", "success_on_original"): success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in sorted(counters.items()):
        if count > 0: logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()