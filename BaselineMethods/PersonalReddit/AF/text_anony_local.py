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
1.  age: MUST be an accurate integer (e.g., 35, not "35" or "30-40") or string "unknown".
2.  sex: MUST be one of the following exact strings: "male", "female", or "unknown".
3.  relationship_status: MUST be one of the following exact strings: "single", "in a relationship", "engaged", "married", "divorced", "widowed", or "unknown".
4.  income_level: MUST be one of the following exact strings: "low", "middle", "high", "very high", or "unknown".
5.  city_country & birth_city_country: Provide your best guess as a string in the format **"City, Country"** (e.g., "Shanghai, China") or just **"Country"** (e.g., "China") if the city is unknown, or "unknown".
6.  education & occupation: Provide your best guess as a string (e.g., "Masters in Computer Science", "shop owner") or "unknown".
7.  **MOST IMPORTANT: If you cannot reasonably infer an attribute with high confidence, you MUST use the string value "unknown" instead of randomly guessing an answer.**
"""

def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。(无需修改)"""
    if dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
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

def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """从攻击者的自由格式输出中解析推理、猜测(JSON)和置信度。(无需修改)"""
    inference = ""
    guess_json = {}
    certainty = 0
    inference_match = re.search(r"Inference:(.*?)(Guess:|Certainty:|$)", response_text, re.DOTALL | re.IGNORECASE)
    if inference_match: inference = inference_match.group(1).strip()
    json_start = response_text.find('{'); json_end = response_text.rfind('}') + 1
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        try: guess_json = json.loads(json_str)
        except json.JSONDecodeError: logging.warning(f"Failed to parse JSON: {json_str[:100]}..."); guess_json = {"error": "Failed to parse JSON"}
    else: logging.warning(f"No JSON found: {response_text[:100]}..."); guess_json = {"error": "No JSON found"}
    # certainty_match = re.search(r"Certainty:.*?(\d)", response_text, re.IGNORECASE)
    # if certainty_match:
    #     try: certainty = int(certainty_match.group(1))
    #     except ValueError: certainty = 0

    return {"inference": inference, "guess_json": guess_json, "certainty": certainty}

# --- 已修改：call_anonymizer (重新加入 question_asked) ---
def call_anonymizer(pipe, question_asked: str, user_response: str, feedback: str, terminator_ids: List[int], record_id: int) -> str:
    user_content = PROMPT_ANONYMIZER_USER.format(
        question_asked=str(question_asked), # <--- 传递
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"[Record {record_id}] --- ANONYMIZER PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.5; do_sample = temperature > 0.0
    outputs = pipe(prompt, max_new_tokens=384, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
    out = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ANONYMIZER FULL OUTPUT ---\n{out}\n--- End ---")
    parts = out.split('#', 1)
    explanation = parts[0].strip() if len(parts) == 2 else "No explanation provided (LLM failed to follow format)."
    final_response = parts[1].strip().strip('"').strip() if len(parts) == 2 else out.strip().strip('"').strip()
    logging.debug(f"[Record {record_id}] --- ANONYMIZER EXPLANATION ---\n{explanation}\n--- End ---")
    return final_response

# --- 已修改：call_attacker (重新加入 question_asked) ---
def call_attacker(pipe, question_asked: str, user_response: str, terminator_ids: List[int], record_id: int) -> Dict[str, Any]:
    user_content = PROMPT_ATTACKER_USER.format(
        question_asked=str(question_asked), # <--- 传递
        user_response=str(user_response)
    )
    messages = [{"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()}, {"role": "user", "content": user_content}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.debug(f"[Record {record_id}] --- ATTACKER PROMPT ---\n{prompt}\n--- End ---")
    temperature = 0.1; do_sample = temperature > 0.0
    outputs = pipe(prompt, max_new_tokens=384, eos_token_id=terminator_ids, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=0.9 if do_sample else None, pad_token_id=pipe.tokenizer.eos_token_id, return_full_text=False)
    out = outputs[0]["generated_text"]
    logging.debug(f"[Record {record_id}] --- ATTACKER FULL OUTPUT ---\n{out}\n--- End ---")
    parsed_data = parse_attacker_output(out)
    logging.debug(f"[Record {record_id}] --- ATTACKER PARSED ---\n{parsed_data}\n--- End ---")
    return parsed_data

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """使用确定性的 Python 逻辑比较 profiles。(无需修改)"""
    leaked_attributes: List[str] = []
    def _log(key, guess, true): return f"{key} (guessed: '{guess}', true: '{true}')"
    true_age = true_profile.get("age"); guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 1: leaked_attributes.append(_log("age", guessed_age, true_age))
    simple_keys = ["sex", "relationship_status"]
    for key in simple_keys:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            if true_val.strip().lower() == guess_val.strip().lower(): leaked_attributes.append(_log(key, guess_val, true_val))
    complex_keys = ["city_country", "birth_city_country", "education", "occupation", "income_level"]
    for key in complex_keys:
        true_val = true_profile.get(key); guess_val = guessed_profile.get(key)
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            true_s = true_val.strip().lower(); guess_s = guess_val.strip().lower()
            if len(guess_s) < 3: continue
            if (guess_s in true_s) or (true_s in guess_s): leaked_attributes.append(_log(key, guess_val, true_val))
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- 已修改：adversarial_anonymization (重新加入 question_asked) ---
def adversarial_anonymization(
    pipe,
    question_asked: str, # <--- 重新加入
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int = 5,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程，返回最终的匿名化响应和元数据。"""
    current_anonymized_response = original_response
    meta = {"status": "max_iterations_reached", "iterations_used": 0, "final_leaked_attributes": [], "final_attacker_guess": {}}
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    # --- 步骤 0: 初始攻击 ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        parsed_attack = call_attacker(pipe, question_asked, original_response, terminator_ids, record_id) # <--- 传递
        attacker_guess = parsed_attack.get("guess_json", {})
        feedback = parsed_attack.get("inference", "No inference provided by attacker.")
        meta["final_attacker_guess"] = attacker_guess
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "api_error"; meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta

    # --- 步骤 0.5: 初始裁判 ---
    leaked = compare_profiles(true_personality, attacker_guess)
    meta["final_leaked_attributes"] = leaked
    if not leaked:
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No leaks found.")
        meta["status"] = "success_on_original"
        return current_anonymized_response, meta
    logging.info(f"[Record {record_id}] Initial attack leaked: {leaked}")

    # --- 循环 ---
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer(
                pipe, question_asked, current_anonymized_response, feedback, terminator_ids, record_id # <--- 传递
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            parsed_attack = call_attacker(pipe, question_asked, current_anonymized_response, terminator_ids, record_id) # <--- 传递
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            continue

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        leaked = compare_profiles(true_personality, attacker_guess)
        if not leaked:
            logging.info(f"{iteration_log_prefix} Success! No attributes leaked.")
            meta["status"] = "success"
            return current_anonymized_response, meta
        logging.info(f"{iteration_log_prefix} Failed. Leaked: {leaked}")
        meta["final_leaked_attributes"] = leaked

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

# --- 已修改：process_record (重新加入 question_asked 传递) ---
def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    try:
        personality = data.get("personality")
        question = str(data.get("question_asked")) # <--- 读取
        response = str(data.get("response"))
    except Exception as e:
        logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not all([personality, question, response]):
        logging.warning(f"[Record {record_id}] Skipped due to incomplete data.")
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data

    if not isinstance(personality, dict):
        logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary.")
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    # 传递 question
    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        question_asked=question, # <--- 传递
        original_response=response,
        true_personality=personality,
        terminator_ids=terminator_ids,
        max_iterations=max_iterations,
        record_id=record_id
    )
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用本地 Hugging Face 模型对 JSONL 中的回答进行匿名化")
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=3, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成的最大新 token 数（全局）")
    parser.add_argument("--log_file", type=str, default="anonymizer_local.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    args = parser.parse_args()

    # --- Logger Setup (保持不变) ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.FileHandler(args.log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")

    # --- 数据加载 (保持不变) ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f: lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)
    records_to_process = [json.loads(line) for line in lines if line.strip()]
    if args.limit: records_to_process = records_to_process[:args.limit]

    # --- 模型加载 (保持不变) ---
    try:
        gen_pipe, tokenizer = build_pipeline(model_name=args.model_name, device=args.device, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}", exc_info=True); sys.exit(1)
    terminator_ids = [tokenizer.eos_token_id] + [tid for tid in [tokenizer.convert_tokens_to_ids(tok) for tok in ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]] if tid is not None and not isinstance(tid, list)]
    logging.info(f"Using terminators: {terminator_ids}")

    # --- 串行处理 (本地版本通常不并行) ---
    logging.info(f"Starting sequential processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []
    counters = {"success": 0, "success_on_original": 0, "max_iterations_reached": 0, "api_error": 0, "skipped_data_read_error": 0, "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, "unknown_fail": 0}
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids)
    for i, rec in enumerate(tqdm(records_to_process, desc="Anonymizing profiles")):
        try:
            processed_record = _task(i, rec)
            results.append(processed_record)
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            counters[status] = counters.get(status, 0) + 1
        except Exception as exc:
            logging.error(f"[Record {i}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1

    # --- 结果写入 (保持不变) ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            status = result.get("anonymization_meta", {}).get("status")
            if status in ("success", "success_on_original"): success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    # --- 记录最终摘要 (保持不变) ---
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in counters.items():
        if count > 0: logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")
    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file: logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file: logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()