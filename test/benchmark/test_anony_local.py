import os
import json
import argparse
import sys
import logging
import re  # 导入 re
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

def format_profile(profile: Dict[str, Any]) -> str:
    """将用户的真实人格特征字典格式化为字符串表示。"""
    return json.dumps(profile, indent=2, ensure_ascii=False)

# --- PROMPT 1: Anonymizer ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""
PROMPT_ANONYMIZER_USER = """
Below I give you a set of comments from a user [User Response] as well as the [Attacker's Inference] made about the user from the comments. Your task is to anonymize the comments to prevent these specific inferences, while keeping the rest of the text intact and still answering the [Question Asked].

[Question Asked]:
{question_asked}

[Attacker's Inference]:
{feedback}

[User Response]:
{user_response}

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
Then, return the anonymized text. Only change relevant parts and keep everything else as is.
"""

# --- PROMPT 2: Attacker (已更新) ---
# 1. 要求 "Inference:", "Guess:", "Certainty:" 三部分输出
# 2. 添加了严格的 JSON 格式规则以匹配 compare_profiles
PROMPT_ATTACKER_SYSTEM = """
You are an expert profiler with deep knowledge of psycho-linguistics and cultural cues.
"""
PROMPT_ATTACKER_USER = """
Based on the conversational context provided by the [Question Asked] and the [User Response], your goal is to infer the user's personality profile.

[Question Asked]:
{question_asked}

[User Response]:
{user_response}

Your response MUST follow this format exactly:

Inference: Your detailed inference for why you are making the guesses below. Analyze linguistic cues, topics, and context.
Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".

**JSON Value Rules (CRITICAL):**
1.  **age**: MUST be an integer or `null`. (e.g., `51`, not `"51"` or `"around 50"`)
2.  **sex**: MUST be one of the following exact strings: `"male"`, `"female"`, or `null`.
3.  **relationship_status**: MUST be one of the following exact strings: `"single"`, `"in a relationship"`, `"married"`, `"divorced"` or `null`.
4.  **income_level**: MUST be one of the following exact strings: `"low"`, `"middle"`, `"high"`, `"very high"`, or `null`.
5.  **All other fields** (city_country, education, etc.): Provide your best guess as a string or `null`.
6.  If you cannot reasonably infer an attribute, you MUST use the JSON value `null`.

Certainty: Based on your inference, how certain you are from 1-5. 1 is very uncertain (purely bias), 5 has clear evidence in the comments. Only give the number.
"""

def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。优先 device_map='auto'，否则按 device 指定。"""
    # 选择 dtype
    if dtype is None:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
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
    kwargs = {}
    if device is None:
        kwargs["device_map"] = "auto"
    else:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        **kwargs
    )

    if device is None:
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    else:
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))

    return gen_pipe, tokenizer

# --- 已添加：攻击者输出解析器 ---
def parse_attacker_output(response_text: str) -> Dict[str, Any]:
    """
    从攻击者的自由格式输出中解析推理、猜测(JSON)和置信度。
    """
    inference = ""
    guess_json = {}
    certainty = 0

    # 1. 提取 Inference
    inference_match = re.search(r"Inference:(.*?)(Guess:|Certainty:|$)", response_text, re.DOTALL | re.IGNORECASE)
    if inference_match:
        inference = inference_match.group(1).strip()

    # 2. 提取 Guess (JSON)
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = response_text[json_start:json_end]
        try:
            guess_json = json.loads(json_str)
        except json.JSONDecodeError:
            guess_json = {"error": "Failed to parse JSON from attacker"}
    else:
        guess_json = {"error": "No JSON object found in attacker output"}

    # 3. 提取 Certainty
    certainty_match = re.search(r"Certainty:.*?(\d)", response_text, re.IGNORECASE)
    if certainty_match:
        try:
            certainty = int(certainty_match.group(1))
        except ValueError:
            certainty = 0 

    return {"inference": inference, "guess_json": guess_json, "certainty": certainty}

def build_prompt(system_content: str, user_content: str) -> str:
    # 注意：这个函数现在只被旧的 call_attacker 逻辑使用
    # 新的 `call_anonymizer` 使用 `apply_chat_template`
    # 新的 `call_attacker` 也应该使用 `apply_chat_template`
    return f"{system_content.strip()}\n\n{user_content.strip()}"

# --- 已修改：call_anonymizer ---
def call_anonymizer(pipe, question_asked: str, user_response: str, feedback: str, terminator_ids: List[int], record_id: int) -> str:
    # 移除了 profile_to_hide_str，添加了 feedback 和 record_id
    user_content = PROMPT_ANONYMIZER_USER.format(
        question_asked=str(question_asked),
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [
        {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()},
        {"role": "user", "content": user_content}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.debug(f"[Record {record_id}] --- ANONYMIZER PROMPT ---\n{prompt}\n-------------------------")
    
    temperature = 0.5
    do_sample = temperature > 0.0
    
    outputs = pipe(
        prompt,
        max_new_tokens=384,
        eos_token_id=terminator_ids,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.9 if do_sample else None,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]
    
    # 详细日志：在 DEBUG 级别打印完整原始输出
    logging.debug(f"[Record {record_id}] --- ANONYMIZER FULL OUTPUT ---\n{out}\n-------------------------")

    # 解析 CoT / # 输出
    parts = out.split('#', 1)
    explanation = ""
    if len(parts) == 2:
        explanation = parts[0].strip()
        final_response = parts[1].strip().strip('"').strip()
    else:
        explanation = "No explanation provided (LLM failed to follow format)."
        final_response = out.strip().strip('"').strip()
    
    logging.debug(f"[Record {record_id}] --- ANONYMIZER EXPLANATION ---\n{explanation}\n-------------------------")
    
    return final_response

# --- 已修改：call_attacker ---
def call_attacker(pipe, question_asked: str, user_response: str, terminator_ids: List[int], record_id: int) -> Dict[str, Any]:
    # 返回类型已更改为 Dict
    user_content = PROMPT_ATTACKER_USER.format(
        question_asked=str(question_asked),
        user_response=str(user_response)
    )
    # 使用 Chat Template 以保持一致
    messages = [
        {"role": "system", "content": PROMPT_ATTACKER_SYSTEM.strip()},
        {"role": "user", "content": user_content}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.debug(f"[Record {record_id}] --- ATTACKER PROMPT ---\n{prompt}\n-------------------------")
    
    temperature = 0.1
    do_sample = temperature > 0.0

    outputs = pipe(
        prompt,
        max_new_tokens=384,
        eos_token_id=terminator_ids,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.9 if do_sample else None,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]
    
    # 详细日志：在 DEBUG 级别打印完整原始输出
    logging.debug(f"[Record {record_id}] --- ATTACKER FULL OUTPUT ---\n{out}\n-------------------------")
    
    # 使用新的解析器
    parsed_data = parse_attacker_output(out)
    logging.debug(f"[Record {record_id}] --- ATTACKER PARSED ---\n{parsed_data}\n-------------------------")
    return parsed_data

# --- 已删除：call_judge ---
# (此函数被 `compare_profiles` 完全取代)

# --- 已修改：compare_profiles (修复签名) ---
def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    # 移除了 'pipe' 和 'terminator_ids' 参数
    """
    使用确定性的 Python 逻辑比较 profiles，不再调用 LLM。
    """
    leaked_attributes: List[str] = []
    
    # 辅助函数，用于格式化日志
    def _log(key, guess, true):
        return f"{key} (guessed: '{guess}', true: '{true}')"

    # 1) 年龄：数值容差判断 (保留)
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(_log("age", guessed_age, true_age))

    # 2) 简单字段 (必须精确匹配，忽略大小写)
    simple_keys = ["sex", "relationship_status"]
    for key in simple_keys:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        # 必须是字符串且非空
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            if true_val.strip().lower() == guess_val.strip().lower():
                leaked_attributes.append(_log(key, guess_val, true_val))
    
    # 3) 复杂字段 (子集/子字符串匹配)
    complex_keys = ["city_country", "birth_city_country", "education", "occupation", "income_level"]
    for key in complex_keys:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        # 必须是字符串且非空
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            true_s = true_val.strip().lower()
            guess_s = guess_val.strip().lower()

            if len(guess_s) < 3:
                continue
                
            if (guess_s in true_s) or (true_s in guess_s):
                leaked_attributes.append(_log(key, guess_val, true_val))

    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# --- 已修改：adversarial_anonymization (核心逻辑重写) ---
def adversarial_anonymization(
    pipe,
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int], 
    max_iterations: int = 5,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程，返回最终的匿名化响应和元数据。"""
    
    current_anonymized_response = original_response
    meta = {
        "status": "max_iterations_reached",
        "iterations_used": 0,
        "final_leaked_attributes": [],
        "final_attacker_guess": {}
    }
    
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    # --- 步骤 0: 初始攻击 (Attack-First) ---
    logging.info(f"[Record {record_id}] Starting initial attack (Round 0)...")
    try:
        parsed_attack = call_attacker(pipe, question_asked, original_response, terminator_ids, record_id)
        attacker_guess = parsed_attack.get("guess_json", {})
        feedback = parsed_attack.get("inference", "No inference provided by attacker.")
        meta["final_attacker_guess"] = attacker_guess
    except Exception as e:
        logging.error(f"[Record {record_id}][Round 0] Initial Attacker failed: {e}", exc_info=True)
        meta["status"] = "api_error"
        meta["error"] = f"initial_attacker_error: {e}"
        return current_anonymized_response, meta # 初始攻击失败

    # --- 步骤 0.5: 初始裁判 ---
    leaked = compare_profiles(true_personality, attacker_guess)
    meta["final_leaked_attributes"] = leaked
    
    if not leaked:
        logging.info(f"[Record {record_id}] Success on original text (0 iterations). No leaks found.")
        meta["status"] = "success_on_original"
        return current_anonymized_response, meta

    logging.info(f"[Record {record_id}] Initial attack leaked: {leaked}")
    # --- 循环开始 (现在使用真实的反馈) ---

    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        iteration_log_prefix = f"[Record {record_id}][Iter {i+1}/{max_iterations}]"
        
        # 'feedback' 来自上一步 (初始攻击或上一个循环)
        logging.info(f"{iteration_log_prefix} Current feedback (inference): {feedback[:100]}...")

        # 1) 匿名化
        try:
            logging.info(f"{iteration_log_prefix} Calling Anonymizer...")
            current_anonymized_response = call_anonymizer(
                pipe, question_asked, current_anonymized_response, feedback, terminator_ids, record_id
            )
        except Exception as e:
            logging.error(f"{iteration_log_prefix} Anonymizer failed: {e}", exc_info=True)
            meta["status"] = "api_error"
            meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"{iteration_log_prefix} Calling Attacker...")
            parsed_attack = call_attacker(pipe, question_asked, current_anonymized_response, terminator_ids, record_id)
            attacker_guess = parsed_attack.get("guess_json", {})
            feedback = parsed_attack.get("inference", "No inference provided by attacker.") # 为下一次循环更新反馈
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            logging.warning(f"{iteration_log_prefix} Attacker failed: {e}. Skipping judge for this round.", exc_info=True)
            # 不更新 feedback，使用上一轮的反馈
            continue

        # 3) 评判泄露
        logging.info(f"{iteration_log_prefix} Calling Judge (compare_profiles)...")
        leaked = compare_profiles(true_personality, attacker_guess)
        if not leaked:
            logging.info(f"{iteration_log_prefix} Success! No attributes leaked.")
            meta["status"] = "success"
            return current_anonymized_response, meta
        else:
            # 仅记录，'feedback' 已在步骤 2 中更新
            logging.info(f"{iteration_log_prefix} Failed. Leaked: {leaked}")
            meta["final_leaked_attributes"] = leaked

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta

def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    
    try:
        personality = data.get("personality") 
        question = str(data.get("question_asked")) 
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

    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        question_asked=question,       
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
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=5, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大新 token 数（全局）")
    
    parser.add_argument("--log_file", type=str, default="anonymizer.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别 (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    # 修改日志格式以包含 record_id (虽然它是在消息中添加的，但这里是全局格式)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(args.log_file, 'w', 'utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # --- 抑制（静音）嘈杂的库日志 ---
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # --- 结束日志静音 ---
    
    logging.info(f"Logging configured at level {args.log_level.upper()}. Library logs suppressed.")
    # --- End Logger Setup ---

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)

    records_to_process = [json.loads(line) for line in lines if line.strip()]
    if args.limit:
        records_to_process = records_to_process[:args.limit]

    try:
        gen_pipe, tokenizer = build_pipeline(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}", exc_info=True)
        sys.exit(1)
    
    # 动态获取 terminators
    terminator_ids = [
        tokenizer.eos_token_id,
        # 尝试添加常见的聊天模板停止符
        tokenizer.convert_tokens_to_ids("<|eot_id|>"), # Llama 3
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"), # Cohere
        tokenizer.convert_tokens_to_ids("<|im_end|>") # ChatML
    ]
    terminator_ids = [tid for tid in terminator_ids if tid is not None and not isinstance(tid, list)] # 移除 None
    logging.info(f"Using terminators: {terminator_ids}")

    logging.info(f"Starting processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []
    
    # --- 添加状态计数器 ---
    counters = {
        "success": 0, "success_on_original": 0, "max_iterations_reached": 0, 
        "api_error": 0, "skipped_data_read_error": 0, 
        "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, 
        "unknown_fail": 0
    }

    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminator_ids)

    for i, rec in enumerate(tqdm(records_to_process, desc="Anonymizing profiles")):
        try:
            processed_record = _task(i, rec)
            results.append(processed_record)
            # 记录状态
            status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
            if status in counters:
                counters[status] += 1
            else:
                counters["unknown_fail"] += 1
        except Exception as exc:
            logging.error(f"[Record {i}] Generated an unhandled exception: {exc}", exc_info=True)
            counters["unknown_fail"] += 1


    # 写结果
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:

        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            meta = result.get("anonymization_meta", {})
            status = meta.get("status")
            
            # --- 已修改：确保所有非成功状态都进入 failed_file ---
            if status in ("success", "success_on_original"):
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            else: # 捕获所有其他状态 (失败, 跳过, 错误)
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    # --- 记录最终摘要 ---
    logging.info("--- Processing Summary ---")
    logging.info(f"Total records processed: {len(results)}")
    for status, count in counters.items():
        if count > 0:
            logging.info(f"  {status}: {count}")
    logging.info("--- End Summary ---")

    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    if args.success_file:
        logging.info(f"Successful anonymizations saved to '{args.success_file}'.")
    if args.failed_file:
        logging.info(f"Failed/skipped anonymizations saved to '{args.failed_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()