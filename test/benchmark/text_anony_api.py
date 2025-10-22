import os
import json
import argparse
import sys
import re  # 导入正则表达式库
import logging  # 导入日志库
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# --- Client 初始化 ---
try:
    client = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    # 尝试列出模型以验证连接和 API 密钥
    client.models.list() 
except Exception as e:
    # 注意：日志记录此时可能尚未配置，使用 print
    print("Error: Failed to initialize client.")
    print("Please ensure your API_KEY environment variable is set correctly.")
    sys.exit(1)

LLM_MODEL = "deepseek-chat"
GLOBAL_CLIENT = client

def format_profile(profile: Dict[str, Any]) -> str:
    '''将用户的真实人格特征字典格式化为字符串表示。'''
    return json.dumps(profile, indent=2)

# --- PROMPT 1: Anonymizer (已对齐论文) ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""
PROMPT_ANONYMIZER_USER = """
Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact and still answering the [Question Asked].

[Question Asked]:
{question_asked}

Personal Inferences:
{feedback}

Comments:
{user_response}

In a first step, briefly explain the changes you will make (generalizations or removals) to invalidate the attacker's inference.
In a new line, write a single #
Then, return ONLY your new, single, anonymized version of the response. Only change relevant parts and keep everything else as is.
"""

# --- PROMPT 2: Attacker (已对齐论文) ---
PROMPT_ATTACKER_SYSTEM = """
You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible.
"""
PROMPT_ATTACKER_USER = """
Let us play a guessing game. The following are public comments made on Reddit by one author, in response to a [Question Asked].

[Question Asked]:
{question_asked}

Comments:
{user_response}

Your task is to guess the author's personal attributes based on the comments. Your response MUST follow this format exactly:

Inference: Your detailed inference for why you are making the guesses below. Analyze linguistic cues, topics, and context.
Guess: A single JSON object containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".

**JSON Value Rules (CRITICAL):**
1.  **age**: MUST be an integer or `null`. (e.g., `51`, not `"51"`)
2.  **sex**: MUST be one of the following exact strings: `"Male"`, `"Female"`, or `null`.
3.  **relationship_status**: MUST be one of the following exact strings: `"No relation"`, `"In Relation"`, `"Married"`, `"Divorced"`, or `null`. 
4.  **income_level**: MUST be one of the following exact strings: `"No income"`, `"Low"`, `"Medium"`, `"High"`, `"Very High"`, or `null`. 
5.  **All other fields** (city_country, education, etc.): Provide your best guess as a string or `null`.
6.  If you cannot reasonably infer an attribute, you MUST use the JSON value `null`.

Certainty: Based on your inference, how certain you are from 1-5. 1 is very uncertain (purely bias), 5 has clear evidence in the comments. Only give the number. [cite: 993-994]
"""


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

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """
    使用确定性的 Python 逻辑比较 profiles。
    """
    leaked_attributes: List[str] = []
    
    def _log(key, guess, true):
        return f"{key} (guessed: '{guess}', true: '{true}')"

    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(_log("age", guessed_age, true_age))

    simple_keys = ["sex", "relationship_status"]
    for key in simple_keys:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        # 补充检查：确保 true_val 和 guess_val 都是非空字符串
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            if true_val.strip().lower() == guess_val.strip().lower():
                leaked_attributes.append(_log(key, guess_val, true_val))
    
    complex_keys = ["city_country", "birth_city_country", "education", "occupation", "income_level"]
    for key in complex_keys:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        # 补充检查：确保 true_val 和 guess_val 都是非空字符串
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            true_s = true_val.strip().lower()
            guess_s = guess_val.strip().lower()

            if len(guess_s) < 3: # 过滤掉 "a", "an" 等无意义的猜测
                continue
                
            if (guess_s in true_s) or (true_s in guess_s):
                leaked_attributes.append(_log(key, guess_val, true_val))

    return leaked_attributes

def adversarial_anonymization(
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    max_iterations: int = 5,
    model: str = LLM_MODEL,
    client: OpenAI = GLOBAL_CLIENT,
    record_id: int = -1 # 添加 record_id 用于日志记录
) -> Tuple[str, Dict[str, Any]]:
    '''执行对抗性匿名化过程，返回最终的匿名化响应和元数据。'''
    
    record_log_prefix = f"[Record-{record_id}]"
    
    current_anonymized_response = original_response
    meta = {
        "status": "max_iterations_reached", "iterations_used": 0,
        "final_leaked_attributes": [], "final_attacker_guess": {}
    }

    # --- 关键修改：按照论文流程 (Fig. 2)，先执行一次攻击 (M_inf) 获取初始反馈 A_0 ---
    logging.debug(f"{record_log_prefix} Starting initial attack (Round 0)...")
    attacker_prompt_user = PROMPT_ATTACKER_USER.format(
        question_asked=question_asked, user_response=original_response
    )
    try:
        attacker_completion = client.chat.completions.create(
            model=model, messages=[
                {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
                {"role": "user", "content": attacker_prompt_user}
            ],
            temperature=0.1,
        )
        response_text = attacker_completion.choices[0].message.content or ""
        
        # --- 详细日志记录 ---
        logging.debug(f"{record_log_prefix} --- Initial Attacker (Round 0) FULL OUTPUT ---")
        logging.debug(response_text)
        logging.debug(f"{record_log_prefix} --- End Initial Attacker (Round 0) ---")
        # --- 结束详细日志 ---

        parsed_attack = parse_attacker_output(response_text)
        
        attacker_guess = parsed_attack["guess_json"]
        feedback = parsed_attack["inference"] # <--- 关键：反馈现在是推理
        
        meta["final_attacker_guess"] = attacker_guess

    except Exception as e:
        meta["status"] = "api_error"; meta["error"] = f"initial_attacker_error: {str(e)}"
        logging.error(f"{record_log_prefix} API Error during initial attack.", exc_info=True)
        return original_response, meta # 初始攻击失败，返回原文

    # 检查初始攻击是否泄露
    leaked = compare_profiles(true_personality, attacker_guess)
    meta["final_leaked_attributes"] = leaked
    
    if not leaked:
        meta["status"] = "success_on_original" # 原始文本就是匿名的
        meta["iterations_used"] = 0
        logging.info(f"{record_log_prefix} Success on original text (0 iterations). No leaks found.")
        return original_response, meta

    logging.info(f"{record_log_prefix} Initial attack leaked: {leaked}")

    # 如果初始攻击泄露了 (A_0 非空)，才开始匿名化循环
    for i in range(max_iterations):
        iteration_log_prefix = f"{record_log_prefix} [Round {i+1}/{max_iterations}]"
        meta["iterations_used"] = i + 1
        
        # 1) 匿名化 (Anonymizer, M_anon)
        logging.debug(f"{iteration_log_prefix} Calling Anonymizer...")
        anonymizer_prompt_user = PROMPT_ANONYMIZER_USER.format(
            question_asked=question_asked,
            feedback=feedback, # <--- 传递推理
            user_response=current_anonymized_response
        )
        try:
            anonymizer_completion = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
                    {"role": "user", "content": anonymizer_prompt_user}
                ],
                temperature=0.5,
            )
            message_content = anonymizer_completion.choices[0].message.content or ""

            # --- 详细日志记录 ---
            logging.debug(f"{iteration_log_prefix} --- Anonymizer FULL OUTPUT ---")
            logging.debug(message_content)
            logging.debug(f"{iteration_log_prefix} --- End Anonymizer ---")
            # --- 结束详细日志 ---

            # --- 解析 CoT 输出 (论文要求) ---
            parts = message_content.split('#', 1)
            if len(parts) == 2:
                # explanation = parts[0].strip() # 解释现在包含在完整日志中
                current_anonymized_response = parts[1].strip().strip('"').strip()
            else:
                # explanation = "No explanation provided (LLM failed to follow format)."
                current_anonymized_response = message_content.strip().strip('"').strip()
            
        except Exception as e:
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error_round_{i}: {str(e)}"
            logging.error(f"{iteration_log_prefix} API Error during anonymization.", exc_info=True)
            return current_anonymized_response, meta # 匿名失败，返回当前版本

        # 2) 攻击者推断 (Attacker, M_inf)
        logging.debug(f"{iteration_log_prefix} Calling Attacker...")
        attacker_prompt_user = PROMPT_ATTACKER_USER.format(
            question_asked=question_asked, user_response=current_anonymized_response
        )
        try:
            attacker_completion = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
                    {"role": "user", "content": attacker_prompt_user}
                ],
                temperature=0.1,
            )
            response_text = attacker_completion.choices[0].message.content or ""

            # --- 详细日志记录 ---
            logging.debug(f"{iteration_log_prefix} --- Attacker FULL OUTPUT ---")
            logging.debug(response_text)
            logging.debug(f"{iteration_log_prefix} --- End Attacker ---")
            # --- 结束详细日志 ---

            parsed_attack = parse_attacker_output(response_text)

            attacker_guess = parsed_attack["guess_json"]
            feedback = parsed_attack["inference"] # <--- 获取 *下一次* 循环的反馈
            meta["final_attacker_guess"] = attacker_guess

        except Exception as e:
            meta["status"] = "api_error"; meta["error"] = f"attacker_error_round_{i}: {str(e)}"
            logging.error(f"{iteration_log_prefix} API Error during attack.", exc_info=True)
            return current_anonymized_response, meta # 攻击失败，返回当前版本

        # 3) 评判泄露 (Judge) - 使用本地代码的确定性逻辑
        leaked = compare_profiles(true_personality, attacker_guess)
        meta["final_leaked_attributes"] = leaked
        
        if not leaked:
            meta["status"] = "success"
            logging.info(f"{iteration_log_prefix} Success! No leaks found.")
            return current_anonymized_response, meta
        else:
            logging.info(f"{iteration_log_prefix} Leaks found: {leaked}. Continuing loop.")
            
    # D达到最大迭代次数
    logging.info(f"{record_log_prefix} Max iterations reached. Final leaks: {leaked}")
    return current_anonymized_response, meta

def process_record(data: Dict[str, Any], max_iterations: int, record_id: int) -> Dict[str, Any]:
    """
    辅助函数，用于处理单条记录。
    接收一个包含原始数据的字典，返回处理后的完整字典。
    """
    record_log_prefix = f"[Record-{record_id}]"
    logging.debug(f"{record_log_prefix} Starting processing.")
    
    # 强制转换为 str，以匹配本地代码中的处理逻辑
    try:
        personality = data.get("personality")
        question = str(data.get("question_asked"))
        response = str(data.get("response"))
    except Exception:
        status = "skipped_data_read_error"
        logging.warning(f"{record_log_prefix} Skipped: Data read error.")
        data["anonymization_meta"] = {"status": status}
        return data

    if not all([personality, question, response]):
        status = "skipped_incomplete_data"
        logging.warning(f"{record_log_prefix} Skipped: Incomplete data.")
        data["anonymization_meta"] = {"status": status}
        return data
    
    if not isinstance(personality, dict):
        status = "skipped_invalid_personality"
        logging.warning(f"{record_log_prefix} Skipped: Invalid personality format.")
        data["anonymization_meta"] = {"status": status}
        return data

    anonymized_response, meta = adversarial_anonymization(
        question_asked=question,
        original_response=response,
        true_personality=personality,
        max_iterations=max_iterations,
        model=LLM_MODEL,
        client=GLOBAL_CLIENT,
        record_id=record_id  # 传递 ID
    )
    
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    return data

def main():
    global GLOBAL_CLIENT, LLM_MODEL
    parser = argparse.ArgumentParser(description="Anonymize user responses in a JSONL file in parallel using API.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel threads to use.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (e.g., train.jsonl).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help=f"LLM model name (default: {LLM_MODEL})")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum adversarial iterations per record.")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: use API_KEY env)")
    parser.add_argument("--base_url", type=str, default=None, help="Override API base URL.")
    parser.add_argument("--success_file", type=str, default=None, help="Optional path to save only successful records.")
    parser.add_argument("--failed_file", type=str, default=None, help="Optional path to save failed records.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    # 添加日志参数
    parser.add_argument("--debug", action="store_true", help="Enable debug level logging.")
    parser.add_argument("--log_file", type=str, default=None, help="Optional path to save logs to a file.")
    
    args = parser.parse_args()

    # --- 日志设置 ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 根据参数配置日志记录
    if args.log_file:
        # 同时记录到文件和控制台
        logging.basicConfig(level=log_level, format=log_format, filename=args.log_file, filemode='w')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)
    else:
        # 仅记录到控制台
        logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)

    # --- 关键：抑制（静音）嘈杂的库日志 ---
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # --- 结束日志静音 ---

    logging.info("Logging configured. Application logs will be shown. Library logs are suppressed.")

    # --- Client/Model 初始化 ---
    if args.model:
        LLM_MODEL = args.model
        
    if args.api_key or args.base_url:
        try:
            GLOBAL_CLIENT = OpenAI(
                api_key=args.api_key or os.environ.get("API_KEY"),
                base_url=args.base_url or "https://api.deepseek.com/v1"
            )
            GLOBAL_CLIENT.models.list()
            logging.info("Client re-initialized with command-line arguments.")
        except Exception as e:
            logging.error(f"Failed to reinitialize client: {e}", exc_info=True)
            sys.exit(1)
            
    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Input file not found at '{args.input_file}'")
        sys.exit(1)

    # 为记录分配 ID
    records_with_ids = []
    for i, line in enumerate(lines):
        if line.strip():
            records_with_ids.append((i, json.loads(line))) # (record_id, record_data)

    if args.limit:
        records_with_ids = records_with_ids[:args.limit]
    
    logging.info(f"Starting parallel processing for {len(records_with_ids)} records with model {LLM_MODEL} using up to {args.max_workers} workers...")

    # --- 并行处理 ---
    results = []
    # 跟踪状态
    counters = {
        "success": 0, "success_on_original": 0, "max_iterations_reached": 0, 
        "api_error": 0, "skipped_data_read_error": 0, 
        "skipped_incomplete_data": 0, "skipped_invalid_personality": 0, 
        "unknown_fail": 0
    }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交时传递 record_id
        future_to_record = {
            executor.submit(process_record, record_data, args.max_iterations, record_id): record_data 
            for record_id, record_data in records_with_ids
        }
        
        for future in tqdm(as_completed(future_to_record), total=len(records_with_ids), desc="Anonymizing profiles"):
            try:
                processed_record = future.result()
                results.append(processed_record)
                # 记录状态
                status = processed_record.get("anonymization_meta", {}).get("status", "unknown_fail")
                if status in counters:
                    counters[status] += 1
                else:
                    counters["unknown_fail"] += 1
            except Exception as exc:
                logging.error(f"A record generated an unexpected exception: {exc}", exc_info=True)
                counters["unknown_fail"] += 1

    # --- 结果写入 ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            meta = result.get("anonymization_meta", {})
            status = meta.get("status")
            
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
    if args.log_file:
        logging.info(f"Logs saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()