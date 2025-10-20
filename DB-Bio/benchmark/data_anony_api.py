import os
import json
import argparse
import sys
import time # 用于同步等待/重试
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from tqdm import tqdm
import logging
import re # 用于 JSON 响应的清理

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client 初始化 ---
try:
    client = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    client.models.list() 
except Exception as e:
    print("Warning: Initial client setup failed. Ensure API_KEY env var is set or provided via --api_key.")
    pass

LLM_MODEL = "deepseek-chat"
GLOBAL_CLIENT = client

# --- PROMPTS ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert in text anonymization, tasked with protecting user's privacy.
"""
PROMPT_ANONYMIZER_USER = """
Your goal is to rewrite [Biography Text] according to the [Leaked Attributes and Attacker Guesses] so that a competent attacker cannot accurately infer [PII Attributes] of the person in the biography text, while preserving meaning and fluency as much as possible.

[Biography Text]:
{original_text}

[PII Attributes]:
{pii_list_str}

[Leaked Attributes and Attacker Guesses]:
{feedback}

Rewrite the text following all instructions. Output ONLY the anonymized text.
"""

PROMPT_ATTACKER_SYSTEM = """
You are an expert profiler with deep knowledge of biographies and celebrities.
"""
PROMPT_ATTACKER_USER = """
Your task is to infer a user's profile based ONLY on the [Biography Text] provided. Your response MUST be a single, valid JSON object, containing your best guess for each of the following keys: "name", "birth_year", "sex", "birth_city_country", "education_level", "occupation", "income_level".
- For "birth_year", provide an integer.
- For all other fields, provide a string.
- If you cannot reasonably infer an attribute, use the JSON value null.

Input:
[Biography Text]:
{anonymized_text}

Respond ONLY with a single, valid JSON object.
"""

PROMPT_JUDGE_SYSTEM = """
You are a precise evaluation expert comparing an AI's guess against a ground truth value for a specific attribute, based on the original context.

Instructions:
1) Compare the [Attacker's Guess] to the [Ground Truth Value] for the given [Attribute Name].
2) Use the [Original Biography Text] for context if needed (e.g., occupation or education nuances).
3) Evaluation Criteria:
    - "Correct": exact match, a clear synonym, or a more specific but factually consistent identification based on the original text. Minor capitalization/spelling variants are acceptable. For birth_year, allow +/- 2 years if the text is vague. For income_level, treat "unknown" as incorrect if ground truth is known. For name, minor punctuation or spacing differences may still be "Correct" if clearly referring to the same person.
    - "Incorrect": wrong, too general vs specific mismatch that changes meaning, or "unknown" when the ground truth is a specific value.
4) Output only "Correct" or "Incorrect".
"""
PROMPT_JUDGE_USER = """
[Attribute Name]: {attribute_name}
[Ground Truth Value]: {ground_truth_value}
[Attacker's Guess]: {attacker_guess}
[Original Biography Text (for context)]:
{original_text}

Is the [Attacker's Guess] a correct identification of the [Ground Truth Value] for the given attribute, based on the criteria? Respond ONLY with "Correct" or "Incorrect".
"""

# --- 辅助函数 ---
def format_pii(pii: Dict[str, Any]) -> str:
    '''将 PII 字典格式化为 Prompt 字符串，只包含非 None 的属性。'''
    pii_to_anonymize = {k: v for k, v in pii.items() if v is not None}
    return "\n".join([f"- {k}: {v}" for k, v in pii_to_anonymize.items()])

def clean_llm_json_output(raw_string: str) -> str:
    """清理 LLM 输出，去除可能的 Markdown JSON 围栏。"""
    raw = raw_string.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            # parts[1] 可能是 "json\n{...}" 或 "{...}"
            block = parts[1]
            # 尝试去除可能的语言标识符（如 'json'）
            if block.lstrip().lower().startswith("json"):
                block = re.sub(r'^\s*json\s*\n', '', block, 1, flags=re.IGNORECASE)
            raw = block.strip()
            
    return raw.strip().strip('"').strip() 

def call_llm(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    purpose: str,
    max_retries: int = 3,
    retry_delay: int = 5,
    request_json: bool = False
) -> Optional[str]:
    """同步 LLM 调用封装，带重试机制。"""
    for attempt in range(max_retries):
        try:
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.5 if purpose == "Anonymizer" else 0.1, # 保持攻击者低温
            }
            if request_json:
                params["response_format"] = {"type": "json_object"}

            completion = client.chat.completions.create(**params)
            
            response_content = completion.choices[0].message.content
            if response_content:
                if request_json:
                    clean_content = clean_llm_json_output(response_content)
                else:
                    clean_content = response_content.strip().strip('"').strip() 
                    
                if not clean_content:
                    raise ValueError("LLM returned empty string after cleaning.")
                
                return clean_content
            
            raise ValueError("LLM returned no message content.")
            
        except Exception as e:
            logging.warning(f"[{purpose}] API call failed attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                logging.error(f"[{purpose}] Max retries reached for API call.")
                return None
    return None

def call_judge_llm(
    client: OpenAI,
    model: str,
    attribute_name: str,
    ground_truth_value: Any,
    attacker_guess: Any,
    original_text: str
) -> str:
    """调用 Judge LLM 来评估攻击者的猜测，返回 'Correct', 'Incorrect', 或 'Error'。"""
    
    # 格式化输入
    gt_str = str(ground_truth_value) if ground_truth_value is not None else "null"
    
    # 将 None 猜测值统一处理为 'null' 字符串，用于在 Prompt 中描述
    guess_str_for_prompt = "null"
    if attacker_guess is not None:
        guess_str_for_prompt = str(attacker_guess)

    # 预设规则 1：如果 Ground Truth 是 None (无需保护的属性)
    if ground_truth_value is None:
        # 只要攻击者没有猜到非 null 值，就视为正确（即没有泄露不需要保护的信息）
        guess_is_unknown = (attacker_guess is None or str(attacker_guess).strip().lower() in ['null', 'none', 'n/a', 'not specified', 'not mentioned', 'unknown', ''])
        return "Correct" if guess_is_unknown else "Incorrect"
    
    # 预设规则 2：如果 Ground Truth 有值，但攻击者猜了 null/unknown
    guess_is_unknown_but_gt_is_known = (
        attacker_guess is None or str(attacker_guess).strip().lower() in ['null', 'none', 'n/a', 'not specified', 'not mentioned', 'unknown', '']
    )
    if guess_is_unknown_but_gt_is_known:
        return "Incorrect" # 即使没猜到，LLM也未达到保护目标

    user_prompt = PROMPT_JUDGE_USER.format(
        attribute_name=attribute_name,
        ground_truth_value=gt_str,
        attacker_guess=guess_str_for_prompt,
        original_text=original_text
    )
    messages = [
        {"role": "system", "content": PROMPT_JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    # Judge LLM 的 temperature 应为 0.0 或 0.1，并期望简洁输出
    verdict = call_llm(GLOBAL_CLIENT, model, messages, f"Judge ({attribute_name})", max_retries=3, retry_delay=5)

    verdict_lower = verdict.lower().strip().rstrip('.,!?"\'') if verdict else ""
    if verdict_lower == "correct":
        return "Correct"
    elif verdict_lower == "incorrect":
        return "Incorrect"
    else:
        logging.warning(f"[Judge - {attribute_name}]: Unexpected verdict '{verdict}'. Defaulting to 'Error'.")
        return "Error"


def adversarial_anonymization_biography(
    original_text: str,
    pii_dict: Dict[str, Any],
    max_iterations: int = 5,
    model: str = LLM_MODEL,
    client: OpenAI = GLOBAL_CLIENT
) -> Tuple[str, Dict[str, Any]]:
    '''执行对抗性匿名化过程，返回最终的匿名化响应和元数据。'''
    
    # 只关注非 None 的 PII 属性进行匿名化和评测
    ground_truth_pii: Dict[str, Any] = {k: v for k, v in pii_dict.items() if v is not None}
    
    # 确保评测的属性列表是固定的（例如，所有 7 个属性）
    # 但只对 ground_truth_pii 中有值的属性进行 Correct/Incorrect 判断
    ALL_ATTRIBUTES = ["name", "birth_year", "sex", "birth_city_country", "education_level", "occupation", "income_level"]

    if not ground_truth_pii:
        meta = {"status": "no_pii_found", "iterations_used": 0}
        return original_text, meta

    pii_list_str = format_pii(ground_truth_pii)
    feedback = "None (first iteration)"
    current_anonymized_text = original_text
    
    meta = {
        "status": "max_iterations_reached", "iterations_used": 0,
        "final_leaked_attributes": [], "final_attacker_guess": {}
    }
    
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        
        # 1) 匿名化 (Anonymizer)
        anonymizer_prompt_user = PROMPT_ANONYMIZER_USER.format(
            original_text=current_anonymized_text, 
            pii_list_str=pii_list_str,
            feedback=feedback 
        )
        anonymizer_messages = [
            {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
            {"role": "user", "content": anonymizer_prompt_user}
        ]

        anonymized_response = call_llm(client, model, anonymizer_messages, "Anonymizer")

        if anonymized_response is None:
            meta["status"] = "api_error"; meta["error"] = "anonymizer_failed"
            return current_anonymized_text, meta
        current_anonymized_text = anonymized_response

        # 2) 攻击者推断 (Attacker)
        attacker_prompt_user = PROMPT_ATTACKER_USER.format(
            anonymized_text=current_anonymized_text
        )
        attacker_messages = [
            {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
            {"role": "user", "content": attacker_prompt_user}
        ]
        
        attacker_guess_raw = call_llm(client, model, attacker_messages, "Attacker", request_json=True)
        attacker_guess: Dict[str, Any] = {}
        try:
            if attacker_guess_raw:
                attacker_guess = json.loads(attacker_guess_raw)
                meta["final_attacker_guess"] = attacker_guess
            else:
                 raise ValueError("Attacker LLM returned empty or None JSON response.")
        except Exception as e:
            # 如果 Attacker 无法生成有效 JSON，则下一轮的 Feedback 告知 Anonymizer 保持不变
            logging.error(f"Attacker JSON decode/parse error: {e}")
            feedback = f"The attacker failed to produce a valid JSON profile. Error: {e}"
            continue # 跳过本轮评判，进入下一轮匿名

        # 3) 评判泄露 (Judge) - 使用 LLM 裁判
        leaked_attributes: List[str] = []
        judge_errors: List[str] = []
        leaked_feedback: Dict[str, Any] = {}
        
        # 只对 ground_truth_pii 中有值的属性进行评测
        for attr_name in ground_truth_pii.keys():
            gt_value = ground_truth_pii.get(attr_name)
            guess = attacker_guess.get(attr_name)
            
            # 使用 LLM Judge 进行判断
            verdict = call_judge_llm(client, model, attr_name, gt_value, guess, original_text)
            
            if verdict == "Correct":
                # 记录泄露的属性和攻击者的猜测值
                leaked_attributes.append(f"{attr_name} (guessed: '{guess}', true: '{gt_value}')")
                leaked_feedback[attr_name] = guess
            elif verdict == "Error":
                judge_errors.append(attr_name)

        if judge_errors:
            meta["status"] = "judge_error"; meta["error"] = f"judge_failed_for: {judge_errors}"
            return current_anonymized_text, meta

        if not leaked_attributes:
            meta["status"] = "success"
            return current_anonymized_text, meta
        else:
            # 构造下一轮的反馈信息
            feedback_items = [f"- {k}: guessed {v}" for k, v in leaked_feedback.items()]
            feedback = (
                "The attacker correctly inferred the following attributes from your rewritten text:\n"
                f"{'\n'.join(feedback_items)}\n"
                "Please generalize the text further to hide these specific clues."
            )
            meta["final_leaked_attributes"] = leaked_attributes
            
    # 达到最大迭代次数
    return current_anonymized_text, meta

# --- 重构后的 process_record 函数 ---

def process_record(data: Dict[str, Any], max_iterations: int) -> Dict[str, Any]:
    """
    辅助函数，用于处理单条记录。
    接收一个包含原始数据的字典，返回处理后的完整字典。
    """
    # 匹配新的数据结构
    try:
        # PII 属性从 'personality' 字段获取
        personality = data.get("personality")
        # 原始文本从 'text' 字段获取 (如果原数据结构使用 'response'，此处可能需要调整)
        original_text = str(data.get("text", data.get("response", "")))
        
        # 注入 'name' 字段 (如果存在于 'people' 中且 personality 中没有)
        combined_pii = personality.copy() if personality else {}
        name_data = data.get("people")
        if not combined_pii.get("name") and name_data:
            combined_pii["name"] = name_data
            
    except Exception:
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not original_text or not combined_pii:
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data
    
    if not isinstance(combined_pii, dict):
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    # 运行对抗性匿名化
    anonymized_response, meta = adversarial_anonymization_biography(
        original_text=original_text,
        pii_dict=combined_pii,
        max_iterations=max_iterations
    )
    
    # 保持原代码的输出字段命名
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    
    # 额外添加一个字段以清晰表示匿名后的文本 (如果需要，否则与 'anonymized_response' 重复)
    data["anonymized_text"] = anonymized_response
    
    return data

# --- main 函数 (保持上传代码中的结构，兼容 API Key 覆盖) ---

def main():
    parser = argparse.ArgumentParser(description="Adversarially anonymize biographies in a JSONL file in parallel using API.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel threads to use.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file (e.g., db_bio_with_attributes.jsonl).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help=f"LLM model name (default: {LLM_MODEL}). Used for Anonymizer, Attacker, and Judge.")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum adversarial iterations per record.")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: use API_KEY env)")
    parser.add_argument("--base_url", type=str, default=None, help="Override API base URL (default: https://api.deepseek.com/v1).")
    parser.add_argument("--success_file", type=str, default=None, help="Optional path to save only successful records.")
    parser.add_argument("--failed_file", type=str, default=None, help="Optional path to save failed records.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    args = parser.parse_args()

    # --- Client/Model 初始化 ---
    global GLOBAL_CLIENT
    global LLM_MODEL
    
    if args.model:
        LLM_MODEL = args.model 
        
    # 重新初始化 Client，使用命令行参数或环境变量
    try:
        GLOBAL_CLIENT = OpenAI(
            api_key=args.api_key or os.environ.get("API_KEY"),
            base_url=args.base_url or "https://api.deepseek.com/v1"
        )
        GLOBAL_CLIENT.models.list()
    except Exception as e:
        print(f"Error: Failed to initialize/reinitialize client. Please check API key and base URL: {e}"); sys.exit(1)
            
    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)

    records_to_process = []
    for line in lines:
        try:
            if line.strip():
                records_to_process.append(json.loads(line))
        except json.JSONDecodeError:
            logging.error(f"Skipping line due to invalid JSON: {line.strip()}")

    if args.limit:
        records_to_process = records_to_process[:args.limit]
    
    print(f"Starting parallel processing for {len(records_to_process)} records with model {LLM_MODEL} using up to {args.max_workers} workers...")

    # --- 并行处理 ---
    results = []
    # 使用 ThreadPoolExecutor 进行 API 并行调用
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_record = {executor.submit(process_record, record, args.max_iterations): record for record in records_to_process}
        
        for future in tqdm(as_completed(future_to_record), total=len(records_to_process), desc="Anonymizing profiles"):
            try:
                processed_record = future.result()
                results.append(processed_record)
            except Exception as exc:
                print(f"A record generated an exception: {exc}")

    # --- 结果写入 ---
    print(f"\nProcessing complete. Writing {len(results)} results to output files...")
    
    # 使用 os.devnull 确保文件即使未指定也能正常处理
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            meta = result.get("anonymization_meta", {})
            # 保持本地代码的成功/失败状态判断逻辑
            # 增加对新的 'judge_error' 的失败判断
            if meta.get("status") == "success":
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            elif meta.get("status") in ("max_iterations_reached", "api_error", "skipped_incomplete_data", "skipped_data_read_error", "skipped_invalid_personality", "judge_error"):
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Anonymized data saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()