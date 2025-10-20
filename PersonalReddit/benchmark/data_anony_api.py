import os
import json
import argparse
import sys
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
    print("Error: Failed to initialize client.")
    print("Please ensure your API_KEY environment variable is set correctly.")
    sys.exit(1)

LLM_MODEL = "deepseek-chat"
GLOBAL_CLIENT = client

def format_profile(profile: Dict[str, Any]) -> str:
    '''将用户的真实人格特征字典格式化为字符串表示。'''
    return json.dumps(profile, indent=2)

# --- PROMPTS ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert in text anonymization, tasked with protecting user's privacy.
"""
PROMPT_ANONYMIZER_USER = """
Your objective is to rewrite the [User Response] according to the [Feedback from Attacker], so that the personal attributes listed in the [True Personality Profile] cannot be inferred by attacker. The rewritten response must still be a natural and coherent answer to the [Question Asked].

[True Personality Profile]:
{profile_to_hide}

[Question Asked]:
{question_asked}

[Feedback from Attacker]:
{feedback}

[User Response]:
{user_response}

Now, provide ONLY your new, single, anonymized version of the response. Do not add any extra commentary or preamble.
"""

PROMPT_ATTACKER_SYSTEM = """
You are an expert profiler with deep knowledge of psycho-linguistics and cultural cues.
"""
PROMPT_ATTACKER_USER = """
Based on the conversational context provided by the [Question Asked] and the [User Response], your goal is to infer the user's personality profile.

[Question Asked]:
{question_asked}

[User Response]:
{user_response}

Your response MUST be a single, valid JSON object, containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
- For "age", provide an integer.
- For all other fields, provide a string.
- If you cannot reasonably infer an attribute, use the JSON value null.
"""

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    """
    使用确定性的 Python 逻辑比较 profiles。
    """
    leaked_attributes: List[str] = []
    
    # 辅助函数，用于格式化日志/输出
    def _log(key, guess, true):
        # 兼容旧代码的输出格式，并包含真实值
        return f"{key} (guessed: '{guess}', true: '{true}')"

    # 1) 年龄：数值容差判断 (<= 3)
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(_log("age", guessed_age, true_age))

    # 2) 简单字段 (精确匹配，忽略大小写)
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
            # 规范化为小写
            true_s = true_val.strip().lower()
            guess_s = guess_val.strip().lower()

            # 过滤掉无意义的短猜测
            if len(guess_s) < 3:
                continue
                
            # 核心逻辑：猜测是真实值的一部分 OR 真实值是猜测的一部分
            if (guess_s in true_s) or (true_s in guess_s):
                leaked_attributes.append(_log(key, guess_val, true_val))

    return leaked_attributes

def adversarial_anonymization(
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    max_iterations: int = 5,
    model: str = LLM_MODEL,
    client: OpenAI = GLOBAL_CLIENT
) -> Tuple[str, Dict[str, Any]]:
    '''执行对抗性匿名化过程，返回最终的匿名化响应和元数据。'''
    feedback = "No feedback yet, because this is the first attempt."
    current_anonymized_response = original_response
    profile_to_hide_str = format_profile(true_personality)
    meta = {
        "status": "max_iterations_reached", "iterations_used": 0,
        "final_leaked_attributes": [], "final_attacker_guess": {}
    }
    
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        
        # 1) 匿名化 (Anonymizer)
        anonymizer_prompt_user = PROMPT_ANONYMIZER_USER.format(
            profile_to_hide=profile_to_hide_str, question_asked=question_asked,
            feedback=feedback, user_response=current_anonymized_response
        )
        try:
            anonymizer_completion = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
                    {"role": "user", "content": anonymizer_prompt_user}
                ],
                temperature=0.5,
            )
            message_content = anonymizer_completion.choices[0].message.content
            if message_content is None:
                raise ValueError("Anonymizer LLM returned no message content.")
            else:
                current_anonymized_response = message_content.strip().strip('"').strip() 
        except Exception as e:
            meta["status"] = "api_error"; meta["error"] = f"anonymizer_error: {str(e)}"
            return current_anonymized_response, meta

        # 2) 攻击者推断 (Attacker)
        attacker_prompt_user = PROMPT_ATTACKER_USER.format(
            question_asked=question_asked, user_response=current_anonymized_response
        )
        attacker_guess = None
        try:
            attacker_completion = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
                    {"role": "user", "content": attacker_prompt_user}
                ],
                temperature=0.1, # 与本地代码中的温度一致
                response_format={"type": "json_object"},
            )
            message_content = attacker_completion.choices[0].message.content
            if message_content is None:
                raise ValueError("Attacker LLM returned no message content.")
            
            response_text = message_content.strip()
            attacker_guess = json.loads(response_text)
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            feedback = f"The attacker failed to produce a valid JSON profile. Error: {e}"
            continue

        # 3) 评判泄露 (Judge) - 使用本地代码的确定性逻辑
        leaked = compare_profiles(true_personality, attacker_guess)
        
        if not leaked:
            meta["status"] = "success"
            return current_anonymized_response, meta
        else:
            # 构造反馈信息
            feedback = (
                "The attacker correctly inferred the following attributes from your rewritten text:\n"
                f"- {', '.join(leaked)}\n"
                "Please generalize the text further to hide these specific clues."
            )
            meta["final_leaked_attributes"] = leaked
            
    # 达到最大迭代次数
    return current_anonymized_response, meta

def process_record(data: Dict[str, Any], max_iterations: int) -> Dict[str, Any]:
    """
    辅助函数，用于处理单条记录。
    接收一个包含原始数据的字典，返回处理后的完整字典。
    """
    # 强制转换为 str，以匹配本地代码中的处理逻辑
    try:
        personality = data.get("personality")
        question = str(data.get("question_asked"))
        response = str(data.get("response"))
    except Exception:
        # 如果字段不存在或无法转换为 str，则跳过
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not all([personality, question, response]):
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data
    
    if not isinstance(personality, dict):
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    anonymized_response, meta = adversarial_anonymization(
        question_asked=question,
        original_response=response,
        true_personality=personality,
        max_iterations=max_iterations
    )
    
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    return data

def main():
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
    args = parser.parse_args()

    # --- Client/Model 初始化 ---
    global GLOBAL_CLIENT
    global LLM_MODEL
    
    if args.model:
        LLM_MODEL = args.model # 允许通过命令行覆盖模型
        
    if args.api_key or args.base_url:
        try:
            GLOBAL_CLIENT = OpenAI(
                api_key=args.api_key or os.environ.get("API_KEY"),
                base_url=args.base_url or "https://api.deepseek.com/v1"
            )
            GLOBAL_CLIENT.models.list()
        except Exception as e:
            print(f"Error: Failed to reinitialize client: {e}"); sys.exit(1)
            
    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'"); sys.exit(1)

    records_to_process = [json.loads(line) for line in lines if line.strip()]
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
            if meta.get("status") == "success":
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            elif meta.get("status") in ("max_iterations_reached", "api_error", "skipped_incomplete_data", "skipped_data_read_error", "skipped_invalid_personality"):
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Anonymized data saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()