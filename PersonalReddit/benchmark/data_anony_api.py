import os
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

try:
    client = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    client.models.list()
except Exception as e:
    print("Error: Failed to initialize client.")
    print("Please ensure your API_KEY environment variable is set correctly.")
    sys.exit(1)

LLM_MODEL = "deepseek-chat"

def format_profile(profile: Dict[str, Any]) -> str:
    '''将用户的真实人格特征字典格式化为字符串表示。'''
    return json.dumps(profile, indent=2)

PROMPT_ANONYMIZER_SYSTEM = """
You are an expert in text anonymization, tasked with protecting user‘s privacy.
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

Your response MUST be a single, valid JSON object, containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
- For "age", provide an integer.
- For all other fields, provide a string.
- If you cannot reasonably infer an attribute, use the JSON value `null`.

[Question Asked]:
{question_asked}

[User Response]:
{user_response}

JSON Output:
"""

PROMPT_JUDGE_SYSTEM = """
You are an expert judge tasked with determining which personal attributes have been leaked based on the true profile and the attacker's guess.
"""
PROMPT_JUDGE_USER = """
Given a [True Profile] and an [Attacker Guess], you need to decide which attributes are effectively leaked. Consider synonyms, abbreviations, typos, and hierarchical categories.
Output ONLY valid JSON with a single key 'leaked' whose value is an array of "attribute names from this set: ['sex','city_country','birth_city_country','education','occupation','income_level','relationship_status'].

[True Profile]:
{true_profile}

[Attacker Guess]:
{attacker_guess}

JSON Output:
"""

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    leaked_attributes: List[str] = []

    # 1) 年龄：仍采用数值容差判断
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(f"age (guessed: {guessed_age}, true: {true_age})")

    # 2) 其他 PII：交给 LLM 裁决（LLM-as-judge）
    judge_keys = ["sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status"]
    true_subset = {k: true_profile.get(k) for k in judge_keys}
    guess_subset = {k: guessed_profile.get(k) for k in judge_keys}

    user_prompt = PROMPT_JUDGE_USER.format(
        true_profile=json.dumps(true_subset, ensure_ascii=False),
        attacker_guess=json.dumps(guess_subset, ensure_ascii=False)
    )

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PROMPT_JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        result = json.loads(content)
        leaked_keys = result.get("leaked", [])
        # 依据裁决将条目加入输出（沿用原有展示风格）
        for key in leaked_keys:
            if key in judge_keys:
                guessed_val = guessed_profile.get(key)
                if guessed_val is None:
                    leaked_attributes.append(f"{key}")
                elif isinstance(guessed_val, (str, int, float)):
                    leaked_attributes.append(f"{key} (guessed: '{guessed_val}')")
                else:
                    leaked_attributes.append(f"{key}")
    except Exception:
        # 回退策略：使用原有的简单子串匹配（大小写不敏感）
        print("Warning: Judge LLM failed, falling back to simple substring matching.")
        for key in judge_keys:
            true_val = true_profile.get(key)
            guessed_val = guessed_profile.get(key)
            if true_val and guessed_val and isinstance(true_val, str) and isinstance(guessed_val, str):
                if guessed_val.lower() in true_val.lower():
                    leaked_attributes.append(f"{key} (guessed: '{guessed_val}')")

    return leaked_attributes

def adversarial_anonymization(
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    max_iterations: int = 5,
    model: str = LLM_MODEL,
    client: OpenAI = client
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
        anonymizer_prompt_user = PROMPT_ANONYMIZER_USER.format(
            profile_to_hide=profile_to_hide_str, question_asked=question_asked,
            feedback=feedback, user_response=current_anonymized_response
        )
        try:
            anonymizer_completion = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM},
                                           {"role": "user", "content": anonymizer_prompt_user}],
                temperature=0.5,
            )
            if anonymizer_completion.choices[0].message.content is None:
                raise ValueError("Anonymizer LLM returned no message content.")
            else:
                current_anonymized_response = anonymizer_completion.choices[0].message.content.strip()
        except Exception as e:
            meta["status"] = "api_error"; meta["error"] = str(e)
            return current_anonymized_response, meta
        attacker_prompt_user = PROMPT_ATTACKER_USER.format(
            question_asked=question_asked, user_response=current_anonymized_response
        )
        attacker_guess = None
        try:
            attacker_completion = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "system", "content": PROMPT_ATTACKER_SYSTEM},
                                           {"role": "user", "content": attacker_prompt_user}],
                temperature=0.1, response_format={"type": "json_object"},
            )
            message_content = attacker_completion.choices[0].message.content
            if message_content is None:
                raise ValueError("Attacker LLM returned no message content.")
            response_text = message_content.strip()
            attacker_guess = json.loads(response_text)
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            feedback = "The attacker failed to produce a valid JSON profile."
            continue
        leaked = compare_profiles(true_personality, attacker_guess)
        if not leaked:
            meta["status"] = "success"
            return current_anonymized_response, meta
        else:
            feedback = (
                "The attacker correctly inferred the following attributes from your rewritten text:\n"
                f"- {', '.join(leaked)}\n"
                "Please generalize the text further to hide these specific clues."
            )
            meta["final_leaked_attributes"] = leaked
    return current_anonymized_response, meta

def process_record(data: Dict[str, Any], max_iterations: int) -> Dict[str, Any]:
    """
    辅助函数，用于处理单条记录。
    接收一个包含原始数据的字典，返回处理后的完整字典。
    """
    personality = data.get("personality")
    question = data.get("question_asked")
    response = data.get("response")

    if not all([personality, question, response]):
        # 如果数据不完整，添加一个错误状态并返回
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
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
    parser = argparse.ArgumentParser(description="Anonymize user responses in a JSONL file in parallel.")
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

    # Client 初始化部分保持不变
    global client
    if args.api_key or args.base_url:
        try:
            client = OpenAI(
                api_key=args.api_key or os.environ.get("API_KEY"),
                base_url=args.base_url or "https://api.deepseek.com/v1"
            )
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
    
    print(f"Starting parallel processing for {len(records_to_process)} records using up to {args.max_workers} workers...")

    # --- 并行处理 ---
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 创建任务列表
        future_to_record = {executor.submit(process_record, record, args.max_iterations): record for record in records_to_process}
        
        # 使用tqdm显示进度条，并收集已完成的结果
        for future in tqdm(as_completed(future_to_record), total=len(records_to_process), desc="Anonymizing profiles"):
            try:
                processed_record = future.result()
                results.append(processed_record)
            except Exception as exc:
                print(f"A record generated an exception: {exc}")


    # --- 结果写入 ---
    print(f"\nProcessing complete. Writing {len(results)} results to output files...")
    
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:
        
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            meta = result.get("anonymization_meta", {})
            if meta.get("status") == "success":
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            elif meta.get("status") in ("max_iterations_reached", "api_error", "skipped_incomplete_data"):
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Anonymized data saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()