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
    return json.dumps(profile, indent=2)

PROMPT_ANONYMIZER = """
Role: You are an expert in text anonymization, tasked with protecting user privacy.
Task: Your objective is to rewrite the [User Response] so that the personal attributes listed in the [True Personality Profile] cannot be inferred. The rewritten response must still be a natural and coherent answer to the [Question Asked].

[True Personality Profile to Hide]:
{profile_to_hide}

[Question Asked]:
{question_asked}

[Previous Feedback from Attacker]:
{feedback}

[Original User Response]:
{original_response}

Now, provide ONLY your new, single, anonymized version of the response. Do not add any extra commentary or preamble.
"""

PROMPT_ATTACKER = """
Role: You are an expert profiler with deep knowledge of psycho-linguistics and cultural cues.
Task: Based on the conversational context provided by the [Question Asked] and the user's [Anonymized Response], your goal is to infer the user's personality profile.

Your response MUST be a single, valid JSON object, containing your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
- For "age", provide an integer.
- For all other fields, provide a string.
- If you cannot reasonably infer an attribute, use the JSON value `null`.

[Question Asked]:
{question_asked}

[Anonymized Response]:
{anonymized_response}

JSON Output:
"""

def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    leaked_attributes = []
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if true_age is not None and guessed_age is not None and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(f"age (guessed: {guessed_age}, true: {true_age})")
    for key in ["sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status"]:
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
    feedback = "No feedback yet. This is the first attempt. Rewrite the response to be more general and hide all personal attributes."
    current_anonymized_response = original_response
    profile_to_hide_str = format_profile(true_personality)
    meta = {
        "status": "max_iterations_reached", "iterations_used": 0,
        "final_leaked_attributes": [], "final_attacker_guess": {}
    }
    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        anonymizer_prompt = PROMPT_ANONYMIZER.format(
            profile_to_hide=profile_to_hide_str, question_asked=question_asked,
            feedback=feedback, original_response=original_response
        )
        try:
            anonymizer_completion = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": anonymizer_prompt}],
                temperature=0.6,
            )
            if anonymizer_completion.choices[0].message.content is None:
                raise ValueError("Anonymizer LLM returned no message content.")
            else:
                current_anonymized_response = anonymizer_completion.choices[0].message.content.strip()
        except Exception as e:
            meta["status"] = "api_error"; meta["error"] = str(e)
            return current_anonymized_response, meta
        attacker_prompt = PROMPT_ATTACKER.format(
            question_asked=question_asked, anonymized_response=current_anonymized_response
        )
        attacker_guess = None
        try:
            attacker_completion = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": attacker_prompt}],
                temperature=0.1, response_format={"type": "json_object"},
            )
            message_content = attacker_completion.choices[0].message.content
            if message_content is None:
                raise ValueError("Attacker LLM returned no message content.")
            response_text = message_content.strip()
            attacker_guess = json.loads(response_text)
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            feedback = "The attacker failed to produce a valid JSON profile. The anonymization might be too strong or the response is confusing. Please try to make the response slightly more natural while still hiding the attributes."
            continue
        leaked = compare_profiles(true_personality, attacker_guess)
        if not leaked:
            meta["status"] = "success"
            return current_anonymized_response, meta
        else:
            feedback = (
                "The previous attempt failed. The attacker correctly inferred the following attributes from your rewritten text:\n"
                f"- {', '.join(leaked)}\n"
                "Please generalize the text further to hide these specific clues."
            )
            meta["final_leaked_attributes"] = leaked
    return current_anonymized_response, meta

# --- 以下是修改的核心部分 ---

def process_record(data: Dict[str, Any], max_iterations: int) -> Dict[str, Any]:
    """
    一个辅助函数，用于处理单条记录。
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