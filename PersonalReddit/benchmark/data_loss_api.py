import os
import json
import argparse
import sys
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# --- 配置 ---
LLM_MODEL = "deepseek-chat"

# --- API客户端初始化 ---
try:
    client = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("BASE_URL", "https://api.deepseek.com/v1")
    )
    client.models.list()
    print("API client initialized successfully.")
except Exception as e:
    print(f"Error: Failed to initialize API client: {e}")
    print("Please ensure your API_KEY and optionally BASE_URL environment variables are set correctly.")
    sys.exit(1)

# --- Prompt模板 ---
PROMPT_LOSS_SYSTEM = """
You are an expert in analyzing text and extracting relevant information about occupations.
"""
PROMPT_LOSS_USER = """
Compare the [Original Text] and [Anonymized Text]. Your goal is to create a very concise note describing only the information that was lost and is useful for guessing the user's occupation.

Follow these rules:

- Focus on Occupation: Only extract clues about the user's job, skills, education, or work environment.

- Ignore Irrelevant Details: Do not include lost information about hobbies, music, personal relationships, or specific non-work-related events.

- Handle "No Clues": If no occupation-relevant information was lost, simply write "No occupation-relevant information was lost."

[Original Text]:
{original_text}

[Anonymized Text]:
{anonymized_text}

Your Output:
"""

# --- 辅助函数 ---
def generate_api_response(prompt_text: str, model: str, temperature: float, max_tokens: int = 100) -> str:
    """使用API生成信息损失描述。"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_LOSS_SYSTEM},
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if completion.choices[0].message.content is None:
            return "Error: API returned no message content."
        return completion.choices[0].message.content.strip()
    except Exception as e:
        # 在并行模式下，直接返回错误信息比打印更好
        return f"Error: API call failed with exception: {e}"

# --- 并行处理辅助函数 ---
def process_record(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    处理单条记录：生成信息损失描述并将其添加到数据字典中。
    """
    original_text = data.get('response')
    anonymized_text = data.get('anonymized_response')
    
    if not original_text or not anonymized_text:
        data['loss_description_sentence'] = "Error: Missing original or anonymized response."
        return data

    prompt = PROMPT_LOSS_USER.format(original_text=original_text, anonymized_text=anonymized_text)
    description = generate_api_response(prompt, model=model, temperature=0.2)
    
    data['loss_description_sentence'] = description
    return data

# --- 主流程 ---
def main():
    parser = argparse.ArgumentParser(description="Generate information loss descriptions in parallel using an API.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel threads to use.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with anonymized data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file where results will be saved.")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help=f"LLM model name for the API (default: {LLM_MODEL}).")
    parser.add_argument("--api_key", type=str, default=None, help="Override API key (default: use environment variable).")
    parser.add_argument("--base_url", type=str, default=None, help="Override API base URL (default: use environment variable or DeepSeek default).")
    args = parser.parse_args()

    if args.api_key or args.base_url:
        try:
            global client
            client = OpenAI(
                api_key=args.api_key or os.environ.get("API_KEY"),
                base_url=args.base_url or os.environ.get("BASE_URL", "https://api.deepseek.com/v1")
            )
            print("API client re-initialized with command-line arguments.")
        except Exception as e:
            print(f"Error: Failed to reinitialize API client with provided args: {e}")
            sys.exit(1)

    # --- 数据加载 ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            records_to_process = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from input file: {e}")
        sys.exit(1)

    print(f"Starting parallel processing for {len(records_to_process)} records using up to {args.max_workers} workers...")

    # --- 并行处理 ---
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务到线程池
        future_to_record = {executor.submit(process_record, record, args.model): record for record in records_to_process}
        
        # 使用tqdm收集结果并显示进度
        for future in tqdm(as_completed(future_to_record), total=len(records_to_process), desc="Generating Loss Descriptions"):
            try:
                processed_record = future.result()
                results.append(processed_record)
            except Exception as exc:
                print(f"A task generated an exception: {exc}")

    # --- 结果写入 ---
    print(f"\nProcessing complete. Writing {len(results)} results to '{args.output_file}'...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in results:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

    print("Task finished successfully.")

if __name__ == "__main__":
    main()