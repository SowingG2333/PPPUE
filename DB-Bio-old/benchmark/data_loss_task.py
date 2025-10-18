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
PROMPT_TEMPLATE = """
Role: You are an AI assistant specializing in biographical text analysis.

Task: Compare the [Original Biography] and [Anonymized Biography]. Your goal is to create a very concise note describing only the information that was lost and is useful for identifying the person's profession or occupation.

Follow these rules:

- Focus on Professional Identity: Only extract clues about specific job titles, professional achievements, institutional affiliations, historical events related to their career, or technical expertise.

- Ignore Irrelevant Details: Do not include lost information about birth/death dates, personal relationships, hobbies unrelated to their profession, or generic location names.

- Handle "No Clues": If no profession-relevant information was lost, simply write "No occupation-relevant information was lost."

Example 1 (Relevant clues):
[Original Biography]: "John Smith (1850-1920) was a British engineer who designed the Thames Bridge and worked at Imperial College London."
[Anonymized Biography]: "An engineer from the 19th century designed a major bridge and worked at a prestigious university."
Your Output: The person designed the Thames Bridge and worked at Imperial College London.

Example 2 (Filtering is needed):
[Original Biography]: "Maria Garcia won three Olympic gold medals in tennis between 1990-1998. She later married Carlos and had two children."
[Anonymized Biography]: "An athlete won multiple Olympic medals in a racquet sport during the 1990s."
Your Output: The person won three Olympic gold medals in tennis between 1990-1998.

Example 3 (No relevant clues):
[Original Biography]: "Born in London on March 15, 1965, the individual later moved to Paris where they lived with their spouse."
[Anonymized Biography]: "Born in a European capital in the mid-1960s, the person later relocated to another major city."
Your Output: No occupation-relevant information was lost.

[Original Biography]:
{original_text}

[Anonymized Biography]:
{anonymized_text}

Your Output:
"""

# --- 辅助函数 ---
def generate_api_response(prompt_text: str, model: str, temperature: float, max_tokens: int = 150) -> str:
    """使用API生成信息损失描述。"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if completion.choices[0].message.content is None:
            return "Error: API returned no message content."
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: API call failed with exception: {e}"

# --- 并行处理辅助函数 ---
def process_record(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    处理单条记录:生成信息损失描述并将其添加到数据字典中。
    """
    original_text = data.get('text')
    anonymized_text = data.get('anonymized_text')
    
    if not original_text or not anonymized_text:
        data['loss_description_sentence'] = "Error: Missing original or anonymized biography."
        return data

    prompt = PROMPT_TEMPLATE.format(original_text=original_text, anonymized_text=anonymized_text)
    description = generate_api_response(prompt, model=model, temperature=0.2)
    
    data['loss_description_sentence'] = description
    return data

# --- 主流程 ---
def main():
    parser = argparse.ArgumentParser(description="Generate information loss descriptions for biographical data in parallel using an API.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel threads to use.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with anonymized biographies.")
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
        future_to_record = {executor.submit(process_record, record, args.model): record for record in records_to_process}
        
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