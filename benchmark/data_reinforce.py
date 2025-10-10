import os
import json
import time
import argparse
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Optional

# --- 1. 配置 ---

# DeepSeek API的配置，也可以通过环境变量设置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
JUDGE_MODEL_NAME = "deepseek-chat"

# --- 2. 提示词模板 ---

PROMPT_SYSTEM_PARAPHRASE = """
You are an expert in natural language, specializing in paraphrasing text while preserving all factual information and persona.
Your goal is to rewrite a given text in a new style, as if a different person with the same underlying knowledge wrote it.
"""

PROMPT_USER_PARAPHRASE = """
Please rewrite the following [Original Text].

Your rewritten text MUST adhere to these strict rules:
1.  **Preserve all facts**: Do not add, remove, or change any factual information (e.g., names, places, events, opinions, feelings, stated profession, etc.).
2.  **Maintain the persona**: The new text should sound like it comes from the same person described in the original context (same age, gender, occupation, etc.).
3.  **Change the style**: Use different sentence structures, vocabulary, and tone. Do not simply swap a few words.
4.  **First-person perspective**: The new text must be written in the first person ("I", "my", "we").
5.  **Output only the text**: Your final output should ONLY be the rewritten text, without any explanations, preambles like "Here is the rewritten text:", or markdown formatting.

[Original Text]:
{text_to_paraphrase}
"""

# --- 3. 辅助函数 ---

def get_api_client() -> OpenAI:
    """初始化并返回API客户端"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("错误：请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

def call_deepseek_api(client: OpenAI, text_to_paraphrase: str, retries: int = 3, delay: int = 5) -> str:
    """
    调用DeepSeek API进行文本复述，包含重试逻辑。
    """
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM_PARAPHRASE},
        {"role": "user", "content": PROMPT_USER_PARAPHRASE.format(text_to_paraphrase=text_to_paraphrase)}
    ]

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=messages,
                temperature=0.7, # 稍微提高温度以增加多样性
                top_p=0.9,
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
            else:
                tqdm.write("  - 警告: API返回内容为空。")
                return "API_EMPTY_RESPONSE"
        except Exception as e:
            tqdm.write(f"  - API调用错误 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                tqdm.write(f"    将在 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                tqdm.write("    已达到最大重试次数。")
                return "API_ERROR"
    return "API_ERROR"

def process_item(item: Dict, index: int, client: OpenAI) -> Optional[Dict]:
    """
    处理单个数据项的工作函数：调用API并构建新的JSON对象。
    此函数将在单独的线程中执行。
    """
    original_response = item.get('response')

    if not original_response:
        tqdm.write(f"跳过索引 {index} 的记录，因为 'response' 字段为空。")
        return None

    # 调用API进行增强
    paraphrased_response = call_deepseek_api(client, original_response)

    # 如果API调用成功，创建并返回新记录
    if paraphrased_response not in ["API_ERROR", "API_EMPTY_RESPONSE"]:
        new_item = item.copy()  # 复制所有原始信息
        new_item['response'] = paraphrased_response  # 用新生成的文本替换'response'字段
        new_item['augmentation_source'] = f'augmented_from_original_index_{index}'  # 添加来源标记

        # 删除可能过时的字段，因为后续需要重新生成
        new_item.pop('anonymized_response', None)
        new_item.pop('loss_description_sentence', None)

        return new_item
    
    return None


# --- 4. 主流程 ---

def main(args):
    """主函数，执行数据增强流程"""
    api_client = get_api_client()

    # 读取原始数据
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{args.input_file}'")
        return

    # 断点续传逻辑：检查输出文件并确定从哪里开始
    start_index = 0
    if os.path.exists(args.output_file) and not args.overwrite:
        processed_ids = set()
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'augmentation_source' in data and data['augmentation_source'].startswith('augmented_from_original_index_'):
                    try:
                        # 从 'augmented_from_original_index_123' 中提取 123
                        processed_index = int(data['augmentation_source'].split('_')[-1])
                        processed_ids.add(processed_index)
                    except (ValueError, IndexError):
                        continue # 如果格式不正确，则跳过
        
        # 找到第一个未被处理的索引
        while start_index < len(original_data) and start_index in processed_ids:
            start_index += 1
            
        if start_index > 0:
            print(f"检测到输出文件，已处理 {len(processed_ids)} 条增强数据。将从索引 {start_index} 继续...")
    else:
        # 如果文件不存在或指定了覆盖，则从头开始并将原始数据写入
        print("从头开始处理，首先将所有原始数据写入输出文件...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in original_data:
                item_copy = item.copy()
                item_copy['augmentation_source'] = 'original'
                f.write(json.dumps(item_copy, ensure_ascii=False) + '\n')
        print(f"已将 {len(original_data)} 条原始数据写入输出文件。")


    # 确定要处理的数据范围
    records_to_process = original_data
    if args.limit:
        records_to_process = original_data[:args.limit]
    
    # 筛选出实际需要处理的任务
    tasks_to_run = [(item, i) for i, item in enumerate(records_to_process) if i >= start_index]

    if not tasks_to_run:
        print("所有数据均已处理。程序退出。")
        return

    print(f"\n--- 开始数据增强，使用最多 {args.max_workers} 个并发线程，共 {len(tasks_to_run)} 条记录待处理 ---")

    # 以追加模式打开输出文件
    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        # 使用ThreadPoolExecutor进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(process_item, item, index, api_client): index for item, index in tasks_to_run}
            
            # 使用tqdm创建进度条，并在任务完成时进行处理
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(tasks_to_run), desc="Augmenting Data"):
                result = future.result()
                if result:
                    # 将成功的结果写入文件
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush() # 立即写入磁盘，防止程序中断时丢失数据

    print(f"\n--- 数据增强完成！---")
    print(f"增强后的数据已追加到: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用DeepSeek API并行地对JSONL数据集进行数据增强。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的训练集文件路径 (e.g., train_val_grouped.jsonl)。")
    parser.add_argument("--output_file", type=str, required=True, help="合并了原始数据和增强后数据的新输出文件路径。")
    parser.add_argument("--max_workers", type=int, default=10, help="并发执行的最大线程数。")
    parser.add_argument("--limit", type=int, default=None, help="（可选）限制处理的原始记录总数，用于测试。")
    parser.add_argument("--overwrite", action='store_true', help="如果设置此项，将覆盖输出文件并从头开始，而不是断点续传。")
    
    args = parser.parse_args()
    main(args)