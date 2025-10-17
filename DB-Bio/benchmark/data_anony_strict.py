'''
脚本命令行参数说明：
    --input_file: 输入 JSONL 文件路径
    --output_file: 输出 JSONL 文件路径
    --model: LLM 模型名称 (默认: deepseek-chat)
    --k: 攻击者需要返回的 Top-K 猜测数量 (默认: 5)
    --max_iterations: 最大对抗匿名化迭代次数 (默认: 5)
    --api_key: 覆盖 API Key (默认: 使用环境变量 API_KEY)
    ---base_url: 覆盖 API 基础 URL (例如: https://api.deepseek.com/v1)
    --success_file: 可选，保存成功匿名化记录的文件路径 (JSONL)
    --failed_file: 可选，保存达到最大迭代或 API 错误记录的文件路径 (JSONL)
    --limit: 仅处理输入文件的前 N 条记录
    --workers: 并行处理的工作线程数 (默认: 10)
'''
import os
import re
import json
import argparse
import sys
import unicodedata
import concurrent.futures
import functools

from typing import List, Dict, Tuple, Any
from openai import OpenAI
from tqdm import tqdm

# --- 配置 ---
# 在 Linux/macOS 上: export API_KEY='your-key-here'
# 在 Windows 上: set API_KEY='your-key-here'
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

# 模型名称
LLM_MODEL = "deepseek-chat"

# --- 提示模板 ---
PROMPT_ANONYMIZER = """
You are a sophisticated privacy guardian AI. Your primary mission is to protect against re-identification by expert adversaries.

Your task is to perform an aggressive generalization and rewriting of the [Original Text] to neutralize the clues identified in the [Previous Feedback from Attacker]. Your goal is to create a version so abstract that an expert can neither guess the true identity of "{true_identity}" nor their specific profession.

**CRITICAL INSTRUCTIONS:**
1.  **Target the Feedback**: The [Previous Feedback from Attacker] reveals the exact clues the adversary is using. Your new version MUST neutralize these specific clues through further generalization or rewriting.
2.  **Aggressive Generalization**: Do not just hide names, dates, and locations. You must also generalize **specific professional terms, unique achievements, and technical jargon** into broader, more generic categories (e.g., rewrite 'tennis' as 'a sport'; rewrite 'Nobel Prize in Physics' as 'a major award in science').
3.  **Maintain Fluency**: The rewritten text must remain grammatically correct and fluent.

[Original Text]:
{original_text}

[Previous Anonymized Version]:
{previous_anonymized_text}

[Previous Feedback from Attacker]:
{feedback}

Provide ONLY the single, newly anonymized paragraph below. Do not add any commentary.
"""

PROMPT_ATTACKER = """
You are a 'Digital Sherlock Holmes,' a world-class investigator with encyclopedic knowledge of public figures. Your specialty is re-identifying individuals from anonymized text by synthesizing subtle clues.

Carefully analyze the [Anonymized Text] and deduce the most likely public figure being described. Your process must be methodical.

**OUTPUT FORMAT:**
Your response MUST be structured in two distinct parts:
1.  **Reasoning**: A step-by-step deduction. First, list all potential clues you've identified. Second, synthesize these clues to form a coherent profile (e.g., profession, nationality, era). Third, explain how this profile leads to your final guesses.
2.  **Top-{k} Guesses**: A ranked list of exactly {k} names, from most to least likely. The format must be a numbered list.

<Example>
[Anonymized Text]:
"An individual from a nation known for its horology and finance was a pivotal figure in a specific scientific field during the early 20th century. He developed groundbreaking theories regarding the nature of space-time and light, and his work in a patent office is legendary. He later emigrated to a North American country."

**Reasoning**:
- **Clues Identified**:
    - Nation of origin known for watchmaking and banking -> Switzerland.
    - Scientific field related to space-time and light -> Physics, specifically Relativity.
    - Time period -> Early 20th century.
    - Famous job -> Patent office.
    - Emigration -> Moved to the USA.
- **Profile Synthesis**: The profile is a Swiss physicist from the early 20th century who worked in a patent office and developed theories of relativity before moving to the USA.
- **Deduction**: This profile overwhelmingly points to Albert Einstein. Other physicists from that era do not match all clues as precisely.

**Top-3 Guesses**:
1. Albert Einstein
2. Henri Poincaré
3. Max Planck
</Example>

---

<Anonymized Text>:
{current_anonymized_text}
"""


def _normalize_person_name(s: str) -> str:
    """
    对姓名做鲁棒归一化：
    1. NFKD 规范化并去除变音符
    2. 转小写
    3. 去除所有非字母数字字符
    """
    if not s:
        return "" # 处理空字符串
    s = unicodedata.normalize('NFKD', s) # 分解字符，将变音符分离
    s = ''.join(c for c in s if not unicodedata.combining(c)) # 去除变音符
    s = re.sub(r'[^a-z0-9]+', '', s.lower()) # 转小写并去除非字母数字字符
    return s


def _parse_attacker_guesses(attacker_response_text: str) -> List[str]:
    """
    解析攻击者的Top-K猜测名单，兼容两种格式：
    I. 编号列表：
       1. Name A
       2. Name B
    ----------------------------------
    II. 单行Guess:
       Guess: Name A; Name B; Name C
    """
    guesses = []

    # 优先：编号列表
    numbered = re.findall(r"^\s*\d+\.\s*(.+)$", attacker_response_text, re.MULTILINE) # 匹配每行的 "1. Name"
    if numbered: # 如果找到编号列表
        guesses = [g.strip() for g in numbered if g.strip()] # 去除空行

    # 兜底：Guess: a; b; c
    if not guesses: # 如果编号列表为空
        m = re.search(r"Guess:\s*(.+)$", attacker_response_text, re.IGNORECASE | re.MULTILINE) # 匹配 "Guess: Name A; Name B; Name C"
        if m: # 如果找到匹配
            guesses_str = m.group(1).strip() # 提取猜测字符串
            # 先用分号分割，不行再尝试逗号或换行
            if ';' in guesses_str:
                guesses = [g.strip() for g in guesses_str.split(';') if g.strip()]
            else:
                guesses = [g.strip() for g in re.split(r"[,\n]", guesses_str) if g.strip()]

    return guesses


def adversarial_anonymization(
    original_text: str,
    true_identity: str,
    k: int = 5,
    max_iterations: int = 5,
    model: str = LLM_MODEL,
    client: OpenAI = client,
    verbose: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    对给定文本执行迭代对抗性匿名化
    -------------------------
    Returns:
        (anonymized_text, meta)
        meta 包含:
          - status: "success" | "max_iterations_reached" | "api_error"
          - iterations_used: 实际迭代次数
          - last_guesses: 攻击者最后一次Top-K名单
          - true_identity: 真实姓名
          - attacker_response: 攻击者最后一次原始响应文本
    """
    feedback = "No feedback yet. This is the first attempt. Rewrite the text to be more general."
    current_anonymized_text = original_text

    normalized_true_identity = _normalize_person_name(true_identity)
    last_guesses: List[str] = []
    last_attacker_response_text: str = ""
    iterations_used = 0

    for i in range(max_iterations):
        iterations_used = i + 1
        if verbose:
            print(f"\n  Iteration {iterations_used}/{max_iterations} for '{true_identity}'...")

        # === 步骤1：Anonymizer对文本进行匿名化重写 ===
        if verbose:
            print("    - Anonymizer is rewriting the text...")
        anonymizer_prompt = PROMPT_ANONYMIZER.format(
            true_identity=true_identity,
            feedback=feedback,
            original_text=original_text,
            previous_anonymized_text=current_anonymized_text
        ) # 填充提示模板
        try:
            anonymizer_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a text anonymization expert."},
                    {"role": "user", "content": anonymizer_prompt}
                ],
                temperature=0.5,
            )
            msg = anonymizer_response.choices[0].message
            if msg.content is None:
                raise ValueError("Anonymizer LLM returned empty content")
            current_anonymized_text = msg.content.strip()
        except Exception as e:
            if verbose:
                print(f"    - ERROR calling Anonymizer LLM: {e}")
            return current_anonymized_text, {
                "status": "api_error",
                "error": str(e),
                "iterations_used": iterations_used - 1,
                "last_guesses": last_guesses,
                "true_identity": true_identity,
                "attacker_response": last_attacker_response_text
            }

        # === 步骤2：Attacker进行身份识别 ===
        if verbose:
            print("    - Attacker is trying to identify the person...")
        attacker_prompt = PROMPT_ATTACKER.format(
            k=k,
            current_anonymized_text=current_anonymized_text
        )
        try:
            attacker_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a world-class identity inference expert."},
                    {"role": "user", "content": attacker_prompt}
                ],
                temperature=0.25,
            )
            msg2 = attacker_response.choices[0].message
            if msg2.content is None:
                raise ValueError("Attacker LLM returned empty content")
            attacker_response_text = msg2.content.strip()
            last_attacker_response_text = attacker_response_text
        except Exception as e:
            if verbose:
                print(f"    - ERROR calling Attacker LLM: {e}")
            return current_anonymized_text, {
                "status": "api_error",
                "error": str(e),
                "iterations_used": iterations_used,
                "last_guesses": last_guesses,
                "true_identity": true_identity,
                "attacker_response": last_attacker_response_text
            }

        # === 步骤3：解析并评估攻击者的猜测 ===
        if verbose:
            print("    - Evaluating attacker's guesses...")
        try:
            guesses = _parse_attacker_guesses(attacker_response_text) # 解析猜测名单
            last_guesses = guesses
            if verbose:
                print(f"    - Attacker's Top-{len(guesses)} Guesses: {guesses}")

            normalized_guesses = [_normalize_person_name(name) for name in guesses]
            found = any(
                (normalized_true_identity and (normalized_true_identity in ng or ng in normalized_true_identity))
                for ng in normalized_guesses
            ) # 检查是否匹配

            if not found:
                if verbose:
                    print(f" Anonymization successful! '{true_identity}' was not in the attacker's list.")
                return current_anonymized_text, {
                    "status": "success",
                    "iterations_used": iterations_used,
                    "last_guesses": last_guesses,
                    "true_identity": true_identity,
                    "attacker_response": last_attacker_response_text
                }
            if verbose:
                print(f" Anonymization failed! '{true_identity}' was found in the attacker's list.")
            feedback = (
                "The previous attempt failed. The attacker analyzed the text and provided the following reasoning and guesses. "
                "Use this to improve the anonymization.\n\n---\n"
                f"{attacker_response_text}\n---"
            )
        except Exception as e:
            if verbose:
                print(f"    - ERROR parsing attacker response: {e}")
                print(f"    - Full Attacker Response:\n{attacker_response_text}")
            feedback = "The previous attempt was successful, but the response from the attacker was malformed and could not be parsed. Please try again, paying close attention to the requested format."
    if verbose:
        print(f" Warning: Max iterations ({max_iterations}) reached for '{true_identity}'. Returning last attempt.")
    return current_anonymized_text, {
        "status": "max_iterations_reached",
        "iterations_used": iterations_used,
        "last_guesses": last_guesses,
        "true_identity": true_identity,
        "attacker_response": last_attacker_response_text
    }


def process_record(data, k, max_iterations, model, client):
    """处理单个数据记录的辅助函数，用于并行化。"""
    try:
        original_text = data.get("text")
        person_name = data.get("people")

        if not original_text or not person_name:
            print(f"Skipping record due to missing 'text' or 'people' field: {data}")
            return None

        # 运行核心匿名化功能 (verbose=False, 因为tqdm会显示进度)
        anonymized_text, meta = adversarial_anonymization(
            original_text=original_text,
            true_identity=person_name,
            k=k,
            max_iterations=max_iterations,
            model=model,
            client=client,
            verbose=False # 在并行模式下关闭详细日志
        )

        # 添加新字段并返回
        data["anonymized_text"] = anonymized_text
        data["anonymization_meta"] = meta
        return data

    except Exception as e:
        print(f"Error processing record for '{data.get('people')}': {e}")
        # 记录错误信息并返回
        data["anonymized_text"] = data.get("text") # 保留原始文本
        data["anonymization_meta"] = {
            "status": "processing_error",
            "error": str(e)
        }
        return data


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize text data from a JSONL file using an adversarial LLM loop."
    ) # 设置命令行参数
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    ) # 输入文件路径
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file."
    ) # 输出文件路径
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model name (default: {LLM_MODEL})"
    ) # 模型名称
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K guesses required from attacker (default: 5)"
    ) # Top-K参数
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum adversarial anonymization iterations (default: 5)"
    ) # 最大迭代次数
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Override API key (default: use OPENAI_API_KEY env)"
    ) # API Key
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Override API base URL (e.g., https://api.deepseek.com/v1)."
    ) # API Base URL
    parser.add_argument(
        "--success_file",
        type=str,
        default=None,
        help="Optional path to save only successful anonymization records (JSONL)."
    ) # 成功匿名化文件路径
    parser.add_argument(
        "--failed_file",
        type=str,
        default=None,
        help="Optional path to save records that hit max iterations or API errors (JSONL)."
    ) # 失败匿名化文件路径
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records from input_file."
    ) # 处理行数限制
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers for processing (default: 10)."
    ) # 并行工作线程数
    args = parser.parse_args() # 解析命令行参数

    # 允许通过命令行覆盖 client 配置
    if args.api_key or args.base_url:
        try:
            global client
            client = OpenAI(
                api_key=args.api_key or os.environ.get("API_KEY"),
                base_url=args.base_url or os.environ.get("BASE_URL")
            )
        except Exception as e:
            print(f"Error: Failed to reinitialize OpenAI client with provided args: {e}")
            sys.exit(1)

    # 读取所有待处理的记录
    records_to_process = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records_to_process.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)

    # 应用 --limit 参数
    if args.limit:
        records_to_process = records_to_process[:args.limit]

    total_to_process = len(records_to_process)
    print(f"Starting processing for {total_to_process} records from '{args.input_file}' with {args.workers} workers...")

    # 可选的成功/失败单独输出文件
    success_out = open(args.success_file, 'w', encoding='utf-8') if args.success_file else None
    failed_out = open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else None

    # 使用线程池并行处理
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        # 使用 functools.partial 预先填充 process_record 函数的固定参数
        worker_func = functools.partial(
            process_record,
            k=args.k,
            max_iterations=args.max_iterations,
            model=args.model,
            client=client
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # 使用 executor.map 并用 tqdm 显示进度条
            results = list(tqdm(executor.map(worker_func, records_to_process), total=total_to_process, desc="Processing data"))

            # 处理完成后，统一写入文件
            for data in results:
                if data is None:
                    continue

                # 写入主输出文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                # 额外写入成功/失败jsonl（可选）
                meta = data.get("anonymization_meta", {})
                if success_out and meta.get("status") == "success":
                    success_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                if failed_out and meta.get("status") not in ("success", None):
                    failed_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    if success_out:
        success_out.close()
    if failed_out:
        failed_out.close()

    print(f"\nProcessing complete. Anonymized data saved to '{args.output_file}'.")


if __name__ == "__main__":
    main()