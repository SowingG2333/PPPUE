import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

from tqdm import tqdm
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class Config:
    DEEPSEEK_API_KEY = os.environ.get("API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"

    # 路径
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/DB-Bio/benchmark/test/local_anony_test_extract_with_local_loss.jsonl"
    PREDICTIONS_FILE = "/root/autodl-tmp/PPPUE/DB-Bio/results/alignment/output/pred_BASELINE.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/DB-Bio/results/alignment/eval_results"

    # 其它
    LIMIT: Optional[int] = None

PROMPT_DISCRETE_SYSTEM = """
You are a highly precise evaluation expert with a deep understanding of professional roles and their nuances.
"""

PROMPT_DISCRETE_USER = """
Your task is to strictly determine if the "Generated Answer" correctly identifies the "Ground Truth Profession".

A "Generated Answer" is considered **"Correct"** only if it meets one of the following criteria:
1.  It is an **exact match** with the "Ground Truth Profession".
2.  It is a **clear and direct synonym** (e.g., Footballer for Soccer Player).
3.  It is **more specific or more precise** than the ground truth, while still being factually accurate based on a person's career (e.g., "Academy Coach" is a more precise current role for a "Soccer Player" who has transitioned).

A "Generated Answer" is **"Incorrect"** if:
1.  It is a completely different profession.
2.  It is a **more general or less precise** role (e.g., "Cook" for "Chef").

Output your final verdict as a single word: "Correct" or "Incorrect".

Ground Truth Profession: "{ground_truth}"

Generated Answer: "{generated_answer}"

Your verdict:
"""

PROMPT_SCORE_SYSTEM = """
You are a highly discerning evaluation expert with deep contextual understanding of professional roles and their nuances.
"""

PROMPT_SCORE_USER = """
Your goal is to evaluate the quality of a "Generated Answer" regarding a person's occupation, using the full [Original Text] as the ultimate source of truth.

The "Generated Answer" was produced by an AI that only saw an ANONYMIZED version of the text. Your evaluation must focus on how well the AI's answer reflects the specific, nuanced, and current information present only in the [Original Text].

Please rate the "Generated Answer" on a scale of 1 to 10 based on the rubric below. Output ONLY the integer score and nothing else.

SCORING RUBRIC (10-Point Scale):
- 1-2 (Incorrect): The answer is completely wrong or irrelevant.
- 3-4 (Weakly Related): The answer is in a related but incorrect field.
- 5-6 (Correct Generalization): The answer provides a correct but overly general category.
- 7-8 (Good Recovery): The answer correctly identifies a significant *past* profession or a very close synonym, but misses the most current or precise role.
- 9-10 (Excellent Recovery): The answer is highly specific and correctly identifies the person's primary and most current profession.

[Original Text]:
{original_text}

[Generated Answer]:
{generated_answer}

Your Output:
"""

def _api_chat_complete(client: OpenAI, model: str, messages: List[Dict], temperature: float) -> str:
    try:
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("Empty content from API.")
        return content.strip()
    except Exception as e:
        tqdm.write(f"API 调用异常: {e}")
        return "API_ERROR"

def judge_discrete(generated_answer: str, ground_truth_label: str, client: OpenAI, config: Config) -> bool:
    prompt = PROMPT_DISCRETE_USER.format(ground_truth=ground_truth_label, generated_answer=generated_answer)
    res = _api_chat_complete(
        client,
        config.JUDGE_MODEL_NAME,
        [{"role": "system", "content": PROMPT_DISCRETE_SYSTEM},
         {"role": "user", "content": prompt}],
        0.0
    )
    verdict = res.lower().strip().strip('.,!"\'')
    return verdict == "correct"

def judge_score(generated_answer: str, original_text: str, client: OpenAI, config: Config) -> int:
    prompt = PROMPT_SCORE_USER.format(original_text=original_text, generated_answer=generated_answer)
    res = _api_chat_complete(
        client,
        config.JUDGE_MODEL_NAME,
        [{"role": "system", "content": PROMPT_SCORE_SYSTEM},
         {"role": "user", "content": prompt}],
        0.0
    )
    if res == "API_ERROR":
        return 1
    try:
        score = int(res)
        if 1 <= score <= 10:
            return score
    except Exception:
        pass
    tqdm.write(f"评分解析失败，原始返回: {res}")
    return 1

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# ========== 新增：线程本地 client 与工具函数 ==========
_thread_local = threading.local()

def _get_client(config: Config) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE)
        _thread_local.client = client
    return client

def _safe_stem(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    else:
        base = os.path.splitext(base)[0]
    # 简单清理，避免奇怪字符
    base = base.replace(" ", "_").replace("/", "_")
    return base or "unknown"

def _evaluate_one(p: Dict[str, Any], data_by_index: Dict[int, Dict[str, Any]], config: Config) -> Optional[Dict[str, Any]]:
    try:
        idx = p.get("index")
        gen = (p.get("generated_answer") or "").strip()
        rec = data_by_index.get(idx, {})
        label = rec.get("label")
        original_text = rec.get("text", "")
        anonymized_text = rec.get("anonymized_text", "")
        loss_desc = rec.get("loss_description_sentence", "")

        client = _get_client(config)

        # 连续分数
        score = judge_score(gen, original_text, client, config)
        # 离散准确（可选）
        is_correct = None
        if label:
            is_correct = judge_discrete(gen, label, client, config)

        det = {
            "index": idx,
            "Generated Answer": gen,
            "True Profession": label,
            "Original Text": original_text,
            "Anonymized Text": anonymized_text,
            "Loss Description": loss_desc,
            "Score": score,
            "Judgement": "Correct" if is_correct else ("Incorrect" if is_correct is not None else "N/A")
        }
        return det
    except Exception as e:
        tqdm.write(f"跳过样本（评估异常）: {e}")
        return None
# ========== 新增结束 ==========

def main():
    parser = argparse.ArgumentParser(description="合并评估：LLM-as-Judge 同时计算离散准确率与连续分数。")
    parser.add_argument('--input', type=str, help="原始数据 JSONL 路径")
    parser.add_argument('--pred', type=str, help="预测结果 JSONL 路径（由生成脚本输出）")
    parser.add_argument('--limit', type=int, help="限制评估条数")
    parser.add_argument('--outdir', type=str, help="评估结果输出目录")
    parser.add_argument('--workers', type=int, default=50, help="并发线程数（默认 10）")
    args = parser.parse_args()

    config = Config()
    if args.input: config.INPUT_DATA_FILE = args.input
    if args.pred: config.PREDICTIONS_FILE = args.pred
    if args.limit is not None: config.LIMIT = args.limit
    if args.outdir: config.OUTPUT_DIR = args.outdir

    if not config.DEEPSEEK_API_KEY:
        print("Error: 未设置环境变量 API_KEY。")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 根据输入与预测文件名区分输出文件名
    in_stem = _safe_stem(config.INPUT_DATA_FILE)
    pred_stem = _safe_stem(config.PREDICTIONS_FILE)
    prefix = f"{pred_stem}__{in_stem}"
    out_json = os.path.join(config.OUTPUT_DIR, f"{prefix}_judge.json")
    out_jsonl = os.path.join(config.OUTPUT_DIR, f"{prefix}_judge_details.jsonl")

    # 读取数据与预测
    dataset = load_jsonl(config.INPUT_DATA_FILE)
    preds = load_jsonl(config.PREDICTIONS_FILE)
    if config.LIMIT is not None:
        preds = preds[:config.LIMIT]

    # 建立 index -> 样本 的映射（生成阶段按顺序保存了 index）
    data_by_index = {i: rec for i, rec in enumerate(dataset)}

    total = 0
    acc_numer = 0
    acc_denom = 0  # 仅统计有 label 的样本
    score_sum = 0
    excellent = 0  # 评分 >= 9
    details: List[Dict[str, Any]] = []

    print(f"\n--- 开始评估，共 {len(preds)} 条，线程数 {args.workers} ---")

    # 并行执行
    with open(out_jsonl, 'w', encoding='utf-8') as fdetail:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_evaluate_one, p, data_by_index, config) for p in preds]
            with tqdm(total=len(futures), desc="Evaluating") as pbar:
                for fut in as_completed(futures):
                    det = fut.result()
                    pbar.update(1)
                    if det is None:
                        continue

                    # 明细输出
                    fdetail.write(json.dumps(det, ensure_ascii=False) + "\n")
                    details.append(det)

                    # 指标累积
                    total += 1
                    score = det["Score"]
                    score_sum += score
                    if score >= 9:
                        excellent += 1

                    j = det["Judgement"]
                    if j in ("Correct", "Incorrect"):
                        acc_denom += 1
                        if j == "Correct":
                            acc_numer += 1

    avg_score = (score_sum / total) if total > 0 else 0.0
    excellent_rate = (excellent / total) if total > 0 else 0.0
    accuracy = (acc_numer / acc_denom) if acc_denom > 0 else 0.0

    metrics = {
        "total_evaluated": total,
        "average_score": avg_score,
        "excellent_recovery_rate_9_10": excellent_rate,
        "excellent_recovery_count": excellent,
        "discrete_accuracy": accuracy,
        "accuracy_denominator_with_label": acc_denom,
        "accuracy_numerator_correct": acc_numer
    }

    result = {
        "config": {
            "INPUT_DATA_FILE": config.INPUT_DATA_FILE,
            "PREDICTIONS_FILE": config.PREDICTIONS_FILE,
            "OUTPUT_DIR": config.OUTPUT_DIR,
            "workers": args.workers
        },
        "metrics": metrics,
        "qualitative_samples": details[:]
    }

    with open(out_json, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=4, ensure_ascii=False)

    print("\n--- 评估完成 ---")
    print(f"Average Score: {avg_score:.2f}/10")
    print(f"Excellent Recovery Rate (>=9): {excellent_rate:.2%} ({excellent}/{total})")
    if acc_denom > 0:
        print(f"Discrete Accuracy: {accuracy:.2%} ({acc_numer}/{acc_denom})")
    else:
        print("Discrete Accuracy: N/A（无可用标签）")
    print(f"明细保存：{out_jsonl}")
    print(f"汇总保存：{out_json}")

if __name__ == "__main__":
    main()