import os
import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import threading
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI

PROMPT_SCORE_SYSTEM = """
You are a highly discerning evaluation expert with deep contextual understanding of professional roles and their nuances.
"""
PROMPT_SCORE_USER = """
Evaluate the quality of a "Generated Answer" regarding a person's occupation, using the full **[Original User Response]** (provided in the context of the **[Question Asked]**) as the ultimate source of truth. The "Generated Answer" was produced by an AI that saw the [Question Asked] and an ANONYMIZED version of the [User Response]. Focus on **how well the AI's answer reflects specific, nuanced occupation-related information present *only* in the [Original User Response]**.

Rate the "Generated Answer" on a scale of 1 to 10 based on the rubric below. Output ONLY the final numerical score (1-10) and nothing else.

SCORING RUBRIC (10-Point Scale):
- 1-2 (Incorrect): Completely wrong or irrelevant.
- 3-4 (Weakly Related): Related but incorrect field.
- 5-6 (Correct Generalization): Correct but overly general category plausible from anonymized text.
- 7-8 (Good Recovery): Correctly identifies a significant past profession, close synonym, but misses the current/precise role from the original.
- 9-10 (Excellent Recovery): Highly specific and correct identification of the primary, current profession only recoverable from the original response (10 = perfect match).

[Question Asked]:
{question_asked}

[Original User Response]:
{original_text}

[Ground Truth Profession]:
{ground_truth_label}

[Generated Answer]:
{generated_answer}

Your Output:
"""

PROMPT_ACC_SYSTEM = """
You are a highly precise evaluation expert with a deep understanding of professional roles and their nuances.
"""
PROMPT_ACC_USER = """
Your task is to strictly determine if the "Generated Answer" correctly identifies the "Ground Truth Profession".

A "Generated Answer" is considered Correct" only if it meets one of the following criteria:
1.  It is an exact match with the "Ground Truth Profession".
2.  It is a clear and direct synonym (e.g., Footballer for Soccer Player).
3.  It is more specific or more precise than the ground truth, while still being factually accurate based on a person's career (e.g., "Academy Coach" is a more precise current role for a "Soccer Player" who has transitioned).

A "Generated Answer" is "Incorrect" if:
1.  It is a completely different profession.
2.  It is a **more general or less precise** role (e.g., "Cook" for "Chef").

Output your final verdict as a single word: "Correct" or "Incorrect".

Ground Truth Profession: "{ground_truth}"

Generated Answer: "{generated_answer}"

Your verdict:
"""

class Config:
    # DeepSeek API
    DEEPSEEK_API_KEY = os.environ.get("API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"
    MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "30"))

def generate_api_response(client: OpenAI, model: str, messages: List[Dict], temperature: float) -> str:
    try:
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("API returned no message content.")
        return content.strip()
    except Exception as e:
        tqdm.write(f"  - ERROR calling API for model {model}: {e}")
        return "API_ERROR"

def judge_score(client: OpenAI, cfg: Config, question_asked: str, original_text: str, ground_truth_label: str, generated_answer: str) -> Optional[int]:
    user_prompt = PROMPT_SCORE_USER.format(
        question_asked=question_asked,
        original_text=original_text,
        ground_truth_label=ground_truth_label,
        generated_answer=generated_answer
    )
    resp = generate_api_response(
        client,
        cfg.JUDGE_MODEL_NAME,
        [{"role": "system", "content": PROMPT_SCORE_SYSTEM},
         {"role": "user", "content": user_prompt}],
        0.0
    )
    if resp == "API_ERROR":
        return None
    try:
        m = re.search(r'\d+', resp)
        if m:
            score = int(m.group(0))
            if not (1 <= score <= 10):
                score = max(1, min(10, score))
            return score
        return None
    except Exception:
        return None

def judge_discrete(client: OpenAI, cfg: Config, ground_truth_label: str, generated_answer: str) -> Optional[bool]:
    judge_prompt = PROMPT_ACC_USER.format(ground_truth=ground_truth_label, generated_answer=generated_answer)
    resp = generate_api_response(
        client,
        cfg.JUDGE_MODEL_NAME,
        [{"role": "system", "content": PROMPT_ACC_SYSTEM},
         {"role": "user", "content": judge_prompt}],
        0.0
    )
    if resp == "API_ERROR":
        return None
    verdict = resp.lower().strip().strip('.,!"\'')
    if verdict in ["correct", "incorrect"]:
        return verdict == "correct"
    return None

thread_local_client = threading.local()

def get_thread_client(cfg: Config) -> OpenAI:
    if not hasattr(thread_local_client, "client"):
        thread_local_client.client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_API_BASE)
    return thread_local_client.client

def evaluate_record(idx: int, line: str, cfg: Config):
    try:
        rec = json.loads(line)
        client = get_thread_client(cfg)
        question_asked = rec.get("question_asked", "")
        original_text = rec.get("original_response", "")
        ground_truth = rec.get("true_label", "")
        generated_answer = rec.get("generated_answer", "")

        score = judge_score(client, cfg, question_asked, original_text, ground_truth, generated_answer)
        correct = judge_discrete(client, cfg, ground_truth, generated_answer)

        sample = {
            "record_index": rec.get("record_index"),
            "question_asked": question_asked,
            "original_response": original_text,
            "anonymized_response": rec.get("anonymized_response", ""),
            "true_label": ground_truth,
            "generated_answer": generated_answer,
            "discrete_judgement": "Correct" if correct else "Incorrect" if correct is not None else "API_ERROR",
            "score": score if score is not None else "API_ERROR",
            "eval_mode": rec.get("eval_mode"),
            "epsilon": rec.get("epsilon"),
        }

        return {
            "index": idx,
            "sample": sample,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "score": score,
            "correct": correct,
            "excellent": score is not None and score >= 9,
        }
    except Exception as e:
        return {"index": idx, "error": str(e)}

def main(cfg: Config):
    if not cfg.DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set (env: API_KEY).")
        return
    if not cfg.PREDICTIONS_FILE or not os.path.exists(cfg.PREDICTIONS_FILE):
        print(f"Error: predictions file not found: {cfg.PREDICTIONS_FILE}")
        return
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_API_BASE)

    # 读取预测
    with open(cfg.PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = 0
    score_sum = 0
    excellent_count = 0  # 9-10
    correct_count = 0

    samples: List[Dict] = []
    sample_results = []

    print(f"--- Running combined evaluation on {len(lines)} predictions ---")
    progress = tqdm(total=len(lines), desc="Evaluate")
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(evaluate_record, i, line, cfg): i
            for i, line in enumerate(lines)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            progress.update(1)
            result = future.result()
            if "error" in result:
                tqdm.write(f"Skip line {result['index'] + 1} due to error: {result['error']}")
                continue

            score = result["score"]
            correct = result["correct"]

            if score is not None:
                score_sum += score
                if result["excellent"]:
                    excellent_count += 1
            if correct is True:
                correct_count += 1
            total += 1

            sample_results.append((result["index"], result["sample"]))

            avg_score = (score_sum / total) if total else 0.0
            acc = (correct_count / total) if total else 0.0
            exc_rate = (excellent_count / total) if total else 0.0
            tqdm.write(
                f"[{result['index'] + 1}/{len(lines)}] Label: {result['ground_truth']} | "
                f"Pred: '{result['generated_answer']}' | "
                f"Judgement: {('Correct' if correct else 'Incorrect' if correct is not None else 'API_ERROR')}, "
                f"Score: {score if score is not None else 'API_ERROR'} | "
                f"AvgScore: {avg_score:.2f} | Acc: {acc:.2%} | 9-10: {exc_rate:.2%}"
            )
    progress.close()

    samples = [sample for _, sample in sorted(sample_results, key=lambda x: x[0])]

    avg_score = (score_sum / total) if total else 0.0
    acc = (correct_count / total) if total else 0.0
    exc_rate = (excellent_count / total) if total else 0.0

    print("\n--- Combined Evaluation Complete ---")
    print(f"Total Evaluated: {total}")
    print(f"Accuracy (discrete): {acc:.2%} ({correct_count}/{total})")
    print(f"Average Score (1-10): {avg_score:.2f}")
    print(f"Excellent Recovery Rate (9-10): {exc_rate:.2%} ({excellent_count}/{total})")

    metrics = {
        "accuracy": acc if total else None,
        "correct_predictions": correct_count,
        "total_evaluated": total,
        "average_score": avg_score if total else None,
        "excellent_recovery_rate": exc_rate if total else None,
        "excellent_recovery_count": excellent_count
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.OUTPUT_DIR, f"combined_eval_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "predictions_file": cfg.PREDICTIONS_FILE,
                "judge_model": cfg.JUDGE_MODEL_NAME,
                "api_base": cfg.DEEPSEEK_API_BASE
            },
            "metrics": metrics,
            "samples": samples
        }, f, ensure_ascii=False, indent=4)

    print(f"Saved combined evaluation to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined evaluation: LLM-as-Judge for accuracy and score.")
    parser.add_argument('--preds', type=str, help="Path to predictions JSONL from gen_outputs.py")
    parser.add_argument('--outdir', type=str, help="Directory to save evaluation results.")
    args = parser.parse_args()

    cfg = Config()
    cfg.PREDICTIONS_FILE = args.preds
    if args.preds:
        cfg.PREDICTIONS_FILE = args.preds
    if args.outdir:
        cfg.OUTPUT_DIR = args.outdir

    main(cfg)