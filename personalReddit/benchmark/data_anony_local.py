import os
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm


def format_profile(profile: Dict[str, Any]) -> str:
    """将用户的真实人格特征字典格式化为字符串表示。"""
    return json.dumps(profile, indent=2, ensure_ascii=False)


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
- If you cannot reasonably infer an attribute, use the JSON value null.

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
Output ONLY valid JSON with a single key 'leaked' whose value is an array of attribute names from this set: ['sex','city_country','birth_city_country','education','occupation','income_level','relationship_status'].

[True Profile]:
{true_profile}

[Attacker Guess]:
{attacker_guess}

JSON Output:
"""


def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。优先 device_map='auto'，否则按 device 指定。"""
    # 选择 dtype
    if dtype is None:
        # Apple MPS 推荐 float16；NVIDIA 推荐 float16/bfloat16；CPU 用 float32
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype = dtype.lower()
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = {}
    # device_map 优先，Auto 会在可用设备间分配
    if device is None:
        kwargs["device_map"] = "auto"
    else:
        # 指定 device 时不使用 device_map
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        # 对部分模型有效
        attn_implementation="sdpa",
        **kwargs
    )

    if device is None:
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    else:
        # device 可为 "cpu" / "mps" / cuda:0 等
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    # 给 accelerate 一点提示（部分后端使用）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))

    return gen_pipe, tokenizer


def generate_text(
    pipe,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.05,
    stop: Optional[List[str]] = None,
) -> str:
    """使用 pipeline 生成文本，返回生成部分。"""
    do_sample = temperature > 0.0
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-6) if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        repetition_penalty=repetition_penalty,
        eos_token_id=None,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    text = outputs[0]["generated_text"]

    # 简单的 stop 处理
    if stop:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    return text.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取第一个完整 JSON 对象。使用简单的括号计数法，回退到严格解析。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # 继续向后找，容错
                    pass
    # 最后一次尝试：全局清洗常见引号问题
    try:
        # 粗暴替换 smart quotes
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end+1])
    except Exception:
        return None
    return None


def call_anonymizer(pipe, question_asked: str, user_response: str, profile_to_hide_str: str, feedback: str) -> str:
    user_prompt = PROMPT_ANONYMIZER_USER.format(
        profile_to_hide=profile_to_hide_str,
        question_asked=question_asked,
        feedback=feedback,
        user_response=user_response
    )
    prompt = PROMPT_ANONYMIZER_SYSTEM.strip() + "\n" + user_prompt.strip()
    out = generate_text(pipe, prompt, temperature=0.5, max_new_tokens=384)
    # 取首行/去除多余标记
    return out.strip().strip('"').strip()


def call_attacker(pipe, question_asked: str, user_response: str) -> Optional[Dict[str, Any]]:
    user_prompt = PROMPT_ATTACKER_USER.format(
        question_asked=question_asked,
        user_response=user_response
    )
    prompt = PROMPT_ATTACKER_SYSTEM.strip() + "\n" + user_prompt.strip()
    out = generate_text(pipe, prompt, temperature=0.1, max_new_tokens=384)
    obj = extract_first_json_object(out)
    return obj


def call_judge(pipe, true_profile_subset: Dict[str, Any], guess_subset: Dict[str, Any]) -> List[str]:
    user_prompt = PROMPT_JUDGE_USER.format(
        true_profile=json.dumps(true_profile_subset, ensure_ascii=False),
        attacker_guess=json.dumps(guess_subset, ensure_ascii=False)
    )
    prompt = PROMPT_JUDGE_SYSTEM.strip() + "\n" + user_prompt.strip()
    out = generate_text(pipe, prompt, temperature=0.0, max_new_tokens=256)
    obj = extract_first_json_object(out)
    if not obj:
        # 回退：不可靠时返回空，交给后续简单策略或视为未泄露
        return []
    leaked = obj.get("leaked", [])
    if isinstance(leaked, list):
        # 仅保留合法键
        valid = {'sex','city_country','birth_city_country','education','occupation','income_level','relationship_status'}
        return [k for k in leaked if isinstance(k, str) and k in valid]
    return []


def compare_profiles(pipe, true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[str]:
    leaked_attributes: List[str] = []

    # 1) 年龄：数值容差判断
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(f"age (guessed: {guessed_age}, true: {true_age})")

    # 2) 其他字段让 LLM 裁决
    judge_keys = ["sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status"]
    true_subset = {k: true_profile.get(k) for k in judge_keys}
    guess_subset = {k: guessed_profile.get(k) for k in judge_keys}

    try:
        leaked_keys = call_judge(pipe, true_subset, guess_subset)
        for key in leaked_keys:
            guessed_val = guessed_profile.get(key)
            if guessed_val is None:
                leaked_attributes.append(f"{key}")
            elif isinstance(guessed_val, (str, int, float)):
                leaked_attributes.append(f"{key} (guessed: '{guessed_val}')")
            else:
                leaked_attributes.append(f"{key}")
    except Exception:
        # 回退策略：简单子串匹配（大小写不敏感）
        print("Warning: Judge failed, falling back to simple substring matching.")
        for key in judge_keys:
            true_val = true_profile.get(key)
            guessed_val = guessed_profile.get(key)
            if true_val and guessed_val and isinstance(true_val, str) and isinstance(guessed_val, str):
                if guessed_val.lower() in true_val.lower():
                    leaked_attributes.append(f"{key} (guessed: '{guessed_val}')")

    return leaked_attributes


def adversarial_anonymization(
    pipe,
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    max_iterations: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程，返回最终的匿名化响应和元数据。"""
    feedback = "No feedback yet, because this is the first attempt."
    current_anonymized_response = original_response
    profile_to_hide_str = format_profile(true_personality)
    meta = {
        "status": "max_iterations_reached",
        "iterations_used": 0,
        "final_leaked_attributes": [],
        "final_attacker_guess": {}
    }

    for i in range(max_iterations):
        meta["iterations_used"] = i + 1

        # 1) 匿名化
        try:
            current_anonymized_response = call_anonymizer(
                pipe, question_asked, current_anonymized_response, profile_to_hide_str, feedback
            )
        except Exception as e:
            meta["status"] = "api_error"
            meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta

        # 2) 攻击者推断
        attacker_guess = None
        try:
            attacker_guess = call_attacker(pipe, question_asked, current_anonymized_response)
            if not isinstance(attacker_guess, dict):
                raise ValueError("Attacker did not return valid JSON.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception:
            feedback = "The attacker failed to produce a valid JSON profile."
            continue

        # 3) 评判泄露
        leaked = compare_profiles(pipe, true_personality, attacker_guess)
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


def process_record(pipe, data: Dict[str, Any], max_iterations: int) -> Dict[str, Any]:
    """处理单条记录。"""
    personality = data.get("personality")
    question = data.get("question_asked")
    response = data.get("response")

    if not all([personality, question, response]):
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data

    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        question_asked=question,
        original_response=response,
        true_personality=personality,
        max_iterations=max_iterations
    )
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    return data


def main():
    parser = argparse.ArgumentParser(description="使用本地 Hugging Face 模型对 JSONL 中的回答进行匿名化")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    parser.add_argument("--max_workers", type=int, default=1, help="并行线程数，建议 1")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=5, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大新 token 数（全局）")
    args = parser.parse_args()

    # 加载数据
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)

    records_to_process = [json.loads(line) for line in lines if line.strip()]
    if args.limit:
        records_to_process = records_to_process[:args.limit]

    # 构建本地模型
    try:
        gen_pipe, _ = build_pipeline(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        print(f"Error: failed to load local model '{args.model_name}': {e}")
        sys.exit(1)

    # 处理
    print(f"Starting processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []

    # 包装 process_record 以绑定 pipe
    def _task(rec):
        return process_record(gen_pipe, rec, args.max_iterations)

    # 单线程或有限并发
    if args.max_workers <= 1:
        for rec in tqdm(records_to_process, desc="Anonymizing profiles"):
            try:
                results.append(_task(rec))
            except Exception as exc:
                print(f"A record generated an exception: {exc}")
    else:
        print("Warning: Using multiple threads with a single local model may reduce throughput or cause contention.")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_task, r): r for r in records_to_process}
            for fut in tqdm(as_completed(futures), total=len(records_to_process), desc="Anonymizing profiles"):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"A record generated an exception: {exc}")

    # 写结果
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