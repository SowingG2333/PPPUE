import os
import json
import argparse
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm


def format_profile(profile: Dict[str, Any]) -> str:
    """将用户的真实人格特征字典格式化为字符串表示。"""
    return json.dumps(profile, indent=2, ensure_ascii=False)


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

PROMPT_JUDGE_SYSTEM = """
You are an expert judge tasked with determining which personal attributes have been leaked based on the true profile and the attacker's guess.
"""
PROMPT_JUDGE_USER = """
Your goal is to perform a **strict, literal comparison** of the [True Profile] and the [Attacker Guess] attribute by attribute. You must identify ONLY the attributes where the attacker's guess is **correctly inferred**.

A guess is **CORRECT** ONLY IF:
1. It is an exact match (case-insensitive).
2. It is a clear subset (e.g., Guess='Engineer' and True='Software Engineer').

A guess is **INCORRECT** IF:
1. It is a different value, even from the same category.
2. It is an opposite (e.g., 'Single' vs 'married').
3. It is a different type (e.g., 'Bachelor's' vs 'Masters').
4. It is a different field (e.g., 'Software engineer' vs 'structural engineer').
5. One of the values is null.

---
Example:

[True Profile]:
{
  "sex": "male",
  "education": "Masters in Structural Engineering",
  "occupation": "structural engineer",
  "relationship_status": "married"
}

[Attacker Guess]:
{
  "sex": "male",
  "education": "Bachelor's degree",
  "occupation": "Software engineer",
  "relationship_status": "Single"
}

JSON Output:
{
  "leaked": ["sex"]
}

---
**Your Task:**

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
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
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
    if device is None:
        kwargs["device_map"] = "auto"
    else:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
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
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))

    return gen_pipe, tokenizer


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
                    pass
    try:
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end+1])
    except Exception:
        return None
    return None


def build_prompt(system_content: str, user_content: str) -> str:
    return f"{system_content.strip()}\n\n{user_content.strip()}"


def call_anonymizer(pipe, question_asked: str, user_response: str, profile_to_hide_str: str, feedback: str, terminator_ids: List[int]) -> str:
    user_content = PROMPT_ANONYMIZER_USER.format(
        profile_to_hide=str(profile_to_hide_str),
        question_asked=str(question_asked),
        feedback=str(feedback),
        user_response=str(user_response)
    )
    messages = [
        {"role": "system", "content": PROMPT_ANONYMIZER_SYSTEM.strip()},
        {"role": "user", "content": user_content}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.debug(f"--- ANONYMIZER PROMPT ---\n{prompt}\n-------------------------")
    
    temperature = 0.5
    do_sample = temperature > 0.0
    
    outputs = pipe(
        prompt,
        max_new_tokens=384,
        eos_token_id=terminator_ids,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.9 if do_sample else None,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]
    # 打印完整输出以供调试
    logging.info(f"--- ANONYMIZER FULL OUTPUT ---\n{out}\n-------------------------")
    logging.debug(f"--- ANONYMIZER RESPONSE ---\n{out}\n-------------------------")
    return out.strip().strip('"').strip()


def call_attacker(pipe, question_asked: str, user_response: str, terminator_ids: List[int]) -> Optional[Dict[str, Any]]:
    user_content = PROMPT_ATTACKER_USER.format(
        question_asked=str(question_asked),
        user_response=str(user_response)
    )
    system_content = PROMPT_ATTACKER_SYSTEM.strip()
    prompt = build_prompt(system_content, user_content)
    logging.debug(f"--- ATTACKER PROMPT ---\n{prompt}\n-------------------------")
    
    temperature = 0.1
    do_sample = temperature > 0.0

    outputs = pipe(
        prompt,
        max_new_tokens=384,
        eos_token_id=terminator_ids,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.9 if do_sample else None,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]
    
    logging.debug(f"--- ATTACKER RESPONSE ---\n{out}\n-------------------------")
    obj = extract_first_json_object(out)
    logging.debug(f"--- ATTACKER PARSED ---\n{obj}\n-------------------------")
    return obj


def call_judge(pipe, true_profile_subset: Dict[str, Any], guess_subset: Dict[str, Any], terminator_ids: List[int]) -> List[str]:
    user_content = PROMPT_JUDGE_USER.format(
        true_profile=json.dumps(true_profile_subset, ensure_ascii=False),
        attacker_guess=json.dumps(guess_subset, ensure_ascii=False)
    )
    system_content = PROMPT_JUDGE_SYSTEM.strip()
    prompt = build_prompt(system_content, user_content)
    logging.debug(f"--- JUDGE PROMPT ---\n{prompt}\n-------------------------")

    temperature = 0.0
    do_sample = temperature > 0.0 # 这将是 False

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminator_ids,
        do_sample=do_sample,
        temperature=None,
        top_p=None,       
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]

    logging.debug(f"--- JUDGE RESPONSE ---\n{out}\n-------------------------")
    obj = extract_first_json_object(out)
    logging.debug(f"--- JUDGE PARSED ---\n{obj}\n-------------------------")

    if not obj:
        return []
    leaked = obj.get("leaked", [])
    if isinstance(leaked, list):
        valid = {'sex','city_country','birth_city_country','education','occupation','income_level','relationship_status'}
        return [k for k in leaked if isinstance(k, str) and k in valid]
    return []


def compare_profiles(pipe, true_profile: Dict[str, Any], guessed_profile: Dict[str, Any], terminator_ids: List[int]) -> List[str]:
    """
    使用确定性的 Python 逻辑比较 profiles，不再调用 LLM (call_judge)。
    """
    leaked_attributes: List[str] = []
    
    # 辅助函数，用于格式化日志
    def _log(key, guess, true):
        return f"{key} (guessed: '{guess}', true: '{true}')"

    # 1) 年龄：数值容差判断 (保留)
    true_age = true_profile.get("age")
    guessed_age = guessed_profile.get("age")
    if isinstance(true_age, int) and isinstance(guessed_age, int):
        if abs(true_age - guessed_age) <= 3:
            leaked_attributes.append(_log("age", guessed_age, true_age))

    # 2) 简单字段 (必须精确匹配，忽略大小写)
    simple_keys = ["sex", "relationship_status"]
    for key in simple_keys:
        true_val = true_profile.get(key)
        guess_val = guessed_profile.get(key)
        # 必须是字符串且非空
        if isinstance(true_val, str) and isinstance(guess_val, str) and true_val and guess_val:
            if true_val.strip().lower() == guess_val.strip().lower():
                leaked_attributes.append(_log(key, guess_val, true_val))
    
    # 3) 复杂字段 (子集/子字符串匹配)
    # 规则：如果猜测是真实值的子集 (e.g., Guess="Masters" in True="Masters in Engineering")，则为泄露。
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
                
            # 核心逻辑：
            # 1. 猜测是真实值的一部分 (e.g., Guess "engineer" in True "software engineer")
            # 2. 真实值是猜测的一部分 (e.g., True "NYC" in Guess "New York City")
            if (guess_s in true_s) or (true_s in guess_s):
                leaked_attributes.append(_log(key, guess_val, true_val))

    # 4) LLM Judge 调用已被完全移除
    logging.debug(f"Rule-based comparison complete. Leaked: {leaked_attributes}")
    return leaked_attributes

# def compare_profiles(pipe, true_profile: Dict[str, Any], guessed_profile: Dict[str, Any], terminator_ids: List[int]) -> List[str]:
#     leaked_attributes: List[str] = []

#     # 1) 年龄：数值容差判断
#     true_age = true_profile.get("age")
#     guessed_age = guessed_profile.get("age")
#     if isinstance(true_age, int) and isinstance(guessed_age, int):
#         if abs(true_age - guessed_age) <= 3:
#             leaked_attributes.append(f"age (guessed: {guessed_age}, true: {true_age})")

#     # 2) 其他字段让 LLM 裁决
#     judge_keys = ["sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status"]
#     true_subset = {k: true_profile.get(k) for k in judge_keys}
#     guess_subset = {k: guessed_profile.get(k) for k in judge_keys}
    
#     logging.debug(f"Comparing True: {true_subset} vs Guess: {guess_subset}")

#     try:
#         leaked_keys = call_judge(pipe, true_subset, guess_subset, terminator_ids)
#         logging.debug(f"Judge returned: {leaked_keys}")
#         for key in leaked_keys:
#             guessed_val = guessed_profile.get(key)
#             if guessed_val is None:
#                 leaked_attributes.append(f"{key}")
#             elif isinstance(guessed_val, (str, int, float)):
#                 leaked_attributes.append(f"{key} (guessed: '{guessed_val}', true: '{true_profile.get(key)}')")
#             else:
#                 leaked_attributes.append(f"{key}")
#     except Exception as e:
#         logging.warning(f"Judge failed ({e}), falling back to simple substring matching.")
#         for key in judge_keys:
#             true_val = true_profile.get(key)
#             guessed_val = guessed_profile.get(key)
#             if true_val and guessed_val and isinstance(true_val, str) and isinstance(guessed_val, str):
#                 if guessed_val.lower() in true_val.lower():
#                     leaked_attributes.append(f"{key} (guessed: '{guessed_val}')")

#     return leaked_attributes


# MODIFIED: 修正了参数顺序
def adversarial_anonymization(
    pipe,
    question_asked: str,
    original_response: str,
    true_personality: Dict[str, Any],
    terminator_ids: List[int], # <- 移到了这里
    max_iterations: int = 5,
    record_id: int = -1
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
    
    logging.info(f"[Record {record_id}] Starting adversarial process (Max {max_iterations} iterations).")

    for i in range(max_iterations):
        meta["iterations_used"] = i + 1
        logging.info(f"[Record {record_id}][Iter {i+1}/{max_iterations}] Current feedback: {feedback}")

        # 1) 匿名化
        try:
            logging.info(f"[Record {record_id}][Iter {i+1}] Calling Anonymizer...")
            current_anonymized_response = call_anonymizer(
                pipe, question_asked, current_anonymized_response, profile_to_hide_str, feedback, terminator_ids
            )
        except Exception as e:
            logging.error(f"[Record {record_id}][Iter {i+1}] Anonymizer failed: {e}")
            meta["status"] = "api_error"
            meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_response, meta

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"[Record {record_id}][Iter {i+1}] Calling Attacker...")
            attacker_guess = call_attacker(pipe, question_asked, current_anonymized_response, terminator_ids)
            if not isinstance(attacker_guess, dict):
                raise ValueError("Attacker did not return valid JSON.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            feedback = f"The attacker failed to produce a valid JSON profile. Error: {e}"
            logging.warning(f"[Record {record_id}][Iter {i+1}] {feedback}")
            continue

        # 3) 评判泄露
        logging.info(f"[Record {record_id}][Iter {i+1}] Calling Judge...")
        leaked = compare_profiles(pipe, true_personality, attacker_guess, terminator_ids)
        if not leaked:
            logging.info(f"[Record {record_id}][Iter {i+1}] Success! No attributes leaked.")
            meta["status"] = "success"
            return current_anonymized_response, meta
        else:
            feedback = (
                "The attacker correctly inferred the following attributes from your rewritten text:\n"
                f"- {', '.join(leaked)}\n"
                "Please generalize the text further to hide these specific clues."
            )
            logging.info(f"[Record {record_id}][Iter {i+1}] Failed. Leaked: {leaked}")
            meta["final_leaked_attributes"] = leaked

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_response, meta


def process_record(pipe, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    
    # MODIFIED: 在这里就地强制转换为 str，确保下游拿到的是字符串
    try:
        personality = data.get("personality") # This should be a dict
        question = str(data.get("question_asked")) # 强制转换为 str
        response = str(data.get("response"))       # 强制转换为 str
    except Exception as e:
        logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not all([personality, question, response]):
        logging.warning(f"[Record {record_id}] Skipped due to incomplete data.")
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data

    # 额外检查：确保 personality 是一个字典
    if not isinstance(personality, dict):
        logging.warning(f"[Record {record_id}] Skipped: 'personality' field is not a dictionary.")
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        question_asked=question,       # 现在保证是 str
        original_response=response,    # 现在保证是 str
        true_personality=personality,
        terminator_ids=terminator_ids, 
        max_iterations=max_iterations,
        record_id=record_id
    )
    data["anonymized_response"] = anonymized_response
    data["anonymization_meta"] = meta
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data


def main():
    parser = argparse.ArgumentParser(description="使用本地 Hugging Face 模型对 JSONL 中的回答进行匿名化")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 train.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=5, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大新 token 数（全局）")
    
    parser.add_argument("--log_file", type=str, default="anonymizer.log", help="日志文件路径")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别 (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()

    # --- Logger Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file, 'w', 'utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # --- End Logger Setup ---

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)

    records_to_process = [json.loads(line) for line in lines if line.strip()]
    if args.limit:
        records_to_process = records_to_process[:args.limit]

    try:
        gen_pipe, tokenizer = build_pipeline(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}")
        sys.exit(1)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    logging.info(f"Using Llama 3 terminators: {terminators}")

    logging.info(f"Starting processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []

    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, rec, args.max_iterations, rec_idx, terminators)

    for i, rec in enumerate(tqdm(records_to_process, desc="Anonymizing profiles")):
        try:
            results.append(_task(i, rec))
        except Exception as exc:
            logging.error(f"[Record {i}] Generated an exception: {exc}", exc_info=True)


    # 写结果
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
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

    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")


if __name__ == "__main__":
    main()