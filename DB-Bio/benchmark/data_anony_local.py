import os
import json
import argparse
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# --- PROMPT ---
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert in text anonymization, tasked with protecting user's privacy.
"""
PROMPT_ANONYMIZER_USER = """
Your goal is to rewrite [Biography Text] according to the [Leaked Attributes and Attacker Guesses] so that a competent attacker cannot accurately infer [PII Attributes] of the person in the biography text, while preserving meaning and fluency as much as possible.

[Biography Text]:
{original_text}

[PII Attributes]:
{pii_list_str}

[Leaked Attributes and Attacker Guesses]:
{feedback}

Now, provide ONLY your new, single, anonymized version of the text. Do not add any extra commentary or preamble.
"""

PROMPT_ATTACKER_SYSTEM = """
You are an expert profiler with deep knowledge of biographies and celebrities.
"""
PROMPT_ATTACKER_USER = """
Your task is to infer a user's profile based ONLY on the [Biography Text] provided. Your response MUST be a single, valid JSON object, containing your best guess for each of the following keys: "name", "birth_year", "sex", "birth_city_country", "education_level", "occupation", "income_level".
- For "birth_year", provide an integer.
- For all other fields, provide a string.
- If you cannot reasonably infer an attribute, use the JSON value null.

Input:
[Biography Text]:
{anonymized_text}

Respond ONLY with a single, valid JSON object.
"""

# --- 辅助函数 ---
def format_pii(pii: Dict[str, Any]) -> str:
    '''将 PII 字典格式化为 Prompt 字符串，只包含非 None 的属性。'''
    # 注意：本地化版本使用 format_profile，这里统一使用新格式
    pii_to_anonymize = {k: v for k, v in pii.items() if v is not None}
    return "\n".join([f"- {k}: {v}" for k, v in pii_to_anonymize.items()])


def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float) -> Tuple[Any, Any, List[int]]:
    """构建本地推理 pipeline。优先 device_map='auto'，否则按 device 指定。"""
    # 选择 dtype
    if dtype is None:
        if torch.cuda.is_available():
            # 兼容性检查和默认设置
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        # 手动设置 dtype
        dtype = dtype.lower()
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # 构建 kwargs
    kwargs = {}
    if device is None:
        kwargs["device_map"] = "auto"
    else:
        # 如果指定了 device，它将在 pipeline 中传递
        pass 
        
    # 增加 Llama 3/Mistral 等模型支持的特殊参数
    if 'llama' in model_name.lower() or 'mistral' in model_name.lower() or 'gemma' in model_name.lower():
        kwargs["attn_implementation"] = "sdpa"

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **kwargs
    )
    
    # 调整 pad_token/eos_token 兼容性
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建 pipeline
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if device else None # device_map='auto' 时 device=None
    )

    # 设置环境变量（在 build_pipeline 外部设置可能更稳定，但保留在函数内）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))

    # 终止符列表（兼容 Llama 3 的 <|eot_id|>）
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.vocab else None
    ]
    terminator_ids = [t for t in terminators if t is not None]
    
    return gen_pipe, tokenizer, terminator_ids


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取第一个完整 JSON 对象。使用简单的括号计数法，回退到严格解析。
    与 data_anony_local.py 版本保持一致，并新增对 Markdown 围栏的兼容。
    """
    # 1. 尝试清理 Markdown 围栏
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            block = parts[1]
            # 尝试去除可能的语言标识符（如 'json'）
            block = re.sub(r'^\s*json\s*\n', '', block, 1, flags=re.IGNORECASE)
            text = block.strip()
            
    # 2. 标准 JSON 提取逻辑
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
                    pass # 尝试找下一个 }
    
    # 3. 回退到宽松解析（如果上述失败）
    try:
        cleaned = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end+1])
    except Exception:
        return None
    return None

def build_prompt_chat(tokenizer, system_content: str, user_content: str) -> str:
    """使用 tokenizer 的 apply_chat_template 构建 Prompt"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def call_anonymizer(pipe, tokenizer, original_text: str, pii_list_str: str, feedback: str, terminator_ids: List[int]) -> str:
    """调用本地 LLM 匿名化器。"""
    user_content = PROMPT_ANONYMIZER_USER.format(
        original_text=str(original_text),
        pii_list_str=str(pii_list_str),
        feedback=str(feedback)
    )
    prompt = build_prompt_chat(tokenizer, PROMPT_ANONYMIZER_SYSTEM, user_content)
    logging.debug(f"--- ANONYMIZER PROMPT ---\n{prompt}\n-------------------------")
    
    temperature = 0.5
    do_sample = temperature > 0.0
    
    outputs = pipe(
        prompt,
        max_new_tokens=512, # 传记可能较长
        eos_token_id=terminator_ids,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.9 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]
    
    logging.debug(f"--- ANONYMIZER RESPONSE ---\n{out}\n-------------------------")
    return out.strip().strip('"').strip()


def call_attacker(pipe, tokenizer, anonymized_text: str, terminator_ids: List[int]) -> Optional[Dict[str, Any]]:
    """调用本地 LLM 攻击者。"""
    user_content = PROMPT_ATTACKER_USER.format(
        anonymized_text=str(anonymized_text)
    )
    prompt = build_prompt_chat(tokenizer, PROMPT_ATTACKER_SYSTEM, user_content)
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
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    out = outputs[0]["generated_text"]
    
    logging.debug(f"--- ATTACKER RESPONSE ---\n{out}\n-------------------------")
    obj = extract_first_json_object(out)
    logging.debug(f"--- ATTACKER PARSED ---\n{obj}\n-------------------------")
    return obj


# LLM Judge 函数被移除，使用 Python 确定性逻辑 compare_profiles_local。
def compare_profiles(true_profile: Dict[str, Any], guessed_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按键名比较属性，返回泄露列表。"""
    leaked_attributes: List[Dict[str, Any]] = []

    def _append(key: str, guess: Any, true: Any) -> None:
        leaked_attributes.append({"key": key, "guess": guess, "true": true})

    def _is_unknown(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "null", "none", "n/a", "not specified", "not mentioned", "unknown"}
        return False

    def _normalize(text: Any) -> Optional[str]:
        if not isinstance(text, str):
            return None
        if _is_unknown(text):
            return None
        return re.sub(r"\s+", " ", text.strip().lower())

    # birth_year 容差匹配（±3 年）
    true_birth_year = true_profile.get("birth_year")
    guessed_birth_year = guessed_profile.get("birth_year")
    if isinstance(true_birth_year, int) and isinstance(guessed_birth_year, int):
        if abs(true_birth_year - guessed_birth_year) <= 3:
            _append("birth_year", guessed_birth_year, true_birth_year)

    # 简单精确字段（忽略大小写与多余空白）
    for key in ["name", "sex"]:
        true_val = _normalize(true_profile.get(key))
        guess_val = _normalize(guessed_profile.get(key))
        if true_val and guess_val and true_val == guess_val:
            _append(key, guessed_profile.get(key), true_profile.get(key))

    # 子串/包含字段
    for key in ["birth_city_country", "education_level", "occupation", "income_level"]:
        true_val = _normalize(true_profile.get(key))
        guess_val = _normalize(guessed_profile.get(key))
        if not true_val or not guess_val:
            continue
        if len(guess_val) < 3:
            continue
        if guess_val in true_val or true_val in guess_val:
            _append(key, guessed_profile.get(key), true_profile.get(key))

    return leaked_attributes


def adversarial_anonymization(
    pipe,
    tokenizer,
    original_text: str,
    pii_dict: Dict[str, Any],
    terminator_ids: List[int],
    max_iterations: int = 5,
    record_id: int = -1
) -> Tuple[str, Dict[str, Any]]:
    """执行对抗性匿名化过程，返回最终的匿名化响应和元数据。"""
    
    # 只关注非 None 的 PII 属性进行匿名化和评测
    ground_truth_pii: Dict[str, Any] = {k: v for k, v in pii_dict.items() if v is not None}
    
    if not ground_truth_pii:
        meta = {"status": "no_pii_found", "iterations_used": 0}
        return original_text, meta

    pii_list_str = format_pii(ground_truth_pii)
    feedback = "None (first iteration)"
    current_anonymized_text = original_text
    
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
            current_anonymized_text = call_anonymizer(
                pipe, tokenizer, current_anonymized_text, pii_list_str, feedback, terminator_ids
            )
        except Exception as e:
            logging.error(f"[Record {record_id}][Iter {i+1}] Anonymizer failed: {e}")
            meta["status"] = "api_error"
            meta["error"] = f"anonymizer_error: {e}"
            return current_anonymized_text, meta

        # 2) 攻击者推断
        attacker_guess = None
        try:
            logging.info(f"[Record {record_id}][Iter {i+1}] Calling Attacker...")
            attacker_guess = call_attacker(pipe, tokenizer, current_anonymized_text, terminator_ids)
            if not isinstance(attacker_guess, dict):
                raise ValueError("Attacker did not return valid JSON.")
            meta["final_attacker_guess"] = attacker_guess
        except Exception as e:
            feedback = f"The attacker failed to produce a valid JSON profile. Error: {e}"
            logging.warning(f"[Record {record_id}][Iter {i+1}] {feedback}")
            continue

        # 3) 评判泄露 (使用本地 Python 规则)
        logging.info(f"[Record {record_id}][Iter {i+1}] Calling Judge (Local Rules)...")
        leaked_results = compare_profiles(ground_truth_pii, attacker_guess)

        if not leaked_results:
            logging.info(f"[Record {record_id}][Iter {i+1}] Success! No attributes leaked.")
            meta["status"] = "success"
            return current_anonymized_text, meta
        else:
            # 构建反馈
            leaked_feedback: Dict[str, Any] = {}
            leaked_attributes_str = []
            for item in leaked_results:
                leaked_attributes_str.append(
                    f"{item['key']} (guessed: '{item['guess']}', true: '{item['true']}')"
                )
                leaked_feedback[item["key"]] = item["guess"]

            feedback_items = [f"- {k}: guessed {v}" for k, v in leaked_feedback.items()]
            feedback = (
                "The attacker correctly inferred the following attributes from your rewritten text:\n"
                f"{chr(10).join(feedback_items)}\n"
                "Please generalize the text further to hide these specific clues."
            )
            logging.info(f"[Record {record_id}][Iter {i+1}] Failed. Leaked: {leaked_attributes_str}")
            meta["final_leaked_attributes"] = leaked_attributes_str

    logging.warning(f"[Record {record_id}] Max iterations reached. Final leaked: {meta['final_leaked_attributes']}")
    return current_anonymized_text, meta

def process_record(pipe, tokenizer, data: Dict[str, Any], max_iterations: int, record_id: int, terminator_ids: List[int]) -> Dict[str, Any]:
    """处理单条记录。"""
    logging.info(f"[Record {record_id}] Starting processing.")
    
    try:
        # PII 属性从 'personality' 字段获取
        personality = data.get("personality")
        # 原始文本从 'text' 或 'response' 字段获取
        original_text = str(data.get("text", data.get("response", "")))
        
        # 注入 'name' 字段 (如果存在于 'people' 中且 personality 中没有) - 待迁移代码的逻辑
        combined_pii = personality.copy() if personality else {}
        name_data = data.get("people")
        if not combined_pii.get("name") and name_data:
            combined_pii["name"] = name_data
            
    except Exception as e:
        logging.error(f"[Record {record_id}] Failed to read data fields: {e}")
        data["anonymization_meta"] = {"status": "skipped_data_read_error"}
        return data

    if not original_text or not combined_pii:
        logging.warning(f"[Record {record_id}] Skipped due to incomplete data.")
        data["anonymization_meta"] = {"status": "skipped_incomplete_data"}
        return data
    
    if not isinstance(combined_pii, dict):
        logging.warning(f"[Record {record_id}] Skipped: 'personality'/'combined_pii' field is not a dictionary.")
        data["anonymization_meta"] = {"status": "skipped_invalid_personality"}
        return data

    anonymized_response, meta = adversarial_anonymization(
        pipe=pipe,
        tokenizer=tokenizer,
        original_text=original_text,
        pii_dict=combined_pii,
        terminator_ids=terminator_ids,
        max_iterations=max_iterations,
        record_id=record_id
    )
    
    # 保持原代码的输出字段命名
    data["anonymized_response"] = anonymized_response
    data["anonymized_text"] = anonymized_response # 待迁移代码新增的字段
    data["anonymization_meta"] = meta
    
    logging.info(f"[Record {record_id}] Finished processing. Status: {meta.get('status')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用本地 Hugging Face 模型对 JSONL 中的传记文本进行匿名化")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Hugging Face 模型名")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 路径（例如 db_bio_with_attributes.jsonl）")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--success_file", type=str, default=None, help="仅成功记录输出路径")
    parser.add_argument("--failed_file", type=str, default=None, help="失败记录输出路径")
    parser.add_argument("--max_iterations", type=int, default=5, help="每条记录最大对抗轮数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条")
    
    parser.add_argument("--log_file", type=str, default="anonymizer_bio.log", help="日志文件路径")
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

    records_to_process = []
    for line in lines:
        try:
            if line.strip():
                records_to_process.append(json.loads(line))
        except json.JSONDecodeError:
            logging.error(f"Skipping line due to invalid JSON: {line.strip()}")


    if args.limit:
        records_to_process = records_to_process[:args.limit]

    # --- 模型加载 ---
    try:
        gen_pipe, tokenizer, terminators = build_pipeline(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        logging.error(f"Error: failed to load local model '{args.model_name}': {e}")
        sys.exit(1)
    
    logging.info(f"Using terminators: {terminators}")

    logging.info(f"Starting processing for {len(records_to_process)} records with model {args.model_name} ...")
    results = []

    # --- 串行处理 (本地 LLM 常用) ---
    def _task(rec_idx: int, rec: Dict[str, Any]):
        return process_record(gen_pipe, tokenizer, rec, args.max_iterations, rec_idx, terminators)

    for i, rec in enumerate(tqdm(records_to_process, desc="Anonymizing biographies")):
        try:
            results.append(_task(i, rec))
        except Exception as exc:
            logging.error(f"[Record {i}] Generated an exception: {exc}", exc_info=True)


    # --- 写结果 ---
    logging.info(f"\nProcessing complete. Writing {len(results)} results to output files...")
    
    # 使用 os.devnull 确保文件即使未指定也能正常处理
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(args.success_file, 'w', encoding='utf-8') if args.success_file else open(os.devnull, 'w') as success_out, \
         open(args.failed_file, 'w', encoding='utf-8') if args.failed_file else open(os.devnull, 'w') as failed_out:

        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            meta = result.get("anonymization_meta", {})
            # 兼容两份代码的失败/成功状态
            if meta.get("status") == "success":
                success_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            elif meta.get("status") in ("max_iterations_reached", "api_error", "skipped_incomplete_data", "skipped_data_read_error", "skipped_invalid_personality", "no_pii_found"):
                failed_out.write(json.dumps(result, ensure_ascii=False) + '\n')

    logging.info(f"Anonymized data saved to '{args.output_file}'.")
    logging.info(f"Log file saved to '{args.log_file}'.")

if __name__ == "__main__":
    main()