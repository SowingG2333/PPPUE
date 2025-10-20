import os
import json
import argparse
import sys
import logging
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# --- 配置 ---
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

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

# --- 辅助函数：模型加载 ---

def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。优先 device_map='auto'，否则按 device 指定。"""
    
    # 1. 选择 dtype
    if dtype is None:
        if torch.cuda.is_available():
            # 优先使用 bf16
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

    print(f"Loading model '{model_name}' with dtype: {str(torch_dtype).split('.')[-1]}")
    
    # 2. 加载 Tokenizer 和 Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    kwargs = {}
    if device is None or device == 'auto':
        kwargs["device_map"] = "auto"
    else:
        # 如果指定了具体设备，例如 'cuda:0'
        pass

    # 设置环境变量以优化 GPU 内存
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa", # 优先使用 Flash Attention 2
            **kwargs
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to default implementation and trying again...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **kwargs
        )

    # 3. 构建 Pipeline
    if device is None or device == 'auto':
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

    return gen_pipe, tokenizer

def build_prompt(system_content: str, user_content: str) -> str:
    """使用简单的 System/User 拼接模板来构建 Prompt。"""
    # 这里的简单拼接旨在模拟原始代码中的提示结构。
    return f"{system_content.strip()}\n\n{user_content.strip()}"

def generate_local_response(pipe, tokenizer, prompt_text: str, temperature: float, max_tokens: int = 100) -> str:
    """使用本地 pipeline 生成信息损失描述。"""
    
    # 将 System 和 User 内容封装为消息列表，以便兼容 Llama/Mistral 等 Instruct 模型
    messages = [
        {"role": "system", "content": PROMPT_LOSS_SYSTEM.strip()},
        {"role": "user", "content": prompt_text}
    ]
    
    # 使用 chat template 格式化 prompt
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 终端符列表 (包含通用的 EOS token 和 Llama/Mistral 的 EOT token)
    terminator_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    do_sample = temperature > 0.0

    try:
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminator_ids,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False, # 仅返回生成的内容
        )
        
        out = outputs[0]["generated_text"]
        return out.strip().strip('"').strip()
    except Exception as e:
        # 在串行模式下，捕获异常并返回错误信息
        return f"Error: Local LLM generation failed with exception: {e}"

# --- 串行处理辅助函数 ---
def process_record(data: Dict[str, Any], pipe, tokenizer) -> Dict[str, Any]:
    """
    处理单条记录：生成信息损失描述并将其添加到数据字典中。
    """
    original_text = data.get('response')
    anonymized_text = data.get('anonymized_response')
    
    if not original_text or not anonymized_text:
        data['loss_description_sentence'] = "Error: Missing original or anonymized response."
        return data

    prompt_user_content = PROMPT_LOSS_USER.format(original_text=original_text, anonymized_text=anonymized_text)
    
    # 调用本地生成函数
    description = generate_local_response(
        pipe=pipe, 
        tokenizer=tokenizer, 
        prompt_text=prompt_user_content, 
        temperature=0.2
    )
    
    data['loss_description_sentence'] = description
    return data

# --- 主流程 ---
def main():
    parser = argparse.ArgumentParser(description="Generate information loss descriptions using a local Hugging Face LLM.")
    # 移除了 --max_workers
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with anonymized data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file where results will be saved.")
    
    # 添加本地模型参数
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face 模型名（例如 mistralai/Mistral-7B-Instruct-v0.3）")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0/auto。默认自动选择")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）")
    
    args = parser.parse_args()

    # --- 模型加载 ---
    try:
        pipe, tokenizer = build_pipeline(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        print(f"Error: Failed to load local model '{args.model_name}': {e}")
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

    print(f"Starting serial processing for {len(records_to_process)} records with model {args.model_name}...")

    # --- 串行处理 ---
    results = []
    for i, record in enumerate(tqdm(records_to_process, desc="Generating Loss Descriptions")):
        try:
            processed_record = process_record(record, pipe, tokenizer)
            results.append(processed_record)
        except Exception as exc:
            print(f"Record {i} generated an exception: {exc}")

    # --- 结果写入 ---
    print(f"\nProcessing complete. Writing {len(results)} results to '{args.output_file}'...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in results:
                # 确保写入时中文不被转义
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

    print("Task finished successfully.")

if __name__ == "__main__":
    # 配置日志系统，仅打印 WARNING 及以上信息，以避免 HuggingFace 自身的 INFO 级别输出干扰
    logging.basicConfig(level=logging.WARNING) 
    main()