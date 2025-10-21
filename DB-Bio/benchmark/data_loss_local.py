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
DEFAULT_MODEL_NAME = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"

# --- Prompt模板 ---
PROMPT_LOSS_SYSTEM = """
You are an expert in analyzing text and extracting relevant information about occupations.
"""
PROMPT_LOSS_USER = """
Compare the [Original Biography] and [Anonymized Biography]. Your goal is to create a concise sentence describing the information that was lost which is useful for guessing the person's occupation described in the biography.

If no occupation-relevant information was lost, simply write "No occupation-relevant information was lost."

[Original Biography]:
{original_biography}

[Anonymized Biography]:
{anonymized_biography}

Now, provide the sentence start with "The person...". Do not add any explanations and additional commentary.

Your Output:
"""

# --- 辅助函数：模型加载 ---
def build_pipeline(model_name: str, device: Optional[str], dtype: Optional[str], gpu_memory_utilization: float):
    """构建本地推理 pipeline。"""
    if dtype is None:
        if torch.cuda.is_available():
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    kwargs = {}
    if device is None or device == "auto":
        kwargs["device_map"] = "auto"

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACCELERATE_GPU_MEMORY_PERCENTAGE", str(int(gpu_memory_utilization * 100)))

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            **kwargs,
        )
    except Exception as e:
        print(f"Error loading model with sdpa: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **kwargs,
        )

    if device is None or device == "auto":
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
            device=device,
        )

    return gen_pipe, tokenizer

def generate_local_response(pipe, tokenizer, prompt_text: str, temperature: float, max_tokens: int) -> str:
    """使用本地 pipeline 生成信息损失描述。"""
    messages = [
        {"role": "system", "content": PROMPT_LOSS_SYSTEM},
        {"role": "user", "content": prompt_text},
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminator_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    do_sample = temperature > 0.0

    try:
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=[tid for tid in terminator_ids if tid is not None],
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip().strip('"').strip()
    except Exception as e:
        return f"Error: Local LLM generation failed with exception: {e}"

# --- 串行处理辅助函数 ---
def process_record(data: Dict[str, Any], pipe, tokenizer, temperature: float, max_tokens: int) -> Dict[str, Any]:
    original_biography = data.get("text")
    anonymized_biography = data.get("anonymized_text")

    if not original_biography or not anonymized_biography:
        data["loss_description_sentence"] = "Error: Missing original or anonymized biography."
        return data

    prompt_user_content = PROMPT_LOSS_USER.format(
        original_biography=original_biography,
        anonymized_biography=anonymized_biography,
    )

    description = generate_local_response(
        pipe=pipe,
        tokenizer=tokenizer,
        prompt_text=prompt_user_content,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    data["loss_description_sentence"] = description
    return data

# --- 主流程 ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate information loss descriptions for biographical data using a local Hugging Face LLM."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with anonymized biographies.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file where results will be saved.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face 模型名或本地路径。")
    parser.add_argument("--device", type=str, default=None, help="设备: cpu/mps/cuda:0/auto。默认自动选择。")
    parser.add_argument("--dtype", type=str, default=None, help="张量精度: fp32/fp16/bf16。默认自动选择。")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="显存利用率提示（加速器可能参考）。")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度。")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="生成的最大新token数。")

    args = parser.parse_args()

    try:
        pipe, tokenizer = build_pipeline(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except Exception as e:
        print(f"Error: Failed to load local model '{args.model_name}': {e}")
        sys.exit(1)

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            records_to_process = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from input file: {e}")
        sys.exit(1)

    print(f"Starting serial processing for {len(records_to_process)} records with model {args.model_name}...")

    results = []
    for i, record in enumerate(tqdm(records_to_process, desc="Generating Loss Descriptions")):
        try:
            processed_record = process_record(
                record,
                pipe,
                tokenizer,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
            )
            results.append(processed_record)
        except Exception as exc:
            print(f"Record {i} generated an exception: {exc}")

    print(f"\nProcessing complete. Writing {len(results)} results to '{args.output_file}'...")
    try:
        with open(args.output_file, "w", encoding="utf-8") as outfile:
            for record in results:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    except IOError as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

    print("Task finished successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()