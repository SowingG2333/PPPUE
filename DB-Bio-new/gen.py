import os
import json
import argparse
from typing import Dict, Optional

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from train import BioTrainableEnhancer, BIO_PROMPT_SYSTEM, BIO_PROMPT_USER

class Config:
    # 设备
    UEM_DEVICE = "cuda:0"
    LLM_DEVICE = "cuda:0"

    # 模型路径
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    UEM_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    CKPT_PATH = "/root/autodl-tmp/PPPUE/DB-Bio-new/ckpt/best_model"

    # 数据与输出
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/DB-Bio-new/benchmark/test/test_anony_with_loss.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/DB-Bio-new/results/ouput"

    # 生成设置
    EVAL_MODE = "DP"  # Options: BASELINE, STANDARD, CLIPPING_ONLY, DP, ORIGINAL_TEXT
    EPSILON = 50.0
    CLIPPING_NORM = 1.0
    PREFIX_LENGTH = 5
    LIMIT: Optional[int] = None

    # 生成超参
    GENERATE_KWARGS = {"max_new_tokens": 10, "temperature": 0.0, "do_sample": False}

@torch.no_grad()
def generate_single_output(
    data: Dict,
    eval_model: Optional[BioTrainableEnhancer],
    shared_llm: AutoModelForCausalLM,
    shared_tokenizer: AutoTokenizer,
    config: Config
) -> str:
    original_bio = data.get('text', '')
    anonymized_bio = data.get('anonymized_text', '')

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT"]:
        user_prompt = BIO_PROMPT_USER.format(
            biography_text=anonymized_bio if config.EVAL_MODE == "BASELINE" else original_bio
        )
        conversation = [
            {"role": "system", "content": BIO_PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ]
        prompt = shared_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        llm_inputs = shared_tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)
        outputs = shared_llm.generate(**llm_inputs, **config.GENERATE_KWARGS)
        generated_ids = outputs[:, llm_inputs['input_ids'].shape[1]:]
        return shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # UEM + 前缀注入路径
    loss_desc_sentence = data.get('loss_description_sentence', '')
    assert eval_model is not None, "eval_model 未加载，但当前模式需要 UEM 与投影层。"

    with autocast(enabled=eval_model.uem.device.type == 'cuda'):
        uem_inputs = eval_model.uem_tokenizer(
            loss_desc_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(eval_model.uem.device)
        uem_outputs = eval_model.uem(**uem_inputs)
        sentence_representation = uem_outputs.last_hidden_state[:, 0, :]
        clean_prefix = eval_model.projection_layer(sentence_representation)

    clean_prefix = clean_prefix.view(-1, config.PREFIX_LENGTH, shared_llm.config.hidden_size)

    # 前缀处理：STANDARD / CLIPPING_ONLY / DP
    if config.EVAL_MODE == "STANDARD":
        prefix_vector = clean_prefix
    else:
        original_norm = torch.linalg.norm(clean_prefix, dim=-1, keepdim=True)
        scale_factor = (config.CLIPPING_NORM / (original_norm + 1e-9)).clamp(max=1.0)
        clipped_prefix = clean_prefix * scale_factor
        if config.EVAL_MODE == "CLIPPING_ONLY":
            prefix_vector = clipped_prefix
        elif config.EVAL_MODE == "DP":
            sigma = (2 * config.CLIPPING_NORM) / (config.EPSILON + 1e-9)
            noise = torch.randn_like(clipped_prefix) * sigma
            prefix_vector = clipped_prefix + noise
        else:
            prefix_vector = clean_prefix

    prefix_embeds = prefix_vector.to(device=config.LLM_DEVICE, dtype=shared_llm.dtype)

    # 手动构建模板嵌入（与训练一致）
    embedding_layer = shared_llm.get_input_embeddings()

    student_user_prompt = BIO_PROMPT_USER.format(biography_text=anonymized_bio)
    system_part = shared_tokenizer.apply_chat_template(
        [{"role": "system", "content": BIO_PROMPT_SYSTEM}], tokenize=False, add_generation_prompt=False
    )
    user_start_part = shared_tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False
    ).replace(shared_tokenizer.eos_token, '')
    user_content_part = student_user_prompt

    dummy_conv = [{"role": "user", "content": "DUMMY"}]
    full_template = shared_tokenizer.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=True)
    dummy_template = shared_tokenizer.apply_chat_template(dummy_conv, tokenize=False, add_generation_prompt=False)
    assistant_start_part = full_template.replace(dummy_template, "")

    system_embeds = embedding_layer(shared_tokenizer(system_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
    user_start_embeds = embedding_layer(shared_tokenizer(user_start_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
    user_content_embeds = embedding_layer(shared_tokenizer(user_content_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
    assistant_start_embeds = embedding_layer(shared_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))

    combined_embeds = torch.cat([
        system_embeds, user_start_embeds, prefix_embeds, user_content_embeds, assistant_start_embeds
    ], dim=1)

    attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device)

    outputs = eval_model.llm_student.generate(
        inputs_embeds=combined_embeds,
        attention_mask=attention_mask,
        **config.GENERATE_KWARGS
    )
    generated_ids = outputs
    return shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

def load_models_for_mode(config: Config):
    print("--- 加载基础 LLM 与 Tokenizer ---")
    shared_llm = AutoModelForCausalLM.from_pretrained(
        config.LLM_PATH, torch_dtype=torch.float32, device_map={"": config.LLM_DEVICE}
    )
    shared_tokenizer = AutoTokenizer.from_pretrained(config.LLM_PATH)
    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        print(f"运行模式: {config.EVAL_MODE}. 不加载 UEM/LoRA。")
        return None, shared_llm, shared_tokenizer

    print("--- 加载 UEM 与 LoRA 组件 ---")
    uem_tokenizer = AutoTokenizer.from_pretrained(config.UEM_PATH, use_fast=False)

    lora_adapter_path = os.path.join(config.CKPT_PATH, "lora_adapters")
    other_weights_path = os.path.join(config.CKPT_PATH, "uem_projection.pt")
    if not os.path.isdir(lora_adapter_path) or not os.path.isfile(other_weights_path):
        raise FileNotFoundError(f"未找到模型组件: {config.CKPT_PATH}")

    print(f"加载 LoRA adapters: {lora_adapter_path}")
    llm_student = PeftModel.from_pretrained(shared_llm, lora_adapter_path)
    llm_student = llm_student.merge_and_unload()
    print("LoRA 合并完成。")

    eval_model = BioTrainableEnhancer(config.UEM_PATH, uem_tokenizer, None, llm_student, shared_tokenizer)
    eval_model.uem.to(config.UEM_DEVICE)
    eval_model.projection_layer.to(config.UEM_DEVICE)

    print(f"加载 UEM 与投影权重: {other_weights_path}")
    checkpoint = torch.load(other_weights_path, map_location="cpu")
    eval_model.uem.load_state_dict(checkpoint['uem_state_dict'])
    eval_model.projection_layer.load_state_dict(checkpoint['projection_layer_state_dict'])
    eval_model.eval()

    print("所有组件加载完成。")
    return eval_model, shared_llm, shared_tokenizer

def main():
    parser = argparse.ArgumentParser(description="仅生成模型输出（与评估解耦）。")
    parser.add_argument('--mode', type=str, help="模式: BASELINE, STANDARD, CLIPPING_ONLY, DP, ORIGINAL_TEXT_BASELINE")
    parser.add_argument('--limit', type=int, help="限制生成条数")
    parser.add_argument('--input', type=str, help="输入数据 JSONL 路径")
    parser.add_argument('--ckpt', type=str, help="UEM/LoRA checkpoint 目录")
    parser.add_argument('--eps', type=float, help="DP 模式的 epsilon")
    parser.add_argument('--output', type=str, help="输出预测 JSONL 路径")
    args = parser.parse_args()

    config = Config()
    if args.mode: config.EVAL_MODE = args.mode
    if args.limit is not None: config.LIMIT = args.limit
    if args.input: config.INPUT_DATA_FILE = args.input
    if args.ckpt: config.CKPT_PATH = args.ckpt
    if args.eps: config.EPSILON = args.eps

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(config.OUTPUT_DIR, f"pred_{config.EVAL_MODE}_eps{config.EPSILON}.jsonl")

    eval_model, shared_llm, shared_tokenizer = load_models_for_mode(config)

    with open(config.INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    records = lines[:config.LIMIT] if config.LIMIT else lines

    print(f"\n--- 开始生成，共 {len(records)} 条 ---")
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, line in enumerate(tqdm(records, desc="Generating")):
            try:
                data = json.loads(line)
                pred = generate_single_output(data, eval_model, shared_llm, shared_tokenizer, config)
                # 仅输出最小必要字段，使用 index 保持顺序对齐
                out = {
                    "index": idx,
                    "generated_answer": pred,
                    "mode": config.EVAL_MODE
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            except Exception as e:
                tqdm.write(f"跳过样本（解析或生成异常）: {e}")

    print(f"\n生成完成。输出已保存：{output_path}")

if __name__ == "__main__":
    main()