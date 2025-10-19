import os
import json
import argparse
from datetime import datetime
from typing import Dict, Optional, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.cuda.amp import autocast

from train import TrainableEnhancer, REDDIT_PROMPT_SYSTEM, REDDIT_PROMPT_USER

class Config:
    # 设备
    IEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    LLM_DEVICE = "cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else IEM_DEVICE

    # 模型路径
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    IEM_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    CKPT_PATH = "/root/autodl-tmp/PPPUE/personalReddit/ckpt/Prefix_LoRA"

    # 数据与输出
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/personalReddit/benchmark/reprocess/test/test_anony_with_loss.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/personalReddit/results/LoRA_Prefix/strict_prompt/preds"

    # 评估/生成模式
    EVAL_MODE = "DP"  # Options: BASELINE, STANDARD, CLIPPING_ONLY, DP, ORIGINAL_TEXT_BASELINE
    EPSILON = 50.0
    CLIPPING_NORM = 1.0
    PREFIX_LENGTH = 5
    LIMIT: Optional[int] = None

    # LoRA 配置（用于找回与合并）
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

def load_local_model(model_path: str, device: str):
    print(f"Loading local model from: {model_path} onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    print("Local model loaded successfully.")
    return model, tokenizer

@torch.no_grad()
def generate_answer_for_entry(
    data: Dict,
    eval_model: Optional[TrainableEnhancer],
    shared_llm: AutoModelForCausalLM,
    shared_tokenizer: AutoTokenizer,
    config: Config
) -> str:
    question_asked = data.get('question_asked', '')

    if config.EVAL_MODE == "ORIGINAL_TEXT_BASELINE":
        text_for_inference = data.get('response', '')
    else:
        text_for_inference = data.get('anonymized_response', '')

    generate_kwargs = {
        "max_new_tokens": 30,
        "temperature": 0.1,
        "do_sample": True,
        "repetition_penalty": 1.2
    }

    # 无前缀增强（BASELINE / ORIGINAL_TEXT_BASELINE）
    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"] or eval_model is None:
        user_prompt = REDDIT_PROMPT_USER.format(question_asked=question_asked, user_response=text_for_inference)
        conversation = [
            {"role": "system", "content": REDDIT_PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ]
        prompt = shared_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        llm_inputs = shared_tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)

        if shared_tokenizer.pad_token_id is None:
            shared_tokenizer.pad_token_id = shared_tokenizer.eos_token_id

        outputs = shared_llm.generate(
            input_ids=llm_inputs['input_ids'],
            attention_mask=llm_inputs['attention_mask'],
            pad_token_id=shared_tokenizer.pad_token_id,
            **generate_kwargs
        )
        generated_ids = outputs[:, llm_inputs['input_ids'].shape[1]:]
    else:
        # 前缀增强路径（STANDARD / CLIPPING_ONLY / DP）
        loss_desc_sentence = data.get('loss_description_sentence', '')
        anonymized_text = data.get('anonymized_response', '')

        if not hasattr(eval_model, 'uem') or not hasattr(eval_model, 'uem_tokenizer') or not hasattr(eval_model, 'projection_layer'):
            raise ValueError("eval_model缺失必要属性(uem/uem_tokenizer/projection_layer)。")

        with autocast(enabled=eval_model.uem.device.type == 'cuda'):
            iem_inputs = eval_model.uem_tokenizer(
                loss_desc_sentence, return_tensors="pt",
                padding=True, truncation=True, max_length=128
            ).to(config.IEM_DEVICE)
            iem_outputs = eval_model.uem(**iem_inputs)
            sentence_representation = iem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = eval_model.projection_layer(sentence_representation)

        clean_prefix = clean_prefix.view(-1, config.PREFIX_LENGTH, shared_llm.config.hidden_size)

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
        embedding_layer = shared_llm.get_input_embeddings()

        student_user_prompt = REDDIT_PROMPT_USER.format(question_asked=question_asked, user_response=anonymized_text)

        system_part = shared_tokenizer.apply_chat_template(
            [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}],
            tokenize=False, add_generation_prompt=False
        )
        user_start_part = shared_tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=False
        ).replace(shared_tokenizer.eos_token, '')
        user_content_part = student_user_prompt

        dummy_conversation = [{"role": "user", "content": "DUMMY"}]
        full_template_with_dummy = shared_tokenizer.apply_chat_template(
            dummy_conversation, tokenize=False, add_generation_prompt=True
        )
        dummy_template = shared_tokenizer.apply_chat_template(
            dummy_conversation, tokenize=False, add_generation_prompt=False
        )
        assistant_start_part = full_template_with_dummy.replace(dummy_template, "")

        system_embeds = embedding_layer(shared_tokenizer(system_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
        user_start_embeds = embedding_layer(shared_tokenizer(user_start_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
        user_content_embeds = embedding_layer(shared_tokenizer(user_content_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
        assistant_start_embeds = embedding_layer(shared_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))

        combined_embeds = torch.cat([
            system_embeds,
            user_start_embeds,
            prefix_embeds,
            user_content_embeds,
            assistant_start_embeds
        ], dim=1)

        attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device)

        if shared_tokenizer.pad_token_id is None:
            shared_tokenizer.pad_token_id = shared_tokenizer.eos_token_id

        if not hasattr(eval_model, 'llm_student'):
            raise AttributeError("eval_model.llm_student 不存在，无法进行增强模式生成。")

        outputs = eval_model.llm_student.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            pad_token_id=shared_tokenizer.pad_token_id,
            **generate_kwargs
        )
        generated_ids = outputs

    decoded_text = shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # 后处理：尽量提取职业短语（与原评测一致）
    lines = decoded_text.split('\n')
    potential_occupation = lines[-1].strip().rstrip('.,!?"\'')
    common_prefixes = [
        "Based on the text, the user is likely a ",
        "The user is likely a ",
        "I believe this person is a ",
        "Occupation: ",
        "Profession: "
    ]
    for prefix in common_prefixes:
        if potential_occupation.lower().startswith(prefix.lower()):
            potential_occupation = potential_occupation[len(prefix):].strip()
            break
    return potential_occupation

def main(config: Config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("--- Loading base LLM ---")
    shared_llm, shared_tokenizer = load_local_model(config.LLM_PATH, device=config.LLM_DEVICE)

    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
        shared_tokenizer.pad_token_id = shared_tokenizer.eos_token_id

    eval_model: Optional[TrainableEnhancer] = None
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        print("--- Loading IEM and LoRA components ---")
        lora_adapter_path = os.path.join(config.CKPT_PATH, "lora_adapters")
        other_weights_path = os.path.join(config.CKPT_PATH, "uem_projection.pt")

        if not os.path.isdir(lora_adapter_path) or not os.path.isfile(other_weights_path):
            print(f"Error: components missing in '{config.CKPT_PATH}' (need 'lora_adapters' & 'uem_projection.pt').")
            return

        print(f"Loading LoRA adapters from: {lora_adapter_path}")
        llm_student = PeftModel.from_pretrained(shared_llm, lora_adapter_path)
        llm_student = llm_student.merge_and_unload()
        print("LoRA adapters loaded and merged.")

        try:
            iem_tokenizer = AutoTokenizer.from_pretrained(config.IEM_PATH, use_fast=False)
        except Exception as e:
            print(f"Error loading IEM tokenizer: {e}")
            return

        eval_model = TrainableEnhancer(config.IEM_PATH, iem_tokenizer, None, llm_student, shared_tokenizer)
        eval_model.uem.to(config.IEM_DEVICE)
        eval_model.projection_layer.to(config.IEM_DEVICE)

        print(f"Loading IEM & projection weights: {other_weights_path}")
        checkpoint = torch.load(other_weights_path, map_location="cpu")
        eval_model.uem.load_state_dict(checkpoint['uem_state_dict'])
        eval_model.projection_layer.load_state_dict(checkpoint['projection_layer_state_dict'])
        eval_model.eval()
        print("IEM & projection loaded.")
    else:
        print(f"Running in {config.EVAL_MODE} mode (no IEM/LoRA).")

    # 读取数据
    try:
        with open(config.INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading input data: {e}")
        return

    records_to_process = lines[:config.LIMIT] if config.LIMIT is not None else lines
    print(f"--- Generating predictions for {len(records_to_process)} records ---")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_filename = f"predictions_{config.EVAL_MODE}_eps{config.EPSILON}_{timestamp}.jsonl"
    pred_path = os.path.join(config.OUTPUT_DIR, pred_filename)

    total = 0
    with open(pred_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(tqdm(records_to_process, desc="Generate")):
            try:
                data = json.loads(line)
                original_response = data.get("response")
                anonymized_response = data.get("anonymized_response")
                true_label = (data.get("personality") or {}).get("occupation")
                question_asked = data.get("question_asked")

                # 字段校验
                missing = []
                if not question_asked: missing.append("question_asked")
                if config.EVAL_MODE == "ORIGINAL_TEXT_BASELINE":
                    if not original_response: missing.append("response")
                else:
                    if not anonymized_response: missing.append("anonymized_response")
                if not true_label: missing.append("personality.occupation")

                if missing:
                    tqdm.write(f"Skip record {i+1}: missing {', '.join(missing)}")
                    continue

                # 生成
                pred = generate_answer_for_entry(
                    data,
                    eval_model,
                    shared_llm,
                    shared_tokenizer,
                    config
                )

                out_rec = {
                    "record_index": i + 1,
                    "question_asked": question_asked,
                    "original_response": original_response,
                    "anonymized_response": anonymized_response,
                    "loss_description_sentence": data.get("loss_description_sentence", ""),
                    "true_label": true_label,
                    "generated_answer": pred,
                    "eval_mode": config.EVAL_MODE,
                    "epsilon": config.EPSILON,
                    "clipping_norm": config.CLIPPING_NORM,
                    "prefix_length": config.PREFIX_LENGTH
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                total += 1

            except Exception as e:
                tqdm.write(f"Skip record {i+1} due to error: {e}")
                continue

    print(f"\nDone. Saved {total} predictions to: {pred_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model outputs under different modes (decoupled from evaluation).")
    parser.add_argument('--mode', type=str, help="Evaluation/inference mode.")
    parser.add_argument('--limit', type=int, help="Limit number of records.")
    parser.add_argument('--input', type=str, help="Path to input JSONL.")
    parser.add_argument('--ckpt', type=str, help="Checkpoint dir containing model components.")
    parser.add_argument('--outdir', type=str, help="Directory to save predictions.")
    args = parser.parse_args()

    cfg = Config()
    if args.mode: cfg.EVAL_MODE = args.mode
    if args.limit is not None: cfg.LIMIT = args.limit
    if args.input: cfg.INPUT_DATA_FILE = args.input
    if args.ckpt: cfg.CKPT_PATH = args.ckpt
    if args.outdir: cfg.OUTPUT_DIR = args.outdir

    # 基本路径检查
    ok = True
    for name, path in {
        "LLM_PATH": cfg.LLM_PATH,
        "INPUT_DATA_FILE": cfg.INPUT_DATA_FILE
    }.items():
        if not os.path.exists(path):
            print(f"Error: {name} not found at '{path}'")
            ok = False

    if cfg.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        for name, path in {
            "IEM_PATH": cfg.IEM_PATH,
            "CKPT_PATH": cfg.CKPT_PATH
        }.items():
            if not os.path.exists(path):
                print(f"Error: {name} not found at '{path}'")
                ok = False

    if ok:
        main(cfg)
    else:
        print("Abort due to missing paths.")