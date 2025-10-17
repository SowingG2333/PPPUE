import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, List
from datetime import datetime
from openai import OpenAI
from torch.cuda.amp import autocast

# 从训练脚本中导入模型定义和提示词
from train_wandb import BioTrainableEnhancer, BIO_PROMPT_SYSTEM, BIO_PROMPT_USER

# --- 1. 配置类 ---
class Config:
    # --- 设备配置 ---
    UEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else UEM_DEVICE

    # --- DeepSeek API 配置 ---
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"

    # --- 模型路径 (与 train_wandb.py 保持一致) ---
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    UEM_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    CKPT_PATH = "/root/autodl-tmp/PPPUE/DB-Bio/ckpt/new_strategy/best_model_epoch_8" # 示例：指向一个具体的最佳模型目录
    
    # --- 数据文件 ---
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/DB-Bio/benchmark/reprocess/new_strategy/test/anony_test_with_loss_relabel.jsonl" 
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/DB-Bio/results"
    
    # --- 实验设置 ---
    EVAL_MODE = "DP" # Options: BASELINE, STANDARD, CLIPPING_ONLY, DP, ORIGINAL_TEXT_BASELINE
    EPSILON = 50.0
    CLIPPING_NORM = 1.0
    PREFIX_LENGTH = 5
    LIMIT = None # 限制处理的记录数量，None 表示处理全部

# --- 2. 提示词模板和辅助函数 ---
PROMPT_JUDGE = """
You are a highly precise evaluation expert with a deep understanding of professional roles and their nuances.

Your task is to strictly determine if the "Generated Answer" correctly identifies the "Ground Truth Profession".

A "Generated Answer" is considered **"Correct"** only if it meets one of the following criteria:
1.  It is an **exact match** with the "Ground Truth Profession".
2.  It is a **clear and direct synonym** (e.g., Footballer for Soccer Player).
3.  It is **more specific or more precise** than the ground truth, while still being factually accurate based on a person's career (e.g., "Academy Coach" is a more precise current role for a "Soccer Player" who has transitioned).

A "Generated Answer" is **"Incorrect"** if:
1.  It is a completely different profession.
2.  It is a **more general or less precise** role (e.g., "Cook" for "Chef").

Output your final verdict as a single word: "Correct" or "Incorrect".

--- EXAMPLES ---

**Example 1: Clear Synonym (Correct)**
Ground Truth Profession: "Soccer Player"
Generated Answer: "Based on the text, I believe this person is a Footballer."
Your verdict: Correct

**Example 2: More Precise Role (Correct)**
Ground Truth Profession: "Soccer Player"
Generated Answer: "His current job is a football coach."
Your verdict: Correct

**Example 3: Less Precise Role (Incorrect)**
Ground Truth Profession: "Chef"
Generated Answer: "Cook"
Your verdict: Incorrect

**Example 4: Related but Different Role (Incorrect)**
Ground Truth Profession: "Data Scientist"
Generated Answer: "Data Analyst."
Your verdict: Incorrect

**Example 5: Clearly Different Profession (Incorrect)**
Ground Truth Profession: "Architect"
Generated Answer: "The person is likely an Engineer."
Your verdict: Incorrect
--- END EXAMPLES ---

Ground Truth Profession: "{ground_truth}"

Generated Answer: "{generated_answer}"

Your verdict:
"""

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

def save_results(config: Config, metrics: Dict, samples: List[Dict]):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{config.EVAL_MODE}_eps{config.EPSILON}_{timestamp}.json"
    filepath = os.path.join(config.OUTPUT_DIR, filename)
    
    output_data = {
        "config": {k: v for k, v in vars(config).items() if not k.startswith('__')},
        "metrics": metrics,
        "qualitative_samples": samples
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)
    print(f"\nEvaluation results saved to: {filepath}")

def judge_prediction(generated_answer: str, ground_truth_label: str, judge_client: OpenAI, config: Config) -> bool:
    judge_prompt = PROMPT_JUDGE.format(ground_truth=ground_truth_label, generated_answer=generated_answer)
    verdict = generate_api_response(judge_client, config.JUDGE_MODEL_NAME, 
                                  [{"role": "system", "content": "You are a highly precise evaluation expert."}, 
                                   {"role": "user", "content": judge_prompt}], 0.0)
    cleaned_verdict = verdict.lower().strip().strip('.,!"\'')
    return cleaned_verdict == "correct"

# --- 3. 评估模块 ---
@torch.no_grad()
def evaluate_single_entry(
    data: Dict, eval_model: BioTrainableEnhancer, shared_llm: AutoModelForCausalLM, 
    shared_tokenizer: AutoTokenizer, config: Config
) -> str:
    
    original_bio = data.get('text', '')
    anonymized_bio = data.get('anonymized_text', '')

    generate_kwargs = {"max_new_tokens": 10, "temperature": 0.0, "do_sample": False}

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        # 根据模式选择提示词内容
        user_prompt = BIO_PROMPT_USER.format(
            biography_text=anonymized_bio if config.EVAL_MODE == "BASELINE" else original_bio
        )
        conversation = [
            {"role": "system", "content": BIO_PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ]
        prompt = shared_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        llm_inputs = shared_tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)
        outputs = shared_llm.generate(**llm_inputs, **generate_kwargs)
        generated_ids = outputs[:, llm_inputs['input_ids'].shape[1]:]
    else:
        # --- UEM增强模式: 使用前缀注入 ---
        loss_desc_sentence = data.get('loss_description_sentence', '')

        # 1. UEM编码描述句
        with autocast(enabled=eval_model.uem.device.type == 'cuda'):
            uem_inputs = eval_model.uem_tokenizer(loss_desc_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(eval_model.uem.device)
            uem_outputs = eval_model.uem(**uem_inputs)
            sentence_representation = uem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = eval_model.projection_layer(sentence_representation)
            
        clean_prefix = clean_prefix.view(-1, config.PREFIX_LENGTH, shared_llm.config.hidden_size)
        
        # 2. 根据评估模式处理前缀
        if config.EVAL_MODE == "STANDARD":
            prefix_vector = clean_prefix
        else: # CLIPPING_ONLY or DP
            original_norm = torch.linalg.norm(clean_prefix, dim=-1, keepdim=True)
            scale_factor = (config.CLIPPING_NORM / (original_norm + 1e-9)).clamp(max=1.0)
            clipped_prefix = clean_prefix * scale_factor
            if config.EVAL_MODE == "CLIPPING_ONLY":
                prefix_vector = clipped_prefix
            elif config.EVAL_MODE == "DP":
                sigma = (2 * config.CLIPPING_NORM) / (config.EPSILON + 1e-9)
                noise = torch.randn_like(clipped_prefix) * sigma
                prefix_vector = clipped_prefix + noise
            else: # Fallback
                prefix_vector = clean_prefix
        
        prefix_embeds = prefix_vector.to(device=config.LLM_DEVICE, dtype=shared_llm.dtype)

        # 3. 手动构建聊天模板的嵌入 (与训练逻辑完全一致)
        embedding_layer = shared_llm.get_input_embeddings()

        student_user_prompt = BIO_PROMPT_USER.format(biography_text=anonymized_bio)
        system_part = shared_tokenizer.apply_chat_template([{"role": "system", "content": BIO_PROMPT_SYSTEM}], tokenize=False, add_generation_prompt=False)
        user_start_part = shared_tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False).replace(shared_tokenizer.eos_token, '')
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

        # 4. 使用拼接好的嵌入进行生成
        outputs = eval_model.llm_student.generate(
            inputs_embeds=combined_embeds, 
            attention_mask=attention_mask,
            **generate_kwargs
        )
        generated_ids = outputs

    return shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

# --- 4. 主流程 ---
def main(config: Config):
    if not config.DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        return

    api_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE)

    print("--- Loading Base LLM and Tokenizer ---")
    # 基础LLM使用float32以匹配训练时的类型
    shared_llm = AutoModelForCausalLM.from_pretrained(config.LLM_PATH, torch_dtype=torch.float32, device_map={"": config.LLM_DEVICE})
    shared_tokenizer = AutoTokenizer.from_pretrained(config.LLM_PATH)
    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
    
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        uem_tokenizer = AutoTokenizer.from_pretrained(config.UEM_PATH, use_fast=False)
        print("--- Loading UEM and LoRA components ---")
        
        lora_adapter_path = os.path.join(config.CKPT_PATH, "lora_adapters")
        other_weights_path = os.path.join(config.CKPT_PATH, "uem_projection.pt")

        if not os.path.isdir(lora_adapter_path) or not os.path.isfile(other_weights_path):
            print(f"Error: Model components not found in '{config.CKPT_PATH}'")
            return

        print(f"Loading LoRA adapters from: {lora_adapter_path}")
        llm_student = PeftModel.from_pretrained(shared_llm, lora_adapter_path)
        llm_student = llm_student.merge_and_unload()
        print("LoRA adapters loaded and merged.")
        
        # 初始化 BioTrainableEnhancer
        eval_model = BioTrainableEnhancer(config.UEM_PATH, uem_tokenizer, None, llm_student, shared_tokenizer)
        eval_model.uem.to(config.UEM_DEVICE)
        eval_model.projection_layer.to(config.UEM_DEVICE)
        
        print(f"Loading UEM and projection weights from: {other_weights_path}")
        checkpoint = torch.load(other_weights_path, map_location="cpu")
        eval_model.uem.load_state_dict(checkpoint['uem_state_dict'])
        eval_model.projection_layer.load_state_dict(checkpoint['projection_layer_state_dict'])
        
        print(f"Successfully loaded all model components from {config.CKPT_PATH}")
        eval_model.eval()
    else:
        eval_model = None
        print(f"Running in {config.EVAL_MODE} mode. UEM and LoRA models are not loaded.")

    with open(config.INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    records_to_process = lines[:config.LIMIT] if config.LIMIT else lines
    
    correct_predictions, total_predictions = 0, 0
    qualitative_samples = []

    print(f"\n--- Starting Evaluation for {len(records_to_process)} records ---")
    for line in tqdm(records_to_process, desc="Evaluation"):
        try:
            data = json.loads(line)
            # true_label = data.get("label")
            true_label = data.get("label_accurate") # 使用更准确的标签字段
            if not true_label:
                tqdm.write(f"Skipping record with missing label: {line.strip()}")
                continue
            
            predicted_text = evaluate_single_entry(data, eval_model, shared_llm, shared_tokenizer, config)
            is_correct = judge_prediction(predicted_text, true_label, api_client, config)

            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            judgement_str = "Correct" if is_correct else "Incorrect"
            tqdm.write(f"\n[Record {total_predictions}] Label: {true_label} | Predicted: '{predicted_text}' | Judgement: {judgement_str}")

            qualitative_samples.append({
                "Original Text": data.get("text"),
                "Anonymized Text": data.get("anonymized_text"),
                "Loss Description": data.get("loss_description_sentence"),
                "True Profession": true_label, 
                "Generated Answer": predicted_text,
                "Judgement": judgement_str,
            })

        except Exception as e:
            tqdm.write(f"Skipping record due to unexpected error: {e}\nLine: {line.strip()}")
            continue

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("\n--- Evaluation Complete ---")
    print(f"Mode: {config.EVAL_MODE}")
    if config.EVAL_MODE == "DP": print(f"Epsilon: {config.EPSILON}")
    print(f"Final Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

    metrics_to_save = { "accuracy": accuracy, "total_processed": total_predictions, "correct_predictions": correct_predictions }
    save_results(config, metrics_to_save, qualitative_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PAUE Evaluation for DB-Bio.")
    parser.add_argument('--mode', type=str, help="Evaluation mode (e.g., BASELINE, STANDARD, DP).")
    parser.add_argument('--limit', type=int, help="Limit the number of records to process.")
    parser.add_argument('--input', type=str, help="Path to the input data file.")
    parser.add_argument('--ckpt', type=str, help="Path to the checkpoint directory containing model components.")
    parser.add_argument('--eps', type=float, help="Epsilon value for DP mode.")
    args = parser.parse_args()

    config = Config()
    if args.mode: config.EVAL_MODE = args.mode
    if args.limit: config.LIMIT = args.limit
    if args.input: config.INPUT_DATA_FILE = args.input
    if args.ckpt: config.CKPT_PATH = args.ckpt
    if args.eps: config.EPSILON = args.eps

    main(config)