import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

from torch.cuda.amp import autocast
from train import TrainableEnhancer, REDDIT_PROMPT_SYSTEM, REDDIT_PROMPT_USER

# --- 1. 配置类 ---
class Config:
    # --- 设备配置 ---
    IEM_DEVICE = "cuda:0"
    LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0"

    # --- DeepSeek API 配置 ---
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"

    # --- 模型路径 ---
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    IEM_PATH = "/root/autodl-tmp/huggingface/hub/models--FacebookAI--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    CKPT_PATH = "/root/autodl-tmp/PAUE-II/ckpts/multi_tokens/best_model.pth"
    
    # --- 数据文件 ---
    INPUT_DATA_FILE = "/root/autodl-tmp/PAUE-II/benchmark/reprocess/test/test_anony_with_loss.jsonl" 
    OUTPUT_DIR = "/root/autodl-tmp/PAUE-II/results"
    
    # --- 实验设置 ---
    EVAL_MODE = "CLIPPING_ONLY" # Options: BASELINE, STANDARD, CLIPPING_ONLY, DP, ORIGINAL_TEXT_BASELINE
    EPSILON = 100.0
    CLIPPING_NORM = 1.0
    PREFIX_LENGTH = 5
    LIMIT = None

# --- 2. 提示词模板和辅助函数 ---
PROMPT_JUDGE = """
You are a highly precise evaluation expert. Your task is to determine if the "Generated Answer" correctly identifies the "Ground Truth Profession".
Follow these steps:
1. Identify the core profession mentioned in the "Generated Answer".
2. Compare it to the "Ground Truth Profession". It is "Correct" if it is an exact match or a clear synonym (e.g., "Soccer Player" is a synonym for "Footballer").
3. Output your final verdict as a single word: "Correct" or "Incorrect".

--- EXAMPLES ---
Ground Truth Profession: "Architect"
Generated Answer: "The person is likely an Engineer."
Your verdict: Incorrect

Ground Truth Profession: "Soccer Player"
Generated Answer: "Based on the text, I believe this person is a Footballer."
Your verdict: Correct

Ground Truth Profession: "Chef"
Generated Answer: "Comedy"
Your verdict: Incorrect
--- END EXAMPLES ---

Ground Truth Profession: "{ground_truth}"

Generated Answer: "{generated_answer}"

Your verdict:
"""

from _utils.model import load_local_model

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
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"direct_eval_{config.EVAL_MODE}_eps{config.EPSILON}_{timestamp}.json"
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
    data: Dict, eval_model: TrainableEnhancer, shared_llm: AutoModelForCausalLM, 
    shared_tokenizer: AutoTokenizer, config: Config
) -> str:
    question_asked = data.get('question_asked', '')

    # 根据EVAL_MODE选择使用哪个文本
    if config.EVAL_MODE == "ORIGINAL_TEXT_BASELINE":
        text_for_inference = data.get('response', '') # 使用原始文本
    else:
        text_for_inference = data.get('anonymized_response', '') # 其他模式使用匿名化文本

    generate_kwargs = {"max_new_tokens": 30, "temperature": 0.1, "do_sample": True, "repetition_penalty": 1.2}

    # 将ORIGINAL_TEXT_BASELINE和BASELINE都归为无UEM的推理
    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        user_prompt = REDDIT_PROMPT_USER.format(question_asked=question_asked, user_response=text_for_inference)
        conversation = [
            {"role": "system", "content": REDDIT_PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ]
        prompt = shared_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        llm_inputs = shared_tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)
        outputs = shared_llm.generate(**llm_inputs, **generate_kwargs)
        generated_ids = outputs[:, llm_inputs['input_ids'].shape[1]:]
    else:
        # --- UEM增强模式: 使用前缀注入 ---
        loss_desc_sentence = data.get('loss_description_sentence', '')
        anonymized_text = data.get('anonymized_response', '')

        # 1. UEM现在只编码描述句 (在 UEM_DEVICE 上)
        with autocast(enabled=eval_model.uem.device.type == 'cuda'):
            uem_inputs = eval_model.uem_tokenizer(loss_desc_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(eval_model.uem.device)
            uem_outputs = eval_model.uem(**uem_inputs)
            sentence_representation = uem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = eval_model.projection_layer(sentence_representation)
            
        # <--- 修改点 2.b (1/2): 添加 reshape 操作，与 train.py 保持一致
        clean_prefix = clean_prefix.view(-1, config.PREFIX_LENGTH, shared_llm.config.hidden_size)
        
        # 2. 根据评估模式处理前缀 (裁剪或加噪)
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
            else: # Fallback
                prefix_vector = clean_prefix
        
        # 3. 将前缀嵌入移动到 LLM 的设备上
        # <--- 修改点 2.b (2/2): 移除 .unsqueeze(1) 操作
        prefix_embeds = prefix_vector.to(device=config.LLM_DEVICE, dtype=shared_llm.dtype)

        # 4. 手动构建聊天模板的嵌入
        embedding_layer = shared_llm.get_input_embeddings()
        
        # 修正4: 使用与训练时学生模型一致的提示词结构
        student_user_prompt = REDDIT_PROMPT_USER.format(question_asked=question_asked, user_response=anonymized_text)

        system_part = shared_tokenizer.apply_chat_template([{"role": "system", "content": REDDIT_PROMPT_SYSTEM}], tokenize=False, add_generation_prompt=False)
        user_start_part = shared_tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False).replace(shared_tokenizer.eos_token, '')
        user_content_part = student_user_prompt
        
        dummy_conversation = [{"role": "user", "content": "DUMMY"}]
        full_template_with_dummy = shared_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, add_generation_prompt=True)
        dummy_template = shared_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, add_generation_prompt=False)
        assistant_start_part = full_template_with_dummy.replace(dummy_template, "")

        system_embeds = embedding_layer(shared_tokenizer(system_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
        user_start_embeds = embedding_layer(shared_tokenizer(user_start_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
        user_content_embeds = embedding_layer(shared_tokenizer(user_content_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))
        assistant_start_embeds = embedding_layer(shared_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(config.LLM_DEVICE))

        # 5. 按正确顺序连接所有部分
        combined_embeds = torch.cat([
            system_embeds,
            user_start_embeds,
            prefix_embeds,
            user_content_embeds,
            assistant_start_embeds
        ], dim=1)

        # 新增：为拼接好的嵌入创建 attention_mask
        attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device)

        # 6. 使用拼接好的嵌入和 attention_mask 进行生成
        outputs = shared_llm.generate(
            inputs_embeds=combined_embeds, 
            attention_mask=attention_mask,  # 传入 attention_mask
            **generate_kwargs
        )
        generated_ids = outputs

    return shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

# --- 4. 主流程 ---
def main(config: Config):
    if not config.DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set. It is required for the Judge model.")
        return

    api_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE)

    print("--- Loading Models for Direct Evaluation ---")
    shared_llm, shared_tokenizer = load_local_model(config.LLM_PATH, device=config.LLM_DEVICE)
    
    # UEM模型只在非BASELINE模式下需要
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        uem_tokenizer = AutoTokenizer.from_pretrained(config.IEM_PATH)
        print("--- Loading UEM Model ---")
        eval_model = TrainableEnhancer(config.IEM_PATH, uem_tokenizer, shared_llm, shared_tokenizer)
        eval_model.uem.to(config.IEM_DEVICE)
        eval_model.projection_layer.to(config.IEM_DEVICE)
        
        checkpoint = torch.load(config.CKPT_PATH, map_location="cpu")
        eval_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Successfully loaded trained UEM weights from {config.CKPT_PATH}")
        eval_model.eval()
    else:
        eval_model = None
        print(f"Running in {config.EVAL_MODE} mode. UEM model is not loaded.")

    with open(config.INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    records_to_process = lines[:config.LIMIT] if config.LIMIT else lines
    
    correct_predictions = 0
    total_predictions = 0
    qualitative_samples = []

    print(f"\n--- Starting Direct Evaluation for {len(records_to_process)} records ---")
    for line in tqdm(records_to_process, desc="Direct Evaluation"):
        try:
            data = json.loads(line)
            response = data.get("response")
            anonymized_response = data.get("anonymized_response")
            loss_desc = data.get("loss_description_sentence")
            
            # --- 修正: 从 personality.occupation 获取真实标签 ---
            personality = data.get("personality", {})
            true_label = personality.get("occupation")
            
            question_asked = data.get("question_asked")

            if not all([response, anonymized_response, true_label, question_asked]):
                tqdm.write(f"Skipping malformed record (missing required fields): {line.strip()}")
                continue
            
            data['loss_description_sentence'] = loss_desc if loss_desc is not None else ""

            # 推理
            predicted_text = evaluate_single_entry(data, eval_model, shared_llm, shared_tokenizer, config)
            
            # LLM 评委判断
            is_correct = judge_prediction(predicted_text, true_label, api_client, config)

            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            judgement_str = "Correct" if is_correct else "Incorrect"
            current_accuracy = correct_predictions / total_predictions
            tqdm.write(f"\n[Record {total_predictions}/{len(records_to_process)}] Label: {true_label}\n"
                       f"  - Predicted: '{predicted_text}'\n"
                       f"  - Judgement: {judgement_str}\n"
                       f"  - Running Accuracy: {current_accuracy:.2%} ({correct_predictions}/{total_predictions})")


            qualitative_samples.append({
                "Original Response": response,
                "Anonymized Response": anonymized_response,
                "Loss Description": loss_desc,
                "True Profession": true_label, 
                "Generated Answer": predicted_text,
                "Judgement": judgement_str,
            })

        except Exception as e:
            tqdm.write(f"Skipping record due to unexpected error: {e}\nLine: {line.strip()}")
            continue

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("\n--- Direct Evaluation Complete ---")
    print(f"Mode: {config.EVAL_MODE}")
    if config.EVAL_MODE == "DP":
        print(f"Epsilon: {config.EPSILON}")
    print(f"Processed Records: {total_predictions}")
    print(f"Final Accuracy: {accuracy:.2%}")

    metrics_to_save = { "accuracy": accuracy, "total_processed": total_predictions, "correct_predictions": correct_predictions }
    save_results(config, metrics_to_save, qualitative_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Direct PAUE Evaluation on pre-anonymized data (Prefix version).")
    parser.add_argument('--mode', type=str, help=f"Evaluation mode.")
    parser.add_argument('--limit', type=int, help="Limit the number of records to process.")
    parser.add_argument('--input', type=str, help="Path to the pre-anonymized input data file.")
    parser.add_argument('--ckpt', type=str, help="Path to the IEM checkpoint file.")
    args = parser.parse_args()

    config = Config()
    if args.mode: config.EVAL_MODE = args.mode
    if args.limit: config.LIMIT = args.limit
    if args.input: config.INPUT_DATA_FILE = args.input
    if args.ckpt: config.CKPT_PATH = args.ckpt

    paths_to_check = {
        "Shared LLM": config.LLM_PATH,
        "Input Data": config.INPUT_DATA_FILE,
    }
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        paths_to_check["UEM"] = config.IEM_PATH
        paths_to_check["UEM Checkpoint"] = config.CKPT_PATH

    all_paths_exist = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"Error: {name} path not found at '{path}'")
            all_paths_exist = False

    if all_paths_exist:
        main(config)
    else:
        print("\nPlease update the paths in the Config class or provide them via command line arguments.")