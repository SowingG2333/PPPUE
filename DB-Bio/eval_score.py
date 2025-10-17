import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, List, Any
from datetime import datetime
from openai import OpenAI
from torch.cuda.amp import autocast

# 从训练脚本中导入模型定义和提示词
from train import BioTrainableEnhancer, BIO_PROMPT_SYSTEM, BIO_PROMPT_USER

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
Role: You are a highly discerning evaluation expert with deep contextual understanding of professional roles and their nuances.

Task: Your goal is to evaluate the quality of a "Generated Answer" regarding a person's occupation, using the full **[Original Text]** as the ultimate source of truth.

The "Generated Answer" was produced by an AI that only saw an ANONYMIZED version of the text. Your evaluation must focus on **how well the AI's answer reflects the specific, nuanced, and current information present only in the [Original Text]**.

Please rate the "Generated Answer" on a scale of 1 to 10 based on the rubric below, and then provide a brief justification for your score. Output ONLY in the following JSON format.

**SCORING RUBRIC (10-Point Scale):**
* **1-2 (Incorrect)**: The answer is completely wrong or irrelevant.
* **3-4 (Weakly Related)**: The answer is in a related but incorrect field (e.g., "Engineer" for an "Architect").
* **5-6 (Correct Generalization)**: The answer provides a correct but overly general category. This is a plausible guess that could be made from the anonymized text alone, without recovering specific details.
* **7-8 (Good Recovery)**: The answer correctly identifies a significant *past* profession or a very close synonym, demonstrating some information recovery, but misses the most current or precise role.
* **9-10 (Excellent Recovery)**: The answer is highly specific and correctly identifies the person's **primary and most current** profession. A score of 10 indicates a perfect, precise match that could only be achieved by successfully recovering critical information lost during anonymization.

--- EXAMPLES ---

**Example 1: Perfect Recovery**
[Original Text]: "Alan Desmond Lee is a former footballer who played as a striker; he is now working as an academy coach at Ipswich Town."
[Ground Truth Profession]: "Academy Coach"
[Generated Answer]: "Academy Coach"
Your JSON Output:
{
  "score": 10,
  "justification": "The model achieved a perfect score by identifying the most current and specific role ('Academy Coach'), which required recovering precise information that was absent in the anonymized text."
}

**Example 2: Good Recovery (Past Profession)**
[Original Text]: "Alan Desmond Lee is a former footballer who played as a striker; he is now working as an academy coach at Ipswich Town."
[Ground Truth Profession]: "Academy Coach"
[Generated Answer]: "Soccer Player"
Your JSON Output:
{
  "score": 7,
  "justification": "The model correctly identified a significant past profession ('Soccer Player'), showing good information recovery, but failed to identify the more precise and current role of 'Academy Coach'."
}

**Example 3: Correct Generalization (Lacks Specificity)**
[Original Text]: "As a structural engineer, my marriage is like a well-oiled piece of intricate structural machinery..."
[Ground Truth Profession]: "structural engineer"
[Generated Answer]: "Engineer"
Your JSON Output:
{
  "score": 6,
  "justification": "The model provided a correct but general category ('Engineer'). It successfully identified the field but missed the key specific detail ('structural') which was present in the original text, indicating only partial information recovery."
}

**Example 4: Weakly Related**
[Original Text]: "As a data scientist, I build predictive models..."
[Ground Truth Profession]: "Data Scientist"
[Generated Answer]: "Data Analyst"
Your JSON Output:
{
  "score": 4,
  "justification": "The answer 'Data Analyst' is in a related field but is a distinct and incorrect role. It shows a shallow understanding without recovering the specific nature of the user's work."
}
--- END EXAMPLES ---

[Original Text]:
{original_text}

[Ground Truth Profession]:
{ground_truth_label}

[Generated Answer]:
{generated_answer}

Your JSON Output:
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

def judge_prediction(generated_answer: str, ground_truth_label: str, original_text: str, judge_client: OpenAI, config: Config) -> Dict[str, Any]:
    judge_prompt = PROMPT_JUDGE.format(
        original_text=original_text,
        ground_truth_label=ground_truth_label, 
        generated_answer=generated_answer
    )
    system_message = "You are a highly discerning evaluation expert with deep contextual understanding. You must only respond in the specified JSON format."
    
    verdict_str = generate_api_response(
        judge_client, 
        config.JUDGE_MODEL_NAME, 
        [{"role": "system", "content": system_message}, 
         {"role": "user", "content": judge_prompt}], 
        0.0
    )

    if verdict_str == "API_ERROR":
        return {"score": 1, "justification": "API call failed."}

    try:
        # Clean the string to better handle potential markdown code blocks
        if verdict_str.startswith("```json"):
            verdict_str = verdict_str[7:-3].strip()
        
        verdict_json = json.loads(verdict_str)
        score = verdict_json.get("score")
        justification = verdict_json.get("justification")

        if score is None or not isinstance(score, (int, float)):
            raise ValueError("Score is missing or not a number.")
        if justification is None:
             justification = "No justification provided."
             
        return {"score": int(score), "justification": str(justification)}

    except (json.JSONDecodeError, ValueError) as e:
        tqdm.write(f"  - ERROR parsing judge response: {e}. Raw response: '{verdict_str}'")
        return {"score": 1, "justification": f"Failed to parse JSON response. Raw: {verdict_str}"}

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
    
    total_score, excellent_recovery_count, total_predictions = 0, 0, 0
    qualitative_samples = []

    print(f"\n--- Starting Evaluation for {len(records_to_process)} records ---")
    for line in tqdm(records_to_process, desc="Evaluation"):
        try:
            data = json.loads(line)
            true_label = data.get("label_accurate")
            original_text = data.get("text", "")
            if not true_label:
                tqdm.write(f"Skipping record with missing label: {line.strip()}")
                continue
            
            predicted_text = evaluate_single_entry(data, eval_model, shared_llm, shared_tokenizer, config)
            judgement_result = judge_prediction(predicted_text, true_label, original_text, api_client, config)
            
            score = judgement_result['score']
            justification = judgement_result['justification']

            total_score += score
            if score >= 9:
                excellent_recovery_count += 1
            total_predictions += 1

            tqdm.write(f"\n[Record {total_predictions}] Label: {true_label} | Predicted: '{predicted_text}' | Score: {score}/10")
            tqdm.write(f"  └─ Justification: {justification}")

            qualitative_samples.append({
                "Original Text": original_text,
                "Anonymized Text": data.get("anonymized_text"),
                "Loss Description": data.get("loss_description_sentence"),
                "True Profession": true_label, 
                "Generated Answer": predicted_text,
                "Score": score,
                "Justification": justification,
            })

        except Exception as e:
            tqdm.write(f"Skipping record due to unexpected error: {e}\nLine: {line.strip()}")
            continue

    avg_score = total_score / total_predictions if total_predictions > 0 else 0
    excellent_rate = excellent_recovery_count / total_predictions if total_predictions > 0 else 0

    print("\n--- Evaluation Complete ---")
    print(f"Mode: {config.EVAL_MODE}")
    if config.EVAL_MODE == "DP": print(f"Epsilon: {config.EPSILON}")
    print(f"Total Records Processed: {total_predictions}")
    print(f"Average Score: {avg_score:.2f}/10")
    print(f"Excellent Recovery Rate (Score 9-10): {excellent_rate:.2%} ({excellent_recovery_count}/{total_predictions})")

    metrics_to_save = {
        "average_score": avg_score,
        "excellent_recovery_rate": excellent_rate,
        "excellent_recovery_count": excellent_recovery_count,
        "total_score": total_score,
        "total_processed": total_predictions,
    }
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