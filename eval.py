import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime
from openai import OpenAI
from torch import amp
from transformers import AutoTokenizer, AutoModelForCausalLM, XLMRobertaTokenizer
from peft import PeftModel

from train import TrainableEnhancer, REDDIT_PROMPT_SYSTEM, REDDIT_PROMPT_USER, CrossAttentionLayer, GetDataset
from _utils.model import load_local_model

# --- 1. 配置类 ---
class Config:
    IEM_DEVICE = "cuda:0"
    LLM_DEVICE = "cuda:0"
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    UEM_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt/LoRA_CrossAttention_II"
    STUDENT_ADAPTER_PATH = os.path.join(CKPT_DIR, "student_lora_adapter")
    UEM_CKPT_PATH = os.path.join(CKPT_DIR, "uem.pth")
    PROJ_CKPT_PATH = os.path.join(CKPT_DIR, "projection_layer.pth")
    XATTN_CKPT_PATH = os.path.join(CKPT_DIR, "cross_attention.pth")
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/task_loss_description/test/test_anony_with_loss.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/results"
    
    # 在此处设置您想运行的模式
    EVAL_MODE = "CLIPPING_ONLY" # 可选: "STANDARD", "CLIPPING_ONLY", "DP", "BASELINE", "ORIGINAL_TEXT_BASELINE"
    CLIPPING_NORM = 1.0
    EPSILON = 10.0 # 差分隐私的 Epsilon 值
    LIMIT = None

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
--- END EXAMPLES ---
Ground Truth Profession: "{ground_truth}"
Generated Answer: "{generated_answer}"
Your verdict:
"""
def generate_api_response(client, model, messages, temperature):
    try:
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        content = completion.choices[0].message.content
        return content.strip() if content else "API_EMPTY"
    except Exception as e:
        return "API_ERROR"

def save_results(config, metrics, samples):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(config.OUTPUT_DIR, f"eval_{config.EVAL_MODE}_eps{config.EPSILON}_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({"config": {k: v for k, v in vars(config).items() if not k.startswith('__')}, "metrics": metrics, "samples": samples}, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def judge_prediction(answer, label, client, config):
    prompt = PROMPT_JUDGE.format(ground_truth=label, generated_answer=answer)
    verdict = generate_api_response(client, config.JUDGE_MODEL_NAME, [{"role": "system", "content": "You are a precise evaluator."}, {"role": "user", "content": prompt}], 0.0)
    return "correct" in verdict.lower()


@torch.no_grad()
def evaluate_single_entry(data: Dict, eval_model: TrainableEnhancer, config: Config) -> str:
    llm = eval_model.llm_student
    tokenizer = eval_model.llm_tokenizer
    generate_kwargs = {"max_new_tokens": 30, "temperature": 0.0, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        text = data['response'] if config.EVAL_MODE == "ORIGINAL_TEXT_BASELINE" else data['anonymized_response']
        user_prompt = REDDIT_PROMPT_USER.format(question_asked=data['question_asked'], user_response=text)
        conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)
        outputs = llm.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # --- 精准注入流程 ---
    with amp.autocast("cuda"):
        uem_inputs = eval_model.uem_tokenizer([data['loss_description_sentence']], return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.IEM_DEVICE)
        uem_outputs = eval_model.uem(**uem_inputs)
        sum_embeds = (uem_outputs.last_hidden_state * uem_inputs['attention_mask'].unsqueeze(-1)).sum(1)
        uem_repr = sum_embeds / uem_inputs['attention_mask'].sum(1).unsqueeze(-1).clamp(min=1e-9)
        prefix_context = eval_model.projection_layer(uem_repr).unsqueeze(1)
    
    prefix_context = prefix_context.to(config.LLM_DEVICE, dtype=torch.bfloat16)

    # --- 新增/修正：应用裁剪和差分隐私噪声 ---
    if config.EVAL_MODE != "STANDARD":
        # 1. 裁剪 (Clipping)
        norm = torch.linalg.norm(prefix_context, dim=-1, keepdim=True)
        scale = (config.CLIPPING_NORM / (norm + 1e-9)).clamp(max=1.0)
        prefix_context = prefix_context * scale

        # 2. 加噪 (DP Noise Addition)
        if config.EVAL_MODE == "DP":
            # 根据(epsilon, 0)-DP计算高斯噪声的标准差
            # 注意：这是一个简化的实现，实际应用中可能需要更严格的隐私核算
            sigma = (2 * config.CLIPPING_NORM) / (config.EPSILON + 1e-9)
            noise = torch.randn_like(prefix_context) * sigma
            prefix_context = prefix_context + noise
    # --- 修正结束 ---

    embedding_layer = llm.get_input_embeddings()
    system_conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}]
    user_conv = [{"role": "user", "content": REDDIT_PROMPT_USER.format(question_asked=data['question_asked'], user_response=data['anonymized_response'])}]
    
    system_ids = tokenizer(tokenizer.apply_chat_template(system_conv, tokenize=False), return_tensors="pt").input_ids.to(config.LLM_DEVICE)
    user_ids = tokenizer(tokenizer.apply_chat_template(user_conv, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, ""), return_tensors="pt").input_ids.to(config.LLM_DEVICE)
    assistant_ids = tokenizer(tokenizer.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, ""), return_tensors="pt").input_ids.to(config.LLM_DEVICE)
    
    system_embeds, user_embeds, assistant_embeds = map(embedding_layer, (system_ids, user_ids, assistant_ids))
    
    enhanced_user_embeds = eval_model.cross_attention(user_embeds, prefix_context)
    combined_embeds = torch.cat([system_embeds, enhanced_user_embeds, assistant_embeds], dim=1)
    attention_mask = torch.ones(combined_embeds.shape[:2], device=config.LLM_DEVICE)

    outputs = llm.generate(inputs_embeds=combined_embeds, attention_mask=attention_mask, **generate_kwargs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded if decoded else "[EMPTY_OUTPUT]"

# ... (main 函数和 if __name__ == "__main__" 部分无变化) ...
def main(config: Config):
    api_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE)

    print("--- Loading Models for Evaluation ---")
    base_llm, llm_tokenizer = load_local_model(config.LLM_PATH, device=config.LLM_DEVICE)
    llm_student = PeftModel.from_pretrained(base_llm, config.STUDENT_ADAPTER_PATH)
    llm_student.eval()

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        eval_model = TrainableEnhancer(config.UEM_PATH, None, llm_student, None, llm_tokenizer)
    else:
        uem_tokenizer = AutoTokenizer.from_pretrained(config.UEM_PATH, use_fast=False)
        eval_model = TrainableEnhancer(config.UEM_PATH, uem_tokenizer, llm_student, None, llm_tokenizer)
        eval_model.uem.load_state_dict(torch.load(config.UEM_CKPT_PATH, map_location="cpu"))
        eval_model.projection_layer.load_state_dict(torch.load(config.PROJ_CKPT_PATH, map_location="cpu"))
        eval_model.cross_attention.load_state_dict(torch.load(config.XATTN_CKPT_PATH, map_location="cpu"))
        eval_model.uem.to(config.IEM_DEVICE)
        eval_model.projection_layer.to(config.IEM_DEVICE)
        eval_model.cross_attention.to(config.LLM_DEVICE)
    eval_model.eval()

    records_to_process = [json.loads(line) for line in open(config.INPUT_DATA_FILE, 'r')][:config.LIMIT or None]
    
    correct_predictions, total_predictions = 0, 0
    qualitative_samples = []

    print(f"\n--- Starting Evaluation for {len(records_to_process)} records ---")
    for data in tqdm(records_to_process, desc="Evaluating"):
        true_label = data.get("personality", {}).get("occupation")
        if not true_label: continue
        
        predicted_text = evaluate_single_entry(data, eval_model, config)
        is_correct = judge_prediction(predicted_text, true_label, api_client, config)
        
        if is_correct: correct_predictions += 1
        total_predictions += 1
        
        tqdm.write(f"[Record {total_predictions}] Label: {true_label}, Predicted: '{predicted_text}', Correct: {is_correct}")
        if len(qualitative_samples) < 20:
             qualitative_samples.append({"data": data, "prediction": predicted_text, "correct": is_correct})

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"Mode: {config.EVAL_MODE}, Final Accuracy: {accuracy:.2%}")
    save_results(config, {"accuracy": accuracy, "total": total_predictions, "correct": correct_predictions}, qualitative_samples)

if __name__ == "__main__":
    main(Config())