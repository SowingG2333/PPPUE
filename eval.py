import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime
from openai import OpenAI
from torch import amp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Import the new, refactored modules from train.py ---
from train import (
    PAUE_LlamaForCausalLM, 
    TrainableEnhancer, 
    REDDIT_PROMPT_SYSTEM, 
    REDDIT_PROMPT_USER, 
    GetDataset
)

# --- 1. 配置类 (Updated Checkpoint Paths) ---
class Config:
    UEM_DEVICE = "cuda:0"
    LLM_DEVICE = "cuda:0"
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"
    
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    UEM_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    
    # --- Corrected paths for the new architecture ---
    CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt/LoRA_CrossAttention_III"
    STUDENT_ADAPTER_PATH = os.path.join(CKPT_DIR, "student_lora_adapter")
    UEM_ENHANCER_CKPT_PATH = os.path.join(CKPT_DIR, "uem_enhancer.pth")
    XATTN_ADAPTERS_CKPT_PATH = os.path.join(CKPT_DIR, "cross_attention_adapters.pth")
    
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/task_loss_description/test/test_anony_with_loss.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/results"
    
    # 在此处设置您想运行的模式
    EVAL_MODE = "CLIPPING_ONLY" # 可选: "STANDARD", "CLIPPING_ONLY", "DP", "BASELINE", "ORIGINAL_TEXT_BASELINE"
    CLIPPING_NORM = 1.0
    EPSILON = 10.0 # 差分隐私的 Epsilon 值
    LIMIT = None # Set to a number (e.g., 50) for quick testing, or None to run all

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
        print(f"API Error: {e}")
        return "API_ERROR"

def save_results(config, metrics, samples):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = f"{config.EVAL_MODE}"
    if config.EVAL_MODE == "DP":
        mode_str += f"_eps{config.EPSILON}"
    filepath = os.path.join(config.OUTPUT_DIR, f"eval_{mode_str}_{timestamp}.json")
    
    # Create a serializable version of the config
    serializable_config = {k: v for k, v in vars(config).items() if not k.startswith('__') and isinstance(v, (str, int, float, bool, type(None)))}

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({"config": serializable_config, "metrics": metrics, "samples": samples}, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def judge_prediction(answer, label, client, config):
    prompt = PROMPT_JUDGE.format(ground_truth=label, generated_answer=answer)
    verdict = generate_api_response(client, config.JUDGE_MODEL_NAME, [{"role": "system", "content": "You are a precise evaluator."}, {"role": "user", "content": prompt}], 0.0)
    
    # 🔴 修复：需要精确匹配，避免 "Incorrect" 中的 "correct" 被误判
    verdict_lower = verdict.lower().strip()
    
    # 方案1：检查是否以 "correct" 开头（推荐）
    if verdict_lower.startswith("correct"):
        return True
    elif verdict_lower.startswith("incorrect"):
        return False
    else:
        # 如果API返回了意外格式，打印警告
        print(f"Warning: Unexpected verdict format: '{verdict}'. Treating as incorrect.")
        return False


@torch.no_grad()
def evaluate_single_entry(data: Dict, uem_model: TrainableEnhancer, student_llm: PeftModel, tokenizer: AutoTokenizer, config: Config) -> str:
    generate_kwargs = {"max_new_tokens": 30, "temperature": 0.0, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}

    # Baseline modes do not use the enhancer and use a standard LLM
    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        text = data['response'] if config.EVAL_MODE == "ORIGINAL_TEXT_BASELINE" else data['anonymized_response']
        user_prompt = REDDIT_PROMPT_USER.format(question_asked=data['question_asked'], user_response=text)
        conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)
        outputs = student_llm.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # --- REFACTORED: In-Layer Injection Flow ---
    
    # 1. Get context vector from the UEM Enhancer
    context_vector = uem_model([data['loss_description_sentence']])
    context_vector = context_vector.to(config.LLM_DEVICE)

    # 2. Apply optional clipping and differential privacy noise
    if config.EVAL_MODE != "STANDARD":
        norm = torch.linalg.norm(context_vector, dim=-1, keepdim=True)
        scale = (config.CLIPPING_NORM / (norm + 1e-9)).clamp(max=1.0)
        context_vector = context_vector * scale

        if config.EVAL_MODE == "DP":
            sigma = (2 * config.CLIPPING_NORM) / (config.EPSILON + 1e-9)
            noise = torch.randn_like(context_vector) * sigma
            context_vector = context_vector + noise

    # 3. Prepare standard LLM inputs (NO embedding manipulation)
    user_prompt = REDDIT_PROMPT_USER.format(question_asked=data['question_asked'], user_response=data['anonymized_response'])
    conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(config.LLM_DEVICE)
    
    # 4. Generate response, passing the context_vector as a keyword argument
    outputs = student_llm.generate(
        **inputs,
        context=context_vector,
        **generate_kwargs
    )
    
    # 5. Decode the generated part of the output
    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return decoded if decoded else "[EMPTY_OUTPUT]"


def main(config: Config):
    api_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE)

    print("--- Loading Models for Evaluation ---")
    llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_PATH)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    uem_model = None
    llm_student = None

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        print("Loading standard LLM for baseline evaluation...")
        llm_student = AutoModelForCausalLM.from_pretrained(config.LLM_PATH, torch_dtype=torch.bfloat16).to(config.LLM_DEVICE)
    else:
        print("Loading custom PAUE_LlamaForCausalLM with adapters...")
        # 1. Load the custom base model which includes the CrossAttentionAdapter layers
        llm_student_base = PAUE_LlamaForCausalLM.from_pretrained(config.LLM_PATH, torch_dtype=torch.bfloat16)
        
        # 2. Load the trained weights for the CrossAttentionAdapter modules
        adapter_weights = torch.load(config.XATTN_ADAPTERS_CKPT_PATH, map_location="cpu")
        llm_student_base.load_state_dict(adapter_weights, strict=False) # strict=False is crucial here
        print("Cross-attention adapter weights loaded.")

        # 3. Apply the LoRA adapter on top of the custom model
        llm_student = PeftModel.from_pretrained(llm_student_base, config.STUDENT_ADAPTER_PATH)
        llm_student.to(config.LLM_DEVICE)
        print("LoRA adapter loaded.")

        # 4. Load the UEM enhancer model
        uem_tokenizer = AutoTokenizer.from_pretrained(config.UEM_PATH, use_fast=False)
        # We need the LLM's hidden size to initialize the projection layer correctly
        uem_model = TrainableEnhancer(config.UEM_PATH, uem_tokenizer, llm_student.config.hidden_size)
        uem_model.load_state_dict(torch.load(config.UEM_ENHANCER_CKPT_PATH, map_location="cpu"))
        uem_model.to(config.UEM_DEVICE)
        uem_model.eval()
        print("UEM enhancer model loaded.")

    llm_student.eval()

    records_to_process = [json.loads(line) for line in open(config.INPUT_DATA_FILE, 'r')]
    if config.LIMIT:
        records_to_process = records_to_process[:config.LIMIT]
    
    correct_predictions, total_predictions = 0, 0
    qualitative_samples = []

    print(f"\n--- Starting Evaluation for {len(records_to_process)} records using mode '{config.EVAL_MODE}' ---")
    for data in tqdm(records_to_process, desc="Evaluating"):
        true_label = data.get("personality", {}).get("occupation")
        if not true_label: continue
        
        predicted_text = evaluate_single_entry(data, uem_model, llm_student, llm_tokenizer, config)
        is_correct = judge_prediction(predicted_text, true_label, api_client, config)
        
        if is_correct: correct_predictions += 1
        total_predictions += 1
        
        tqdm.write(f"[Record {total_predictions}] Label: {true_label}, Predicted: '{predicted_text}', Correct: {is_correct}")
        if len(qualitative_samples) < 20 or not is_correct: # Log more incorrect samples
             qualitative_samples.append({"data": data, "prediction": predicted_text, "correct": is_correct})

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"Mode: {config.EVAL_MODE}, Final Accuracy: {accuracy:.2%}")
    save_results(config, {"accuracy": accuracy, "total": total_predictions, "correct": correct_predictions}, qualitative_samples)

if __name__ == "__main__":
    main(Config())