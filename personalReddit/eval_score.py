import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from torch.cuda.amp import autocast
import re

# Assume train.py defines these correctly for the personalReddit task
# Make sure train.py is accessible
try:
    from train import TrainableEnhancer, REDDIT_PROMPT_SYSTEM, REDDIT_PROMPT_USER
except ImportError:
    print("Error: Could not import from train.py. Make sure it's in the Python path or current directory.")
    exit()

# Ensure _utils/model.py exists and load_local_model is correctly defined
# Make sure _utils/model.py is accessible
try:
    from _utils.model import load_local_model
except ImportError:
    print("Error: Could not import from _utils/model.py. Make sure it exists and is accessible.")
    exit()

# --- 1. 配置类 ---
class Config:
    # --- 设备配置 ---
    IEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else IEM_DEVICE

    # --- DeepSeek API 配置 ---
    # 确保 API_KEY 在您的环境中设置
    DEEPSEEK_API_KEY = os.environ.get("API_KEY")
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    JUDGE_MODEL_NAME = "deepseek-chat"

    # --- 模型路径 ---
    LLM_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    IEM_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    CKPT_PATH = "/root/autodl-tmp/PPPUE/personalReddit/ckpt/Prefix_LoRA"

    # --- 数据文件 ---
    INPUT_DATA_FILE = "/root/autodl-tmp/PPPUE/personalReddit/benchmark/reprocess/test/test_anony_with_loss.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/PPPUE/personalReddit/results/LoRA_Prefix/strict_prompt/score"

    # --- 实验设置 ---
    EVAL_MODE = "DP" # Options: BASELINE, STANDARD, CLIPPING_ONLY, DP, ORIGINAL_TEXT_BASELINE
    EPSILON = 50.0
    CLIPPING_NORM = 1.0
    PREFIX_LENGTH = 5
    LIMIT = None 

    # LoRA 配置
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- 2. 提示词模板和辅助函数 ---

# --- MODIFIED Judge Prompt for Score-Only Output ---
PROMPT_JUDGE = """
Role: You are a highly discerning evaluation expert with deep contextual understanding of professional roles and their nuances.

Task: Evaluate the quality of a "Generated Answer" regarding a person's occupation, using the full **[Original User Response]** (provided in the context of the **[Question Asked]**) as the ultimate source of truth. The "Generated Answer" was produced by an AI that saw the [Question Asked] and an ANONYMIZED version of the [User Response]. Focus on **how well the AI's answer reflects specific, nuanced occupation-related information present *only* in the [Original User Response]**.

Rate the "Generated Answer" on a scale of 1 to 10 based on the rubric below. **Output ONLY the final numerical score (1-10) and nothing else.**

**SCORING RUBRIC (10-Point Scale):**
* **1-2 (Incorrect)**: Completely wrong or irrelevant.
* **3-4 (Weakly Related)**: Related but incorrect field.
* **5-6 (Correct Generalization)**: Correct but overly general category plausible from anonymized text.
* **7-8 (Good Recovery)**: Correctly identifies a significant past profession, close synonym, but misses the current/precise role from the original.
* **9-10 (Excellent Recovery)**: Highly specific and correct identification of the primary, current profession only recoverable from the original response (10 = perfect match).

--- EXAMPLES ---

**Example 1:**
[Question Asked]: "..."
[Original User Response]: "...As an engineer... intricate structural machinery..."
[Ground Truth Profession]: "structural engineer"
[Generated Answer]: "Structural Engineer"
Your Output: 10

**Example 2:**
[Question Asked]: "..."
[Original User Response]: "...retired-CEO gig..."
[Ground Truth Profession]: "retired CEO"
[Generated Answer]: "CEO"
Your Output: 8

**Example 3:**
[Question Asked]: "..."
[Original User Response]: "...Being an astronomer... spinning theories, deciphering the secrets of the cosmic ballet."
[Ground Truth Profession]: "astronomer"
[Generated Answer]: "Scientist"
Your Output: 6

**Example 4:**
[Question Asked]: "..."
[Original User Response]: "...Bezier curves... Pantone colors..."
[Ground Truth Profession]: "graphic designer"
[Generated Answer]: "Software Developer"
Your Output: 4

**Example 5:**
[Question Asked]: "..."
[Original User Response]: "...celestial phenomenon I've observed... Royal Observatory..."
[Ground Truth Profession]: "astronomer"
[Generated Answer]: "Librarian"
Your Output: 2
--- END EXAMPLES ---

[Question Asked]:
{question_asked}

[Original User Response]:
{original_text}

[Ground Truth Profession]:
{ground_truth_label}

[Generated Answer]:
{generated_answer}

Your Output:
"""

def generate_api_response(client: OpenAI, model: str, messages: List[Dict], temperature: float) -> str:
    # (保持不变)
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
    # (保持不变, 确保创建 OUTPUT_DIR)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{config.EVAL_MODE}_eps{config.EPSILON}_{timestamp}.json" # Updated filename
    filepath = os.path.join(config.OUTPUT_DIR, filename)

    output_data = {
        "config": {k: v for k, v in vars(config).items() if not k.startswith('__') and isinstance(v, (str, int, float, bool, list, dict, type(None)))},
        "metrics": metrics,
        "qualitative_samples": samples # Samples will no longer have justification
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False, default=str)

    print(f"\nEvaluation results saved to: {filepath}")

# --- MODIFIED Judge Function for Score-Only Output ---
def judge_prediction(generated_answer: str, ground_truth_label: str, original_text: str, question_asked: str, judge_client: OpenAI, config: Config) -> Optional[int]: # Return Optional[int]
    judge_prompt = PROMPT_JUDGE.format(
        question_asked=question_asked,
        original_text=original_text,
        ground_truth_label=ground_truth_label,
        generated_answer=generated_answer
    )
    # 更新的系统消息 - 要求只输出数字
    system_message = "You are an evaluation expert. Output ONLY a single integer score between 1 and 10 based on the provided rubric and context."

    score_str = generate_api_response(
        judge_client,
        config.JUDGE_MODEL_NAME,
        [{"role": "system", "content": system_message},
         {"role": "user", "content": judge_prompt}],
        0.0 # Temperature 0
    )

    if score_str == "API_ERROR":
        tqdm.write(f"  - API Error during judging.")
        return 1 # Default error score

    try:
        # Clean the string: remove surrounding whitespace, code blocks etc.
        # Find the first sequence of digits
        match = re.search(r'\d+', score_str)
        if match:
             score = int(match.group(0))
             # Clamp score to 1-10
             if not (1 <= score <= 10):
                 tqdm.write(f"  - WARNING: Judge returned score outside 1-10 range: {score}. Clamping.")
                 score = max(1, min(10, score))
             return score
        else:
             # Handle cases where LLM might output text like "Score: 7" or just "7."
             tqdm.write(f"  - ERROR: Could not parse score from judge response: '{score_str}'. Returning default score 1.")
             return 1 # Default error score if no digits found

    except ValueError:
        tqdm.write(f"  - ERROR: Judge response was not a valid number: '{score_str}'. Returning default score 1.")
        return 1 # Default error score if conversion fails

# --- 3. 评估模块 (evaluate_single_entry) ---
# (保持不变)
@torch.no_grad()
def evaluate_single_entry(
    data: Dict, eval_model: Optional[TrainableEnhancer], shared_llm: AutoModelForCausalLM, # Added Optional type hint
    shared_tokenizer: AutoTokenizer, config: Config
) -> str:
    question_asked = data.get('question_asked', '')

    if config.EVAL_MODE == "ORIGINAL_TEXT_BASELINE":
        text_for_inference = data.get('response', '') # Use original response
    else:
        text_for_inference = data.get('anonymized_response', '') # Use anonymized response

    generate_kwargs = {"max_new_tokens": 30, "temperature": 0.1, "do_sample": True, "repetition_penalty": 1.2}

    if config.EVAL_MODE in ["BASELINE", "ORIGINAL_TEXT_BASELINE"] or eval_model is None: # Added check for eval_model is None
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
    else: # This block now requires eval_model to be not None
        loss_desc_sentence = data.get('loss_description_sentence', '')
        anonymized_text = data.get('anonymized_response', '')

        # 1. IEM (UEM) encoding on IEM_DEVICE
        if not hasattr(eval_model, 'uem') or not hasattr(eval_model, 'uem_tokenizer') or not hasattr(eval_model, 'projection_layer'):
             raise ValueError("eval_model is missing required attributes (uem, uem_tokenizer, projection_layer).")

        with autocast(enabled=eval_model.uem.device.type == 'cuda'):
            iem_inputs = eval_model.uem_tokenizer(loss_desc_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.IEM_DEVICE)
            iem_outputs = eval_model.uem(**iem_inputs)
            sentence_representation = iem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = eval_model.projection_layer(sentence_representation)

        clean_prefix = clean_prefix.view(-1, config.PREFIX_LENGTH, shared_llm.config.hidden_size)

        # 2. Process prefix (Clipping/DP)
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
            else: # Fallback should ideally not happen if mode is validated
                prefix_vector = clean_prefix

        prefix_embeds = prefix_vector.to(device=config.LLM_DEVICE, dtype=shared_llm.dtype)

        # 3. Manually build chat template embeddings
        if not hasattr(shared_llm, 'get_input_embeddings'):
             raise AttributeError("shared_llm does not have method get_input_embeddings.")
        embedding_layer = shared_llm.get_input_embeddings()

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

        # 4. Concatenate embeddings
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

        # 5. Generate using the student model
        if not hasattr(eval_model, 'llm_student'):
             raise AttributeError("eval_model does not have llm_student attribute for enhanced mode generation.")
        outputs = eval_model.llm_student.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            pad_token_id=shared_tokenizer.pad_token_id,
            **generate_kwargs
        )
        generated_ids = outputs

    decoded_text = shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # --- 后处理：尝试提取单个职业 --- (保持不变)
    lines = decoded_text.split('\n')
    potential_occupation = lines[-1].strip().rstrip('.,!?"\'')
    common_prefixes = ["Based on the text, the user is likely a ", "The user is likely a ", "I believe this person is a ", "Occupation: ", "Profession: "]
    for prefix in common_prefixes:
        if potential_occupation.lower().startswith(prefix.lower()):
            potential_occupation = potential_occupation[len(prefix):].strip()
            break
    return potential_occupation

# --- 4. 主流程 ---
def main(config: Config):
    if not config.DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set. It is required for the Judge model.")
        return

    api_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_BASE)

    print("--- Loading Models for Evaluation ---")
    try:
        shared_llm, shared_tokenizer = load_local_model(config.LLM_PATH, device=config.LLM_DEVICE)
    except Exception as e:
        print(f"Error loading base LLM model from {config.LLM_PATH}: {e}")
        return

    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token
        shared_tokenizer.pad_token_id = shared_tokenizer.eos_token_id

    eval_model: Optional[TrainableEnhancer] = None # Explicitly type hint
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        try:
             iem_tokenizer = AutoTokenizer.from_pretrained(config.IEM_PATH, use_fast=False)
        except Exception as e:
            print(f"Error loading IEM tokenizer from {config.IEM_PATH}: {e}")
            return
        print("--- Loading IEM and LoRA components ---")

        lora_adapter_path = os.path.join(config.CKPT_PATH, "lora_adapters")
        other_weights_path = os.path.join(config.CKPT_PATH, "uem_projection.pt")

        if not os.path.isdir(lora_adapter_path) or not os.path.isfile(other_weights_path):
            print(f"Error: Model components not found in '{config.CKPT_PATH}'. Expected 'lora_adapters' dir and 'uem_projection.pt' file.")
            print("Cannot run enhanced modes. Exiting.")
            return # Exit if components are missing for enhanced modes

        try:
            print(f"Loading LoRA adapters from: {lora_adapter_path}")
            llm_student = PeftModel.from_pretrained(shared_llm, lora_adapter_path)
            llm_student = llm_student.merge_and_unload()
            print("LoRA adapters loaded and merged.")

            eval_model = TrainableEnhancer(config.IEM_PATH, iem_tokenizer, None, llm_student, shared_tokenizer)
            eval_model.uem.to(config.IEM_DEVICE)
            eval_model.projection_layer.to(config.IEM_DEVICE)

            print(f"Loading IEM (UEM) and projection weights from: {other_weights_path}")
            checkpoint = torch.load(other_weights_path, map_location="cpu")
            eval_model.uem.load_state_dict(checkpoint['uem_state_dict'])
            eval_model.projection_layer.load_state_dict(checkpoint['projection_layer_state_dict'])

            print(f"Successfully loaded all model components from {config.CKPT_PATH}")
            eval_model.eval()
        except Exception as e:
            print(f"Error loading LoRA or IEM components from {config.CKPT_PATH}: {e}")
            print("Cannot run enhanced modes. Exiting.")
            return # Exit if components fail to load for enhanced modes
    else:
        print(f"Running in {config.EVAL_MODE} mode. IEM and LoRA models are not loaded.")

    try:
        with open(config.INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input data file not found at {config.INPUT_DATA_FILE}")
        return
    except Exception as e:
        print(f"Error reading input data file {config.INPUT_DATA_FILE}: {e}")
        return

    records_to_process = lines[:config.LIMIT] if config.LIMIT is not None else lines

    total_score = 0
    excellent_recovery_count = 0
    valid_records_processed = 0
    qualitative_samples = []

    print(f"\n--- Starting Evaluation for {len(records_to_process)} records ---")
    try:
        for i, line in enumerate(tqdm(records_to_process, desc="Evaluation")):
            data = None
            try:
                data = json.loads(line)

                original_response = data.get("response")
                anonymized_response = data.get("anonymized_response")
                loss_desc = data.get("loss_description_sentence")
                personality = data.get("personality", {})
                true_label = personality.get("occupation")
                question_asked = data.get("question_asked")

                missing_fields = []
                if not original_response: missing_fields.append("response")
                if not true_label: missing_fields.append("personality.occupation")
                if not question_asked: missing_fields.append("question_asked")
                if config.EVAL_MODE != "ORIGINAL_TEXT_BASELINE" and not anonymized_response:
                    missing_fields.append("anonymized_response")

                if missing_fields:
                    tqdm.write(f"Skipping record {i+1} due to missing fields: {', '.join(missing_fields)}.")
                    continue

                data['loss_description_sentence'] = loss_desc if loss_desc is not None else ""

                # --- Inference ---
                predicted_text = evaluate_single_entry(data, eval_model, shared_llm, shared_tokenizer, config)

                # --- Judging ---
                score = judge_prediction(predicted_text, true_label, original_response, question_asked, api_client, config)

                # --- Update Metrics ---
                if score is not None: # Check if judging was successful
                    total_score += score
                    if score >= 9:
                        excellent_recovery_count += 1
                    valid_records_processed += 1
                else:
                    # If judge_prediction returned None (e.g., severe parsing error, though unlikely with current code)
                    tqdm.write(f"Skipping metrics update for record {i+1} due to judging error.")
                    continue # Skip logging and sample saving for this record

                # --- Logging ---
                current_avg_score = total_score / valid_records_processed if valid_records_processed > 0 else 0
                current_excellent_rate = excellent_recovery_count / valid_records_processed if valid_records_processed > 0 else 0
                tqdm.write(f"\n[Record {i+1}/{len(records_to_process)} | Processed: {valid_records_processed}] Label: {true_label}\n"
                           f"  - Predicted: '{predicted_text}'\n"
                           f"  - Score: {score}/10\n" # Removed justification from log
                           f"  - Running Avg Score: {current_avg_score:.2f}/10\n"
                           f"  - Running Excellent Rate (9-10): {current_excellent_rate:.2%}")

                # --- Store Sample ---
                qualitative_samples.append({
                    "Record Index": i+1,
                    "Question Asked": question_asked,
                    "Original Response": original_response,
                    "Anonymized Response": anonymized_response,
                    "Loss Description": loss_desc,
                    "True Profession": true_label,
                    "Generated Answer": predicted_text,
                    "Score": score, # Removed justification
                })

            except json.JSONDecodeError:
                 tqdm.write(f"Skipping record {i+1} due to invalid JSON: {line.strip()}")
                 continue
            except Exception as e:
                import traceback
                error_context = f"Data: {json.dumps(data) if data else 'None'}"
                tqdm.write(f"Skipping record {i+1} due to unexpected error: {e}\n{traceback.format_exc()}Line: {line.strip()}\n{error_context}")
                continue
    finally:
        pass

    # --- Final Metrics ---
    average_score = total_score / valid_records_processed if valid_records_processed > 0 else 0
    excellent_rate = excellent_recovery_count / valid_records_processed if valid_records_processed > 0 else 0
    total_predictions = valid_records_processed

    print("\n--- Evaluation Complete ---")
    print(f"Mode: {config.EVAL_MODE}")
    if config.EVAL_MODE == "DP":
        print(f"Epsilon: {config.EPSILON}")
    print(f"Total Records Attempted: {len(records_to_process)}")
    print(f"Total Records Successfully Processed: {total_predictions}")
    if total_predictions == 0:
        print("No records were successfully processed. Cannot calculate metrics.")
    else:
        print(f"Average Score: {average_score:.2f}/10")
        print(f"Excellent Recovery Rate (Score 9-10): {excellent_rate:.2%} ({excellent_recovery_count}/{total_predictions})")

    metrics_to_save = {
        "average_score": average_score if total_predictions > 0 else None,
        "excellent_recovery_rate": excellent_rate if total_predictions > 0 else None,
        "excellent_recovery_count": excellent_recovery_count,
        "total_score": total_score,
        "total_processed": total_predictions,
        "total_attempted": len(records_to_process),
    }
    if total_predictions > 0:
        save_results(config, metrics_to_save, qualitative_samples)
    else:
        print("\nNo results saved as no records were successfully processed.")

if __name__ == "__main__":
    # 命令行解析部分已被移除。
    # 要更改设置，请直接修改上面的 Config 类。
    config = Config()

    # --- 路径验证 ---
    print("\n--- Configuration ---")
    print(f"Mode: {config.EVAL_MODE}")
    if config.EVAL_MODE == "DP": print(f"Epsilon: {config.EPSILON}")
    print(f"LLM Path: {config.LLM_PATH}")
    print(f"Input Data: {config.INPUT_DATA_FILE}")
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        print(f"IEM Path: {config.IEM_PATH}")
        print(f"Checkpoint Path: {config.CKPT_PATH}")
    print(f"Output Dir: {config.OUTPUT_DIR}")
    print(f"Limit: {config.LIMIT}")
    print("---------------------\n")

    paths_to_check = {
        "Shared LLM": config.LLM_PATH,
        "Input Data": config.INPUT_DATA_FILE,
    }
    if config.EVAL_MODE not in ["BASELINE", "ORIGINAL_TEXT_BASELINE"]:
        paths_to_check["IEM Model"] = config.IEM_PATH
        paths_to_check["Checkpoint Dir"] = config.CKPT_PATH
        if os.path.exists(config.CKPT_PATH):
             paths_to_check["LoRA Adapters"] = os.path.join(config.CKPT_PATH, "lora_adapters")
             paths_to_check["IEM/Projection Weights"] = os.path.join(config.CKPT_PATH, "uem_projection.pt")

    all_paths_exist = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        print(f"Checking path for {name}: '{path}' ... {'Found' if exists else 'NOT FOUND'}")
        if not exists:
            all_paths_exist = False

    if not config.DEEPSEEK_API_KEY:
        print("Checking DEEPSEEK_API_KEY (env var API_KEY)... NOT FOUND")
        all_paths_exist = False
    else:
        print("Checking DEEPSEEK_API_KEY (env var API_KEY)... Found")

    if all_paths_exist:
        print("\nAll required paths and API key found. Starting evaluation...")
        main(config)
    else:
        print("\nError: One or more required paths were not found, or API key is missing.")
        print("Please update the paths in the Config class and ensure the API_KEY environment variable is set.")
        print("Required paths depend on the chosen EVAL_MODE.")