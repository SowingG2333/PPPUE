import json
import argparse
import logging
import asyncio
import copy
import sys
from typing import Optional, List, Dict, Any, Literal, Tuple

# 导入必要的库
try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    print("Pydantic is not installed. Please install it: pip install pydantic", file=sys.stderr)
    sys.exit(1)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, Timeout, APIConnectionError
except ImportError:
    print("OpenAI library is not installed. Please install it: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm as sync_tqdm
except ImportError:
    print("tqdm is not installed. Please install it: pip install tqdm", file=sys.stderr)
    # Define dummy tqdm functions if not installed
    async def async_tqdm(iterable, *args, **kwargs):
        for item in iterable:
            yield item
    def sync_tqdm(iterable, *args, **kwargs):
        return iterable

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic 模型定义 ---
class Personality(BaseModel):
    # 新增 name 字段，便于将 people（姓名）纳入匿名与评测链路
    name: Optional[str] = Field(None, description="Person's full name if present")
    birth_year: Optional[int] = Field(None, description="Year of birth as a 4-digit integer")
    sex: Optional[Literal["male", "female", "unknown"]] = Field(None, description="Biological sex inferred from text")
    birth_city_country: Optional[str] = Field(None, description="City and country of birth")
    education_level: Optional[str] = Field(None, description="Highest or most specific education level mentioned")
    occupation: Optional[str] = Field(None, description="Primary occupation mentioned")
    income_level: Optional[Literal["low", "middle", "high", "very high", "unknown"]] = Field(None, description="Inferred income level based on text clues")

    @field_validator('birth_year')
    @classmethod
    def check_birth_year_range(cls, v):
        if v is not None and (v < 1800 or v > 2025):
            logging.warning(f"Extracted birth_year {v} seems out of reasonable range.")
        return v

    @field_validator('sex')
    @classmethod
    def normalize_sex(cls, v):
        """归一化 Attacker 可能返回的 'Male', 'FEMALE', '' 等值"""
        if v is None:
            return None
        vv = str(v).strip().lower()
        if vv in {"male", "female"}:
            return vv
        # 将 'null', '', 'none' 等统一视为 'unknown'
        if vv in {'unknown', 'null', 'none', ''}:
             return "unknown"
        return "unknown" # 兜底

    @field_validator('income_level')
    @classmethod
    def normalize_income_level(cls, v):
        """归一化 Attacker 可能返回的 'High', ' middle ' 等值"""
        if v is None:
            return None
        vv = str(v).strip().lower()
        if vv in {"low", "middle", "high", "very high"}:
             return vv
        if vv in {'unknown', 'null', 'none', ''}:
             return "unknown"
        return "unknown" # 兜底

    @field_validator('name', 'birth_city_country', 'education_level', 'occupation')
    @classmethod
    def normalize_string_fields(cls, v):
        """将 Attacker 返回的 'null' 字符串或空字符串转为 None"""
        if v is None:
            return None
        vv = str(v).strip()
        if vv.lower() in {'null', 'none', 'n/a', ''}:
            return None # 转换为 None
        return vv # 返回清理过的字符串


# 1. 匿名者 (Anonymizer) Prompt
SYSTEM_PROMPT_ANONYMIZER = """
You are a privacy engineer specializing in adversarial text anonymization against a strong profiler.
Your goal is to rewrite biographies so that a competent attacker cannot accurately infer target PII attributes,
including personal names, while preserving meaning and fluency as much as possible.
"""

USER_PROMPT_ANONYMIZER_TEMPLATE = """
Your task is to rewrite the [Biography Text] to obscure ONLY the specific [PII Attributes] listed, and also any cues that enable inferring those attributes according to the [Leaked Attributes and Attacker Guesses]. Preserve overall meaning and readability.

[Biography Text]:
{original_text}

[PII Attributes to Anonymize]:
{pii_list_str}

[Leaked Attributes and Attacker Guesses]:
{leak_feedback_str}

Instructions:
1) Remove direct mentions of the listed attributes (including personal names if listed).
2) Neutralize indirect cues that enable inference (gendered pronouns, years/ages, elite institution names, senior titles, precise geographies, salary/wealth hints).
3) Prefer generalization over deletion. If needed at higher intensity, drop or restructure sentences.
4) Do NOT invent new facts. Do NOT add placeholders unless instructed by intensity.
5) Output ONLY the anonymized biography text. No explanations.

{intensity_hint}
Rewrite the text following all instructions. Output ONLY the anonymized text.
"""

# 2. 攻击者 (Attacker) Prompt
SYSTEM_PROMPT_ATTACKER = """
You are an expert profiler. Maximize inference of PII from a biography using text evidence plus general world knowledge
(roles, seniority, industry, geography, institution prestige, lifestyle cues, numbers/dates). Prefer the most likely value
when signals are weak but consistent. Output JSON only; no explanations.
"""

USER_PROMPT_ATTACKER_TEMPLATE = """
Your task is to infer a user's profile based ONLY on the [Biography Text] provided.

CRITICAL INSTRUCTIONS:
1) Read the [Biography Text] carefully.
2) Make your best guess for EACH attribute using text evidence AND general world knowledge priors (salary bands by role/seniority/industry and city, typical education for professions, gendered pronouns/titles/relations, etc.).
3) Prefer a most-likely guess when signals are weak but consistent. Use JSON null ONLY if there is truly no usable signal.
4) Strict JSON output with exactly these keys: "name", "birth_year", "sex", "birth_city_country", "education_level", "occupation", "income_level".
5) Formatting:
   - name: If an explicit personal name is present, output it as a string; otherwise null. Do not fabricate names.
   - birth_year: integer or null; you may derive from anchored ages/dates.
   - sex: "male", "female", or "unknown" (use "unknown" if gender cues are inconsistent).
   - birth_city_country: "City, Country" if clear; else null.
   - education_level: highest/most specific plausible credential (e.g., "Master's in Economics", "Doctorate", "Bachelor's in Computer Science"); null if no signal.
   - occupation: primary/current role in normalized form (e.g., "software engineer", "professor", "artist").
   - income_level: ONLY "low" | "middle" | "high" | "very high" | "unknown". Use priors by role/seniority/industry and geography; null only if no signal at all.

Input:
[Biography Text]:
{anonymized_text}

Respond ONLY with a single, valid JSON object.
"""

# 3. 裁判 (Judge) Prompt
SYSTEM_PROMPT_JUDGE = """
You are a precise, rule-following evaluation expert comparing an AI's guess against a ground truth value.
"""

USER_PROMPT_JUDGE_TEMPLATE = """
Instructions:
1) Compare the [Attacker's Guess] to the [Ground Truth Value].
2) Evaluation Criteria:
   - "Correct": exact match, a clear synonym, or a factually consistent specific guess (e.g., GT "scientist", Guess "physicist").
   - "Incorrect": wrong, OR "unknown", OR **too general** (e.g., GT "physicist", Guess "scientist").
3) Output only "Correct" or "Incorrect".

[Attribute Name]: {attribute_name}
[Ground Truth Value]: {ground_truth_value}
[Attacker's Guess]: {attacker_guess}

Is the [Attacker's Guess] a correct identification of the [Ground Truth Value] for the given attribute, based on the criteria? Respond ONLY with "Correct" or "Incorrect".
"""

# --- Async API Call Helper ---
async def call_llm(
    client: AsyncOpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    purpose: str,
    entry_index: int,
    max_retries: int = 3,
    retry_delay: int = 5,
    request_json: bool = False
) -> Optional[str]:
    """Generic async LLM call with retries."""
    last_error = None
    for attempt in range(max_retries):
        try:
            logging.debug(f"Entry {entry_index+1} ({purpose}): Attempt {attempt + 1}/{max_retries} calling {model_name}.")

            completion_params = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
            }
            # 使用 JSON 模式（DeepSeek / OpenAI 支持）
            if request_json and ("gpt-" in model_name or "deepseek" in model_name):
                completion_params["response_format"] = {"type": "json_object"}

            completion = await client.chat.completions.create(**completion_params)

            response_content = completion.choices[0].message.content
            logging.debug(f"Entry {entry_index+1} ({purpose}): Raw response: {response_content}")
            if response_content is None:
                logging.warning(f"Entry {entry_index+1} ({purpose}): API returned None content on attempt {attempt+1}.")
                response_content = ""

            if not response_content.strip() and attempt < max_retries - 1:
                logging.warning(f"Entry {entry_index+1} ({purpose}): API returned empty string on attempt {attempt+1}. Retrying...")
                # 强制重试
                raise APIError(f"Empty response received on attempt {attempt+1}", http_status=500, request=None)

            return response_content.strip() if response_content else None

        except (APIError, RateLimitError, Timeout, APIConnectionError) as e:
            last_error = e
            logging.warning(f"Entry {entry_index+1} ({purpose}): API call failed attempt {attempt + 1}/{max_retries}: {e}")
            sleep_time = retry_delay * (2 ** attempt)
            if isinstance(e, RateLimitError):
                logging.warning(f"Rate limit hit, increasing retry delay to {sleep_time}s.")
            if attempt < max_retries - 1:
                logging.info(f"Entry {entry_index+1} ({purpose}): Retrying after {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
            else:
                logging.error(f"Entry {entry_index+1} ({purpose}): Max retries reached for API call.")
                return None
        except Exception as e:
            last_error = e
            logging.exception(f"Entry {entry_index+1} ({purpose}): Unexpected error attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Entry {entry_index+1} ({purpose}): Retrying after {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error(f"Entry {entry_index+1} ({purpose}): Max retries reached after unexpected error.")
                return None

    logging.error(f"Entry {entry_index+1} ({purpose}): Failed all retries. Last error: {last_error}")
    return None

# --- Core Logic Functions ---
async def anonymize_text(
    client: AsyncOpenAI,
    model_name: str,
    original_text: str,
    pii_dict: Dict[str, Any],
    iteration: int,
    entry_index: int,
    max_retries: int,
    retry_delay: int,
    leak_feedback: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Calls the Anonymizer LLM with leak-aware context."""
    pii_to_anonymize = {k: v for k, v in pii_dict.items() if v is not None}
    if not pii_to_anonymize:
        logging.info(f"Entry {entry_index+1}: No non-null PII found, returning original text.")
        return original_text

    pii_list_str = "\n".join([f"- {k}: {v}" for k, v in pii_to_anonymize.items()])

    # 逐轮升级强度
    if iteration == 0:
        intensity_hint = "**Intensity:** Standard anonymization. Generalize specific entities, neutralize obvious gender/number/geo cues, remove personal names."
    elif iteration == 1:
        intensity_hint = "**Intensity:** More aggressive. Remove numbers/dates entirely, downgrade roles, replace elite institutions with generic phrases, convert cities to country/region, avoid any name mentions."
    else:
        intensity_hint = "**Intensity:** Highly aggressive. Prefer sentence restructuring or removal where needed. Use generic placeholders like [Location], [Profession], [Year], [Name] if a sentence cannot be safely rewritten."

    leak_feedback_str = "None (first iteration)"
    if leak_feedback:
        leak_feedback_str = "\n".join([f"- {k}: {v}" for k, v in leak_feedback.items()])

    user_prompt = USER_PROMPT_ANONYMIZER_TEMPLATE.format(
        original_text=original_text,
        pii_list_str=pii_list_str,
        leak_feedback_str=leak_feedback_str,
        intensity_hint=intensity_hint
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ANONYMIZER},
        {"role": "user", "content": user_prompt}
    ]
    return await call_llm(client, model_name, messages, "Anonymizer", entry_index, max_retries, retry_delay)

async def attack_text_and_parse(
    client: AsyncOpenAI,
    model_name: str,
    anonymized_text: str,
    entry_index: int,
    max_retries: int,
    retry_delay: int
) -> Optional[Dict[str, Any]]:
    """Calls the Attacker LLM once and parses the JSON response."""
    user_prompt = USER_PROMPT_ATTACKER_TEMPLATE.format(
        anonymized_text=anonymized_text
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ATTACKER},
        {"role": "user", "content": user_prompt}
    ]

    json_string = await call_llm(
        client, model_name, messages,
        "Attacker (JSON)", entry_index,
        max_retries, retry_delay,
        request_json=True
    )

    if json_string is None:
        logging.error(f"Entry {entry_index+1} (Attacker): Failed to get response after retries.")
        return None

    try:
        raw = json_string.strip()
        # 兜底处理三引号围栏
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 3:
                block = parts[1]
                if block.lstrip().lower().startswith("json"):
                    block = block.split("\n", 1)[1] if "\n" in block else ""
                raw = block.strip()

        if not raw:
            logging.error(f"Entry {entry_index+1} (Attacker): Received empty string after cleaning. Response: '{json_string}'")
            return None

        parsed_json = json.loads(raw)

        # 验证并归一化 (关键步骤)
        validated_guesses = Personality.model_validate(parsed_json)
        # Pydantic 验证器会自动将 'Male'->'male', 'null'->None, ''->'unknown' 等
        return validated_guesses.model_dump(mode='json') 

    except json.JSONDecodeError as e:
        logging.error(f"Entry {entry_index+1} (Attacker): Failed to decode JSON: {e}. Response: '{json_string}'")
        return None
    except ValidationError as e:
        logging.error(f"Entry {entry_index+1} (Attacker): JSON validation failed: {e}. Response: '{json_string}'")
        return None
    except Exception as e:
        logging.exception(f"Entry {entry_index+1} (Attacker): Unexpected error parsing/validating: {e}. Response: '{json_string}'")
        return None

async def judge_guess(
    client: AsyncOpenAI,
    model_name: str,
    attribute_name: str,
    ground_truth_value: Any,
    attacker_guess: Any,
    original_text: str,
    entry_index: int,
    max_retries: int,
    retry_delay: int
) -> str:
    """Calls the Judge LLM to evaluate the attacker's guess for *complex* fields."""
    # (此函数现在只被 '混合裁判' 用于复杂字段)
    
    gt_str = str(ground_truth_value)
    guess_str_for_prompt = str(attacker_guess)

    user_prompt = USER_PROMPT_JUDGE_TEMPLATE.format(
        attribute_name=attribute_name,
        ground_truth_value=gt_str,
        attacker_guess=guess_str_for_prompt
        # original_text=original_text
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
        {"role": "user", "content": user_prompt}
    ]
    verdict = await call_llm(client, model_name, messages, f"Judge ({attribute_name})", entry_index, max_retries, retry_delay)

    verdict_lower = verdict.lower().strip().rstrip('.,!?"\'') if verdict else ""
    if verdict_lower == "correct":
        return "Correct"
    elif verdict_lower == "incorrect":
        return "Incorrect"
    else:
        logging.warning(f"Entry {entry_index+1} (Judge - {attribute_name}): Unexpected verdict '{verdict}'. Defaulting to 'Error'.")
        return "Error"

# --- 优化后的核心处理逻辑 (使用混合裁判) ---
async def process_single_biography(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    entry_index: int,
    original_data: Dict[str, Any],
    models: Dict[str, str],
    max_iterations: int,
    max_retries: int,
    retry_delay: int
) -> Tuple[int, Dict[str, Any], Optional[str], Dict[str, Any]]:
    """
    Performs the iterative anonymization, attack, and judging process for one biography.
    [OPTIMIZED]: Now uses a HYBRID JUDGE (Python + LLM) for efficiency.
    """
    original_text = original_data.get("text")
    personality_data = original_data.get("personality")
    name_data = original_data.get("people")

    if not original_text or not personality_data:
        logging.warning(f"Entry {entry_index + 1}: Skipping due to missing 'text' or 'personality' field.")
        return entry_index, original_data, None, {"status": "skipped", "reason": "Missing text or personality"}

    # 1. 合并 PII (将 'people' 字段注入为 'name')
    combined_pii_data = copy.deepcopy(personality_data)
    if not combined_pii_data.get("name") and name_data:
        combined_pii_data["name"] = name_data
        logging.debug(f"Entry {entry_index + 1}: Injected 'name' from 'people' field.")

    try:
        ground_truth_pii_model = Personality.model_validate(combined_pii_data)
        ground_truth_pii = ground_truth_pii_model.model_dump(mode='json')
    except ValidationError as e:
        logging.error(f"Entry {entry_index + 1}: Invalid PII data after merging 'people': {e}. Skipping. Data: {combined_pii_data}")
        return entry_index, original_data, None, {"status": "skipped", "reason": "Invalid input PII"}

    attributes_to_anonymize: List[str] = [k for k, v in ground_truth_pii.items() if v is not None]
    attributes_to_judge: List[str] = list(Personality.model_fields.keys())

    if not attributes_to_anonymize:
        logging.info(f"Entry {entry_index + 1}: No non-null PII attributes (including name) to anonymize. Returning original text.")
        return entry_index, original_data, original_text, {"status": "no_pii_found", "iterations_used": 0}

    current_text = original_text
    final_anonymized_text: Optional[str] = None
    anonymization_metadata: Dict[str, Any] = {}
    last_attacker_guesses: Dict[str, Any] = {}
    last_judgements: Dict[str, str] = {}
    leak_feedback: Optional[Dict[str, Any]] = None

    async with semaphore:
        logging.debug(f"Entry {entry_index + 1}: Acquired main semaphore.")
        try:
            for iteration in range(max_iterations):
                logging.info(f"Entry {entry_index + 1}: Starting anonymization iteration {iteration + 1}/{max_iterations}.")

                # --- 1. Anonymize (泄露感知) ---
                anonymized_text = await anonymize_text(
                    client, models["anonymizer"], current_text, ground_truth_pii,
                    iteration, entry_index, max_retries, retry_delay,
                    leak_feedback=leak_feedback
                )
                if anonymized_text is None:
                    logging.error(f"Entry {entry_index + 1}: Anonymizer failed after retries on iteration {iteration + 1}.")
                    anonymization_metadata = {"status": "error", "iterations_used": iteration + 1, "reason": "Anonymizer failed"}
                    return entry_index, original_data, current_text if iteration > 0 else original_text, anonymization_metadata

                current_text = anonymized_text

                # --- 2. Attack (Single JSON Call) ---
                logging.debug(f"Entry {entry_index + 1} Iter {iteration + 1}: Calling Attacker (JSON)...")
                attacker_guesses = await attack_text_and_parse(
                    client, models["attacker"], current_text, entry_index, max_retries, retry_delay
                )

                if attacker_guesses is None:
                    logging.error(f"Entry {entry_index + 1}: Attacker (JSON) failed on iteration {iteration + 1}. Stopping.")
                    anonymization_metadata = {"status": "error", "iterations_used": iteration + 1, "reason": "Attacker (JSON) failed"}
                    return entry_index, original_data, current_text, anonymization_metadata

                last_attacker_guesses = attacker_guesses # Pydantic 已归一化

                # --- 3. HYBRID JUDGE (Python + LLM) ---
                llm_judge_tasks: Dict[str, asyncio.Task] = {} # 存储[属性名, LLM任务]
                judgements: Dict[str, str] = {} # 存储所有裁决结果

                for attr_name in attributes_to_judge:
                    guess = attacker_guesses.get(attr_name)
                    gt_value = ground_truth_pii.get(attr_name)
                    
                    verdict = None # Python 裁判的临时结果
                    
                    # 规则 1: GT 为 null
                    # Pydantic 已将 'null', '', 'none' 转为 None (string fields) 或 'unknown' (enum fields)
                    if gt_value is None:
                        is_guess_unknown = (guess is None) or (guess == "unknown")
                        verdict = "Correct" if is_guess_unknown else "Incorrect"
                    
                    # 规则 2: GT 存在, Guess 为 null/unknown
                    elif (guess is None) or (guess == "unknown"):
                         verdict = "Incorrect"

                    # 规则 3: birth_year (允许 +/- 2 年)
                    elif attr_name == "birth_year" and isinstance(guess, int):
                        verdict = "Correct" if abs(gt_value - guess) <= 2 else "Incorrect"
                    
                    # 规则 4: sex, income_level (精确匹配, Pydantic 已归一化)
                    elif attr_name in ("sex", "income_level"):
                        verdict = "Correct" if gt_value == guess else "Incorrect"
                    
                    # --- Python 裁判结束 ---
                    
                    if verdict is not None:
                        judgements[attr_name] = verdict # 存入 Python 裁决
                    else:
                        # 复杂字段: name, education, occupation, birth_city_country
                        # 这些字段需要 LLM 理解上下文/同义词，且 GT 和 Guess 都不为 null
                        task = asyncio.create_task(
                            judge_guess(client, models["judge"], attr_name, gt_value, guess, original_text, entry_index, max_retries, retry_delay)
                        )
                        llm_judge_tasks[attr_name] = task # 存入任务字典
                
                # 运行所有必需的 LLM 裁判任务
                if llm_judge_tasks:
                    logging.debug(f"Entry {entry_index + 1} Iter {iteration + 1}: Calling LLM Judge for {len(llm_judge_tasks)} complex attributes...")
                    await asyncio.gather(*llm_judge_tasks.values())
                    
                    # 收集 LLM 裁判的结果
                    for attr_name, task in llm_judge_tasks.items():
                        judgements[attr_name] = task.result()
                
                last_judgements = judgements
                # --- 3. END HYBRID JUDGE ---


                # --- 4. Check for Success ---
                judge_errors = [attr for attr, verdict in judgements.items() if verdict == "Error"]

                if judge_errors:
                    logging.error(f"Entry {entry_index + 1}: Judge failed for attributes: {judge_errors} on iteration {iteration + 1}. Stopping.")
                    anonymization_metadata = {"status": "error", "iterations_used": iteration + 1, "reason": f"Judge failed for {judge_errors}", "final_attacker_guesses": last_attacker_guesses}
                    return entry_index, original_data, current_text, anonymization_metadata

                # 成功的标准：真实值 (GT) 不为 null 且被猜中 (Correct)
                leaked_with_value = [
                    attr for attr, verdict in judgements.items()
                    if verdict == "Correct" and ground_truth_pii.get(attr) is not None
                ]

                if not leaked_with_value:
                    logging.info(f"Entry {entry_index + 1}: Anonymization successful (no non-null GT leaked) after {iteration + 1} iterations.")
                    final_anonymized_text = current_text
                    anonymization_metadata = {"status": "success", "iterations_used": iteration + 1, "final_attacker_guesses": last_attacker_guesses}
                    break
                else:
                    # 失败：构建反馈，进入下一轮
                    leak_feedback = {k: last_attacker_guesses.get(k) for k in leaked_with_value}
                    logging.info(f"Entry {entry_index + 1}: Iteration {iteration + 1} failed. Leaked (with value): {leaked_with_value}. Guesses: { {k: last_attacker_guesses.get(k) for k in leaked_with_value} }")

            else: # for 循环正常结束 (未 break)
                logging.warning(f"Entry {entry_index + 1}: Max iterations ({max_iterations}) reached.")
                final_anonymized_text = current_text
                final_leaked_with_value = [
                    attr for attr, verdict in last_judgements.items()
                    if verdict == "Correct" and ground_truth_pii.get(attr) is not None
                ]
                anonymization_metadata = {"status": "max_iterations_reached", "iterations_used": max_iterations, "final_leaked_attributes": final_leaked_with_value, "final_attacker_guesses": last_attacker_guesses}

            logging.debug(f"Entry {entry_index + 1}: Releasing main semaphore.")
            return entry_index, original_data, final_anonymized_text, anonymization_metadata

        except Exception as e:
            logging.exception(f"Entry {entry_index + 1}: Unexpected error during processing loop: {e}")
            return entry_index, original_data, None, {"status": "error", "reason": f"Unexpected loop error: {e}"}


# --- Async 主函数 ---
async def run_main():
    parser = argparse.ArgumentParser(description="Adversarially anonymize biographies using LLMs for Anonymizer, Attacker, and Judge.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file (e.g., db_bio_with_attributes.jsonl).")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file for anonymized results.")
    parser.add_argument("--api_key", required=True, help="API key for the LLM service.")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1", help="Base URL for the API endpoint (default: DeepSeek).")
    parser.add_argument("--anonymizer_model", default="deepseek-chat", help="Model for anonymizing text.")
    parser.add_argument("--attacker_model", default="deepseek-chat", help="Model for attacking/guessing PII.")
    parser.add_argument("--judge_model", default="deepseek-chat", help="Model for judging attacker guesses.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N entries (optional).")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum anonymization iterations per entry.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of API call retries on failure.")
    parser.add_argument("--retry_delay", type=int, default=5, help="Base delay (in seconds) between API call retries.")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum number of concurrent biographies to process.")

    args = parser.parse_args()

    models = {
        "anonymizer": args.anonymizer_model,
        "attacker": args.attacker_model,
        "judge": args.judge_model
    }

    client: Optional[AsyncOpenAI] = None
    try:
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
        logging.info(f"Initialized AsyncOpenAI client for endpoint: {args.base_url}")
        logging.info(f"Models - Anonymizer: {models['anonymizer']}, Attacker: {models['attacker']}, Judge: {models['judge']}")
        logging.info(f"Concurrency: {args.concurrency}, Max Iterations: {args.max_iterations}")
    except Exception as e:
        logging.error(f"Failed to initialize AsyncOpenAI client: {e}")
        return

    entries_to_process: List[Tuple[int, Dict[str, Any]]] = []
    total_lines_read = 0
    skipped_json_errors = 0
    skipped_missing_data = 0
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            logging.info(f"Reading input file: {args.input_file}")
            for i, line in enumerate(infile):
                total_lines_read += 1
                if args.limit is not None and i >= args.limit:
                    logging.info(f"Reached input limit of {args.limit} lines.")
                    break
                try:
                    original_data = json.loads(line.strip())
                    if "text" in original_data and "personality" in original_data and original_data["personality"] is not None:
                        entries_to_process.append((i, original_data))
                    else:
                        logging.warning(f"Skipping line {i + 1} due to missing 'text' or 'personality' key, or personality is null.")
                        skipped_missing_data += 1
                except json.JSONDecodeError:
                    logging.error(f"Skipping line {i + 1} due to invalid JSON.")
                    skipped_json_errors += 1

        logging.info(f"Read {total_lines_read} lines (up to limit), found {len(entries_to_process)} valid entries to process.")
        if not entries_to_process:
            logging.warning("No valid entries found to process.")
            if client: await client.close()
            return

    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return
    except IOError as e:
        logging.error(f"Error reading input file: {e}")
        return

    # --- 创建并发任务 ---
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for index, data in entries_to_process:
        tasks.append(
            asyncio.create_task(
                process_single_biography(
                    client, semaphore, index, data, models,
                    args.max_iterations, args.max_retries, args.retry_delay
                )
            )
        )

    logging.info(f"Created {len(tasks)} tasks for concurrent processing.")

    # --- 执行任务并收集结果 ---
    results_list: List[Tuple[int, Dict[str, Any], Optional[str], Dict[str, Any]]] = []
    task_exception = None
    try:
        # tqdm.asyncio.gather 可能不可用，保留兼容分支
        if 'async_tqdm' in globals() and hasattr(async_tqdm, 'gather'):
            logging.info("Using tqdm.asyncio.gather for progress tracking.")
            raw_results = await async_tqdm.gather(*tasks, desc="Anonymizing Biographies")
        else:
            logging.info("tqdm.asyncio.gather not available, using standard asyncio.gather.")
            raw_results = await asyncio.gather(*tasks)
        results_list.extend(raw_results)
        logging.info("All tasks completed gathering.")
    except Exception as e:
        task_exception = e
        logging.exception(f"An error occurred during asyncio task gathering: {e}")
        logging.error("Attempting to process completed tasks before the error...")
        for task in tasks:
            if task.done():
                try:
                    results_list.append(task.result())
                except Exception as task_exc:
                    logging.error(f"Error retrieving result from a completed task: {task_exc}")

    finally:
        if client:
            await client.close()
            logging.info("AsyncOpenAI client closed.")

    # --- 处理结果并写入输出文件 ---
    success_count = 0
    max_iter_count = 0
    error_count = 0

    results_list.sort(key=lambda x: x[0])

    try:
        logging.info(f"Writing {len(results_list)} results to {args.output_file}...")
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for index, original_data, anonymized_text, metadata in sync_tqdm(results_list, desc="Writing results"):
                output_entry = copy.deepcopy(original_data)
                output_entry["anonymized_text"] = anonymized_text
                output_entry["anonymization_meta"] = metadata
                outfile.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

                status = metadata.get("status", "unknown")
                if status == "success":
                    success_count += 1
                elif status == "max_iterations_reached":
                    max_iter_count += 1
                elif status == "error":
                    error_count += 1

    except IOError as e:
        logging.error(f"Error writing output file: {e}")
    finally:
        logging.info("--- Anonymization Summary ---")
        logging.info(f"Total lines read (up to limit): {total_lines_read}")
        logging.info(f"JSON decode errors skipped: {skipped_json_errors}")
        logging.info(f"Skipped (missing text/personality or null personality): {skipped_missing_data}")
        logging.info(f"Valid entries attempted: {len(entries_to_process)}")
        logging.info(f"Results obtained (incl. errors): {len(results_list)}")
        logging.info(f"Successfully anonymized: {success_count}")
        logging.info(f"Reached max iterations: {max_iter_count}")
        logging.info(f"Errors during processing: {error_count}")
        logging.info(f"Output written to: {args.output_file}")
        if task_exception:
            logging.error(f"Processing stopped prematurely due to error during task execution: {task_exception}")

if __name__ == "__main__":
    try:
        if sys.version_info < (3, 7):
            print("This script requires Python 3.7+ for asyncio.run()", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_main())
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")