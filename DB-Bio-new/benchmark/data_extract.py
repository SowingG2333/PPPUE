import json
import argparse
import logging
import asyncio
from typing import Optional, List, Dict, Any, Literal, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from tqdm import tqdm as sync_tqdm

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM 提取的系统 Prompt ---
SYSTEM_PROMPT_FOR_EXTRACTION = """
You are an expert biographical analyst and an assertive JSON labeler.

**Core Task:**
Analyze the text. Use disciplined inference and world knowledge (roles, geography, seniority) to maximize labels for the schema below. Use `null` (JSON null, not string) *only* when no signal exists.

**Output Rules:**
Respond ONLY with a single, valid JSON object. Do not add extra keys. Enum values (`sex`, `income_level`) must be lowercase. Do not output reasoning.

**Output Schema (exact keys, no others):**
{
  "birth_year": <int or null>,
  "sex": <"male" | "female" | null>,
  "birth_city_country": <string or null>,
  "education_level": <string or null>,
  "occupation": <string or null>,
  "income_level": <"low" | "middle" | "high" | "very high" | null>
}

**Example:**
Input:
[Biography Text]:
"Susan Jim (Born in Feb 22th, 1990) is a senior data scientist at a major tech firm in San Francisco, previously a research assistant at Stanford. She leads model deployment and mentors junior staff."

Expected output:
{
  "birth_year": 1990,
  "sex": "female",
  "birth_city_country": "San Francisco, USA",
  "education_level": "Master's in Data Science",
  "occupation": "data scientist",
  "income_level": "high"
}
"""

# --- Pydantic 模型定义 ---
class Personality(BaseModel):
    birth_year: Optional[int] = Field(None, description="Year of birth as a 4-digit integer")
    sex: Optional[Literal["male", "female"]] = Field(None, description="Sex inferred from explicit mentions")
    birth_city_country: Optional[str] = Field(None, description="City and country of birth")
    education_level: Optional[str] = Field(None, description="Highest or most specific education level mentioned")
    occupation: Optional[str] = Field(None, description="Primary occupation mentioned")
    income_level: Optional[Literal["low", "middle", "high", "very high"]] = Field(None, description="Inferred income level based on explicit text clues")

    @field_validator('birth_year')
    @classmethod
    def check_birth_year_range(cls, v):
        if v is not None and (v < 1800 or v > 2025):
            logging.warning(f"Extracted birth_year {v} out of range; coercing to None.")
            return None
        return v

    @field_validator('sex')
    @classmethod
    def normalize_sex(cls, v):
        if v is None:
            return None
        v = v.strip().lower()
        return v if v in {"male", "female"} else None

    @field_validator('income_level')
    @classmethod
    def normalize_income_level(cls, v):
        if v is None:
            return None
        v = v.strip().lower()
        return v if v in {"low", "middle", "high", "very high"} else None

# --- Async API 调用和处理函数 ---
async def process_entry(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    entry_index: int,
    original_data: Dict[str, Any],
    model_name: str = "deepseek-chat",
    max_retries: int = 3,
    retry_delay: int = 5
) -> Tuple[int, Dict[str, Any], Optional[Personality]]:

    biography_text = original_data.get("text")
    if not biography_text or not biography_text.strip():
        logging.warning(f"Entry {entry_index + 1}: Received empty biography text, skipping processing.")
        return entry_index, original_data, None

    user_prompt = f"[Biography Text]:\n{biography_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_EXTRACTION},
        {"role": "user", "content": user_prompt}
    ]

    async with semaphore:
        logging.debug(f"Entry {entry_index + 1}: Acquired semaphore, starting API call.")
        for attempt in range(max_retries):
            try:
                logging.debug(f"Entry {entry_index + 1}: Attempt {attempt + 1}/{max_retries} calling LLM API.")
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_response = completion.choices[0].message.content
                logging.debug(f"Entry {entry_index + 1}: Raw API response: {raw_response}")

                if not raw_response:
                    logging.warning(f"Entry {entry_index + 1}: API returned an empty response on attempt {attempt + 1}.")
                    extracted_data = {}
                else:
                    if raw_response.strip().startswith("```json"):
                        raw_response = raw_response.strip()[7:-3].strip()
                    elif raw_response.strip().startswith("```"):
                         raw_response = raw_response.strip()[3:-3].strip()
                    extracted_data = json.loads(raw_response)

                validated_personality = Personality.model_validate(extracted_data)
                logging.debug(f"Entry {entry_index + 1}: Validation successful.")
                return entry_index, original_data, validated_personality

            except json.JSONDecodeError as e:
                logging.error(f"Entry {entry_index + 1}: Failed JSON decode attempt {attempt + 1}. Error: {e}. Response: '{raw_response}'")
                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached for JSON decoding.")
                    return entry_index, original_data, None

            except ValidationError as e:
                logging.error(f"Entry {entry_index + 1}: Pydantic validation failed attempt {attempt + 1}. Error: {e}. Data: '{extracted_data}'")
                if attempt < max_retries - 1:
                     logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                     await asyncio.sleep(retry_delay)
                else:
                     logging.error(f"Entry {entry_index + 1}: Max retries reached for Pydantic validation.")
                     return entry_index, original_data, None

            except (APIError, RateLimitError, APITimeoutError, APIConnectionError) as e:
                logging.warning(f"Entry {entry_index + 1}: API call failed attempt {attempt + 1}/{max_retries}: {e}")
                sleep_time = retry_delay * (attempt + 1)
                if isinstance(e, RateLimitError):
                     logging.warning(f"Rate limit hit, increasing retry delay to {sleep_time}s.")

                if attempt < max_retries - 1:
                    logging.info(f"Entry {entry_index + 1}: Retrying after {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    logging.error(f"Entry {entry_index + 1}: Max retries reached for API call.")
                    return entry_index, original_data, None

            except Exception as e:
                logging.exception(f"Entry {entry_index + 1}: Unexpected error attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                     logging.info(f"Entry {entry_index + 1}: Retrying after {retry_delay} seconds...")
                     await asyncio.sleep(retry_delay)
                else:
                     logging.error(f"Entry {entry_index + 1}: Max retries reached after unexpected error.")
                     return entry_index, original_data, None

        logging.debug(f"Entry {entry_index + 1}: Releasing semaphore.")
    return entry_index, original_data, None

# --- Async 主函数 ---
async def main():
    parser = argparse.ArgumentParser(description="Extract PII attributes from DB-Bio JSONL file using an LLM API concurrently.")
    parser.add_argument("--input_file", help="Path to the input JSONL file (e.g., train.jsonl).")
    parser.add_argument("--output_file", help="Path to the output JSONL file (e.g., db_bio_with_attributes.jsonl).")
    parser.add_argument("--api_key", required=True, help="API key for the LLM service.")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1", help="Base URL for the API endpoint (default: DeepSeek).")
    parser.add_argument("--model", default="deepseek-chat", help="Name of the LLM model to use (default: deepseek-chat).")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N entries (optional).")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of API call retries on failure.")
    parser.add_argument("--retry_delay", type=int, default=5, help="Base delay (in seconds) between API call retries.")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum number of concurrent API requests.") # Concurrency argument

    args = parser.parse_args()

    # --- 初始化 Async API 客户端 ---
    try:
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
        logging.info(f"Initialized AsyncOpenAI client for endpoint: {args.base_url} with concurrency {args.concurrency}")
    except Exception as e:
        logging.error(f"Failed to initialize AsyncOpenAI client: {e}")
        return

    # --- 读取输入文件 ---
    lines_to_process = []
    total_lines_read = 0
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                if args.limit is not None and i >= args.limit:
                    logging.info(f"Reached input limit of {args.limit} lines.")
                    break
                try:
                    original_data = json.loads(line.strip())
                    lines_to_process.append((i, original_data)) # Store index and data
                except json.JSONDecodeError:
                    logging.error(f"Skipping line {i + 1} due to invalid JSON.")
                    # Optionally write skipped lines immediately to output?
                total_lines_read += 1
        logging.info(f"Read {len(lines_to_process)} valid entries to process from {args.input_file}")
        if total_lines_read == 0:
            logging.warning("Input file was empty or contained no valid JSON.")
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
    for index, data in lines_to_process:
        tasks.append(
            asyncio.create_task(
                process_entry(
                    client,
                    semaphore,
                    index,
                    data,
                    model_name=args.model,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay
                )
            )
        )

    logging.info(f"Created {len(tasks)} tasks for concurrent processing.")

    # --- 执行任务并收集结果 ---
    results_list: List[Tuple[int, Dict[str, Any], Optional[Personality]]] = []
    try:
        raw_results = await asyncio.gather(*tasks)
        results_list.extend(raw_results)
        logging.info("All tasks completed.")
    except Exception as e:
        logging.exception(f"An error occurred during asyncio.gather: {e}")
        logging.error("Aborting due to error during task execution.")
        await client.close()
        return

    # --- 处理结果并写入输出文件 ---
    processed_count = 0
    error_count = 0

    results_list.sort(key=lambda x: x[0])

    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for index, original_data, personality_result in sync_tqdm(results_list, desc="Writing results"):
                if personality_result:
                    original_data["personality"] = personality_result.model_dump(mode='json')
                    processed_count += 1
                else:
                    original_data["personality"] = None
                    if original_data.get("text"): 
                         error_count += 1
                    else:
                         pass

                outfile.write(json.dumps(original_data, ensure_ascii=False) + '\n')
    except IOError as e:
        logging.error(f"Error writing output file: {e}")
    finally:
        await client.close() # Ensure the async client session is closed
        logging.info("--- Processing Summary ---")
        logging.info(f"Total entries read (up to limit): {total_lines_read}")
        logging.info(f"Entries attempted processing: {len(results_list)}")
        logging.info(f"Successfully processed and extracted: {processed_count}")
        logging.info(f"Errors during processing: {error_count}")
        logging.info(f"Output written to: {args.output_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")