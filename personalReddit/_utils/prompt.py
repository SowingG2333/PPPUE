# ----- 匿名化 LLM 提示词模板 -----
PROMPT_ANONYMIZER_SYSTEM = """
You are an expert in text anonymization, tasked with protecting user privacy. 
"""

PROMPT_ANONYMIZER_USER = """
Your objective is to rewrite the [User Response] so that the personal attributes listed in the [True Personality Profile] cannot be inferred. 
The rewritten response must still be a natural and coherent answer to the [Question Asked]. 
You will receive [Previous Feedback from Attacker] if your last attempt failed. Use it to improve your anonymization.
Now, provide ONLY your new, single, anonymized version of the response. Do not add any extra commentary or preamble.

[True Personality Profile]:
{profile_to_hide}

[Question Asked]:
{question_asked}

[Previous Feedback from Attacker]:
{feedback}

[User Response]:
{original_response}
"""

# ----- 攻击者 LLM 提示词模板 -----
PROMPT_ATTACKER_SYSTEM = """
You are an expert profiler with deep knowledge of psycho-linguistics and cultural cues. 
"""

PROMPT_ATTACKER_USER = """
Based on the conversational context provided by the [Question Asked] and the user's [Anonymized Response], your goal is to infer the user's personality profile.

Your response MUST be a single, valid JSON object. It must contain your best guess for each of the following keys: "age", "sex", "city_country", "birth_city_country", "education", "occupation", "income_level", "relationship_status".
- For "age", provide an integer.
- For all other fields, provide a string.
- If you cannot reasonably infer an attribute, use the JSON value `null`.

--- EXAMPLE ---
[Question Asked]:
What's your favorite season and why?

[Anonymized Response]:
I love it when the weather gets cooler and the leaves start to change color. It's a great time for long walks outdoors.

JSON Output:
{{
  "age": 35,
  "sex": null,
  "city_country": null,
  "birth_city_country": null,
  "education": "Bachelors",
  "occupation": null,
  "income_level": "middle",
  "relationship_status": null
}}
--- END EXAMPLE ---

[Question Asked]:
{question_asked}

[Anonymized Response]:
{anonymized_response}

JSON Output:
"""

# ----- 评估 LLM 提示词模板 -----
PROMPT_JUDGE = """
You are a highly precise evaluation expert. Your task is to determine if the "Generated Answer" correctly identifies the "Ground Truth Profession".
Follow these steps:
1. Identify the core profession mentioned in the "Generated Answer".
2. Compare it to the "Ground Truth Profession". It is "Correct" if it is an exact match or a clear synonym (e.g., "Soccer Player" is a synonym for "Footballer").
3. Output your final verdict as a single word: "Correct" or "Incorrect".

Example 1:
Ground Truth Profession: "Architect"
Generated Answer: "The person is likely an Engineer."
Your verdict: Incorrect

Example 2:
Ground Truth Profession: "Soccer Player"
Generated Answer: "Based on the text, I believe this person is a Footballer."
Your verdict: Correct

Example 3:
Ground Truth Profession: "Chef"
Generated Answer: "Comedy"
Your verdict: Incorrect

Ground Truth Profession: "{ground_truth}"

Generated Answer: "{generated_answer}"

Your verdict:
"""

# ----- 信息损失 LLM 提示词模板 -----
PROMPT_LOSS_SYSTEM = """
You are a professional linguistic analyst.
"""

PROMPT_LOSS_USER = """
Compare the [Original Text] and the [Anonymized Text]. Concisely describe what information was lost or generalized during the anonymization process in key words.

Example 1:
[Original Text]: "As a 35-year-old software engineer from Berlin, I enjoy hiking."
[Anonymized Text]: "As a professional in my thirties from a European city, I enjoy outdoor activities."
Your Description: 35-year-old, software engineer, and Berlin.

Example 2:
[Original Text]: "My favorite band is Queen, and I saw them live in 1986."
[Anonymized Text]: "I enjoy classic rock music and have seen some legendary bands live."
Your Description: Queen band and 1986.

[Original Text]:
{original_text}

[Anonymized Text]:
{anonymized_text}

Your Description:
"""