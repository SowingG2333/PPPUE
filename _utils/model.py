import torch # 导入 PyTorch 库
import tqdm # 导入进度条库
from transformers import AutoModelForCausalLM, AutoTokenizer # 导入模型和分词器

def load_frozen_llm(model_path: str, device: str):
    """在特定设备上加载冻结的 LLM，用于训练 UEM"""
    print(f"Loading and freezing reasoning LLM from {model_path} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map=device, 
        trust_remote_code=True
    )
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.to(torch.float16)
    model.eval()
    print("Reasoning LLM loaded and frozen successfully.")
    return model, tokenizer

def load_local_model(model_path: str, device: str):
    '''加载本地模型和分词器，用于评估任务'''
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
    print(f"Local model loaded successfully.")
    return model, tokenizer

def generate_local_response(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        prompt_system: str,
        prompt_user: str, 
        temperature: float, 
        max_new_tokens: int = 256
    ) -> str:
    '''使用本地模型生成响应，用于评估任务'''
    try:
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ] # 构建消息列表
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # 应用聊天模板
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # 编码输入并移动到设备
        # 生成参数
        generate_kwargs = {"temperature": temperature, "max_new_tokens": max_new_tokens}
        if temperature > 0: # 如果温度大于0，启用采样：从概率分布中随机选择下一个词，而不是总是选择概率最高的词
            generate_kwargs["do_sample"] = True

        outputs = model.generate(**inputs, **generate_kwargs)
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:] # 获取生成的IDs
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip() # 解码生成的IDs为字符串
    except Exception as e:
        tqdm.write(f"  - ERROR during local generation: {e}")
        return "LOCAL_GEN_ERROR"