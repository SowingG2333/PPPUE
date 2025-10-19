import os # 用于文件和目录操作
import json # 用于处理 JSON 数据
import torch # PyTorch 库
import torch.nn.functional as F # 用于神经网络功能
from torch.utils.data import Dataset, DataLoader # 数据集和数据加载器
from torch.optim import AdamW # 优化器
from torch.optim.lr_scheduler import CosineAnnealingLR # 学习率调度器（余弦退火）
from torch.cuda.amp import autocast # 自动混合精度
from tqdm import tqdm # 进度条显示
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, XLMRobertaTokenizer # 预训练模型和分词器
from peft import LoraConfig, get_peft_model, TaskType  # LoRA 相关
from typing import List, Dict, Tuple # 类型提示

# --- CONFIGURATION VARIABLES --- #
LLM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
UEM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"

TRAIN_DATA_FILE = "/root/autodl-tmp/PPPUE/personalReddit/benchmark/reprocess/train/seed_42/train_split.jsonl"
VAL_DATA_FILE = "/root/autodl-tmp/PPPUE/personalReddit/benchmark/reprocess/train/seed_42/val_split.jsonl"
CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt/prefix_lora"

# 训练超参数
LEARNING_RATE_UEM = 1e-5  # UEM 和投影层的学习率
LEARNING_RATE_LORA = 1e-4  # LoRA 的学习率
EPOCHS = 100 # 训练轮数
BATCH_SIZE = 1 # 批量大小
PREFIX_LENGTH = 5 # 前缀长度
UEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # UEM 设备
LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else UEM_DEVICE # LLM 设备
CLIPPING_NORM = 1.0 # 梯度裁剪范数
DISTILLATION_TEMP = 1.0 # 知识蒸馏温度
MAX_GEN_TOKENS = 10 # 教师模型生成的最大 token 数

# LoRA 配置
LORA_R = 16 # LoRA 秩
LORA_ALPHA = 32 # LoRA 缩放因子
LORA_DROPOUT = 0.1 # LoRA dropout 率
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"] # 目标模块列表

# ----- Reddit 系统提示词 -----
REDDIT_PROMPT_SYSTEM = """
You are an expert in analyzing Reddit posts to infer user's occupations.
"""

REDDIT_PROMPT_USER = """
Based on the conversational context provided by the [Question Asked] and the [User Response], your goal is to infer the user's occupation.
Your response MUST be a single occupation name, without any additional text or explanation.

[Question Asked]:
{question_asked}

[User Response]:
{user_response}
"""

class GetDataset(Dataset):
    """数据集类,加载 JSONL 文件"""
    def __init__(self, jsonl_path: str):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        return {
            "question_asked": item['question_asked'],
            "original_text": item['response'],
            "anonymized_text": item['anonymized_response'],
            "loss_description_sentence": item['loss_description_sentence']
        }

class TrainableEnhancer(torch.nn.Module):
    '''可训练的增强模型,结合 UEM 和独立的教师/学生 LLM'''
    def __init__(self, uem_path: str, uem_tokenizer, llm_teacher, llm_student, llm_tokenizer):
        super().__init__()
        self.llm_teacher = llm_teacher
        self.llm_student = llm_student
        self.llm_tokenizer = llm_tokenizer
        self.uem_tokenizer = uem_tokenizer
        self.uem = AutoModel.from_pretrained(uem_path)
        
        # 投影层
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.uem.config.hidden_size, self.uem.config.hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(self.uem.config.hidden_size * 4, self.llm_student.config.hidden_size * PREFIX_LENGTH)
        )
        
        if llm_teacher is not None:
            # 冻结教师模型
            for param in self.llm_teacher.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def get_teacher_logits(self, questions_asked: List[str], original_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用教师模型生成目标序列和 logits"""
        user_response = original_texts[0]
        question_asked = questions_asked[0]

        teacher_user_prompt = REDDIT_PROMPT_USER.format(
            question_asked=question_asked,
            user_response=user_response
        )
        teacher_conversation = [
            {"role": "system", "content": REDDIT_PROMPT_SYSTEM},
            {"role": "user", "content": teacher_user_prompt}
        ]
        teacher_prompt = self.llm_tokenizer.apply_chat_template(teacher_conversation, tokenize=False, add_generation_prompt=True)
        teacher_inputs = self.llm_tokenizer(teacher_prompt, return_tensors="pt").to(self.llm_teacher.device)
        
        # 生成教师模型的输出序列
        generated_ids = self.llm_teacher.generate(
            **teacher_inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        target_ids = generated_ids[:, teacher_inputs.input_ids.shape[1]:]

        if target_ids.shape[1] == 0:
            return None, None

        # 获取完整序列的 logits
        full_input_ids = torch.cat([teacher_inputs.input_ids, target_ids], dim=1)
        teacher_outputs = self.llm_teacher(input_ids=full_input_ids)
        
        start_pos = teacher_inputs.input_ids.shape[1] - 1
        end_pos = start_pos + target_ids.shape[1]
        teacher_logits = teacher_outputs.logits[:, start_pos:end_pos, :]

        return teacher_logits, target_ids

    def forward(self,
                loss_description_sentences: List[str],
                questions_asked: List[str],
                anonymized_texts: List[str],
                logits_teacher: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        """学生模型前向传播(使用 LoRA 微调的模型)"""
        
        # 1. 使用 UEM 生成前缀嵌入
        with autocast(enabled=self.uem.device.type == 'cuda'):
            uem_inputs = self.uem_tokenizer(loss_description_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.uem.device)
            uem_outputs = self.uem(**uem_inputs)
            sentence_representation = uem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = self.projection_layer(sentence_representation)

        clean_prefix = clean_prefix.view(-1, PREFIX_LENGTH, self.llm_teacher.config.hidden_size)

        # 2. 裁剪前缀的 L2 范数
        original_norm = torch.linalg.norm(clean_prefix, dim=-1, keepdim=True)
        scale_factor = (CLIPPING_NORM / (original_norm + 1e-9)).clamp(max=1.0)
        clipped_prefix = clean_prefix * scale_factor

        # 3. 移动到学生模型设备
        prefix_embeds = clipped_prefix.to(device=self.llm_student.device, dtype=self.llm_student.dtype)

        # 4. 构建学生模型输入嵌入
        embedding_layer = self.llm_student.get_input_embeddings()

        student_user_prompt = REDDIT_PROMPT_USER.format(
            question_asked=questions_asked[0],
            user_response=anonymized_texts[0]
        )

        system_part = self.llm_tokenizer.apply_chat_template([{"role": "system", "content": REDDIT_PROMPT_SYSTEM}], tokenize=False, add_generation_prompt=False)
        user_start_part = self.llm_tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False).replace(self.llm_tokenizer.eos_token, '')
        user_content_part = student_user_prompt
        
        dummy_conversation = [{"role": "user", "content": "DUMMY"}]
        full_template_with_dummy = self.llm_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, add_generation_prompt=True)
        dummy_template = self.llm_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, add_generation_prompt=False)
        assistant_start_part = full_template_with_dummy.replace(dummy_template, "")

        # 5. 转换为嵌入
        system_embeds = embedding_layer(self.llm_tokenizer(system_part, return_tensors="pt").input_ids.to(self.llm_student.device))
        user_start_embeds = embedding_layer(self.llm_tokenizer(user_start_part, return_tensors="pt").input_ids.to(self.llm_student.device))
        user_content_embeds = embedding_layer(self.llm_tokenizer(user_content_part, return_tensors="pt").input_ids.to(self.llm_student.device))
        assistant_start_embeds = embedding_layer(self.llm_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(self.llm_student.device))
        target_embeds = embedding_layer(target_ids.to(self.llm_student.device))

        combined_embeds = torch.cat([
            system_embeds,
            user_start_embeds,
            prefix_embeds,
            user_content_embeds,
            assistant_start_embeds,
            target_embeds
        ], dim=1)

        # 6. 学生模型前向传播(带 LoRA)
        student_outputs = self.llm_student(inputs_embeds=combined_embeds)
        
        # 7. 提取学生 logits
        start_pos = combined_embeds.shape[1] - target_ids.shape[1] - 1
        end_pos = start_pos + target_ids.shape[1]
        logits_student = student_outputs.logits[:, start_pos:end_pos, :]
        
        # 8. 确保数据类型一致，并 clip logits 以防止溢出
        logits_student = logits_student.to(dtype=torch.float32)
        logits_teacher = logits_teacher.to(device=logits_student.device, dtype=torch.float32)
        
        # 9. 裁剪教师模型和学生模型的 Logits
        logits_student = torch.clamp(logits_student, min=-10.0, max=10.0)
        logits_teacher = torch.clamp(logits_teacher, min=-10.0, max=10.0)
        
        # 10. 计算 KL 散度损失
        if logits_teacher.shape != logits_student.shape:
            print(f"Warning: Shape mismatch. Teacher: {logits_teacher.shape}, Student: {logits_student.shape}. Skipping batch.")
            return torch.tensor(0.0, device=logits_student.device, requires_grad=True)

        student_log_sm = F.log_softmax(logits_student / DISTILLATION_TEMP, dim=-1)
        teacher_sm = F.softmax(logits_teacher / DISTILLATION_TEMP, dim=-1)
        
        loss = F.kl_div(
            student_log_sm.view(-1, student_log_sm.size(-1)),
            teacher_sm.view(-1, teacher_sm.size(-1)),
            reduction='batchmean'
        ) * (DISTILLATION_TEMP ** 2)

        return loss

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

def evaluate(model, data_loader):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            logits_teacher, target_ids = model.get_teacher_logits(
                questions_asked=batch['question_asked'],
                original_texts=batch['original_text']
            )
            
            if logits_teacher is None:
                continue
                
            loss = model(
                loss_description_sentences=batch['loss_description_sentence'],
                questions_asked=batch['question_asked'],
                anonymized_texts=batch['anonymized_text'],
                logits_teacher=logits_teacher,
                target_ids=target_ids
            )
            total_loss += loss.item()
            num_samples += 1
            
            del logits_teacher, target_ids
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    return avg_loss

def main():
    """主函数"""
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"Using UEM_DEVICE: {UEM_DEVICE} and LLM_DEVICE: {LLM_DEVICE}")
    print(f"Checkpoints will be saved to: {CKPT_DIR}")
    
    # 加载冻结的教师模型
    llm_teacher, llm_tokenizer = load_frozen_llm(LLM_MODEL_PATH, LLM_DEVICE)
    
    # 加载学生模型并应用 LoRA
    from transformers import AutoModelForCausalLM
    print("Loading student model with LoRA...")
    llm_student = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map={"": LLM_DEVICE}
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )
    
    # 应用 LoRA
    llm_student = get_peft_model(llm_student, lora_config)
    llm_student.print_trainable_parameters()
    
    try:
        uem_tokenizer = AutoTokenizer.from_pretrained(UEM_MODEL_PATH, use_fast=False)
    except ValueError:
        uem_tokenizer = XLMRobertaTokenizer.from_pretrained(UEM_MODEL_PATH)

    train_dataset = GetDataset(TRAIN_DATA_FILE)
    val_dataset = GetDataset(VAL_DATA_FILE)
    
    print(f"Dataset loaded: {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    def collate_fn(batch):
        return {
            'loss_description_sentence': [item['loss_description_sentence'] for item in batch],
            'question_asked': [item['question_asked'] for item in batch],
            'original_text': [item['original_text'] for item in batch],
            'anonymized_text': [item['anonymized_text'] for item in batch]
        }

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TrainableEnhancer(UEM_MODEL_PATH, uem_tokenizer, llm_teacher, llm_student, llm_tokenizer)
    model.uem.to(UEM_DEVICE)
    model.projection_layer.to(UEM_DEVICE)
    
    # 优化器:为不同模块设置不同的学习率
    optimizer = AdamW([
        {
            'params': model.uem.parameters(),
            'lr': LEARNING_RATE_UEM,
            'weight_decay': 0.01
        },
        {
            'params': model.projection_layer.parameters(),
            'lr': LEARNING_RATE_UEM,
            'weight_decay': 0.01
        },
        {
            'params': model.llm_student.parameters(),
            'lr': LEARNING_RATE_LORA,
            'weight_decay': 0.01
        }
    ])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    best_val_loss = float('inf')

    print(f"\n--- Starting Knowledge Distillation Training with LoRA ---")
    print(f"UEM Learning Rate: {LEARNING_RATE_UEM}")
    print(f"LoRA Learning Rate: {LEARNING_RATE_LORA}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        num_samples_trained = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            logits_teacher, target_ids = model.get_teacher_logits(
                questions_asked=batch['question_asked'],
                original_texts=batch['original_text']
            )
            
            if logits_teacher is None:
                continue
                
            optimizer.zero_grad()
            loss = model(
                loss_description_sentences=batch['loss_description_sentence'],
                questions_asked=batch['question_asked'],
                anonymized_texts=batch['anonymized_text'],
                logits_teacher=logits_teacher,
                target_ids=target_ids
            )
            total_train_loss += loss.item()
            num_samples_trained += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPPING_NORM)
            optimizer.step()
            
            del logits_teacher, target_ids
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / num_samples_trained if num_samples_trained > 0 else 0
        val_loss = evaluate(model, val_loader)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Current LR - UEM: {optimizer.param_groups[0]['lr']:.2e}, LoRA: {optimizer.param_groups[2]['lr']:.2e}")

        # 保存最佳模型 (解耦保存)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 创建一个目录来保存该 epoch 的所有组件
            epoch_ckpt_dir = os.path.join(CKPT_DIR, f"best_model")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)

            # 1. 保存 LoRA 适配器权重
            lora_path = os.path.join(epoch_ckpt_dir, "lora_adapters")
            model.llm_student.save_pretrained(lora_path)

            # 2. 保存 UEM 和投影层的权重
            other_weights_path = os.path.join(epoch_ckpt_dir, "uem_projection.pt")
            torch.save({
                'uem_state_dict': model.uem.state_dict(),
                'projection_layer_state_dict': model.projection_layer.state_dict()
            }, other_weights_path)
            
            print(f"Saved decoupled model components to {epoch_ckpt_dir}")

    print("Training complete.")

if __name__ == "__main__":
    main()