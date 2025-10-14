import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Tuple

# --- CONFIGURATION VARIABLES --- #
LLM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
UEM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"

TRAIN_DATA_FILE = "/home/sowingg/coding/LLM/PPPUE/DB-Bio/benchmark/reprocess/train.jsonl"
VAL_DATA_FILE = "/home/sowingg/coding/LLM/PPPUE/DB-Bio/benchmark/reprocess/test.jsonl"  # 使用 test.jsonl 作为验证集
CKPT_DIR = "/home/sowingg/coding/LLM/PPPUE/DB-Bio/ckpt/bio_prefix_lora"

# 训练超参数
LEARNING_RATE_UEM = 1e-5
LEARNING_RATE_LORA = 1e-4
EPOCHS = 100
BATCH_SIZE = 1
PREFIX_LENGTH = 5
UEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else UEM_DEVICE
CLIPPING_NORM = 1.0
DISTILLATION_TEMP = 1.0
MAX_GEN_TOKENS = 50  # 传记类别名称可能较长

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ----- 传记系统提示词 -----
BIO_PROMPT_SYSTEM = """
You are an expert in analyzing biographical texts to identify a person's profession or occupation.
"""

BIO_PROMPT_USER = """
Based on the provided [Original Biography] and [Anonymized Biography], your goal is to identify the person's primary occupation or professional category.
Your response MUST be a single occupation name (e.g., "Architect", "Tennis Player", "Engineer"), without any additional text or explanation.

[Original Biography]:
{original_biography}

[Anonymized Biography]:
{anonymized_biography}
"""

class BioDataset(Dataset):
    """传记数据集类"""
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        return {
            "original_text": item['text'],
            "anonymized_text": item['anonymized_text'],
            "loss_description_sentence": item['loss_description_sentence'],
            "label": item['label']  # 职业标签
        }

class BioTrainableEnhancer(torch.nn.Module):
    '''传记任务的可训练增强模型'''
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
            for param in self.llm_teacher.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def get_teacher_logits(self, original_texts: List[str], anonymized_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用教师模型生成目标序列和 logits"""
        original_bio = original_texts[0]
        anonymized_bio = anonymized_texts[0]

        teacher_user_prompt = BIO_PROMPT_USER.format(
            original_biography=original_bio,
            anonymized_biography=anonymized_bio
        )
        teacher_conversation = [
            {"role": "system", "content": BIO_PROMPT_SYSTEM},
            {"role": "user", "content": teacher_user_prompt}
        ]
        teacher_prompt = self.llm_tokenizer.apply_chat_template(
            teacher_conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        teacher_inputs = self.llm_tokenizer(teacher_prompt, return_tensors="pt").to(self.llm_teacher.device)
        
        # 生成教师输出
        generated_ids = self.llm_teacher.generate(
            **teacher_inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        target_ids = generated_ids[:, teacher_inputs.input_ids.shape[1]:]

        if target_ids.shape[1] == 0:
            return None, None

        # 获取 logits
        full_input_ids = torch.cat([teacher_inputs.input_ids, target_ids], dim=1)
        teacher_outputs = self.llm_teacher(input_ids=full_input_ids)
        
        start_pos = teacher_inputs.input_ids.shape[1] - 1
        end_pos = start_pos + target_ids.shape[1]
        teacher_logits = teacher_outputs.logits[:, start_pos:end_pos, :]

        return teacher_logits, target_ids

    def forward(self,
                loss_description_sentences: List[str],
                original_texts: List[str],
                anonymized_texts: List[str],
                logits_teacher: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        """学生模型前向传播"""
        
        # 1. 生成前缀嵌入
        with autocast(enabled=self.uem.device.type == 'cuda'):
            uem_inputs = self.uem_tokenizer(
                loss_description_sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512  # 传记文本通常较长
            ).to(self.uem.device)
            uem_outputs = self.uem(**uem_inputs)
            sentence_representation = uem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = self.projection_layer(sentence_representation)

        clean_prefix = clean_prefix.view(-1, PREFIX_LENGTH, self.llm_teacher.config.hidden_size)

        # 2. 裁剪前缀
        original_norm = torch.linalg.norm(clean_prefix, dim=-1, keepdim=True)
        scale_factor = (CLIPPING_NORM / (original_norm + 1e-9)).clamp(max=1.0)
        clipped_prefix = clean_prefix * scale_factor

        # 3. 移动到学生设备
        prefix_embeds = clipped_prefix.to(device=self.llm_student.device, dtype=self.llm_student.dtype)

        # 4. 构建学生输入嵌入
        embedding_layer = self.llm_student.get_input_embeddings()

        student_user_prompt = BIO_PROMPT_USER.format(
            original_biography=original_texts[0],
            anonymized_biography=anonymized_texts[0]
        )

        system_part = self.llm_tokenizer.apply_chat_template(
            [{"role": "system", "content": BIO_PROMPT_SYSTEM}], 
            tokenize=False, 
            add_generation_prompt=False
        )
        user_start_part = self.llm_tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], 
            tokenize=False, 
            add_generation_prompt=False
        ).replace(self.llm_tokenizer.eos_token, '')
        user_content_part = student_user_prompt
        
        dummy_conversation = [{"role": "user", "content": "DUMMY"}]
        full_template_with_dummy = self.llm_tokenizer.apply_chat_template(
            dummy_conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        dummy_template = self.llm_tokenizer.apply_chat_template(
            dummy_conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        assistant_start_part = full_template_with_dummy.replace(dummy_template, "")

        # 5. 转换为嵌入
        system_embeds = embedding_layer(
            self.llm_tokenizer(system_part, return_tensors="pt").input_ids.to(self.llm_student.device)
        )
        user_start_embeds = embedding_layer(
            self.llm_tokenizer(user_start_part, return_tensors="pt").input_ids.to(self.llm_student.device)
        )
        user_content_embeds = embedding_layer(
            self.llm_tokenizer(user_content_part, return_tensors="pt").input_ids.to(self.llm_student.device)
        )
        assistant_start_embeds = embedding_layer(
            self.llm_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(self.llm_student.device)
        )
        target_embeds = embedding_layer(target_ids.to(self.llm_student.device))

        combined_embeds = torch.cat([
            system_embeds,
            user_start_embeds,
            prefix_embeds,
            user_content_embeds,
            assistant_start_embeds,
            target_embeds
        ], dim=1)

        # 6. 前向传播
        student_outputs = self.llm_student(inputs_embeds=combined_embeds)
        
        # 7. 提取 logits
        start_pos = combined_embeds.shape[1] - target_ids.shape[1] - 1
        end_pos = start_pos + target_ids.shape[1]
        logits_student = student_outputs.logits[:, start_pos:end_pos, :]
        
        # 8. 类型转换和裁剪
        logits_student = logits_student.to(dtype=torch.float32)
        logits_teacher = logits_teacher.to(device=logits_student.device, dtype=torch.float32)
        
        logits_student = torch.clamp(logits_student, min=-10.0, max=10.0)
        logits_teacher = torch.clamp(logits_teacher, min=-10.0, max=10.0)
        
        # 9. 计算 KL 散度损失
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

def load_frozen_llm(model_path, device):
    """加载冻结的 LLM"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map={"": device}
    )
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

def evaluate(model, data_loader):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            logits_teacher, target_ids = model.get_teacher_logits(
                original_texts=batch['original_text'],
                anonymized_texts=batch['anonymized_text']
            )
            
            if logits_teacher is None:
                continue
                
            loss = model(
                loss_description_sentences=batch['loss_description_sentence'],
                original_texts=batch['original_text'],
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
    
    # 加载教师模型
    llm_teacher, llm_tokenizer = load_frozen_llm(LLM_MODEL_PATH, LLM_DEVICE)
    
    # 加载学生模型并应用 LoRA
    from transformers import AutoModelForCausalLM
    print("Loading student model with LoRA...")
    llm_student = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map={"": LLM_DEVICE}
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )
    
    llm_student = get_peft_model(llm_student, lora_config)
    llm_student.print_trainable_parameters()
    
    try:
        uem_tokenizer = AutoTokenizer.from_pretrained(UEM_MODEL_PATH, use_fast=False)
    except ValueError:
        uem_tokenizer = XLMRobertaTokenizer.from_pretrained(UEM_MODEL_PATH)

    train_dataset = BioDataset(TRAIN_DATA_FILE)
    val_dataset = BioDataset(VAL_DATA_FILE)
    
    print(f"Dataset loaded: {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    def collate_fn(batch):
        return {
            'loss_description_sentence': [item['loss_description_sentence'] for item in batch],
            'original_text': [item['original_text'] for item in batch],
            'anonymized_text': [item['anonymized_text'] for item in batch],
            'label': [item['label'] for item in batch]
        }

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BioTrainableEnhancer(UEM_MODEL_PATH, uem_tokenizer, llm_teacher, llm_student, llm_tokenizer)
    model.uem.to(UEM_DEVICE)
    model.projection_layer.to(UEM_DEVICE)
    
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

    print(f"\n--- Starting Biography Knowledge Distillation Training with LoRA ---")
    print(f"UEM Learning Rate: {LEARNING_RATE_UEM}")
    print(f"LoRA Learning Rate: {LEARNING_RATE_LORA}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        num_samples_trained = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            logits_teacher, target_ids = model.get_teacher_logits(
                original_texts=batch['original_text'],
                anonymized_texts=batch['anonymized_text']
            )
            
            if logits_teacher is None:
                continue
                
            optimizer.zero_grad()
            loss = model(
                loss_description_sentences=batch['loss_description_sentence'],
                original_texts=batch['original_text'],
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
        
        scheduler.step()
        
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Current LR - UEM: {optimizer.param_groups[0]['lr']:.2e}, LoRA: {optimizer.param_groups[2]['lr']:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_ckpt_dir = os.path.join(CKPT_DIR, f"best_model_epoch_{epoch + 1}")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)

            # 保存 LoRA 适配器
            lora_path = os.path.join(epoch_ckpt_dir, "lora_adapters")
            model.llm_student.save_pretrained(lora_path)

            # 保存 UEM 和投影层
            other_weights_path = os.path.join(epoch_ckpt_dir, "uem_projection.pt")
            torch.save({
                'uem_state_dict': model.uem.state_dict(),
                'projection_layer_state_dict': model.projection_layer.state_dict()
            }, other_weights_path)
            
            print(f"Saved decoupled model components to {epoch_ckpt_dir}")

    print("Training complete.")

if __name__ == "__main__":
    main()