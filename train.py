import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
from typing import List, Dict, Tuple

# --- CONFIGURATION VARIABLES --- #
# 模型和数据路径
LLM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
UEM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"

# --- 训练和验证文件 ---
TRAIN_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/train/seed_42/train_split.jsonl"  # 训练数据
VAL_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/train/seed_42/val_split.jsonl"   # 独立的验证数据
CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt/prefix_I"

# 训练超参数
LEARNING_RATE = 1e-6
EPOCHS = 100
BATCH_SIZE = 1
PREFIX_LENGTH = 5
UEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else UEM_DEVICE
CLIPPING_NORM = 1.0
DISTILLATION_TEMP = 1.0
MAX_GEN_TOKENS = 10 # 教师模型生成序列的最大长度

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
    """数据集类，加载 JSONL 文件"""
    # 初始化数据集
    def __init__(self, jsonl_path: str):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    # 获取数据集长度
    def __len__(self):
        return len(self.data)

    # 获取单个数据项
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        return {
            "question_asked": item['question_asked'],
            "original_text": item['response'],
            "anonymized_text": item['anonymized_response'],
            "loss_description_sentence": item['loss_description_sentence']
        }

class TrainableEnhancer(torch.nn.Module):
    '''可训练的增强模型，结合 UEM 和冻结的 LLM'''
    def __init__(self, uem_path: str, uem_tokenizer, llm_model, llm_tokenizer):
        super().__init__()
        self.llm_frozen = llm_model # 冻结的推理 LLM
        self.llm_tokenizer = llm_tokenizer # 推理 LLM 的分词器
        self.uem_tokenizer = uem_tokenizer # UEM 的分词器
        self.uem = AutoModel.from_pretrained(uem_path) # 加载 UEM 模型
        # 投影层, 将 UEM 输出映射到 LLM 的隐藏层维度
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.uem.config.hidden_size, self.uem.config.hidden_size * 4),
            torch.nn.GELU(), # 使用GELU非线性激活函数
            torch.nn.Linear(self.uem.config.hidden_size * 4, self.llm_frozen.config.hidden_size * PREFIX_LENGTH)
        )

        # 将推理 LLM 的参数冻结
        for param in self.llm_frozen.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_teacher_logits(self, questions_asked: List[str], original_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        辅助函数：在无梯度上下文中计算教师模型的 Logits (针对整个生成序列)。
        MODIFIED: This function now generates a target sequence and returns logits for that entire sequence.
        """
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
        teacher_inputs = self.llm_tokenizer(teacher_prompt, return_tensors="pt").to(self.llm_frozen.device)
        
        # 1. 生成教师模型的输出序列 (e.g., "Software", "Engineer")
        generated_ids = self.llm_frozen.generate(
            **teacher_inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        # 提取生成的 token，排除输入部分
        target_ids = generated_ids[:, teacher_inputs.input_ids.shape[1]:]

        # 如果生成了空的序列，则返回空结果以跳过该批次
        if target_ids.shape[1] == 0:
            return None, None

        # 2. 将生成的序列与原始输入拼接，以获取每个生成 token 对应的 logits
        full_input_ids = torch.cat([teacher_inputs.input_ids, target_ids], dim=1)
        
        # 3. 再次进行前向传播以获取整个序列的 logits
        teacher_outputs = self.llm_frozen(input_ids=full_input_ids)
        
        # 4. 提取与目标序列对应的 logits
        # 我们需要从输入结束的位置开始，提取长度为 target_ids 长度的 logits
        # Logits at position i are used to predict token i+1.
        # So, we take logits from (input_len - 1) to (input_len + target_len - 2)
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
        """
        MODIFIED: This function now accepts target_ids and calculates loss over the entire sequence.
        """
        
        # --- 学生流程: 使用前缀注入获取 Logits ---
        # 1. 使用 UEM 和投影层生成前缀嵌入
        with autocast(enabled=self.uem.device.type == 'cuda'):
            uem_inputs = self.uem_tokenizer(loss_description_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.uem.device)
            uem_outputs = self.uem(**uem_inputs)
            sentence_representation = uem_outputs.last_hidden_state[:, 0, :]
            clean_prefix = self.projection_layer(sentence_representation)

        clean_prefix = clean_prefix.view(-1, PREFIX_LENGTH, self.llm_frozen.config.hidden_size)

        # 2. 裁剪前缀的 L2 范数
        original_norm = torch.linalg.norm(clean_prefix, dim=-1, keepdim=True)
        scale_factor = (CLIPPING_NORM / (original_norm + 1e-9)).clamp(max=1.0)
        clipped_prefix = clean_prefix * scale_factor

        # 3. 将前缀嵌入移动到 LLM 的设备上
        prefix_embeds = clipped_prefix.to(device=self.llm_frozen.device, dtype=self.llm_frozen.dtype)

        # 4. 手动为聊天模板的每个部分构建嵌入
        embedding_layer = self.llm_frozen.get_input_embeddings()

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

        # 5. 将每个文本部分转换为嵌入
        system_embeds = embedding_layer(self.llm_tokenizer(system_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        user_start_embeds = embedding_layer(self.llm_tokenizer(user_start_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        user_content_embeds = embedding_layer(self.llm_tokenizer(user_content_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        assistant_start_embeds = embedding_layer(self.llm_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        
        # MODIFIED: 6. Teacher Forcing - 将目标序列的嵌入也拼接进去
        target_embeds = embedding_layer(target_ids)

        combined_embeds = torch.cat([
            system_embeds,
            user_start_embeds,
            prefix_embeds,
            user_content_embeds,
            assistant_start_embeds,
            target_embeds # 将目标序列嵌入拼接到末尾
        ], dim=1)

        # 7. 使用拼接好的嵌入进行前向传播
        student_outputs = self.llm_frozen(inputs_embeds=combined_embeds)
        
        # MODIFIED: 8. 提取与目标序列对应的学生 logits
        start_pos = combined_embeds.shape[1] - target_ids.shape[1] - 1
        end_pos = start_pos + target_ids.shape[1]
        logits_student = student_outputs.logits[:, start_pos:end_pos, :]

        # --- 计算KL散度损失 ---
        # 确保 logits_teacher 和 logits_student 形状一致
        if logits_teacher.shape != logits_student.shape:
             # 如果形状不匹配，这通常意味着存在一个 bug，但为了鲁棒性，我们可以跳过这个批次
            print(f"Warning: Shape mismatch. Teacher: {logits_teacher.shape}, Student: {logits_student.shape}. Skipping batch.")
            return torch.tensor(0.0, device=self.llm_frozen.device, requires_grad=True)

        logits_teacher = logits_teacher.to(logits_student.device)

        # Reshape for KLDivLoss: (N, C) where N is batch*seq_len and C is vocab_size
        student_log_sm = F.log_softmax(logits_student / DISTILLATION_TEMP, dim=-1)
        teacher_sm = F.softmax(logits_teacher / DISTILLATION_TEMP, dim=-1)
        
        # 将序列维度合并到批次维度中，以便 kl_div 正确计算 batchmean
        loss = F.kl_div(
            student_log_sm.view(-1, student_log_sm.size(-1)),
            teacher_sm.view(-1, teacher_sm.size(-1)),
            reduction='batchmean'
        ) * (DISTILLATION_TEMP ** 2)

        return loss

from _utils.model import load_frozen_llm

def evaluate(model, data_loader):
    """在给定的数据集上评估模型"""
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            # MODIFIED: 获取教师 logits 和 target_ids
            logits_teacher, target_ids = model.get_teacher_logits(
                questions_asked=batch['question_asked'],
                original_texts=batch['original_text']
            )
            
            # 如果教师没有生成任何 token，跳过这个样本
            if logits_teacher is None:
                continue
                
            # MODIFIED: 传入 target_ids
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
    """主函数，用于协调训练和验证流程"""
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"Using UEM_DEVICE: {UEM_DEVICE} and LLM_DEVICE: {LLM_DEVICE}")
    print(f"Checkpoints will be saved to: {CKPT_DIR}")
    
    llm_frozen, llm_tokenizer = load_frozen_llm(LLM_MODEL_PATH, LLM_DEVICE)
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

    model = TrainableEnhancer(UEM_MODEL_PATH, uem_tokenizer, llm_frozen, llm_tokenizer)
    model.uem.to(UEM_DEVICE)
    model.projection_layer.to(UEM_DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    best_val_loss = float('inf')

    print("\n--- Starting Knowledge Distillation Training (Sequence-level) ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        model.train()
        total_train_loss = 0
        num_samples_trained = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            optimizer.zero_grad(set_to_none=True)
            
            # MODIFIED: 1. 获取教师 logits 和 target_ids
            logits_teacher, target_ids = model.get_teacher_logits(
                questions_asked=batch['question_asked'],
                original_texts=batch['original_text']
            )

            # 如果教师没有生成任何 token，跳过这个样本
            if logits_teacher is None:
                continue

            # 2. 学生模型前向传播和损失计算
            with autocast():
                # MODIFIED: 传入 target_ids
                loss = model(
                    loss_description_sentences=batch['loss_description_sentence'],
                    questions_asked=batch['question_asked'],
                    anonymized_texts=batch['anonymized_text'],
                    logits_teacher=logits_teacher,
                    target_ids=target_ids
                )

            if torch.isnan(loss):
                print("Warning: Loss is NaN before scaling, skipping update.")
                del logits_teacher, target_ids
                torch.cuda.empty_cache()
                continue

            loss_value = loss.item()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIPPING_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss_value
            num_samples_trained += 1

            del logits_teacher, target_ids, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / num_samples_trained if num_samples_trained > 0 else 0
        
        val_loss = evaluate(model, val_loader)
        
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, ckpt_path)
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {ckpt_path}")
        else:
            print(f"Validation loss did not improve from {best_val_loss:.4f}.")
            
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()