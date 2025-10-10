import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict

# --- CONFIGURATION VARIABLES --- #
# 模型和数据路径
LLM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
UEM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"

# --- 训练和验证文件 ---
TRAIN_DATA_FILE = "/root/autodl-tmp/PAUE-II/benchmark/reprocess/train/distinct/train_val_grouped.jsonl"  # 使用增强后的训练数据
VAL_DATA_FILE = "/root/autodl-tmp/PAUE-II/benchmark/reprocess/test/test_anony_with_loss.jsonl"   # 独立的验证数据
CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt"   

# 训练超参数
LEARNING_RATE = 1e-5
EPOCHS = 100
BATCH_SIZE = 4
PREFIX_LENGTH = 5 
UEM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LLM_DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else UEM_DEVICE
CLIPPING_NORM = 1.0
DISTILLATION_TEMP = 1.0

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
    def get_teacher_logits(self, questions_asked: List[str], original_texts: List[str]) -> torch.Tensor:
        """辅助函数：在无梯度上下文中计算教师模型的 Logits"""
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
        teacher_outputs = self.llm_frozen(**teacher_inputs)
        return teacher_outputs.logits[:, -1, :]

    def forward(self, 
                loss_description_sentences: List[str],
                questions_asked: List[str], 
                anonymized_texts: List[str],
                logits_teacher: torch.Tensor) -> torch.Tensor:
        
        # --- 学生流程: 使用前缀注入获取 Logits ---
        # 1. 使用 UEM 和投影层生成前缀嵌入
        with autocast(enabled=self.uem.device.type == 'cuda'):
            uem_inputs = self.uem_tokenizer(loss_description_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.uem.device)
            uem_outputs = self.uem(**uem_inputs)
            # 使用 [CLS] 标记的隐藏状态作为句子表示
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
        
        # 统一学生和教师的提示结构
        student_user_prompt = REDDIT_PROMPT_USER.format(
            question_asked=questions_asked[0],
            user_response=anonymized_texts[0]
        )
        
        system_part = self.llm_tokenizer.apply_chat_template([{"role": "system", "content": REDDIT_PROMPT_SYSTEM}], tokenize=False, add_generation_prompt=False)
        user_start_part = self.llm_tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False).replace(self.llm_tokenizer.eos_token, '')
        user_content_part = student_user_prompt # 使用与教师结构一致的用户提示
        
        dummy_conversation = [{"role": "user", "content": "DUMMY"}]
        full_template_with_dummy = self.llm_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, add_generation_prompt=True)
        dummy_template = self.llm_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, add_generation_prompt=False)
        assistant_start_part = full_template_with_dummy.replace(dummy_template, "")

        # 5. 将每个文本部分转换为嵌入
        system_embeds = embedding_layer(self.llm_tokenizer(system_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        user_start_embeds = embedding_layer(self.llm_tokenizer(user_start_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        user_content_embeds = embedding_layer(self.llm_tokenizer(user_content_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))
        assistant_start_embeds = embedding_layer(self.llm_tokenizer(assistant_start_part, return_tensors="pt").input_ids.to(self.llm_frozen.device))

        # 6. 按正确顺序连接所有部分
        combined_embeds = torch.cat([
            system_embeds,
            user_start_embeds,
            prefix_embeds,
            user_content_embeds,
            assistant_start_embeds
        ], dim=1)

        # 7. 使用拼接好的嵌入进行前向传播
        student_outputs = self.llm_frozen(inputs_embeds=combined_embeds)
        logits_student = student_outputs.logits[:, -1, :]
        
        # --- 计算KL散度损失 ---
        # 将教师 logits 移动到学生 logits 所在的设备以进行损失计算
        logits_teacher = logits_teacher.to(logits_student.device)
        
        loss = F.kl_div(
            F.log_softmax(logits_student / DISTILLATION_TEMP, dim=-1),
            F.softmax(logits_teacher / DISTILLATION_TEMP, dim=-1),
            reduction='batchmean'
        ) * (DISTILLATION_TEMP ** 2)
        
        return loss

from _utils.model import load_frozen_llm

def evaluate(model, data_loader):
    """在给定的数据集上评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            # 分离教师和学生计算
            logits_teacher = model.get_teacher_logits(
                questions_asked=batch['question_asked'],
                original_texts=batch['original_text']
            )
            loss = model(
                loss_description_sentences=batch['loss_description_sentence'],
                questions_asked=batch['question_asked'],
                anonymized_texts=batch['anonymized_text'],
                logits_teacher=logits_teacher
            )
            total_loss += loss.item()
            
            # 尝试释放显存
            del logits_teacher
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def main():
    """主函数，用于协调训练和验证流程"""
    os.makedirs(CKPT_DIR, exist_ok=True) # 创建检查点目录
    
    print(f"Using UEM_DEVICE: {UEM_DEVICE} and LLM_DEVICE: {LLM_DEVICE}")
    print(f"Checkpoints will be saved to: {CKPT_DIR}")
    # 加载模型和分词器
    llm_frozen, llm_tokenizer = load_frozen_llm(LLM_MODEL_PATH, LLM_DEVICE)
    uem_tokenizer = AutoTokenizer.from_pretrained(UEM_MODEL_PATH)
    
    # --- 分别从特定文件加载训练集和验证集 ---
    train_dataset = GetDataset(TRAIN_DATA_FILE)
    val_dataset = GetDataset(VAL_DATA_FILE)
    
    print(f"Dataset loaded: {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    def collate_fn(batch):
        '''自定义数据整理函数'''
        return {
            'loss_description_sentence': [item['loss_description_sentence'] for item in batch],
            'question_asked': [item['question_asked'] for item in batch],
            'original_text': [item['original_text'] for item in batch],
            'anonymized_text': [item['anonymized_text'] for item in batch]
        }

    # 划分训练和验证数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 初始化可训练的增强模型
    model = TrainableEnhancer(UEM_MODEL_PATH, uem_tokenizer, llm_frozen, llm_tokenizer)
    # 将 UEM 和投影层移动到指定设备
    model.uem.to(UEM_DEVICE)
    model.projection_layer.to(UEM_DEVICE)
    # 加载优化器和梯度缩放器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # 用于混合精度训练的梯度缩放器
    # 记录最佳验证损失
    best_val_loss = float('inf')
    # --- 训练循环 ---
    print("\n--- Starting Knowledge Distillation Training ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        model.train()
        total_train_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            optimizer.zero_grad(set_to_none=True)
            
            # 1. 计算教师 logits (无梯度)
            logits_teacher = model.get_teacher_logits(
                questions_asked=batch['question_asked'],
                original_texts=batch['original_text']
            )
            
            # 2. 学生模型前向传播和损失计算 (有梯度)
            with autocast():
                loss = model(
                    loss_description_sentences=batch['loss_description_sentence'],
                    questions_asked=batch['question_asked'],
                    anonymized_texts=batch['anonymized_text'],
                    logits_teacher=logits_teacher
                )

            if torch.isnan(loss):
                print("Warning: Loss is NaN before scaling, skipping update.")
                continue

            loss_value = loss.item()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss_value

            del logits_teacher, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
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