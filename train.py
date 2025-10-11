import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Tuple

# --- CONFIGURATION VARIABLES --- #
# 模型和数据路径
LLM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
UEM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"

# --- 训练和验证文件 ---
TRAIN_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/task_loss_description/train/seed_42/train_split.jsonl"
VAL_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/task_loss_description/train/seed_42/val_split.jsonl"
CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt/LoRA_CrossAttention_SeparateTeacher_Fixed"

# 训练超参数
LEARNING_RATE = 5e-6
EPOCHS = 100
BATCH_SIZE = 1
UEM_DEVICE = "cuda:0"
LLM_DEVICE_STUDENT = "cuda:0"
LLM_DEVICE_TEACHER = "cuda:0"
CLIPPING_NORM = 1.0
DISTILLATION_TEMP = 1.0

# ----- Reddit 系统提示词 -----
REDDIT_PROMPT_SYSTEM = "You are an expert in analyzing Reddit posts to infer user's occupations."
REDDIT_PROMPT_USER = """
Based on the conversational context provided by the [Question Asked] and the [User Response], your goal is to infer the user's occupation.
Your response MUST be a single occupation name, without any additional text or explanation.
[Question Asked]: {question_asked}
[User Response]: {user_response}
Your answer:
"""

# _utils/model.py 里有 load_frozen_llm 函数
def load_frozen_llm(model_path: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model from: {model_path} onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

class GetDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.data = [json.loads(line) for line in open(jsonl_path, 'r')]
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        return {
            "questions_asked": item["question_asked"],
            "original_texts": item["response"],
            "anonymized_texts": item["anonymized_response"],
            "loss_description_sentences": item["loss_description_sentence"],
        }

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.query, self.key, self.value, self.out = [nn.Linear(hidden_size, hidden_size) for _ in range(4)]
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self, x, context):
        residual, orig_dtype = x, x.dtype
        x, context = x.to(self.layer_norm.weight.dtype), context.to(self.layer_norm.weight.dtype)
        x_norm = self.layer_norm(x)
        q = self.query(x_norm).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(context.size(0), context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(context.size(0), context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        weighted_avg = torch.matmul(attention, v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return (self.out(weighted_avg).to(orig_dtype) + residual)

class TrainableEnhancer(nn.Module):
    def __init__(self, uem_path: str, uem_tokenizer, llm_student, llm_teacher, llm_tokenizer):
        super().__init__()
        self.llm_student, self.llm_teacher, self.llm_tokenizer, self.uem_tokenizer = llm_student, llm_teacher, llm_tokenizer, uem_tokenizer
        self.uem = AutoModel.from_pretrained(uem_path)
        hidden_size = self.llm_student.config.hidden_size
        self.cross_attention = CrossAttentionLayer(hidden_size)
        self.projection_layer = nn.Sequential(nn.Linear(self.uem.config.hidden_size, self.uem.config.hidden_size * 4), nn.GELU(),
                                              nn.Linear(self.uem.config.hidden_size * 4, hidden_size))

    @torch.no_grad()
    def compute_teacher_outputs(self, questions_asked: List[str], original_texts: List[str]) -> Dict[str, torch.Tensor]:
        prompts = []
        for q, o in zip(questions_asked, original_texts):
            user_prompt = REDDIT_PROMPT_USER.format(question_asked=q, user_response=o)
            conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
            prompts.append(self.llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
        inputs = self.llm_tokenizer(prompts, return_tensors="pt", padding=True).to(self.llm_teacher.device)
        outputs = self.llm_teacher(**inputs)
        return {"logits": outputs.logits.detach(), "input_ids": inputs.input_ids}

    def forward(self, loss_description_sentences: List[str], questions_asked: List[str],
                anonymized_texts: List[str], original_texts: List[str]) -> torch.Tensor:
        # 1. 教师模型处理 (更高效)
        teacher_info = self.compute_teacher_outputs(questions_asked, original_texts)
        teacher_logits = teacher_info["logits"]
        teacher_input_ids = teacher_info["input_ids"]

        # 2. 增强信息提取 (UEM)
        with autocast(enabled=self.uem.device.type == 'cuda'):
            uem_inputs = self.uem_tokenizer(loss_description_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.uem.device)
            uem_outputs = self.uem(**uem_inputs)
            last_hidden_state = uem_outputs.last_hidden_state
            attention_mask = uem_inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            uem_representation = sum_embeddings / sum_mask
            prefix_context = self.projection_layer(uem_representation).unsqueeze(1)
        prefix_context = prefix_context.to(self.llm_student.device, dtype=torch.bfloat16)

        # 3. 精准构建学生输入
        embedding_layer = self.llm_student.get_input_embeddings()
        batch_student_embeds = []
        for q, a_text in zip(questions_asked, anonymized_texts):
            system_conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}]
            user_conv = [{"role": "user", "content": REDDIT_PROMPT_USER.format(question_asked=q, user_response=a_text)}]
            
            system_ids = self.llm_tokenizer(self.llm_tokenizer.apply_chat_template(system_conv, tokenize=False), return_tensors="pt").input_ids.to(self.llm_student.device)
            user_ids = self.llm_tokenizer(self.llm_tokenizer.apply_chat_template(user_conv, tokenize=False, add_generation_prompt=False).replace(self.llm_tokenizer.bos_token, ""), return_tensors="pt").input_ids.to(self.llm_student.device)
            assistant_ids = self.llm_tokenizer(self.llm_tokenizer.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=False).replace(self.llm_tokenizer.bos_token, ""), return_tensors="pt").input_ids.to(self.llm_student.device)

            system_embeds = embedding_layer(system_ids)
            user_embeds = embedding_layer(user_ids)
            assistant_embeds = embedding_layer(assistant_ids)

            enhanced_user_embeds = self.cross_attention(user_embeds, prefix_context)
            
            combined = torch.cat([system_embeds, enhanced_user_embeds, assistant_embeds], dim=1)
            batch_student_embeds.append(combined)
        
        # 对齐批次中的长度
        max_len = max(e.shape[1] for e in batch_student_embeds)
        padded_embeds = []
        attention_masks = []
        for embeds in batch_student_embeds:
            pad_len = max_len - embeds.shape[1]
            padded_embeds.append(F.pad(embeds, (0, 0, pad_len, 0))) # Pad on the left
            attention_masks.append(F.pad(torch.ones_like(embeds[0, :, 0]), (pad_len, 0), value=0))

        student_inputs_embeds = torch.cat(padded_embeds, dim=0)
        student_attention_mask = torch.stack(attention_masks, dim=0)

        # 4. 学生模型前向传播
        student_outputs = self.llm_student(inputs_embeds=student_inputs_embeds, attention_mask=student_attention_mask)
        student_logits = student_outputs.logits

        # 5. 序列级KL散度损失
        teacher_labels = teacher_input_ids.clone()
        teacher_labels[teacher_labels == self.llm_tokenizer.pad_token_id] = -100 # Ignore pad tokens in loss
        
        student_labels = torch.argmax(student_logits, dim=-1) # Just for length alignment, not used in loss
        
        # 对齐 logits 和 labels
        common_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits_aligned = student_logits[:, :common_len]
        teacher_logits_aligned = teacher_logits[:, :common_len]
        teacher_labels_aligned = teacher_labels[:, :common_len]

        # 计算有效的 token mask
        mask = (teacher_labels_aligned != -100)

        student_log_probs = F.log_softmax(student_logits_aligned / DISTILLATION_TEMP, dim=-1)
        teacher_probs = F.softmax(teacher_logits_aligned / DISTILLATION_TEMP, dim=-1)

        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(-1)
        masked_kl = kl_div.where(mask, torch.tensor(0.0).to(kl_div.device))
        
        loss = (masked_kl.sum() / mask.sum()) * (DISTILLATION_TEMP ** 2)
        return loss

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            loss = model(**batch) # 直接传递整个batch
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Devices: UEM@{UEM_DEVICE}, Student@{LLM_DEVICE_STUDENT}, Teacher@{LLM_DEVICE_TEACHER}")

    print("Loading Frozen Teacher LLM...")
    llm_teacher, llm_tokenizer = load_frozen_llm(LLM_MODEL_PATH, LLM_DEVICE_TEACHER)
    llm_teacher.eval()
    for param in llm_teacher.parameters(): param.requires_grad = False

    print("Loading Student Base LLM for LoRA...")
    llm_student_base, _ = load_frozen_llm(LLM_MODEL_PATH, LLM_DEVICE_STUDENT)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    llm_student_peft = get_peft_model(llm_student_base, lora_config)
    llm_student_peft.print_trainable_parameters()
    llm_student_peft.gradient_checkpointing_enable()

    uem_tokenizer = AutoTokenizer.from_pretrained(UEM_MODEL_PATH, use_fast=False)
    
    train_dataset, val_dataset = GetDataset(TRAIN_DATA_FILE), GetDataset(VAL_DATA_FILE)
    collate_fn = lambda batch: {k: [d[k] for d in batch] for k in batch[0]}
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TrainableEnhancer(UEM_MODEL_PATH, uem_tokenizer, llm_student_peft, llm_teacher, llm_tokenizer)
    model.uem.to(UEM_DEVICE)
    model.projection_layer.to(UEM_DEVICE)
    model.cross_attention.to(LLM_DEVICE_STUDENT)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    scaler = GradScaler()
    best_val_loss = float('inf')

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training"):
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                loss = model(**batch)
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping."); continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=CLIPPING_NORM)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}. Saving model.")
            model.llm_student.save_pretrained(os.path.join(CKPT_DIR, "student_lora_adapter"))
            torch.save(model.uem.state_dict(), os.path.join(CKPT_DIR, "uem.pth"))
            torch.save(model.projection_layer.state_dict(), os.path.join(CKPT_DIR, "projection_layer.pth"))
            torch.save(model.cross_attention.state_dict(), os.path.join(CKPT_DIR, "cross_attention.pth"))
        else:
            print(f"Val loss did not improve from {best_val_loss:.4f}.")

if __name__ == "__main__":
    main()