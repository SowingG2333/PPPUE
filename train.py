import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Tuple, Optional, Union
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

# --- CONFIGURATION VARIABLES --- #
LLM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
UEM_MODEL_PATH = "/root/autodl-tmp/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"
TRAIN_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/task_loss_description/train/seed_42/train_split.jsonl"
VAL_DATA_FILE = "/root/autodl-tmp/PPPUE/benchmark/reprocess/task_loss_description/train/seed_42/val_split.jsonl"
CKPT_DIR = "/root/autodl-tmp/PPPUE/ckpt"

LEARNING_RATE = 5e-6
EPOCHS = 100
BATCH_SIZE = 1
UEM_DEVICE = "cuda:0"
LLM_DEVICE_STUDENT = "cuda:0"
LLM_DEVICE_TEACHER = "cuda:0"
CLIPPING_NORM = 1.0
DISTILLATION_TEMP = 1.0

REDDIT_PROMPT_SYSTEM = "You are an expert in analyzing Reddit posts to infer user's occupations."
REDDIT_PROMPT_USER = """
Based on the conversational context provided by the [Question Asked] and the [User Response], your goal is to infer the user's occupation.
Your response MUST be a single occupation name, without any additional text or explanation.
[Question Asked]: {question_asked}
[User Response]: {user_response}
Your answer:
"""

# --- 交叉注意力适配器模块 ---
class CrossAttentionAdapter(nn.Module):
    def __init__(self, hidden_size, context_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(context_size, hidden_size)
        self.value_proj = nn.Linear(context_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, context):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.shape
        q = self.query_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(context).view(context.size(0), context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(context).view(context.size(0), context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_probs, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.output_proj(attention_output)
        return self.layer_norm(output + residual)

# --- 自定义解码器层 ---
class PerLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.cross_attention_adapter = CrossAttentionAdapter(
            hidden_size=config.hidden_size,
            context_size=config.hidden_size
        )
        # 禁用 gradient checkpointing
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # 调用父类 forward
        raw_outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        # 分离当前层的 hidden_states 和其他输出
        if isinstance(raw_outputs, tuple):
            current_hidden_states = raw_outputs[0]
            other_outputs = raw_outputs[1:]
        else:
            current_hidden_states = raw_outputs
            other_outputs = ()

        # 从层属性获取 context
        context = getattr(self, '_per_context', None)
        if context is not None:
            enhanced_hidden_states = self.cross_attention_adapter(
                current_hidden_states, 
                context
            )
        else:
            enhanced_hidden_states = current_hidden_states
        
        return (enhanced_hidden_states,) + other_outputs

# --- 3. 自定义 Llama 模型 ---
class PerLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            PerLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        # 禁用 gradient checkpointing
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 移除 cache_position 参数
        **kwargs,
    ):
        # 提取 context 并分发到所有层
        context = kwargs.pop("context", None)
        if context is not None:
            for layer in self.layers:
                layer._per_context = context
        
        # 调用父类 forward (不传递 cache_position)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,  # 其他参数
        )
        
        # 清理 context
        if context is not None:
            for layer in self.layers:
                layer._per_context = None
        
        return outputs

# --- 顶层因果语言模型 ---
class PerLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = PerLlamaModel(config)
        self.supports_gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        context: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            context=context,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class TrainableEnhancer(nn.Module):
    def __init__(self, uem_path: str, uem_tokenizer, llm_hidden_size: int):
        super().__init__()
        self.uem_tokenizer = uem_tokenizer
        self.uem = AutoModel.from_pretrained(uem_path)
        self.projection_layer = nn.Sequential(
            nn.Linear(self.uem.config.hidden_size, self.uem.config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.uem.config.hidden_size * 4, llm_hidden_size)
        )

    def forward(self, loss_description_sentences: List[str]) -> torch.Tensor:
        uem_inputs = self.uem_tokenizer(loss_description_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.uem.device)
        with torch.amp.autocast(device_type=self.uem.device.type, enabled=self.uem.device.type == 'cuda'):
            uem_outputs = self.uem(**uem_inputs)
            last_hidden_state = uem_outputs.last_hidden_state
            attention_mask = uem_inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            uem_representation = sum_embeddings / sum_mask
            context_vector = self.projection_layer(uem_representation).unsqueeze(1)
        return context_vector.to(dtype=torch.bfloat16)

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

def compute_distillation_loss(student_logits, teacher_logits, teacher_labels, temperature):
    common_len = min(student_logits.shape[1], teacher_logits.shape[1])
    student_logits_aligned = student_logits[:, :common_len]
    teacher_logits_aligned = teacher_logits[:, :common_len]
    teacher_labels_aligned = teacher_labels[:, :common_len]
    mask = (teacher_labels_aligned != -100)
    student_log_probs = F.log_softmax(student_logits_aligned / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits_aligned / temperature, dim=-1)
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(-1)
    masked_kl = kl_div.where(mask, torch.tensor(0.0, device=kl_div.device))
    loss = (masked_kl.sum() / mask.sum()) * (temperature ** 2)
    return loss

def evaluate(uem_model, student_model, data_loader, llm_teacher, llm_tokenizer):
    uem_model.eval()
    student_model.eval()
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            try:
                teacher_prompts = []
                for q, o in zip(batch["questions_asked"], batch["original_texts"]):
                    user_prompt = REDDIT_PROMPT_USER.format(question_asked=q, user_response=o)
                    conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
                    teacher_prompts.append(llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                teacher_inputs = llm_tokenizer(teacher_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(llm_teacher.device)
                teacher_outputs = llm_teacher(**teacher_inputs)
                teacher_logits = teacher_outputs.logits.detach()
                teacher_labels = teacher_inputs.input_ids.clone()
                teacher_labels[teacher_labels == llm_tokenizer.pad_token_id] = -100

                context_vector = uem_model(batch['loss_description_sentences']).to(student_model.device)
                student_prompts = []
                for q, a in zip(batch["questions_asked"], batch["anonymized_texts"]):
                     user_prompt = REDDIT_PROMPT_USER.format(question_asked=q, user_response=a)
                     conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
                     student_prompts.append(llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                student_inputs = llm_tokenizer(student_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(student_model.device)
                student_outputs = student_model(**student_inputs, context=context_vector)
                student_logits = student_outputs.logits
                loss = compute_distillation_loss(student_logits, teacher_logits, teacher_labels, DISTILLATION_TEMP)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    valid_batches += 1
            except Exception as e:
                print(f"Error in validation: {e}")
                continue
                
    return total_loss / max(valid_batches, 1)

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Devices: UEM@{UEM_DEVICE}, Student@{LLM_DEVICE_STUDENT}, Teacher@{LLM_DEVICE_TEACHER}")

    print("Loading Frozen Teacher LLM and Tokenizer...")
    llm_teacher = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, torch_dtype=torch.bfloat16).to(LLM_DEVICE_TEACHER)
    llm_teacher.eval()
    for param in llm_teacher.parameters(): 
        param.requires_grad = False
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    print("Teacher LLM and Tokenizer loaded.")

    print("Loading custom PAUE_LlamaForCausalLM for Student...")
    llm_student = PerLlamaForCausalLM.from_pretrained(LLM_MODEL_PATH, torch_dtype=torch.bfloat16)
    
    # 在应用 LoRA 之前彻底禁用 gradient checkpointing
    llm_student.config.use_cache = False
    llm_student.supports_gradient_checkpointing = False
    llm_student.gradient_checkpointing = False
    if hasattr(llm_student, 'enable_input_require_grads'):
        llm_student.enable_input_require_grads()
    
    print("Configuring LoRA for the Student LLM...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    llm_student_peft = get_peft_model(llm_student, lora_config)
    llm_student_peft.to(LLM_DEVICE_STUDENT)
    
    # 应用 LoRA 后再次确认禁用 gradient checkpointing
    llm_student_peft.base_model.gradient_checkpointing = False
    if hasattr(llm_student_peft.base_model.model, 'gradient_checkpointing'):
        llm_student_peft.base_model.model.gradient_checkpointing = False
    
    llm_student_peft.print_trainable_parameters()

    uem_tokenizer = AutoTokenizer.from_pretrained(UEM_MODEL_PATH, use_fast=False)
    uem_model = TrainableEnhancer(UEM_MODEL_PATH, uem_tokenizer, llm_student.config.hidden_size)
    uem_model.to(UEM_DEVICE)

    train_dataset, val_dataset = GetDataset(TRAIN_DATA_FILE), GetDataset(VAL_DATA_FILE)
    collate_fn = lambda batch: {k: [d[k] for d in batch] for k in batch[0]}
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    trainable_params = list(uem_model.parameters()) + list(filter(lambda p: p.requires_grad, llm_student_peft.parameters()))
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    # 移除 GradScaler，bfloat16 不需要
    best_val_loss = float('inf')

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        uem_model.train()
        llm_student_peft.train()
        total_train_loss = 0
        valid_batches = 0

        for batch in tqdm(train_loader, desc=f"Training"):
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_prompts = []
                for q, o in zip(batch["questions_asked"], batch["original_texts"]):
                    user_prompt = REDDIT_PROMPT_USER.format(question_asked=q, user_response=o)
                    conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
                    teacher_prompts.append(llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                teacher_inputs = llm_tokenizer(teacher_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(llm_teacher.device)
                teacher_outputs = llm_teacher(**teacher_inputs)
                teacher_logits = teacher_outputs.logits.detach()
                teacher_labels = teacher_inputs.input_ids.clone()
                teacher_labels[teacher_labels == llm_tokenizer.pad_token_id] = -100

            # 使用 autocast 但不用 GradScaler
            with torch.amp.autocast(device_type=LLM_DEVICE_STUDENT.split(':')[0], dtype=torch.bfloat16):
                context_vector = uem_model(batch['loss_description_sentences']).to(llm_student_peft.device)
                student_prompts = []
                for q, a in zip(batch["questions_asked"], batch["anonymized_texts"]):
                     user_prompt = REDDIT_PROMPT_USER.format(question_asked=q, user_response=a)
                     conv = [{"role": "system", "content": REDDIT_PROMPT_SYSTEM}, {"role": "user", "content": user_prompt}]
                     student_prompts.append(llm_tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
                student_inputs = llm_tokenizer(student_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(llm_student_peft.device)
                student_outputs = llm_student_peft(**student_inputs, context=context_vector)
                student_logits = student_outputs.logits
                loss = compute_distillation_loss(student_logits, teacher_logits, teacher_labels, DISTILLATION_TEMP)

            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping."); continue
            
            # 直接 backward，不使用 scaler
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=CLIPPING_NORM)
            optimizer.step()
            total_train_loss += loss.item()
            valid_batches += 1

        avg_train_loss = total_train_loss / max(valid_batches, 1)
        avg_val_loss = evaluate(uem_model, llm_student_peft, val_loader, llm_teacher, llm_tokenizer)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}. Saving models.")
            llm_student_peft.save_pretrained(os.path.join(CKPT_DIR, "student_lora_adapter"))
            torch.save(uem_model.state_dict(), os.path.join(CKPT_DIR, "uem_enhancer.pth"))
            
            adapter_state_dict = {}
            for name, module in llm_student_peft.named_modules():
                if isinstance(module, CrossAttentionAdapter):
                    adapter_state_dict[name] = module.state_dict()
            torch.save(adapter_state_dict, os.path.join(CKPT_DIR, "cross_attention_adapters.pth"))
        else:
            print(f"Val loss did not improve from {best_val_loss:.4f}.")

if __name__ == "__main__":
    main()