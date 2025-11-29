import torch
import json
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from typing import Any, Dict, List

# --- 1. 配置模型和 Tokenizer (和之前一样) ---

model_name = "Qwen/Qwen3-8B"
dataset_path = "/scratch/yw8866/MQuAKE/datasets/MQuAKE-T.json" 
new_model_name = "qwen3-8b-implicit-knowledge-update"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,  # <-- 修复1：使用 'dtype'
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token_id = 151643
tokenizer.padding_side = "right"

# --- 2. 🌟 新的 DataCollator (带打印和修复) 🌟 ---
# 把它放在 tokenizer 定义之后

class DataCollatorWithDebugging:
    """
    一个自定义的 data collator，用于调试并确保 'labels' 被创建。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print("--- DataCollatorWithDebugging 已初始化 ---")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        print(f"\n--- DEBUG: Collator 接收到 {len(batch)} 个项目 ---")
        if batch:
            print(f"DEBUG: Collator 接收到的第一个项目键: {list(batch[0].keys())}")
            # 打印第一个项目的 input_ids (部分)
            # print(f"DEBUG: 第一个项目的 input_ids (前10): {batch[0]['input_ids'][:10]}")

        # 1. 使用 tokenizer.pad 填充批次
        # 这将把 List[Dict] 转换为 Dict[List] 并填充，然后转为 Tensors
        try:
            padded_batch = self.tokenizer.pad(
                batch,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            print(f"DEBUG: Collator padding 失败: {e}")
            print(f"DEBUG: 尝试检查的批次数据: {batch}")
            raise e
            
        print(f"DEBUG: Collator 填充后的键: {list(padded_batch.keys())}")

        # 2. 核心修复：手动创建 'labels'
        # labels 应该是 input_ids 的一个副本
        labels = padded_batch["input_ids"].clone()
        
        # 3. 关键步骤：将 labels 中的 padding token 替换为 -100
        # 这样它们在计算损失时会被忽略
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 4. 将 'labels' 添加到最终的批次中
        padded_batch["labels"] = labels
        
        print(f"DEBUG: Collator 最终发送给模型的键: {list(padded_batch.keys())}")
        # 此时，键应该包含 'input_ids', 'attention_mask', 和 'labels'
        
        return padded_batch

# --- 3. PEFT (LoRA) 配置 (和之前一样) ---

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 4. 数据集处理 (和之前一样，使用 batched .map) ---

def generate_training_examples_batched(batch):
    new_examples = {"text": []}
    num_examples = len(batch[list(batch.keys())[0]])
    
    for i in range(num_examples):
        data_point = {key: batch[key][i] for key in batch}
        try:
            for rewrite in data_point["requested_rewrite"]:
                user_question_single = rewrite["question"]
                new_answer_single = rewrite["target_new"]["str"]
                messages = [
                    {"role": "user", "content": user_question_single},
                    {"role": "assistant", "content": new_answer_single}
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                new_examples["text"].append(text)

            user_question_multi = data_point["questions"][0]
            new_answer_multi = data_point["new_answer"]
            messages = [
                {"role": "user", "content": user_question_multi},
                {"role": "assistant", "content": new_answer_multi}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            new_examples["text"].append(text)
        except Exception as e:
            pass
    return new_examples

dataset = load_dataset("json", data_files=dataset_path, split="train")

processed_dataset = dataset.map(
    generate_training_examples_batched,
    batched=True,
    remove_columns=dataset.column_names
)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

print(f"--- 原始数据集大小: {len(dataset)} ---")
print(f"--- 处理后训练样本总数: {len(tokenized_dataset)} ---")


# --- 5. 🌟 实例化新的 Collator 🌟 ---
collator_with_debug = DataCollatorWithDebugging(tokenizer=tokenizer)


# --- 6. 训练 (和之前一样，但添加了 checkpointing 修复) ---

training_args = TrainingArguments(
    output_dir=f"./{new_model_name}-results",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True, 
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,  # <-- 修复2：添加 checkpointing 修复
    gradient_checkpointing_kwargs={"use_reentrant": False} # <-- 修复2
)

# --- 7. 🌟 更新 Trainer 初始化 🌟 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator_with_debug  # <-- 修复3：使用我们带 debug 的 collator
)

print("--- 开始微调 ---")
trainer.train()
print("--- 微调完成 ---")

# --- 8. 保存模型 (和之前一样) ---
print(f"保存 LoRA 适配器到 {new_model_name}")
trainer.save_model(new_model_name)

print("训练完成。")