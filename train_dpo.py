import torch
import os
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import DPOTrainer, DPOConfig

# --- 路径与配置 ---
model_name = "Qwen/Qwen3-8B" 
dataset_path = "dpo/mquake_dpo.json"      # 输入数据
output_dir = "dpo/qwen3-8b-dpo-results"   # 训练过程输出
final_model_dir = "dpo/qwen3-8b-mquake-dpo-final" # 最终模型保存路径

# --- 1. 加载模型与 Tokenizer (A100 优化) ---
print(f"正在加载模型: {model_name} ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # A100 使用 bf16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # <--- 删除或注释这一行
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # DPO 生成需要左填充

# --- 2. 数据处理 (Prompt/Chosen/Rejected) ---
print(f"正在加载数据集: {dataset_path} ...")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"找不到数据集文件: {dataset_path}，请先运行 process_dpo_data.py")

dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_dpo_chat(row):
    """
    将数据格式化为 User/Assistant 对话格式。
    """
    # Prompt: User 提问
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Chosen/Rejected: Assistant 回答 + EOS
    chosen = row["chosen"] + tokenizer.eos_token
    rejected = row["rejected"] + tokenizer.eos_token
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

processed_dataset = dataset.map(format_dpo_chat, num_proc=8)

# --- 3. LoRA 配置 ---
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
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

# --- 4. DPO 训练参数 ---
training_args = DPOConfig(
    output_dir=output_dir,
    beta=0.1,                       # DPO 温度
    per_device_train_batch_size=8,  # 显存允许的情况下尽量大
    gradient_accumulation_steps=4,
    learning_rate=5e-6,             # DPO 学习率 (比 SFT 低)
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,                      # A100 开启 bf16
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    max_length=1024,
    max_prompt_length=512,
)

# --- 5. 开始训练 ---
print("初始化 DPOTrainer...")
trainer = DPOTrainer(
    model=model,
    ref_model=None, # LoRA 模式不需要加载 ref_model
    args=training_args,
    train_dataset=processed_dataset,
    processing_class=tokenizer, # <--- 修改这里：将 tokenizer 改为 processing_class
    peft_config=peft_config,
)

print("🚀 开始 DPO 微调...")
trainer.train()
print("🎉 微调完成！")

# --- 6. 保存 ---
print(f"正在保存模型到: {final_model_dir}")
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
