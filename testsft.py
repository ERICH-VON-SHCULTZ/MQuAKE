import torch
import json
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
import re
from typing import List, Dict, Any

# --- 1. 配置 ---
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
LORA_ADAPTER_PATH = "./qwen3-8b-implicit-knowledge-update" 
DATASET_PATH = "/scratch/yw8866/MQuAKE/datasets/MQuAKE-T.json" 
THINK_TOKEN_ID = 151668 # </think>
GLOBAL_ENABLE_THINKING = False

EVAL_BATCH_SIZE = 64

# --- 2. 辅助函数 (清理/检查) ---

def clean_answer(text):
    text = text.strip()
    text = re.sub(r"^[.,'\" ]+", "", text)
    text = re.sub(r"[.,'\" ]+$", "", text)
    return text

def check_answer(generated_answer, expected_answer, aliases):
    if generated_answer == expected_answer:
        return True
    if aliases and generated_answer in aliases:
        return True
    if generated_answer.startswith(expected_answer):
        return True
    return False

# --- 3. 🌟 新的批量推理函数 🌟 ---

def get_batch_responses(model, tokenizer, prompts: List[str], enable_thinking=False) -> List[Dict[str, str]]:
    """
    核心的批量生成函数。
    """
    all_responses = []
    
    # 将长列表分成小批次
    for i in tqdm(range(0, len(prompts), EVAL_BATCH_SIZE), desc=f"Batch Inference (thinking={enable_thinking})"):
        batch_prompts = prompts[i:i+EVAL_BATCH_SIZE]
        
        # 1. 准备批量聊天模板
        batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        try:
            texts = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking 
            )
            
            # 2. 批量 Tokenize
            model_inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(model.device)
            
            # 3. 批量生成
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 4. 批量解码 (逐个解析)
            input_ids_len = model_inputs.input_ids.shape[1]
            batch_output_ids = generated_ids[:, input_ids_len:].tolist()

            for output_ids in batch_output_ids:
                thinking_content = ""
                final_answer = ""

                if enable_thinking:
                    try:
                        index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
                        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
                        final_answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
                    except ValueError:
                        final_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                else:
                    final_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

                all_responses.append({
                    "thinking": thinking_content,
                    "answer": clean_answer(final_answer)
                })

        except Exception as e:
            print(f"处理批次时出错: {e}")
            # 为失败的批次添加空响应
            all_responses.extend([{"thinking": "", "answer": ""}] * len(batch_prompts))
            
    return all_responses

# --- 4. 主评估函数 (重构) ---

def main():
    print(f"--- 1. 加载模型: {BASE_MODEL_NAME} ---")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = 151643

    print(f"--- 2. 合并 LoRA 权重: {LORA_ADAPTER_PATH} ---")
    try:
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
        print("LoRA 权重合并成功。")
    except Exception as e:
        print(f"合并 LoRA 权重失败: {e}\n警告：正在使用基础模型进行评估。")

    model.eval()

    print(f"--- 3. 加载并准备数据集: {DATASET_PATH} ---")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    metrics = {
        "edit_wise": {"correct": 0, "total": 0},
        "instance_wise": {"correct": 0, "total": 0},
        "multi_hop": {"correct": 0, "total": 0},
        "multi_hop_cot": {"correct": 0, "total": 0}
    }
    
    # 🌟 步骤 3.1: 准备所有评估任务
    ew_prompts, ew_answers = [], []
    iw_prompt_groups, iw_answer_groups, iw_alias_groups = [], [], []
    mh_prompts, mh_answers, mh_aliases = [], [], []

    for data_point in tqdm(dataset, desc="准备评估数据"):
        # 1. Edit-wise 任务
        for rewrite in data_point["requested_rewrite"]:
            ew_prompts.append(rewrite["question"])
            ew_answers.append({"ans": rewrite["target_new"]["str"], "alias": []})
            metrics["edit_wise"]["total"] += 1

        # 2. Instance-wise 任务 (分组)
        if "new_single_hops" in data_point:
            metrics["instance_wise"]["total"] += 1
            current_hop_prompts = []
            current_hop_answers = []
            current_hop_aliases = []
            for hop in data_point["new_single_hops"]:
                current_hop_prompts.append(hop["question"])
                current_hop_answers.append(hop["answer"])
                current_hop_aliases.append(hop.get("answer_alias", []))
            iw_prompt_groups.append(current_hop_prompts)
            iw_answer_groups.append(current_hop_answers)
            iw_alias_groups.append(current_hop_aliases)

        # 3. Multi-hop 任务
        mh_prompts.append(data_point["questions"][0])
        mh_answers.append(data_point["new_answer"])
        mh_aliases.append(data_point.get("new_answer_alias", []))
        metrics["multi_hop"]["total"] += 1
        metrics["multi_hop_cot"]["total"] += 1


    print("--- 4. 开始批量评估 ---")

    # === 评估 1: Edit-wise Success ===
    print("\n--- 正在运行: Edit-wise (事实记忆) ---")
    ew_results = get_batch_responses(model, tokenizer, ew_prompts, enable_thinking=GLOBAL_ENABLE_THINKING)
    for i, res in enumerate(ew_results):
        if check_answer(res["answer"], ew_answers[i]["ans"], ew_answers[i]["alias"]):
            metrics["edit_wise"]["correct"] += 1

    # === 评估 2: Instance-wise Accuracy ===
    print("\n--- 正在运行: Instance-wise (链条记忆) ---")
    # 1. 展平所有任务 (Flatten all tasks)
    all_iw_prompts = []
    all_iw_expected_answers = []
    all_iw_aliases = []
    # 这个列表用于追踪每个 hop 属于哪个原始实例 (instance_id)
    all_iw_group_indices = [] 
    
    for instance_id, prompt_group in enumerate(iw_prompt_groups):
        for hop_index, prompt in enumerate(prompt_group):
            all_iw_prompts.append(prompt)
            all_iw_expected_answers.append(iw_answer_groups[instance_id][hop_index])
            all_iw_aliases.append(iw_alias_groups[instance_id][hop_index])
            all_iw_group_indices.append(instance_id) # 追踪 instance_id

    # 2. 一次性运行所有 'hop' 的批量推理
    # 这是真正的批量优化，GPU 利用率会很高
    all_iw_results = get_batch_responses(
        model, tokenizer, all_iw_prompts, 
        enable_thinking=GLOBAL_ENABLE_THINKING
    )

    # 3. 重新组合结果 (在 CPU 上快速完成)
    num_instances = len(iw_prompt_groups)
    # 初始化一个列表，假设所有实例都正确
    instance_correct_tracker = [True] * num_instances
    
    # 遍历所有 hops 的结果
    for i, result in enumerate(tqdm(all_iw_results, desc="Re-grouping Instance-wise")):
        instance_id = all_iw_group_indices[i] # 找到这个 hop 属于哪个实例
        
        # 如果这个实例已经因为之前的 hop 失败了，就跳过检查 (小优化)
        if not instance_correct_tracker[instance_id]:
            continue
            
        expected_answer = all_iw_expected_answers[i]
        aliases = all_iw_aliases[i]
        
        # 检查这个 hop 是否正确
        is_correct = check_answer(result["answer"], expected_answer, aliases)
        
        # 如果这个 hop 错了，就将整个实例标记为错误
        if not is_correct:
            instance_correct_tracker[instance_id] = False
    
    # 4. 统计最终结果
    metrics["instance_wise"]["correct"] = sum(instance_correct_tracker)

    # === 评估 3: Multi-hop Accuracy (非 CoT) ===
    print("\n--- 正在运行: Multi-hop (非 CoT) ---")
    mh_results = get_batch_responses(model, tokenizer, mh_prompts, enable_thinking=GLOBAL_ENABLE_THINKING)
    for i, res in enumerate(mh_results):
        if check_answer(res["answer"], mh_answers[i], mh_aliases[i]):
            metrics["multi_hop"]["correct"] += 1

    # === 评估 4: Multi-hop Accuracy (CoT / 'Thinking') ===
    print("\n--- 正在运行: Multi-hop (CoT/Thinking) ---")
    mh_cot_results = get_batch_responses(model, tokenizer, mh_prompts, enable_thinking=True)
    for i, res in enumerate(mh_cot_results):
        if check_answer(res["answer"], mh_answers[i], mh_aliases[i]):
            metrics["multi_hop_cot"]["correct"] += 1
            
    # 打印前 5 个 CoT 示例
    print("\n--- 示例 CoT (Thinking) 结果 (前5) ---")
    for i in range(min(5, len(mh_prompts))):
        print(f"\n--- 示例 {i+1} (CoT 模式) ---")
        print(f"Q: {mh_prompts[i]}")
        print(f"THINKING:\n{mh_cot_results[i]['thinking']}")
        print(f"A (模型): {mh_cot_results[i]['answer']}")
        print(f"A (预期): {mh_answers[i]}")
        print("-" * 20)

    # --- 5. 打印最终结果 ---
    print("\n\n--- 评估完成：最终结果 ---")
    
    ew_acc = (metrics["edit_wise"]["correct"] / metrics["edit_wise"]["total"]) * 100
    iw_acc = (metrics["instance_wise"]["correct"] / metrics["instance_wise"]["total"]) * 100
    mh_acc = (metrics["multi_hop"]["correct"] / metrics["multi_hop"]["total"]) * 100
    mh_cot_acc = (metrics["multi_hop_cot"]["correct"] / metrics["multi_hop_cot"]["total"]) * 100

    print(f"\n📊 MQUAKE 评估指标 (模型: {LORA_ADAPTER_PATH}):")
    print("-" * 40)
    print(f"1. Edit-wise (事实记忆):   {ew_acc:.2f}% ({metrics['edit_wise']['correct']} / {metrics['edit_wise']['total']})")
    print(f"2. Instance-wise (链条记忆): {iw_acc:.2f}% ({metrics['instance_wise']['correct']} / {metrics['instance_wise']['total']})")
    print(f"3. Multi-hop (非 CoT):     {mh_acc:.2f}% ({metrics['multi_hop']['correct']} / {metrics['multi_hop']['total']})")
    print(f"4. Multi-hop (CoT/Thinking): {mh_cot_acc:.2f}% ({metrics['multi_hop_cot']['correct']} / {metrics['multi_hop_cot']['total']})")
    print("-" * 40)

if __name__ == "__main__":
    main()