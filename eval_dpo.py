import torch
import json
import argparse
import re
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --- 默认配置 ---
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
# 这里指向 DPO 训练后的最终模型路径
DEFAULT_LORA_PATH = "dpo/qwen3-8b-mquake-dpo-final" 
DEFAULT_DATASET_PATH = "datasets/MQuAKE-T.json"

# Special tokens
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"

# --- 辅助函数 ---

def normalize_answer(s):
    """
    标准化答案：转小写，去标点，去多余空格
    """
    s = str(s).lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return " ".join(s.split())

def check_answer(generated_answer, expected_answer, aliases):
    """
    检查答案是否正确（包含关系）
    """
    gen_norm = normalize_answer(generated_answer)
    exp_norm = normalize_answer(expected_answer)
    
    # 1. 直接包含检查
    if exp_norm in gen_norm:
        return True
        
    # 2. 别名包含检查
    if aliases:
        for alias in aliases:
            alias_norm = normalize_answer(alias)
            if alias_norm and alias_norm in gen_norm:
                return True
    
    return False

def get_batch_responses(model, tokenizer, prompts: List[str], enable_thinking=False, batch_size=32, max_new_tokens=1024) -> List[Dict[str, str]]:
    """
    批量生成函数
    """
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"推理进度 (think={enable_thinking})"):
        batch_prompts = prompts[i:i+batch_size]
        
        # 1. 准备 Chat 模板
        # 注意：Qwen3 的 chat template 不需要系统提示词，且支持 system role
        batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        try:
            texts = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=False,
                add_generation_prompt=True
                # Qwen3-8B 通常不需要 enable_thinking 参数，除非是 DeepSeek 或特定微调版本
                # 这里为了兼容性保留，但如果报错可能需要移除
            )
            
            # 2. Tokenize (左填充用于推理)
            model_inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048 
            ).to(model.device)
            
            # 3. 生成
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 4. 解码
            input_ids_len = model_inputs.input_ids.shape[1]
            batch_output_ids = generated_ids[:, input_ids_len:]
            decoded_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)

            for raw_text in decoded_texts:
                thinking_content = ""
                final_answer = raw_text

                # CoT 解析逻辑 (如果模型输出 <think> 标签)
                if enable_thinking and THINK_START_TOKEN in raw_text:
                    clean_raw = raw_text.strip()
                    if THINK_END_TOKEN in clean_raw:
                        parts = clean_raw.split(THINK_END_TOKEN)
                        thinking_content = parts[0].replace(THINK_START_TOKEN, "").strip()
                        final_answer = parts[1].strip()
                    else:
                        # 只有开始没有结束，说明截断了或者生成未完成
                        thinking_content = clean_raw.replace(THINK_START_TOKEN, "").strip()
                        final_answer = "[INCOMPLETE_GENERATION]"
                else:
                    final_answer = raw_text

                # 清理特殊 Token
                final_answer = re.sub(r'<\|.*?\|>', '', final_answer).strip()
                # 移除 EOS token 文本表示如果存在
                final_answer = final_answer.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()

                all_responses.append({
                    "thinking": thinking_content,
                    "answer": final_answer, 
                    "raw_output": raw_text
                })

        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            all_responses.extend([{"thinking": "", "answer": "", "raw_output": "ERROR"}] * len(batch_prompts))
            
    return all_responses

# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 DPO on MQuAKE")
    
    # 任务选择
    parser.add_argument("--task", nargs="+", default=["all"], 
                        choices=["all", "edit", "instance", "multihop", "cot"],
                        help="选择要运行的评测任务。")
    
    # 设置
    parser.add_argument("--test_mode", action="store_true", help="测试模式：只运行前 100 个样本。")
    parser.add_argument("--batch_size", type=int, default=32, help="评测 Batch Size (A100 可设较大)。")
    parser.add_argument("--max_tokens", type=int, default=1024, help="最大生成 Token 数。")
    
    # 路径覆盖
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)

    args = parser.parse_args()

    # 确定要运行的任务
    run_all = "all" in args.task
    run_edit = run_all or "edit" in args.task
    run_instance = run_all or "instance" in args.task
    run_multihop = run_all or "multihop" in args.task
    run_cot = run_all or "cot" in args.task

    print(f"--- 配置 ---")
    print(f"任务: {args.task}")
    print(f"测试模式: {args.test_mode}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LoRA 路径: {args.lora_path}")
    print("-" * 30)

    # --- 1. 加载模型 ---
    print(f"正在加载基座模型: {args.base_model} ...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" # <--- 删除或注释这一行
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 2. 合并 LoRA ---
    print(f"正在加载并合并 LoRA Adapter: {args.lora_path} ...")
    if os.path.exists(args.lora_path):
        try:
            model = PeftModel.from_pretrained(model, args.lora_path)
            # 4-bit 模型不能直接 merge_and_unload，通常直接带着 adapter 跑即可，
            # 或者需要先反量化。对于评测，直接挂载 Adapter 是最方便的。
            # model = model.merge_and_unload() 
            print("LoRA Adapter 加载成功。")
        except Exception as e:
            print(f"❌ 加载 LoRA 失败: {e}\n⚠️ 将使用基座模型运行。")
    else:
        print(f"⚠️ 找不到 LoRA 路径: {args.lora_path}，将使用基座模型运行。")

    model.eval()

    # --- 3. 加载数据 ---
    print(f"正在加载数据集: {args.dataset_path} ...")
    if not os.path.exists(args.dataset_path):
         # 尝试从上一级目录找
        if os.path.exists(os.path.join("..", args.dataset_path)):
            args.dataset_path = os.path.join("..", args.dataset_path)
            
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    if args.test_mode:
        print(f"\n⚠️ 测试模式：只使用前 100 条数据。\n")
        dataset = dataset.select(range(min(len(dataset), 100)))

    # 准备数据容器
    metrics = {
        "edit_wise": {"correct": 0, "total": 0},
        "instance_wise": {"correct": 0, "total": 0},
        "multi_hop": {"correct": 0, "total": 0},
        "multi_hop_cot": {"correct": 0, "total": 0}
    }
    
    ew_prompts, ew_answers = [], []
    iw_prompt_groups, iw_answer_groups, iw_alias_groups = [], [], []
    mh_prompts, mh_answers, mh_aliases = [], [], []

    print("正在准备评测数据...")
    for data_point in dataset:
        # 1. Edit-wise (单跳重写)
        for rewrite in data_point.get("requested_rewrite", []):
            ew_prompts.append(rewrite["question"])
            ew_answers.append({"ans": rewrite["target_new"]["str"], "alias": []})
            if run_edit: metrics["edit_wise"]["total"] += 1

        # 2. Instance-wise (单跳事实检查)
        if "new_single_hops" in data_point:
            current_hop_prompts, current_hop_answers, current_hop_aliases = [], [], []
            for hop in data_point["new_single_hops"]:
                current_hop_prompts.append(hop["question"])
                current_hop_answers.append(hop["answer"])
                current_hop_aliases.append(hop.get("answer_alias", []))
            iw_prompt_groups.append(current_hop_prompts)
            iw_answer_groups.append(current_hop_answers)
            iw_alias_groups.append(current_hop_aliases)
            if run_instance: metrics["instance_wise"]["total"] += 1

        # 3. Multi-hop (多跳推理)
        if "questions" in data_point and data_point["questions"]:
            mh_prompts.append(data_point["questions"][0])
            mh_answers.append(data_point["new_answer"])
            mh_aliases.append(data_point.get("new_answer_alias", []))
            if run_multihop: metrics["multi_hop"]["total"] += 1
            if run_cot: metrics["multi_hop_cot"]["total"] += 1

    print("--- 4. 开始评测 ---")

    # === 1. Edit-wise (重写准确率) ===
    if run_edit and ew_prompts:
        print("\n--- 正在运行: Edit-wise (重写准确率) ---")
        ew_results = get_batch_responses(
            model, tokenizer, ew_prompts, 
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )
        
        for i, res in enumerate(ew_results):
            is_correct = check_answer(res["answer"], ew_answers[i]["ans"], ew_answers[i]["alias"])
            if is_correct:
                metrics["edit_wise"]["correct"] += 1
            
            if args.test_mode:
                status = "✅ PASS" if is_correct else "❌ FAIL"
                print(f"\n[Edit #{i}] {status}")
                print(f"Q: {ew_prompts[i]}")
                print(f"Got: {res['answer']}")
                print(f"Exp: {ew_answers[i]['ans']}")

    # === 2. Instance-wise (连贯性检查) ===
    if run_instance and iw_prompt_groups:
        print("\n--- 正在运行: Instance-wise (连贯性检查) ---")
        all_iw_prompts = []
        all_iw_expected_answers = []
        all_iw_aliases = []
        all_iw_group_indices = [] 
        
        for instance_id, prompt_group in enumerate(iw_prompt_groups):
            for hop_index, prompt in enumerate(prompt_group):
                all_iw_prompts.append(prompt)
                all_iw_expected_answers.append(iw_answer_groups[instance_id][hop_index])
                all_iw_aliases.append(iw_alias_groups[instance_id][hop_index])
                all_iw_group_indices.append(instance_id)

        all_iw_results = get_batch_responses(
            model, tokenizer, all_iw_prompts, 
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )

        num_instances = len(iw_prompt_groups)
        instance_correct_tracker = [True] * num_instances
        
        for i, result in enumerate(all_iw_results):
            instance_id = all_iw_group_indices[i]
            expected = all_iw_expected_answers[i]
            aliases = all_iw_aliases[i]
            
            is_correct = check_answer(result["answer"], expected, aliases)
            if not is_correct:
                instance_correct_tracker[instance_id] = False

        metrics["instance_wise"]["correct"] = sum(instance_correct_tracker)

    # === 3. Multi-hop (多跳推理) ===
    if run_multihop and mh_prompts:
        print("\n--- 正在运行: Multi-hop (直接回答) ---")
        mh_results = get_batch_responses(
            model, tokenizer, mh_prompts, 
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )
        for i, res in enumerate(mh_results):
            is_correct = check_answer(res["answer"], mh_answers[i], mh_aliases[i])
            if is_correct: metrics["multi_hop"]["correct"] += 1
            
            if args.test_mode:
                status = "✅ PASS" if is_correct else "❌ FAIL"
                print(f"\n[Multi-hop #{i}] {status}")
                print(f"Q: {mh_prompts[i]}")
                print(f"Got: {res['answer']}")
                print(f"Exp: {mh_answers[i]}")

    # === 4. Multi-hop CoT (思维链) ===
    if run_cot and mh_prompts:
        print("\n--- 正在运行: Multi-hop (CoT 思维链) ---")
        mh_cot_results = get_batch_responses(
            model, tokenizer, mh_prompts, 
            enable_thinking=True, # 这里主要是指如果模型有CoT能力，我们尝试解析
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )
        for i, res in enumerate(mh_cot_results):
            is_correct = check_answer(res["answer"], mh_answers[i], mh_aliases[i])
            if is_correct: metrics["multi_hop_cot"]["correct"] += 1

    # --- 结果汇总 ---
    print("\n\n=== 📊 最终评测结果 ===")
    def calc_acc(key):
        if metrics[key]["total"] == 0: return "N/A"
        val = (metrics[key]["correct"] / metrics[key]["total"]) * 100
        return f"{val:.2f}% ({metrics[key]['correct']}/{metrics[key]['total']})"

    if run_edit: print(f"Edit-wise (单跳重写):   {calc_acc('edit_wise')}")
    if run_instance: print(f"Instance-wise (连贯性): {calc_acc('instance_wise')}")
    if run_multihop: print(f"Multi-hop (多跳推理):   {calc_acc('multi_hop')}")
    if run_cot: print(f"Multi-hop (CoT):        {calc_acc('multi_hop_cot')}")

if __name__ == "__main__":
    main()



# python3 dpo/eval_dpo.py --batch_size 64 --max_tokens=2048 2>&1 | tee dpo/eval_log.txt
# python3 dpo/eval_dpo.py --test_mode --batch_size 64 --max_tokens=2048 2>&1 | tee dpo/eval_log2.txt