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

# --- 1. 配置 ---
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
# 确保这个路径指向你训练好的 LoRA 适配器目录
LORA_ADAPTER_PATH = "./qwen3-8b-implicit-knowledge-update" 
DATASET_PATH = "/scratch/yw8866/MQuAKE/datasets/MQuAKE-T.json" 
THINK_TOKEN_ID = 151668 # </think>

# --- 2. 辅助函数 ---

def clean_answer(text):
    """
    一个简单的清理函数，用于规范化模型的输出，以便进行比较。
    """
    text = text.strip()
    # 移除句点、逗号、引号
    text = re.sub(r"^[.,'\" ]+", "", text)
    text = re.sub(r"[.,'\" ]+$", "", text)
    return text

def get_model_response(model, tokenizer, prompt, enable_thinking=False):
    """
    核心生成函数，可以切换“思考”模式。
    """
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # 1. 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking # <-- 切换“思考”模式
        )
        
        # 2. Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 3. 生成
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256, # 答案通常不长，256 足够
            pad_token_id=tokenizer.pad_token_id
        )
        
        # 4. 解码并解析
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        thinking_content = ""
        final_answer = ""

        if enable_thinking:
            try:
                # 寻找 </think> (151668)
                index = len(output_ids) - output_ids[::-1].index(THINK_TOKEN_ID)
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
                final_answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            except ValueError:
                # 找不到 </think>，说明模型可能直接回答了
                final_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        else:
            # 非思考模式
            final_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return {
            "thinking": thinking_content,
            "answer": clean_answer(final_answer)
        }

    except Exception as e:
        print(f"在处理 prompt 时出错: {prompt}\n错误: {e}")
        return {"thinking": "", "answer": ""}

def check_answer(generated_answer, expected_answer, aliases):
    """
    检查生成的答案是否与预期答案或其别名之一匹配。
    """
    if generated_answer == expected_answer:
        return True
    if aliases and generated_answer in aliases:
        return True
    
    # 尝试更宽松的匹配（例如，模型可能会说 "Eric Adams."）
    if generated_answer.startswith(expected_answer):
        return True
        
    return False

# --- 3. 主评估函数 ---

def main():
    print(f"--- 1. 加载模型: {BASE_MODEL_NAME} ---")
    
    # 使用 4-bit 量化加载基础模型
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
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token_id = 151643 # <|endoftext|>

    print(f"--- 2. 合并 LoRA 权重: {LORA_ADAPTER_PATH} ---")
    try:
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
        print("LoRA 权重合并成功。")
    except Exception as e:
        print(f"合并 LoRA 权重失败: {e}")
        print("警告：正在使用基础模型进行评估。")

    model.eval() # 设置为评估模式

    print(f"--- 3. 加载数据集: {DATASET_PATH} ---")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    metrics = {
        "edit_wise": {"correct": 0, "total": 0},
        "instance_wise": {"correct": 0, "total": 0},
        "multi_hop": {"correct": 0, "total": 0},
        "multi_hop_cot": {"correct": 0, "total": 0}
    }

    print("--- 4. 开始评估 ---")
    
    for i, data_point in enumerate(tqdm(dataset)):
        
        # === 评估 1: Edit-wise Success ===
        # 测试模型是否记住了 'requested_rewrite' 中的新事实
        for rewrite in data_point["requested_rewrite"]:
            metrics["edit_wise"]["total"] += 1
            question = rewrite["question"]
            expected_answer = rewrite["target_new"]["str"]
            
            response = get_model_response(model, tokenizer, question, enable_thinking=False)
            
            if check_answer(response["answer"], expected_answer, []): # 'rewrite' 中没有别名
                metrics["edit_wise"]["correct"] += 1

        # === 评估 2: Instance-wise Accuracy ===
        # 测试模型是否能回忆起 'new_single_hops' 中的所有事实
        # (这是回答多跳问题的前提)
        metrics["instance_wise"]["total"] += 1
        all_hops_correct = True
        if "new_single_hops" not in data_point: continue # 确保数据格式正确

        for hop in data_point["new_single_hops"]:
            question = hop["question"]
            expected_answer = hop["answer"]
            aliases = hop.get("answer_alias", [])
            
            response = get_model_response(model, tokenizer, question, enable_thinking=False)
            
            if not check_answer(response["answer"], expected_answer, aliases):
                all_hops_correct = False
                break # 只要错一个，这个实例就失败了
        
        if all_hops_correct:
            metrics["instance_wise"]["correct"] += 1

        # === 评估 3: Multi-hop Accuracy (非 CoT) ===
        # 测试多跳问题 (enable_thinking=False)
        metrics["multi_hop"]["total"] += 1
        question = data_point["questions"][0]
        expected_answer = data_point["new_answer"]
        aliases = data_point.get("new_answer_alias", [])

        response_no_cot = get_model_response(model, tokenizer, question, enable_thinking=False)
        
        if check_answer(response_no_cot["answer"], expected_answer, aliases):
            metrics["multi_hop"]["correct"] += 1

        # === 评估 4: Multi-hop Accuracy (CoT / 'Thinking') ===
        # 测试多跳问题 (enable_thinking=True)
        metrics["multi_hop_cot"]["total"] += 1
        
        response_cot = get_model_response(model, tokenizer, question, enable_thinking=True)
        
        if check_answer(response_cot["answer"], expected_answer, aliases):
            metrics["multi_hop_cot"]["correct"] += 1
            
        # 打印前 5 个例子的“思考”过程，以供分析
        if i < 5:
            print(f"\n--- 示例 {i+1} (CoT 模式) ---")
            print(f"Q: {question}")
            print(f"THINKING:\n{response_cot['thinking']}")
            print(f"A (模型): {response_cot['answer']}")
            print(f"A (预期): {expected_answer}")
            print(f"是否正确: {check_answer(response_cot['answer'], expected_answer, aliases)}")
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