import json
import os

# --- 路径配置 ---
# 假设从项目根目录运行: python dpo/process_dpo_data.py
input_path = "datasets/MQuAKE-T.json"
output_path = "dpo/mquake_dpo.json" 

def format_dpo_dataset():
    print(f"正在读取原始数据: {input_path} ...")
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到文件 {input_path}")
        print("请确保你在项目根目录下运行此脚本，或者检查文件路径。")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dpo_data = []
    count_single = 0

    for item in data:
        # 只处理单跳重写 (Requested rewrite)
        rewrites = item.get("requested_rewrite", [])
        for rewrite in rewrites:
            q_single = rewrite.get("question")
            target_new = rewrite.get("target_new", {}).get("str", "")
            target_true = rewrite.get("target_true", {}).get("str", "")

            if q_single and target_new and target_true:
                dpo_data.append({
                    "prompt": q_single,
                    "chosen": target_new,
                    "rejected": target_true,
                    "type": "single_hop"
                })
                count_single += 1

    print(f"处理完成！")
    print(f"  - 单跳样本数 (Requested Rewrite): {count_single}")
    print(f"  - 总 DPO 样本数: {len(dpo_data)}")
    
    # 保存到 dpo 文件夹
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ DPO 数据集已保存至: {output_path}")

if __name__ == "__main__":
    format_dpo_dataset()
