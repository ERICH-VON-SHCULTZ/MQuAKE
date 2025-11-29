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

# --- 1. Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
LORA_ADAPTER_PATH = "./qwen3-8b-implicit-knowledge-update" 
DATASET_PATH = "/scratch/yw8866/MQuAKE/datasets/MQuAKE-T.json" 

# Special tokens
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"

# Global Thinking Toggle
GLOBAL_ENABLE_THINKING = False

# Batch size & Generation Limits
EVAL_BATCH_SIZE = 32
# Increased max tokens to prevent cutting off thoughts
MAX_NEW_TOKENS = 1024 

# 🌟 TEST MODE SETTINGS 🌟
TEST_MODE = False       
TEST_SAMPLE_SIZE = 100 

# --- 2. Helper Functions (Robust Matching) ---

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = str(s).lower().strip()
    
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    
    # Remove common articles (optional, but helps with 'The United States' vs 'United States')
    # s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    return " ".join(s.split())

def check_answer(generated_answer, expected_answer, aliases):
    """
    Robust check: Returns True if the expected answer (or alias) 
    is CONTAINED inside the generated answer.
    """
    gen_norm = normalize_answer(generated_answer)
    exp_norm = normalize_answer(expected_answer)
    
    # 1. Direct containment check (e.g., "Joe Biden" in "The president is Joe Biden")
    if exp_norm in gen_norm:
        return True
        
    # 2. Alias containment check
    if aliases:
        for alias in aliases:
            alias_norm = normalize_answer(alias)
            if alias_norm and alias_norm in gen_norm:
                return True
    
    return False

# --- 3. Batch Inference Function (Improved Parsing) ---

def get_batch_responses(model, tokenizer, prompts: List[str], enable_thinking=False) -> List[Dict[str, str]]:
    """
    Core batch generation function with robust CoT parsing.
    """
    all_responses = []
    
    # Process in chunks
    for i in tqdm(range(0, len(prompts), EVAL_BATCH_SIZE), desc=f"Batch Inference (think={enable_thinking})"):
        batch_prompts = prompts[i:i+EVAL_BATCH_SIZE]
        
        # 1. Prepare chat template
        batch_messages = [[{"role": "user", "content": p}] for p in batch_prompts]
        
        try:
            texts = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking 
            )
            
            # 2. Batch Tokenize (Left Padding for Inference!)
            model_inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048 # Increased context length
            ).to(model.device)
            
            # 3. Batch Generate
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 4. Batch Decode
            input_ids_len = model_inputs.input_ids.shape[1]
            batch_output_ids = generated_ids[:, input_ids_len:]

            # Decode full text including special tokens to catch <think> tags
            decoded_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)

            for raw_text in decoded_texts:
                thinking_content = ""
                final_answer = raw_text

                # 🌟 Robust CoT Parsing Logic (String-based) 🌟
                if enable_thinking:
                    # Clean up the output string first
                    clean_raw = raw_text.strip()
                    
                    # Case A: Standard </think> split
                    if THINK_END_TOKEN in clean_raw:
                        parts = clean_raw.split(THINK_END_TOKEN)
                        thinking_content = parts[0].replace(THINK_START_TOKEN, "").strip()
                        # Everything after </think> is the answer
                        final_answer = parts[1].strip()
                    
                    # Case B: <think> exists but no </think> (Cut off)
                    elif THINK_START_TOKEN in clean_raw:
                        thinking_content = clean_raw.replace(THINK_START_TOKEN, "").strip()
                        final_answer = "[INCOMPLETE_GENERATION]" # Model didn't finish thinking
                    
                    # Case C: No <think> tags found (Model skipped thinking)
                    else:
                        final_answer = clean_raw

                # Remove other special tokens (like <|im_end|>) from the final answer
                final_answer = re.sub(r'<\|.*?\|>', '', final_answer).strip()

                all_responses.append({
                    "thinking": thinking_content,
                    "answer": final_answer, # This is the extracted text for checking
                    "raw_output": raw_text  # For debug printing
                })

        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            all_responses.extend([{"thinking": "", "answer": "", "raw_output": "ERROR"}] * len(batch_prompts))
            
    return all_responses

# --- 4. Main Evaluation Function ---

def main():
    print(f"--- 1. Loading Model: {BASE_MODEL_NAME} ---")
    
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
    
    # Padding side LEFT for inference
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643

    print(f"--- 2. Merging LoRA Adapter: {LORA_ADAPTER_PATH} ---")
    try:
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
        print("LoRA weights merged successfully.")
    except Exception as e:
        print(f"Failed to merge LoRA weights: {e}\nWARNING: Running with Base Model.")

    model.eval()

    print(f"--- 3. Loading Dataset: {DATASET_PATH} ---")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    if TEST_MODE:
        print(f"\n⚠️ TEST MODE ENABLED: Running on first {TEST_SAMPLE_SIZE} samples only.\n")
        dataset = dataset.select(range(min(len(dataset), TEST_SAMPLE_SIZE)))

    metrics = {
        "edit_wise": {"correct": 0, "total": 0},
        "instance_wise": {"correct": 0, "total": 0},
        "multi_hop": {"correct": 0, "total": 0},
        "multi_hop_cot": {"correct": 0, "total": 0}
    }
    
    # Prepare evaluation tasks
    ew_prompts, ew_answers = [], []
    iw_prompt_groups, iw_answer_groups, iw_alias_groups = [], [], []
    mh_prompts, mh_answers, mh_aliases = [], [], []

    print("Preparing tasks...")
    for data_point in dataset:
        # 1. Edit-wise
        for rewrite in data_point["requested_rewrite"]:
            ew_prompts.append(rewrite["question"])
            ew_answers.append({"ans": rewrite["target_new"]["str"], "alias": []})
            metrics["edit_wise"]["total"] += 1

        # 2. Instance-wise
        if "new_single_hops" in data_point:
            metrics["instance_wise"]["total"] += 1
            current_hop_prompts, current_hop_answers, current_hop_aliases = [], [], []
            for hop in data_point["new_single_hops"]:
                current_hop_prompts.append(hop["question"])
                current_hop_answers.append(hop["answer"])
                current_hop_aliases.append(hop.get("answer_alias", []))
            iw_prompt_groups.append(current_hop_prompts)
            iw_answer_groups.append(current_hop_answers)
            iw_alias_groups.append(current_hop_aliases)

        # 3. Multi-hop
        mh_prompts.append(data_point["questions"][0])
        mh_answers.append(data_point["new_answer"])
        mh_aliases.append(data_point.get("new_answer_alias", []))
        metrics["multi_hop"]["total"] += 1
        metrics["multi_hop_cot"]["total"] += 1


    print("--- 4. Starting Evaluation ---")

    # === 1. Edit-wise ===
    print("\n--- Running: Edit-wise ---")
    ew_results = get_batch_responses(model, tokenizer, ew_prompts, enable_thinking=GLOBAL_ENABLE_THINKING)
    
    for i, res in enumerate(ew_results):
        is_correct = check_answer(res["answer"], ew_answers[i]["ans"], ew_answers[i]["alias"])
        if is_correct:
            metrics["edit_wise"]["correct"] += 1
        
        if TEST_MODE:
            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"\n[Edit-wise #{i}] {status}")
            print(f"Q: {ew_prompts[i]}")
            print(f"Got: {res['answer']}")
            print(f"Exp: {ew_answers[i]['ans']}")
            if not is_correct: print(f"Raw: {res['raw_output'][:100]}...")

    # === 2. Instance-wise ===
    print("\n--- Running: Instance-wise ---")
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
        enable_thinking=GLOBAL_ENABLE_THINKING
    )

    num_instances = len(iw_prompt_groups)
    instance_correct_tracker = [True] * num_instances
    if TEST_MODE: instance_logs = {idx: [] for idx in range(num_instances)}

    for i, result in enumerate(all_iw_results):
        instance_id = all_iw_group_indices[i]
        expected = all_iw_expected_answers[i]
        aliases = all_iw_aliases[i]
        
        is_correct = check_answer(result["answer"], expected, aliases)
        if not is_correct:
            instance_correct_tracker[instance_id] = False
        
        if TEST_MODE:
            symbol = "✅" if is_correct else "❌"
            instance_logs[instance_id].append(f"{symbol} Q: {all_iw_prompts[i]} | Got: {result['answer']}")

    if TEST_MODE:
        for idx in range(num_instances):
            status = "✅ PASS" if instance_correct_tracker[idx] else "❌ FAIL"
            print(f"\n[Instance-wise #{idx}] {status}")
            for log in instance_logs[idx]: print(log)

    metrics["instance_wise"]["correct"] = sum(instance_correct_tracker)

    # === 3. Multi-hop (Standard) ===
    print("\n--- Running: Multi-hop (Standard) ---")
    mh_results = get_batch_responses(model, tokenizer, mh_prompts, enable_thinking=GLOBAL_ENABLE_THINKING)
    for i, res in enumerate(mh_results):
        is_correct = check_answer(res["answer"], mh_answers[i], mh_aliases[i])
        if is_correct: metrics["multi_hop"]["correct"] += 1
        
        if TEST_MODE:
            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"\n[Multi-hop #{i}] {status}")
            print(f"Q: {mh_prompts[i]}")
            print(f"Got: {res['answer']}")
            print(f"Exp: {mh_answers[i]}")

    # === 4. Multi-hop (CoT) ===
    print("\n--- Running: Multi-hop (CoT) ---")
    mh_cot_results = get_batch_responses(model, tokenizer, mh_prompts, enable_thinking=True)
    for i, res in enumerate(mh_cot_results):
        is_correct = check_answer(res["answer"], mh_answers[i], mh_aliases[i])
        if is_correct: metrics["multi_hop_cot"]["correct"] += 1
        
        if TEST_MODE:
            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"\n[Multi-hop CoT #{i}] {status}")
            print(f"Q: {mh_prompts[i]}")
            # Print last 200 chars of thinking to verify it worked
            think_preview = res['thinking'][-200:] if len(res['thinking']) > 200 else res['thinking']
            print(f"Think: ...{think_preview}")
            print(f"Got: {res['answer']}")
            print(f"Exp: {mh_answers[i]}")

    # --- Print Stats ---
    print("\n\n=== Final Results ===")
    def calc_acc(key):
        if metrics[key]["total"] == 0: return 0.0
        return (metrics[key]["correct"] / metrics[key]["total"]) * 100

    print(f"Edit-wise:   {calc_acc('edit_wise'):.2f}%")
    print(f"Instance:    {calc_acc('instance_wise'):.2f}%")
    print(f"Multi-hop:   {calc_acc('multi_hop'):.2f}%")
    print(f"Multi-hop CoT: {calc_acc('multi_hop_cot'):.2f}%")

if __name__ == "__main__":
    main()