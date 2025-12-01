import torch
import json
import argparse
import re
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LORA_PATH = "./qwen3-8b-CF"
DEFAULT_DATASET_PATH = "/scratch/yw8866/MQuAKE/datasets/MQuAKE-CF-3k-v2.json"

# Special tokens
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"

# --- 2. Helper Functions (Robust Matching) ---

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = str(s).lower().strip()
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    return " ".join(s.split())

def check_answer(generated_answer, expected_answer, aliases):
    """
    Robust check: Returns True if the expected answer (or alias) 
    is CONTAINED inside the generated answer.
    """
    gen_norm = normalize_answer(generated_answer)
    exp_norm = normalize_answer(expected_answer)
    
    # 1. Direct containment check
    if exp_norm in gen_norm:
        return True
        
    # 2. Alias containment check
    if aliases:
        for alias in aliases:
            alias_norm = normalize_answer(alias)
            if alias_norm and alias_norm in gen_norm:
                return True
    
    return False



def get_batch_responses(model, tokenizer, prompts: List[str], enable_thinking=False, batch_size=32, max_new_tokens=1024) -> List[Dict[str, str]]:
    """
    Core batch generation function with robust CoT parsing.
    """
    all_responses = []
    
    # Process in chunks
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Batch Inference (think={enable_thinking})"):
        batch_prompts = prompts[i:i+batch_size]
        
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
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 4. Batch Decode
            input_ids_len = model_inputs.input_ids.shape[1]
            batch_output_ids = generated_ids[:, input_ids_len:]

            # Decode full text including special tokens
            decoded_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)

            for raw_text in decoded_texts:
                thinking_content = ""
                final_answer = raw_text

                # 🌟 Robust CoT Parsing Logic 🌟
                if enable_thinking:
                    clean_raw = raw_text.strip()
                    
                    if THINK_END_TOKEN in clean_raw:
                        parts = clean_raw.split(THINK_END_TOKEN)
                        thinking_content = parts[0].replace(THINK_START_TOKEN, "").strip()
                        final_answer = parts[1].strip()
                    elif THINK_START_TOKEN in clean_raw:
                        thinking_content = clean_raw.replace(THINK_START_TOKEN, "").strip()
                        final_answer = "[INCOMPLETE_GENERATION]" 
                    else:
                        final_answer = clean_raw

                # Remove other special tokens
                final_answer = re.sub(r'<\|.*?\|>', '', final_answer).strip()

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

# --- 4. Main Function with Argument Parsing ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 LoRA on MQuAKE")
    
    # Task selection
    parser.add_argument("--task", nargs="+", default=["all"], 
                        choices=["all", "edit", "instance", "multihop", "cot"],
                        help="Which evaluation tasks to run. Can select multiple.")
    
    # Settings
    parser.add_argument("--test_mode", action="store_true", help="Run on only 100 samples with debug prints.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens for generation.")
    
    # Paths (Optional overrides)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA_PATH)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)

    args = parser.parse_args()

    # Determine which tasks to run
    run_all = "all" in args.task
    run_edit = run_all or "edit" in args.task
    run_instance = run_all or "instance" in args.task
    run_multihop = run_all or "multihop" in args.task
    run_cot = run_all or "cot" in args.task

    print(f"--- Configuration ---")
    print(f"Tasks: {args.task}")
    print(f"Test Mode: {args.test_mode}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"LoRA Path: {args.lora_path}")
    print("-" * 30)

    print(f"--- 1. Loading Model: {args.base_model} ---")
    
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
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643

    print(f"--- 2. Merging LoRA Adapter ---")
    try:
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA weights merged successfully.")
    except Exception as e:
        print(f"Failed to merge LoRA weights: {e}\nWARNING: Running with Base Model.")

    model.eval()

    print(f"--- 3. Loading Dataset: {args.dataset_path} ---")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    if args.test_mode:
        print(f"\n⚠️ TEST MODE ENABLED: Running on first 100 samples only.\n")
        dataset = dataset.select(range(min(len(dataset), 100)))

    metrics = {
        "edit_wise": {"correct": 0, "total": 0},
        "instance_wise": {"correct": 0, "total": 0},
        "multi_hop": {"correct": 0, "total": 0},
        "multi_hop_cot": {"correct": 0, "total": 0}
    }
    
    # Prepare data containers
    ew_prompts, ew_answers = [], []
    iw_prompt_groups, iw_answer_groups, iw_alias_groups = [], [], []
    mh_prompts, mh_answers, mh_aliases = [], [], []

    print("Preparing task data...")
    for data_point in dataset:
        # 1. Edit-wise
        for rewrite in data_point["requested_rewrite"]:
            ew_prompts.append(rewrite["question"])
            ew_answers.append({"ans": rewrite["target_new"]["str"], "alias": []})
            if run_edit: metrics["edit_wise"]["total"] += 1

        # 2. Instance-wise
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

        # 3. Multi-hop
        mh_prompts.append(data_point["questions"][0])
        mh_answers.append(data_point["new_answer"])
        mh_aliases.append(data_point.get("new_answer_alias", []))
        if run_multihop: metrics["multi_hop"]["total"] += 1
        if run_cot: metrics["multi_hop_cot"]["total"] += 1


    print("--- 4. Starting Evaluation ---")

    # === 1. Edit-wise ===
    if run_edit:
        print("\n--- Running: Edit-wise ---")
        ew_results = get_batch_responses(
            model, tokenizer, ew_prompts, 
            enable_thinking=False, 
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )
        
        for i, res in enumerate(ew_results):
            is_correct = check_answer(res["answer"], ew_answers[i]["ans"], ew_answers[i]["alias"])
            if is_correct:
                metrics["edit_wise"]["correct"] += 1
            
            if args.test_mode:
                status = "✅ PASS" if is_correct else "❌ FAIL"
                print(f"\n[Edit-wise #{i}] {status}")
                print(f"Q: {ew_prompts[i]}")
                print(f"Got: {res['answer']}")
                print(f"Exp: {ew_answers[i]['ans']}")

    # === 2. Instance-wise ===
    if run_instance:
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
            enable_thinking=False,
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
        
        if args.test_mode:
            print(f"Instance-wise debug logs suppressed to save space, but {metrics['instance_wise']['correct']}/{metrics['instance_wise']['total']} were correct.")

    # === 3. Multi-hop (Standard) ===
    if run_multihop:
        print("\n--- Running: Multi-hop (Standard) ---")
        mh_results = get_batch_responses(
            model, tokenizer, mh_prompts, 
            enable_thinking=False,
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

    # === 4. Multi-hop (CoT) ===
    if run_cot:
        print("\n--- Running: Multi-hop (CoT) ---")
        mh_cot_results = get_batch_responses(
            model, tokenizer, mh_prompts, 
            enable_thinking=True, # Force Thinking
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )
        for i, res in enumerate(mh_cot_results):
            is_correct = check_answer(res["answer"], mh_answers[i], mh_aliases[i])
            if is_correct: metrics["multi_hop_cot"]["correct"] += 1
            
            if args.test_mode:
                status = "✅ PASS" if is_correct else "❌ FAIL"
                print(f"\n[Multi-hop CoT #{i}] {status}")
                print(f"Q: {mh_prompts[i]}")
                think_preview = res['thinking'][-200:] if len(res['thinking']) > 200 else res['thinking']
                print(f"Think: ...{think_preview}")
                print(f"Got: {res['answer']}")
                print(f"Exp: {mh_answers[i]}")

    # --- Print Stats ---
    print("\n\n=== Final Results ===")
    def calc_acc(key):
        if metrics[key]["total"] == 0: return "N/A"
        val = (metrics[key]["correct"] / metrics[key]["total"]) * 100
        return f"{val:.2f}%"

    print(f"Task selection: {args.task}")
    if run_edit: print(f"Edit-wise:   {calc_acc('edit_wise')}")
    if run_instance: print(f"Instance:    {calc_acc('instance_wise')}")
    if run_multihop: print(f"Multi-hop:   {calc_acc('multi_hop')}")
    if run_cot: print(f"Multi-hop CoT: {calc_acc('multi_hop_cot')}")

if __name__ == "__main__":
    main()