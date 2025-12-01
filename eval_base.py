import torch
import json
import argparse
import re
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DATASET_PATH = "/scratch/yw8866/MQuAKE/datasets/MQuAKE-CF-3k-v2.json"

# Special tokens
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"



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
                max_length=2048 
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
            all_responses.extend([{"thinking": "", "answer": "", "raw_output": "ERROR"}] * len(batch_prompts))
            
    return all_responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base Model (No LoRA) on MQuAKE")
    
    # Task selection
    parser.add_argument("--task", nargs="+", default=["all"], 
                        choices=["all", "edit", "instance", "multihop", "cot"],
                        help="Which evaluation tasks to run.")
    
    # 🌟 Evaluation Target (New vs Original) 🌟
    parser.add_argument("--target", type=str, default="original", choices=["original", "new"],
                        help="Evaluate against 'original' answers (Base Capability) or 'new' answers (Edit Success).")

    # Settings
    parser.add_argument("--test_mode", action="store_true", help="Run on only 100 samples with debug prints.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens.")
    
    # Paths
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)

    args = parser.parse_args()

    # Determine tasks
    run_all = "all" in args.task
    run_edit = run_all or "edit" in args.task
    run_instance = run_all or "instance" in args.task
    run_multihop = run_all or "multihop" in args.task
    run_cot = run_all or "cot" in args.task

    print(f"--- Configuration (Base Model Evaluation) ---")
    print(f"Model: {args.base_model}")
    print(f"Evaluation Target: {args.target.upper()} Answers")
    print(f"Tasks: {args.task}")
    print(f"Test Mode: {args.test_mode}")
    print("-" * 30)

    print(f"--- 1. Loading Base Model: {args.base_model} ---")
    
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

    # ❌ Skipped LoRA Merging step

    model.eval()

    print(f"--- 2. Loading Dataset: {args.dataset_path} ---")
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
    
    # Data Containers
    ew_prompts, ew_answers = [], []
    iw_prompt_groups, iw_answer_groups, iw_alias_groups = [], [], []
    mh_prompts, mh_answers, mh_aliases = [], [], []

    print(f"Preparing data (Target: {args.target})...")
    
    # 🌟 Logic to switch between Original and New answers 🌟
    for data_point in dataset:
        
        # 1. Edit-wise Data
        for rewrite in data_point["requested_rewrite"]:
            ew_prompts.append(rewrite["question"])
            if args.target == "new":
                # Check if model knows the NEW fact
                ew_answers.append({"ans": rewrite["target_new"]["str"], "alias": []})
            else:
                # Check if model knows the ORIGINAL fact
                ew_answers.append({"ans": rewrite["target_true"]["str"], "alias": []})
            
            if run_edit: metrics["edit_wise"]["total"] += 1

        # 2. Instance-wise Data
        # If target is NEW, we check the NEW single hops chain.
        # If target is ORIGINAL, we check the ORIGINAL single hops chain.
        hops_key = "new_single_hops" if args.target == "new" else "single_hops"
        
        if hops_key in data_point:
            current_hop_prompts, current_hop_answers, current_hop_aliases = [], [], []
            for hop in data_point[hops_key]:
                current_hop_prompts.append(hop["question"])
                current_hop_answers.append(hop["answer"])
                current_hop_aliases.append(hop.get("answer_alias", []))
            
            iw_prompt_groups.append(current_hop_prompts)
            iw_answer_groups.append(current_hop_answers)
            iw_alias_groups.append(current_hop_aliases)
            
            if run_instance: metrics["instance_wise"]["total"] += 1

        # 3. Multi-hop Data
        mh_prompts.append(data_point["questions"][0])
        
        if args.target == "new":
            mh_answers.append(data_point["new_answer"])
            mh_aliases.append(data_point.get("new_answer_alias", []))
        else:
            mh_answers.append(data_point["answer"])
            mh_aliases.append(data_point.get("answer_alias", []))
            
        if run_multihop: metrics["multi_hop"]["total"] += 1
        if run_cot: metrics["multi_hop_cot"]["total"] += 1


    print("--- 3. Starting Evaluation ---")

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
            if check_answer(res["answer"], ew_answers[i]["ans"], ew_answers[i]["alias"]):
                metrics["edit_wise"]["correct"] += 1
            elif args.test_mode:
                 # Debug print for failures
                 print(f"[Edit Fail] Q: {ew_prompts[i]} | Got: {res['answer']} | Exp: {ew_answers[i]['ans']}")

    # === 2. Instance-wise ===
    if run_instance:
        print("\n--- Running: Instance-wise ---")
        # Flattening logic
        all_iw_prompts = []
        all_iw_group_indices = []
        
        for instance_id, prompt_group in enumerate(iw_prompt_groups):
            for prompt in prompt_group:
                all_iw_prompts.append(prompt)
                all_iw_group_indices.append(instance_id)

        all_iw_results = get_batch_responses(
            model, tokenizer, all_iw_prompts, 
            enable_thinking=False,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )

        num_instances = len(iw_prompt_groups)
        instance_correct_tracker = [True] * num_instances
        
        # Mapping back results
        # We need to iterate carefully.
        # Structure: Flattened list corresponds to flattened groups
        
        flat_idx = 0
        for instance_id in range(num_instances):
            prompt_group = iw_prompt_groups[instance_id]
            answer_group = iw_answer_groups[instance_id]
            alias_group = iw_alias_groups[instance_id]
            
            for hop_idx in range(len(prompt_group)):
                res = all_iw_results[flat_idx]
                expected = answer_group[hop_idx]
                aliases = alias_group[hop_idx]
                
                if not check_answer(res["answer"], expected, aliases):
                    instance_correct_tracker[instance_id] = False
                
                flat_idx += 1

        metrics["instance_wise"]["correct"] = sum(instance_correct_tracker)

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
            elif args.test_mode:
                 print(f"[MH Fail] Q: {mh_prompts[i]} | Got: {res['answer']} | Exp: {mh_answers[i]}")

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
            elif args.test_mode:
                 print(f"[MH CoT Fail] Q: {mh_prompts[i]} | Think: ...{res['thinking'][-50:]} | Got: {res['answer']} | Exp: {mh_answers[i]}")

    # --- Print Stats ---
    print("\n\n=== Final Results (Base Model) ===")
    def calc_acc(key):
        if metrics[key]["total"] == 0: return "N/A"
        val = (metrics[key]["correct"] / metrics[key]["total"]) * 100
        return f"{val:.2f}%"

    print(f"Model: {args.base_model}")
    print(f"Target Answers: {args.target.upper()}")
    if run_edit: print(f"Edit-wise:   {calc_acc('edit_wise')}")
    if run_instance: print(f"Instance:    {calc_acc('instance_wise')}")
    if run_multihop: print(f"Multi-hop:   {calc_acc('multi_hop')}")
    if run_cot: print(f"Multi-hop CoT: {calc_acc('multi_hop_cot')}")

if __name__ == "__main__":
    main()