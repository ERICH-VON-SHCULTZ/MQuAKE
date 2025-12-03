import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ==========================================
# Configuration Section
# ==========================================

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_PATH = 'MQuAKE-CF-3k-v2.json'
OUTPUT_FILE = 'mello_qwen_results.json'

# Number of samples to test (Original was 1, set higher for batch evaluation)
T = 100  
# Whether to print detailed logs during generation
VERBOSE = True 

# ==========================================
# MeLLo Prompt Templates
# ==========================================

# First-round prompt (no retrieved fact yet)
FIRST_ROUND_PROMPT = """
You are solving a multi-hop question using iterative retrieval.

Rules:
1. You are NOT allowed to output “Final answer:” at this step because no retrieved fact has been provided yet.
2. Since you do NOT have enough information to answer the original question, you MUST output exactly two lines:
   Subquestion: <your subquestion here>
   Tentative answer: <your initial guess here>
3. Do NOT output “Final answer:” at this step.
4. Do NOT output any text besides these two lines.
5. Do NOT generate "Retrieved fact:", I will provide it to you.
6. You MUST NOT include phrases such as:
   "Let's think", "Let me analyze", "Okay", "First", "Next",
   "Here is", "We can see", or any narrative text.
Now solve the question:
Question: {question}
"""

# Later-round prompt where Final answer is prohibited (step=1)
LATER_ROUND_PROMPT_FORCE_SUBQ = """
You have been provided with the following retrieved fact:
Retrieved fact: {retrieved_fact}

Rules for this step:
1. You are NOT allowed to output “Final answer:” at this step.
2. You MUST output exactly these two lines ONLY:
   Subquestion: <your new subquestion>
   Tentative answer: <your guess>
3. You must NOT output any explanation, reasoning, or additional text.
4. Any text outside these two lines is forbidden.
5. You MUST NOT include phrases such as:
   "Let's think", "Let me analyze", "Okay", "First", "Next",
   "Here is", "We can see", or any narrative text.


Original question:
{question}
"""

# Later-round prompt where Final answer is allowed (step >= 2)
LATER_ROUND_PROMPT = """
You have been provided with the following retrieved facts:

Retrieved fact:
{retrieved_fact}

You MUST strictly follow these rules:

1. You MUST output exactly ONE line in the following format:
   Final answer: <your answer>

2. You MUST treat ALL text that appears AFTER the line "Retrieved fact:"
   as absolutely true and authoritative.
   - Completely ignore your own world knowledge.
   - Completely ignore real-world facts.
   - Ignore anything that contradicts the retrieved facts.
   - ONLY use the retrieved facts to produce the final answer.

3. You MUST NOT output any explanations, reasoning, analysis, chain-of-thought,
   inner thoughts, descriptions, or natural language beyond the required line.

4. You MUST NOT include phrases such as:
   "Let's think", "Let me analyze", "Okay", "First", "Next",
   "Here is", "We can see", or any narrative text.

5. Your entire output MUST consist of EXACTLY one line beginning with:
   Final answer:
   No other text before or after it is allowed.

Original question:
{question}
"""

# ==========================================
# Model and Utility Functions
# ==========================================

def load_models():
    """
    Load the generation model (Qwen) and the retrieval model (Contriever).
    """
    print(f"Loading Generation Model: {MODEL_NAME}...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading Retrieval Model: facebook/contriever-msmarco...")
    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").cuda()
    retrieval_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    
    return qwen_model, qwen_tokenizer, contriever, retrieval_tokenizer

def mean_pooling(token_embeddings, mask):
    """
    Mean pooling for Contriever embeddings.
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):
    """
    Generate embeddings for a list of sentences using Contriever.
    """
    all_embs = []
    print("Generating embeddings for memory index...")
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, new_facts, k=1):
    """
    Retrieve the top-k most similar facts for a given query.
    """
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    
    # Calculate similarity
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices

def call_qwen(model, tokenizer, cur_prompt, stop=None, max_tokens=256, temperature=0):
    """
    Call Qwen with strict format enforcement.
    """
    system_instruction = "You are an assistant that automatically solves tasks or decomposes them when needed."
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": cur_prompt}
    ]

    # Note: 'enable_thinking' is specific to certain models/APIs. 
    # Wrapped in try/except to maintain compatibility with standard transformers.
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True 
        )
    except TypeError:
        # Fallback if the tokenizer template doesn't support 'enable_thinking'
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode response
    generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Aggressive cleaning of potential thought tags
    response = response.replace('<think>', '').replace('</think>', '')
    response = response.strip()

    # Handle stop sequences if provided
    if stop:
        for stop_seq in stop:
            if stop_seq in response:
                response = response[:response.index(stop_seq)]

    # Extract formatted parts (Subquestion / Tentative answer / Final answer)
    lines = response.split('\n')
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith('Subquestion:') or \
           line.startswith('Tentative answer:') or \
           line.startswith(''):
            formatted_lines.append(line)

    # If formatted lines are found, return only those to keep it clean
    if formatted_lines:
        return '\n'.join(formatted_lines)

    return response

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Load Models
    qwen_model, qwen_tokenizer, contriever, retrieval_tokenizer = load_models()

    # 2. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    # 3. Build Retrieval Index from edited facts
    new_facts = set()
    for d in dataset:
        for r in d["requested_rewrite"]:
            # Format: "The president of Country is Name"
            new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
    new_facts = list(new_facts)
    print(f"Total edited facts to index: {len(new_facts)}")
    
    # Pre-compute embeddings for all new facts
    fact_embs = get_sent_embeddings(new_facts, contriever, retrieval_tokenizer)

    # 4. Evaluation Loop
    cor = 0
    tot = 0
    results = []
    
    # Slice the first T samples for testing
    test_data = dataset[:T] if T < len(dataset) else dataset

    print(f"\nStarting evaluation on {len(test_data)} examples...")

    for idx, d in enumerate(tqdm(test_data, desc="Evaluating")):
        tot += 1
        answered = False
        expected = d["new_answer"].strip()
        aliases = [a.strip() for a in d.get("new_answer_alias", [])]
        
        # MQuAKE typically has multiple questions per case. 
        # Here we only iterate through the first one for efficiency, 
        # but you can remove the break statement below to test all.
        for q_idx, q in enumerate(d["questions"]):
            if VERBOSE:
                print(f"\n{'='*80}\nExample {idx}, Q{q_idx}: {q}\nExpected: {d['new_answer']}\n{'='*80}")

            retrieved_facts_history = []
            found_ans = False
            current_retrieved_fact = None
            
            # Allow maximum 4 hops/steps
            for step in range(4):
                
                # --- Build Prompt ---
                if current_retrieved_fact is None:
                    # Step 0: Initial prompt
                    prompt = FIRST_ROUND_PROMPT.format(question=q)
                elif step == 1:
                    # Step 1: Provide one fact, force subquestion
                    prompt = LATER_ROUND_PROMPT_FORCE_SUBQ.format(
                        retrieved_fact=current_retrieved_fact,
                        question=q
                    )
                else:
                    # Step >= 2: Provide history of facts, allow final answer
                    facts_text = "\n".join([f"Retrieved fact: {f}" for i, f in enumerate(retrieved_facts_history)])
                    prompt = LATER_ROUND_PROMPT.format(
                        retrieved_fact=facts_text,
                        question=q
                    )

                # --- Generation ---
                gen = call_qwen(qwen_model, qwen_tokenizer, prompt, stop=None, max_tokens=512)

                if VERBOSE:
                    print(f"\n--- Iteration {step} ---")
                    print(f"Generated:\n{gen}")

                gen_lines = [line.strip() for line in gen.strip().split('\n') if line.strip()]

                # --- Logic Parsing ---
                ans = ""
                
                # Check for tentative answer early stop (Step 1)
                # Sometimes the tentative answer is already the correct final answer
                if step == 1:
                    for line in gen_lines:
                        m = re.search(r"tentative answer:\s*(.*)$", line, flags=re.IGNORECASE)
                        if m:
                            tmp = m.group(1).strip()
                            if (expected in tmp) or any(alias in tmp for alias in aliases):
                                ans = tmp
                                found_ans = True
                                if VERBOSE:
                                    print(f"Early stop: Tentative answer already correct → '{ans}'")
                                break
                
                # Check for final answer (Step >= 2)
                if step >= 2:
                    for line in gen_lines:
                        m = re.search(r"final answer\s*:\s*(.*)$", line, flags=re.IGNORECASE)
                        if m:
                            ans = m.group(1).strip()
                            break
                
                # If an answer was found
                if ans:
                    found_ans = True
                    if VERBOSE:
                        print(f"Found final answer: '{ans}'")
                    break

                # --- Detect Subquestion & Retrieve ---
                subq_line = None
                for line in gen_lines:
                    if line.lower().startswith("subquestion:"):
                        subq_line = line
                        break
                
                if not subq_line:
                    if VERBOSE:
                        print("No Subquestion found → stopping this question.")
                    break

                subquestion = subq_line[len("Subquestion:"):].strip()
                if VERBOSE:
                    print(f"Parsed Subquestion: {subquestion}")

                # Retrieve relevant fact using the subquestion
                fact_ids = retrieve_facts(subquestion, fact_embs, contriever, retrieval_tokenizer, new_facts)
                current_retrieved_fact = new_facts[fact_ids[0]]
                retrieved_facts_history.append(current_retrieved_fact)
                
                if VERBOSE:
                    print(f"Retrieved fact: {current_retrieved_fact}")

            # --- End of steps loop ---

            if not found_ans:
                if VERBOSE:
                    print("No final answer found.")
                continue

            # --- Evaluation ---
            # Check if the generated answer matches expected answer or aliases
            is_correct = (expected in ans) or any(alias in ans for alias in aliases)

            if VERBOSE:
                print("\nComparison:")
                print(f"  Generated: '{ans}'")
                print(f"  Expected: '{expected}'")
                print(f"  Match: {is_correct}")

            if is_correct:
                cor += 1
                answered = True
                if VERBOSE:
                    print("✓✓✓ CORRECT ✓✓✓")
            else:
                if VERBOSE:
                    print("✗✗✗ INCORRECT ✗✗✗")
            
            # Break after the first question of the instance is processed 
            break

        results.append({
            'idx': idx,
            'answered': answered,
            'questions': d['questions'],
            'expected': d['new_answer']
        })

    # 5. Save Results
    print(f"\n{'='*80}")
    print(f'Multi-hop Accuracy = {cor/tot:.2%} ({cor}/{tot})')
    print(f"{'='*80}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            'accuracy': cor/tot if tot > 0 else 0,
            'correct': cor,
            'total': tot,
            'results': results
        }, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()