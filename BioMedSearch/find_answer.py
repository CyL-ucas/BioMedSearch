import os
import re
import csv
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

DATASET_DIR = r""

print("🔧 Loading medical embedding model: NeuML/pubmedbert-base-embeddings")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_MODEL = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device=str(device))

from models import claudeLLM_zh, gpt41_llm, gemini_llm, llama_LLM, qwen_llm, claude_llm, claude4_llm, deepseek_llm, openaihub_llm


def load_semantic_blocks(md_path):
    """
    Load structured Markdown report, extract all paragraphs starting with "#" or "##" followed by number+title.
    Returns format {number+title: content} and prints all titles for verification.
    """
    import re
    from collections import OrderedDict

    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    pattern = r"(?:^|\n)#{1,3}\s*(\d+)[\.:]?\s+(.*?)\n(.*?)(?=\n#{1,3}\s*\d+[\.:]?\s+[^\n]+|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    sections = OrderedDict()
    for i, (num, title, content) in enumerate(matches, 1):
        full_title = f"{num} {title.strip()}"
        sections[full_title] = content.strip()
    return sections


def extract_option_judgement(response, qid):
    pattern = rf"Q{qid}:\s*([\s\S]+?)(?=\nQ\d+:|\Z)"
    match = re.search(pattern, response)
    options_block = match.group(1) if match else response

    option_judge = {}
    for opt in ["A", "B", "C", "D"]:
        opt_pat = rf"{opt}:\s*(Correct|Incorrect|Not mentioned)[\.:，: ]*(.*?)(?=\n[A-D]:|\n*$)"
        m = re.search(opt_pat, options_block)
        if m:
            judge, reason = m.group(1), m.group(2).strip()
            option_judge[opt] = {"judge": judge, "reason": reason}
        else:
            option_judge[opt] = {"judge": "Not mentioned", "reason": ""}
    return option_judge


def normalize(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()


def get_option_keywords(option_text):
    return [normalize(w) for w in re.split(r"[，,、\s]+", option_text) if len(w) > 1]


def extract_answer_format(response: str, question_id: str) -> str:
    q_number = re.findall(r"\d+", question_id)
    q_number = q_number[0] if q_number else ""

    pattern = rf"Q{q_number}:\s*([A-D](?:\s*[,，、]\s*[A-D])*)"
    match = re.search(pattern, response)

    if match:
        raw = match.group(1)
        letters = [x for x in re.findall(r"[A-D]", raw.upper())]
        seen = set()
        unique = [x for x in letters if not (x in seen or seen.add(x))]
        return f"Q{q_number}: {', '.join(unique)}"

    return "Unanswered"


def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def final_must_answer_with_search_assumption(qid, question, context):
    prompt = f"""You are a senior biomedical expert with extensive clinical, research, and literature retrieval experience.
This is your third attempt at answering this question. The model failed to answer it previously, so we have provided additional reference material (retrieved through semantic search or external literature) for you to consider. Please read all the content carefully and answer strictly based on the context.

[Instructions for Answering]
(Instructions remain unchanged...)

[Question]
{qid}: {question}

[Original Context]
{context if context else "No original context provided"}

Please follow the example above to analyze the question and options, and complete your answer accordingly.
"""

    try:
        response = gpt41_llm.chat([{"role": "user", "content": prompt}])
        return extract_answer_format(response, qid)
    except:
        return f"{qid}: A"


def parse_question_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    q_entry = {}
    opt_map = {}
    q_pattern = re.compile(r"^(Q\d+):\s*(.+)")
    opt_pattern = re.compile(r"^\s*([A-D])[.:、:：\-]+\s*(.+)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        q_match = q_pattern.match(line)
        opt_match = opt_pattern.match(line)

        if q_match:
            if q_entry:
                q_entry["options"] = opt_map
                questions.append(q_entry)
                q_entry, opt_map = {}, {}
            q_entry["qid"] = q_match.group(1)
            q_entry["stem"] = q_match.group(2)
        elif line.startswith("Options:"):
            continue
        elif opt_match:
            opt_map[opt_match.group(1)] = opt_match.group(2)

    if q_entry:
        q_entry["options"] = opt_map
        questions.append(q_entry)

    return questions


def evaluate_mcq_with_semantic_assist(report_md_path, question_items, answer_items, result_file=None, level_name="Level1"):
    assert len(question_items) == len(answer_items), "❗ Number of questions and answers do not match! Please check the source files."
    blocks = load_semantic_blocks(report_md_path)
    fail_log = []

    result_file = result_file or os.path.join(DATASET_DIR, f"gpt41_llm_Level_{level_name}_BioMed1000_Evaluation.csv")

    def answer_single(idx, question_item, std_answer):
        qid = question_item["qid"]
        stem = question_item["stem"]
        options_dict = question_item["options"]

        block_keys = list(blocks.keys())
        block_texts = [blocks[k] for k in block_keys]
        query_prompt = f"Represent this biomedical multiple-choice question for semantic search:\nQuestion: {stem}\nOptions:\n" + "\n".join([f"{k}. {v}" for k, v in options_dict.items()])
        q_emb = EMBED_MODEL.encode(query_prompt, convert_to_tensor=True).to(device)
        blk_emb = EMBED_MODEL.encode(block_texts, convert_to_tensor=True).to(device)

        scores = util.cos_sim(q_emb, blk_emb)[0]
        top_indices = scores.argsort(descending=True)[:5]
        context_blocks = [f"### {block_keys[i]}\n{block_texts[i]}" for i in top_indices]
        context = "\n\n".join(context_blocks)

        prompt = f"""
(Keep your QA Prompt unchanged)
"""

        start = time.time()
        try:
            response, usage = gpt41_llm.chat(
                [{"role": "user", "content": prompt}],
                return_usage=True
            )
            elapsed = time.time() - start
            token_used = usage.get("total_tokens", 0)
        except Exception:
            response = "No answer"
            elapsed = 0
            token_used = 0

        model_ans_initial = extract_answer_format(response, qid)
        option_judgement = extract_option_judgement(response, idx + 1)
        model_ans = model_ans_initial or "Unanswered"

        MAX_RETRIES = 10
        retry_count = 0
        while model_ans == "Unanswered" and retry_count < MAX_RETRIES:
            retry_count += 1
            model_ans = final_must_answer_with_search_assumption(qid, stem, context)
            response = model_ans
        if model_ans == "Unanswered":
            print(f"⚠️ {qid} reached maximum retry attempts {MAX_RETRIES}, still unanswered.")

        option_judgement = extract_option_judgement(response, idx + 1)

        def extract_option_set(ans_string):
            return set(re.findall(r"[A-D]", ans_string.upper()))

        true_set = extract_option_set(std_answer)
        pred_set = extract_option_set(model_ans)
        strict_correct = (model_ans != "Unanswered") and (pred_set == true_set)

        if not strict_correct:
            fail_log.append({
                "qid": qid,
                "question": stem,
                "true": std_answer,
                "pred": model_ans,
                "option_judgement": option_judgement,
                "context": context,
                "raw_response": response
            })

        is_correct = "✅" if strict_correct else "❌"
        tokens_used = len(stem.split()) + len(model_ans.split())
        print(f"{qid} | Model: {model_ans} | Standard: {std_answer}")
        return [
            qid, model_ans, std_answer,
            is_correct, round(elapsed, 2),
            model_ans_initial, tokens_used
        ], strict_correct, tokens_used, elapsed

    all_rows, total_correct, total_tokens, total_time = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(answer_single, i, q, a): i
            for i, (q, a) in enumerate(zip(question_items, answer_items))
        }
        for future in as_completed(futures):
            try:
                row, correct, tokens, elapsed = future.result()
                all_rows.append(row)
                total_correct += int(correct)
                total_tokens += tokens
                total_time += elapsed
            except Exception as e:
                print(f"❗ Thread execution error: {e}")

    with open(result_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question ID", "Model Answer", "Standard Answer", "Is Correct", "Elapsed Time", "Initial Extraction", "Token Count"])

        for row in sorted(all_rows, key=lambda x: int(x[0][1:])):
            writer.writerow(row)
    avg_token = total_tokens / len(question_items)
    avg_time = total_time / len(question_items)

    print(f"\n🎯 Accuracy: {total_correct}/{len(question_items)} = {total_correct / len(question_items):.2%}")
    print(f"📊 Token Consumption: {total_tokens}, Average per Question: {avg_token:.1f}, Average Time: {avg_time:.2f} seconds")
    print(f"📄 Answer log saved to: {result_file}")
