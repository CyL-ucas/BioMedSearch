import os
import sys
import json
import time
import re
import difflib
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import random
from collections import OrderedDict, Counter
import threading
from sentence_transformers import SentenceTransformer, util

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from models import gpt41_llm, llama_LLM, openaihub_llm
from pmc_search import PMCSearcher
from pubmed_search import PubMedSearcher
from ScienceDirect_Research import ScopusSearcher

SAVE_DIR = os.path.join(project_root, "dataset")
os.makedirs(SAVE_DIR, exist_ok=True)
load_dotenv()

QUESTION_ANGLES = [
    "mechanistic insight",
    "experimental design",
    "clinical application",
    "data interpretation",
    "comparison of interventions",
    "foundational concept"
]
angle = random.choice(QUESTION_ANGLES)

def normalize(text):
    return re.sub(r"[^\w\s]", "", text).lower()

def coverage_rate(text, keywords, threshold=0.85):
    text_norm = normalize(text)
    covered = 0
    for kw in keywords:
        kw_norm = normalize(kw)
        if kw_norm in text_norm:
            covered += 1
            continue
        words = text_norm.split()
        window_size = len(kw_norm.split())
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            if difflib.SequenceMatcher(None, kw_norm, window).ratio() >= threshold:
                covered += 1
                break
    return covered / len(keywords) if keywords else 0

def generate_biomedical_topic():
    prompt = "Forget all previous memory. Randomly generate a biomedical research topic (different from any previously generated ones), in English, and return only the title."
    return openaihub_llm.chat([{"role": "user", "content": prompt}]).strip()

def extract_biomedical_keywords(title):
    prompt = [
        {"role": "system", "content": "You are a biomedical expert. Extract 3~6 core English keywords from the following research title. Only extract from the title. Return a JSON array."},
        {"role": "user", "content": f"Research Title: {title}"}
    ]
    try:
        response = openaihub_llm.chat(prompt).strip().replace("```json", "").replace("```", "").strip()
        try:
            keywords = json.loads(response)
        except json.JSONDecodeError:
            keywords = [kw.strip() for kw in response.strip("[]").split(",") if kw.strip()]
        return [kw.replace('"', '').replace("'", "").strip() for kw in keywords]
    except:
        return [title]

def save_literature_to_file(literature, topic):
    lit_path = os.path.join(SAVE_DIR, "literature.txt")
    seen_titles = set()
    unique_lit = []

    for doc in literature:
        title = doc.get("title", "").strip()
        abstract = doc.get("abstract", "").strip()
        if not title or not abstract:
            continue
        norm_title = re.sub(r"\s+", " ", title.lower())
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        unique_lit.append(doc)

    with open(lit_path, "w", encoding="utf-8") as f:
        f.write(f"Research Topic: {topic}\n\n")
        for idx, doc in enumerate(unique_lit, 1):
            f.write(f"{idx}. Title: {doc['title'].strip()}\nAbstract: {doc['abstract'].strip()}\n\n")

    print(f"📚 Saved {len(unique_lit)} unique literature entries to {lit_path}")

def generate_research_report(literature, save_path=None):
    prompt = "You are a summarization expert. Read the following literature titles and abstracts carefully, and generate a detailed biomedical research report in academic English, at least 7000 words.\n\n"
    for doc in literature:
        title = doc.get("title", "No Title").strip()
        abstract = doc.get("abstract", "No Abstract").strip()
        prompt += f"Title: {title}\nAbstract: {abstract}\n\n"

    report = openaihub_llm.chat([{"role": "user", "content": prompt}]).strip()

    print("\n📝 Generated Research Report:\n" + "="*80)
    print(report)
    print("="*80 + "\n")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📁 Research report saved to: {save_path}")

    return report

def decompose_into_subqueries(title, max_queries=5):
    prompt = [
        {"role": "system", "content": "You are a biomedical task planning expert."},
        {"role": "user", "content": f"""
Based on the following research topic, decompose it into 3~5 specific subqueries. Each subquery should focus on a specific mechanism, molecular interaction, clinical impact, treatment approach, or other precise research point, suitable for literature retrieval.

Research Topic: {title}

Return a JSON array, e.g.:
["How does butyrate affect microglial activation?", "What is the role of gut microbiota in Alzheimer’s disease?", ...]
"""}
    ]
    try:
        response = openaihub_llm.chat(prompt).strip().replace("```json", "").replace("```", "").strip()
        subqueries = json.loads(response)
        return [q.strip() for q in subqueries if q.strip()]
    except Exception as e:
        print(f"❌ Subquery decomposition failed: {e}")
        return [title]

# ... (All other functions continue with the same translation principle)



def fetch_all_literature_from_subqueries(subqueries, topic):
    """Extract keywords and search literature for each subquery, merge and deduplicate."""
    all_docs = []
    seen_titles = set()

    for i, sub in enumerate(subqueries):
        print(f"\n🔎 Processing Subquery {i+1}: {sub}")
        keywords = extract_biomedical_keywords(sub)
        docs = fetch_literature(keywords, sub, retrieve_n=200, top_k=90)
        for doc in docs:
            title = doc.get("title", "").strip()
            if title and title not in seen_titles:
                all_docs.append(doc)
                seen_titles.add(title)
    print(f"📚 Total merged literature count: {len(all_docs)}")
    return all_docs

def save_markdown_report(markdown_report, save_path):
    """Save the generated Markdown format report to file."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"🎯 Markdown report saved to: {save_path}")

def normalize(text):
    return re.sub(r"[^\w\s]", "", text).lower()

def detect_out_of_scope(q_text, report_text, threshold=0.5):
    q_words = set(normalize(q_text).split())
    report_words = set(normalize(report_text).split())
    if not q_words:
        return True
    overlap = len(q_words & report_words) / len(q_words)
    return overlap < threshold

def fetch_literature(keywords, topic, retrieve_n=300, top_k=90):
    """
    1. Retrieve candidate literature from PMC and PubMed using keywords;
    2. Use Sentence-Transformer model to encode topic and documents, sort by cosine similarity;
    3. Return Top_k most relevant documents.
    """
    query = " ".join(keywords)
    print(f"📥 Searching with keywords: {query}")
    pmc = PMCSearcher().search_pmc(query, max_results=retrieve_n)
    pubmed = PubMedSearcher().search_pubmed(query, max_results=retrieve_n)
    candidates = [doc for doc in (pmc + pubmed) if doc.get("abstract", "").strip()]
    print(f"🔍 Total candidate documents: {len(candidates)} (with abstracts)")

    model_name = 'NeuML/pubmedbert-base-embeddings'
    print(f"🔧 Loading embedding model: {model_name} (supports safetensors, no PyTorch upgrade needed)")
    model = SentenceTransformer(model_name)

    texts = [f"{doc['title']} {doc['abstract']}" for doc in candidates]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    topic_emb = model.encode(topic, convert_to_tensor=True)

    sims = util.cos_sim(embeddings, topic_emb).squeeze()
    topk = sims.topk(min(top_k, len(candidates)))
    idxs = topk.indices.cpu().tolist()
    selected = [candidates[i] for i in idxs]

    print(f"✅ Filtering completed, selected Top {len(selected)} relevant documents")
    return selected

def regenerate_question_strict(gpt_func, level, title, report, index):
    prompt = f"""
You are a biomedical question generator. Please regenerate only one multiple-choice question (Q{index+1}) strictly based on the following report.

# ❗ Rules:
- Use only facts that explicitly appear in the report.
- Do NOT use external knowledge or assumptions.
- Format exactly as follows (and only return this format):

Q{index+1}: [question content]
Options:
  A: ...
  B: ...
  C: ...
  D: ...
Q{index+1} Answer: A, B

# Research Report:
{report}
"""
    try:
        response = gpt_func([{"role": "user", "content": prompt}])
        return response.strip()
    except Exception as e:
        print(f"⚠️ Regeneration failed Q{index+1}: {e}")
        return None

def extract_question_blocks(text_block):
    pattern = r"Q\d+:(.*?)Options:(.*?)Q\d+\s+Answer:\s*(.*?)\n"
    matches = re.findall(pattern, text_block, re.DOTALL)
    parsed = []
    for idx, (q, opts, ans) in enumerate(matches, 1):
        opts_list = [line.strip().split(": ", 1)[1] for line in opts.strip().splitlines()]
        ans_list = [x.strip() for x in ans.split(",")]
        parsed.append({
            "question": q.strip(),
            "options": opts_list,
            "answer": [opts_list[ord(x) - ord("A")] for x in ans_list if x in "ABCD"]
        })
    return parsed

def load_md_sections(md_path):
    """Split MD structured report into sections by secondary titles and return {section_title: content}"""
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    # Assume each section starts with '### x.x Title'
    pattern = r"(?:^|\n)(#{2,3} [^\n]+)\n(.*?)(?=\n#{2,3} |\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    sections = OrderedDict()
    for h, content in matches:
        title = h.lstrip("#").strip()
        if content.strip():
            sections[title] = content.strip()
    return sections

EMBED_MODEL = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def detect_out_of_scope(text, context, threshold=0.5):
    # Semantic similarity detection, context is the concatenation of selected theme contents
    q_emb = EMBED_MODEL.encode([text], convert_to_tensor=True)
    ctx_emb = EMBED_MODEL.encode([context], convert_to_tensor=True)
    score = float(util.cos_sim(q_emb, ctx_emb)[0][0])
    # Also check keyword overlap
    words = set(re.sub(r"[^\w]", " ", text).lower().split())
    ctx_words = set(re.sub(r"[^\w]", " ", context).lower().split())
    overlap = len(words & ctx_words) / max(1, len(words))
    return score < threshold or overlap < 0.5



def generate_mcq_on_sections(level, theme_title, context, batch_size=1, min_words=15, q_index=None):
    if not context.strip():
        raise ValueError("Context cannot be empty.")

    if level == 1:
        prompt = f"""
You are a biomedical multiple-choice question (MCQ) writer. Your ONLY knowledge source is the CONTEXT below.

# 【Strict Rules for MCQ Generation - Level 1: Mechanism Recognition】
- Focus on **single molecular effects**, such as activation, inhibition, upregulation, downregulation, stimulation, suppression, enhancement, or causality.
- Design questions around **direct molecular interactions**, where one entity modulates another (e.g., "X activates Y", "A inhibits B via C").
- Avoid combinatorial logic or multi-step reasoning.
- Emphasize **regulatory directionality**: upstream vs. downstream actors.
- Use terms like receptor, enzyme, cytokine, transcription factor, etc.
- Ensure the **question stem and all correct options are explicitly supported by the CONTEXT**.
- Do NOT use external knowledge or assumptions.
- There must be **at least two correct options**.
- The question stem must be at least {min_words} words.

# 【Output format — Strict JSON】
Respond ONLY with a JSON object in the following format:

{{
  "question": "Your question stem here (≥ {min_words} words)",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  "answer": ["A", "C"]
}}

# 【Your topics/sections】:
{theme_title}
# 【CONTEXT】:
{context}

REMEMBER: All correct answers and the question stem must be strictly justified by the CONTEXT.
"""

    elif level == 2:
        prompt = f"""
You are a biomedical multiple-choice question (MCQ) writer. Your ONLY knowledge source is the CONTEXT below.

# 【Strict Rules for MCQ Generation - Level 2: Semantic Integration】
- Each correct option must **synthesize information from multiple places in the CONTEXT**.
- Test cross-paragraph or cross-sentence integration (e.g., mechanism + outcome).
- Avoid fact-based or single-sentence retrieval questions.
- Encourage **logical linking between dispersed biological concepts**.
- Distractors may be plausible but MUST NOT be supported by the CONTEXT.
- The question stem and correct options must be **explicitly grounded in the CONTEXT**.
- Do NOT introduce external facts or background assumptions.
- There must be **at least two correct options**.
- The question stem must be at least {min_words} words.

# 【Output format — Strict JSON】
Respond ONLY with a JSON object in the following format:

{{
  "question": "Your question stem here (≥ {min_words} words)",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  "answer": ["A", "C"]
}}

# 【Your topics/sections】:
{theme_title}
# 【CONTEXT】:
{context}
"""

    elif level == 3:
        # print(f"❗ Level {level} 第 {q_index or '?'} 题：")
        prompt = f"""
You are a biomedical multiple-choice question (MCQ) writer. Your ONLY knowledge source is the CONTEXT below.

# 【Strict Rules for MCQ Generation - Level 3: Temporal and Hierarchical Reasoning】
- Focus on **time-dependent processes**, **sequential biological events**, or **feedback mechanisms**.
- Design questions around **multi-step biological pathways**, e.g., receptor → kinase → gene → behavior.
- Ask about **temporal logic**, like "what happens next", or "which event precedes another".
- Emphasize regulatory cascades, feedback loops, and **layered molecular control**.
- Include biological hierarchy: chromatin → transcription → translation → phenotype.
- Use phrases implying timeline, progression, or phase (early vs. late, acute vs. chronic).
- All correct answers and the question stem must be **explicitly supported by the CONTEXT**.
- Do NOT invent facts or add assumptions.
- There must be **at least two correct options**.
- The question stem must be at least {min_words} words.

# 【Output format — Strict JSON】
Respond ONLY with a JSON object in the following format:

{{
  "question": "Your question stem here (≥ {min_words} words)",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  "answer": ["A", "C"]
}}

# 【Your topics/sections】:
{theme_title}
# 【CONTEXT】:
{context}
"""
    else:
        raise ValueError(f"Unsupported level: {level}")

    response = openaihub_llm.chat([{"role": "user", "content": prompt}])
    print(f"🧠 GPT raw output length: {len(response)}")
    print(f"🧠 GPT preview: {response[:300]}")
    return response.strip()



def extract_mcqs(text, context=None, retry_fn=None, min_words=15): 
    try:
        # Clean up non-JSON leading content
        json_text = (
            text.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        # Fix: Remove possible markdown/newlines wrapping JSON
        first_brace = json_text.find("{")
        last_brace = json_text.rfind("}")
        if first_brace == -1 or last_brace == -1:
            raise ValueError("No valid JSON braces found")
        json_text = json_text[first_brace:last_brace + 1]

        data = json.loads(json_text)
        question = data.get("question", "").strip()
        options = data.get("options", {})
        answer_keys = data.get("answer", [])

        if not question or not options or not answer_keys:
            print("❌ Missing JSON fields")
            return []

        answer_text = [options[k] for k in answer_keys if k in options]

        return [{
            "question": question,
            "options": list(options.values()),
            "answer": answer_text
        }]
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        if retry_fn and context:
            return extract_mcqs(retry_fn(context), context, retry_fn, min_words)
        return []


def is_obviously_related(text, context, keywords=None, threshold=0.3):
    # First: Semantic similarity
    q_emb = EMBED_MODEL.encode([text], convert_to_tensor=True)
    ctx_emb = EMBED_MODEL.encode([context], convert_to_tensor=True)
    score = float(util.cos_sim(q_emb, ctx_emb)[0][0])
    if score >= threshold:
        return True
    # Second: Keyword matching
    if keywords:
        norm_text = normalize(text)
        for kw in keywords:
            if normalize(kw) in norm_text:
                return True
    # Fallback: Word intersection ≥ 2
    text_words = set(normalize(text).split())
    ctx_words = set(normalize(context).split())
    if len(text_words & ctx_words) >= 2:
        return True
    return False  # Truly unrelated, considered out-of-scope


def regenerate_mcq_if_out_of_scope(mcq, context, index, min_words=15):
    question = mcq["question"]
    options = mcq["options"]
    print(f"⚠️ Out-of-scope MCQ detected, regenerating...")

    new_prompt = f"""
You are a biomedical multiple-choice question generator.

You must STRICTLY follow all instructions below:
1. Your question MUST be asked from the angle of: **{angle}**. Avoid question stems that are overly similar in style or substance to previous ones.
2. **The question stem and all correct options MUST have explicit and direct support (wording, fact, or phrase) within the provided context. If no evidence is found in context, do NOT include it.**
3. Incorrect/distractor options can be plausible but do not need to be supported by context.
4. Do NOT use any external knowledge, assumptions, or logical inferences. Use ONLY information explicitly present or implied in the context.
5. Quote or paraphrase context content directly when writing the question and correct answers.
6. Every question must be answerable using ONLY the provided context, and every correct option must be verifiable.
7. The question stem must be at least {min_words} words.
8. All output must be in English. Do NOT include explanations, answer rationales, or any extra content.

**Output format (strictly follow this format, nothing else):**
Q: [Question stem, English only]
Options:
  A: [Option A, English only]
  B: [Option B, English only]
  C: [Option C, English only]
  D: [Option D, English only]
Answer: [A, B, ...]   # Only correct options

# Context (all questions and correct answers must strictly come from here):
{context}
"""

    new_response = openaihub_llm.chat([{"role": "user", "content": new_prompt}])
    mcqs_new = extract_mcqs(new_response + "\n")
    if mcqs_new and not detect_out_of_scope(mcqs_new[0]["question"], context):
        print("✅ Replaced with compliant MCQ")
        return mcqs_new[0]
    else:
        print("❌ Still out-of-scope, retrying with longer stem...")
        new_prompt2 = f"""
You are a biomedical multiple-choice question generator.

You must STRICTLY follow all instructions below:
1. **The question stem and all correct options MUST be directly traceable to the provided context. Quote or closely paraphrase from the context; do NOT invent any new fact.**
2. If a correct answer cannot be found in context, do NOT include it.
3. Distractor options can be plausible but do not need context support.
4. Absolutely NO external knowledge or inference—ONLY information explicitly present in the context.
5. The question stem must be at least {min_words+5} words.
6. All output must be in English. No explanations or answer rationales.
7. You MUST ask your question from the angle of: **{angle}**. Avoid stems that are overly similar in style or substance to previous ones.

Ensure the question stem contains at least {min_words+5} words.

**Output format (strict, nothing else):**
Q: [Question stem, English only]
Options:
  A: [Option A, English only]
  B: [Option B, English only]
  C: [Option C, English only]
  D: [Option D, English only]
Answer: [A, B, ...]   # Only correct options

# Context (your only allowed source):
{context}
"""
        new_response2 = openaihub_llm.chat([{"role": "user", "content": new_prompt2}])
        mcqs_new2 = extract_mcqs(new_response2 + "\n")
        if mcqs_new2 and not detect_out_of_scope(mcqs_new2[0]["question"], context):
            print("✅ Second attempt generated compliant MCQ")
            return mcqs_new2[0]
        else:
            print("❌ Still out-of-scope, discarding this MCQ or fallback to original")
            return None

def generate_mcq_thread_task(block_titles, blocks, level, min_words, q_index=None):
    theme_num = {1: 2, 2: 4, 3: 6}[level]
    print(f"🔍 extract_mcqs starting, generating question #{q_index}")
    for outer_retry in range(5):  # Retry up to 5 times externally, changing topic combinations each time
        sampled_titles = random.sample(block_titles, theme_num)
        theme_title = " / ".join(sampled_titles)
        context = "\n\n".join([blocks[t] for t in sampled_titles])

        for retry in range(5):  # Retry up to 5 times for each theme combination
            raw_mcq_text = generate_mcq_on_sections(
                level, theme_title, context, batch_size=1, min_words=min_words + retry * 5
            )
            mcqs = extract_mcqs(
                raw_mcq_text + "\n",
                context=context,
                retry_fn=lambda ctx, min_words=15: generate_mcq_on_sections(level, theme_title, ctx, min_words=min_words)
            )

            if not mcqs or not isinstance(mcqs, list) or not mcqs[0]:
                print(f"\n⚠️ Level {level} Question #{q_index} failed: No valid MCQ generated, discarded.")
                print("📄 GPT raw output (for debugging):")
                print("=" * 60)
                print(raw_mcq_text.strip())
                print("=" * 60)
                print("📄 extract_mcqs parsed result:")
                print(mcqs)
                print("=" * 60 + "\n")
                continue

            mcq = mcqs[0]
            if not is_obviously_related(mcq.get("question", ""), context):
                print(f"⚠️ Level {level} Question #{q_index} failed: Question stem unrelated to content.")
                continue
            if all(not is_obviously_related(opt, context) for opt in mcq.get("options", [])):
                print(f"⚠️ Level {level} Question #{q_index} failed: All four options unrelated to content.")
                continue
            if len(mcq.get("answer", [])) <= 1:
                print(f"⚠️ Level {level} Question #{q_index} is a single-choice question (only {len(mcq.get('answer', []))} correct option), discarded.")
                continue

            # ✅ All criteria met, return the generated MCQ
            return mcq

    print(f"❗ Level {level} Question #{q_index}: Failed to generate a valid MCQ after multiple attempts, returning None.")
    return None


def _sample_theme_set(block_titles, theme_num, seen_theme_sets):
    """
    Avoid repeating topic combinations
    """
    candidates = []
    for _ in range(100):  # Try multiple rounds of sampling
        sampled = tuple(sorted(random.sample(block_titles, theme_num)))
        if sampled not in seen_theme_sets:
            candidates.append(sampled)
    if candidates:
        # Prefer combinations that have been used the least recently
        sampled = min(candidates, key=lambda x: sum([seen_theme_sets.get(k, 0) for k in x]))
        for k in sampled:
            seen_theme_sets[k] = seen_theme_sets.get(k, 0) + 1
        return sampled
    # If no new combination found, fallback to a random combination
    sampled = tuple(sorted(random.sample(block_titles, theme_num)))
    for k in sampled:
        seen_theme_sets[k] = seen_theme_sets.get(k, 0) + 1
    return sampled



def question_hash(q):
    # Hash based on question stem content
    return re.sub(r"\W+", "", q['question']).lower()

def question_and_options_similar(q1, q2, threshold=0.8):
    # Check if question stems are similar
    from difflib import SequenceMatcher
    if SequenceMatcher(None, q1['question'], q2['question']).ratio() < threshold:
        return False
    # Check if options are exactly the same (order and content)
    return q1['options'] == q2['options']

question_buffer = {}
question_locks = {}
question_counts = {}

for lvl in [1, 2, 3]:
    question_buffer[lvl] = []
    question_locks[lvl] = threading.Lock()
    question_counts[lvl] = 0

def get_question_count(level):
    with question_locks[level]:
        return question_counts[level]

def collect_and_maybe_flush(level, mcq):
    with question_locks[level]:
        question_buffer[level].append(mcq)
        question_counts[level] += 1
        idx = question_counts[level]

        if idx % 300 == 0:
            batch_index = idx // 300
            buffer_copy = question_buffer[level][:]
            flush_to_file(level, batch_index, buffer_copy)
            question_buffer[level] = []

def flush_to_file(level, batch_index, buffer):
    qfile = os.path.join(SAVE_DIR, f"questions_level{level}.txt")
    afile = os.path.join(SAVE_DIR, f"answers_level{level}.txt")
    mode = "a" if os.path.exists(qfile) else "w"

    with open(qfile, mode, encoding="utf-8") as fq, open(afile, mode, encoding="utf-8") as fa:
        base_idx = (batch_index - 1) * 300
        for i, qa in enumerate(buffer, start=1):
            idx = base_idx + i
            fq.write(f"Q{idx}: {qa['question']}\nOptions:\n")
            for opt_idx, opt in zip(['A', 'B', 'C', 'D'], qa['options']):
                fq.write(f"  {opt_idx}: {opt}\n")
            fq.write("\n")
            answers = [opt_idx for opt_idx, opt in zip(['A', 'B', 'C', 'D'], qa['options']) if opt in qa['answer']]
            fa.write(f"Q{idx} Answer: {', '.join(answers)}\n")

    print(f"✅ Level {level} wrote {len(buffer)} questions, up to Q{batch_index * 300}")

def finalize_flush(level):
    with question_locks[level]:
        if question_buffer[level]:
            batch_index = (question_counts[level] // 300) + 1
            buffer_copy = question_buffer[level][:]
            flush_to_file(level, batch_index, buffer_copy)
            print(f"🎯 Level {level} all questions have been written, total {question_counts[level]} questions")
            question_buffer[level] = []

def safe_gen(level, final_md, batch_size, max_workers):
    try:
        print(f"🚀 Starting generation of Level {level} questions, total required: {batch_size}")
        gen_multilevel_mcq_threaded(final_md, level, batch_size, max_workers=max_workers)
        finalize_flush(level)
        print(f"✅ Level {level} all questions have been written, total {question_counts[level]} questions")
    except Exception as e:
        print(f"❌ Error occurred during Level {level} generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"🎉 Level {level} thread execution completed, preparing to exit")
        return

def gen_multilevel_mcq_threaded(md_path, level=1, batch_size=30, max_workers=3):
    from collections import Counter

    blocks = load_md_sections(md_path)
    block_titles = list(blocks.keys())
    min_words = {1: 15, 2: 25, 3: 35}[level]
    seen_hashes = set()
    seen_theme_sets = Counter()

    print(f"🚀 Starting multi-threaded MCQ generation for Level {level}, target number: {batch_size}")

    def worker(q_index):
        mcq = generate_mcq_thread_task(block_titles, blocks, level, min_words, q_index=q_index)
        if not mcq:
            return None
        qh = question_hash(mcq)
        if qh in seen_hashes:
            return None
        with question_locks[level]:
            if qh in seen_hashes:  # Double check
                return None
            seen_hashes.add(qh)
        return mcq

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        i = 0
        while get_question_count(level) < batch_size:
            i += 1
            futures.append(executor.submit(worker, i))

            # Collect results in real-time
            for future in as_completed(futures):
                mcq = future.result()
                if mcq:
                    collect_and_maybe_flush(level, mcq)
                    print(f"✅ Level {level} generated question #{get_question_count(level)} (Attempts: {i})")
                if get_question_count(level) >= batch_size:
                    break
            futures = []  # Clear futures list for next batch

    print(f"🎯 Level {level} multi-threaded MCQ generation completed, total generated: {get_question_count(level)} questions")


    
 
def main():
    # title = generate_biomedical_topic()
    # print("🎯 Research Topic:", title)
    
    # subqueries = decompose_into_subqueries(title)
    # print("✅ Subqueries decomposition completed:", subqueries)

    # raw_lit = fetch_all_literature_from_subqueries(subqueries, title)

    # if not raw_lit:
    #     print("❌ No literature found, exiting.")
    #     return

    # # → The GPT secondary filtering part has been removed ←

    # save_literature_to_file(raw_lit, title)

    # literature = load_literature_from_file(os.path.join(SAVE_DIR, "max_literature.txt"))
    # report_text = generate_research_report(literature, save_path=os.path.join(SAVE_DIR, "report.txt"))
    # if not report_text:
    #     print("❌ Failed to generate report")
    #     return

    # markdown_report = convert_report_to_markdown(report_text)
    # if not markdown_report:
    #     print("❌ Failed to convert to Markdown")
    #     return
    # md_path = os.path.join(SAVE_DIR, "new_origin_biomedical_report.md")
    # save_markdown_report(markdown_report, md_path)

    # restructure_markdown_for_embedding(md_path, final_md)
    final_md = os.path.join(SAVE_DIR, "literature_max.md")
    levels = [1, 2]
    batch_size = 50
    
    for level in levels:
        safe_gen(level, final_md, batch_size, max_workers=5)






if __name__ == "__main__":
    main()