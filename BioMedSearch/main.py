import json
import urllib.error
import requests
import re
from dotenv import load_dotenv
import csv
import textwrap
import os
import requests
from bs4 import BeautifulSoup
import logging
import chardet
# from frontend.find_answer import evaluate_mcq_with_semantic_assist
from find_answer import evaluate_mcq_with_semantic_assist
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
import sys
sys.path.append(project_root)
import time
import asyncio
import concurrent.futures
from copy import deepcopy
from models import gpt41_llm, gemini_llm,claude_llm,qwen_llm,deepseek_llm,gpt41_openai_llm,claude4_llm,llama_LLM
from pubmed_search import PubMedSearcher
from pmc_search import PMCSearcher
from uniprot_search import UniProtSearcher
from ScienceDirect_Research import ScopusSearcher
from alphafold_search import AlphaFoldSearcher
from web_search import web_search
import time
DEFAULT_ORGANISM = "Danio rerio"
DATASET_DIR = r""
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()
from typing import List, Dict

history = []
final_report_cached = None  
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

def parse_answer_line(line):
    match = re.match(r"^(Q\d+)[\s:]*Answer[:：]?\s*([A-D](?:[,，、 ]\s*[A-D])*)", line.strip(), re.IGNORECASE)
    if match:
        qid = match.group(1).strip().upper()
        ans = ', '.join(re.findall(r"[A-D]", match.group(2).upper()))
        return qid, ans
    return None, None
def chat_with_memory(messages, max_retries=5):

    global history
    if not history:
        history = messages  
    else:
        first_system = [m for m in history if m["role"] == "system"]
        history = first_system + [m for m in messages if m["role"] == "user"]

    for attempt in range(max_retries):
        try:
            response = gpt41_llm.chat(history).strip()
            return response
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            time.sleep(wait_time)

    raise RuntimeError("chat_with_memory failed repeatedly, terminating.")

def parse_subtasks(user_input: str):
    prompt = f"""
    You are a biomedical expert. Please decompose the following question into 3-5 biomedical sub-questions. Each sub-question should be concise, complete, cover different perspectives or mechanisms, and should not exceed 20 words. You don't need to consider retrieval tools.
    
    User input: {user_input}

    Return a JSON array, where each object contains a 'subquery' field representing the sub-question.
    """
    content = gpt41_llm.chat([{"role": "user", "content": prompt}])

    print("📝 Raw LLM Response:", content)

    try:
        subtasks = safe_json_load(content)
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        subtasks = []

    if isinstance(subtasks, dict):
        subtasks = [subtasks]
    elif isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
        subtasks = [{"subquery": s} for s in subtasks]

    print("🚀 Step 1: Parsed Subtasks Result")
    for subtask in subtasks:
        try:
            print(f"Sub-question: {subtask['subquery']}")
        except Exception as e:
            print(f"⚠️ Failed to parse sub-question: {e} -> {subtask}")
    return subtasks

def extract_keywords_via_llm(subquery: str) -> List[str]:
    """
    Extract biomedical keywords from the subquery using LLM.
    Returns a list of keyword strings.
    """
    prompt = [
        {"role": "system", "content": "You are a biomedical keyword extraction expert."},
        {"role": "user", "content": f"Please extract biomedical keywords from the following sentence and return a JSON array (no more than 6 keywords):\nSentence: {subquery}"}
    ]
    try:
        response = gpt41_llm.chat(prompt).strip()
        keywords = safe_json_load(response)
        if isinstance(keywords, list):
            return [k.strip() for k in keywords if isinstance(k, str)]
    except Exception as e:
        print(f"❌ Keyword extraction failed: {e}")
    return []

def build_keyword_dag(subquery: str, keywords: List[str]) -> List[str]:
    """
    Construct a logical order (DAG) of keywords using LLM. Only sorting is performed.
    Returns a sorted list of keywords.
    """
    prompt = [
        {"role": "system", "content": "You are a biomedical knowledge graph expert, skilled at constructing logical sequences between keywords based on semantics."},
        {"role": "user", "content": f"""
Please sort the following keywords extracted from the sub-question: {subquery}
Keywords: {keywords}

Sort them in logical order (from basic to complex, from background to mechanism), and return only the sorted JSON array, for example:
["background", "protein interaction", "inflammation"]
"""}
    ]
    try:
        response = gpt41_llm.chat(prompt).strip()
        sorted_keywords = safe_json_load(response)
        if isinstance(sorted_keywords, list):
            return [k.strip() for k in sorted_keywords if isinstance(k, str)]
    except Exception as e:
        print(f"❌ DAG sorting failed: {e}")
    return keywords

def assign_tools_to_keyword_group(subquery: str, keyword_group: List[str]) -> List[str]:
    """
    Assign retrieval tools based on the overall semantic needs of the sorted keyword group, not individually per keyword.
    Returns a list such as: ["literature", "webSearch"]
    """
    prompt = [
        {"role": "system", "content": "You are a biomedical expert skilled at determining required information retrieval tools based on keyword groups."},
        {"role": "user", "content": f"""
I have a sub-question: {subquery}
The extracted and sorted keyword group is: {keyword_group}

Please assign appropriate retrieval tools based on the overall semantic requirements of this keyword group (up to three). Available options are:

- literature: For retrieving biomedical research articles.
- uniprot: For retrieving gene/protein functions, interactions, sequences, etc.
- webSearch: For exploring disease mechanisms, background knowledge, or content not suitable for the above.

Return a JSON array, for example: ["literature", "webSearch"]
"""}
    ]
    try:
        response = gpt41_llm.chat(prompt).strip()
        tools = safe_json_load(response)
        if isinstance(tools, list):
            return [t for t in tools if t in {"literature", "webSearch", "uniprot"}]
    except Exception as e:
        print(f"❌ Tool assignment failed: {e}")
    return ["webSearch"]

def run_dispatch_in_parallel(subtasks, gene=None, organism=None, uniprot_id=None):
    """
    Execute keyword processing and tool-based retrieval for each subtask using unified dispatch_tools.
    """
    results = {}
    all_docs = []
    subanswers = []

    for subtask in subtasks:
        try:
            print(f"\n🚀 Executing subtask: {subtask['subquery']}")
            partial_results, partial_docs, partial_subanswers = dispatch_tools([subtask], gene, organism, uniprot_id)
            results.update(partial_results)
            all_docs.extend(partial_docs)
            subanswers.extend(partial_subanswers)
        except Exception as e:
            print(f"❌ Subtask execution failed: {e}")

    return results, all_docs, subanswers





def extract_webpage_text(url):
    """
    Extract webpage main content using requests + chardet + BeautifulSoup, handling encoding issues.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        raw_bytes = response.content
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8")
        decoded_text = raw_bytes.decode(encoding, errors="ignore")

        soup = BeautifulSoup(decoded_text, "html.parser")

        # Prefer extracting main structural content
        for tag in ['article', 'main', 'section']:
            content = soup.find(tag)
            if content:
                return content.get_text(separator='\n', strip=True)

        # Otherwise, concatenate all paragraph texts
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
        return text.strip()

    except Exception as e:
        print(f"❌ Failed to extract webpage text: {url} - {e}")
        return ""

def clean_webpage_content_via_llm(raw_text, query):
    """
    Clean webpage content using LLM to remove irrelevant information and retain only content highly relevant to the query.
    """
    prompt = [
        {"role": "system", "content": "You are a webpage content cleaning expert, specialized in extracting key biomedical information and fixing broken paragraph structures."},
        {"role": "user", "content": f"""
Please extract content relevant to the user's query from the following webpage text, ensuring logical flow and semantic completeness.

Requirements:

1. Remove irrelevant content:
- Author affiliations, contact info, DOI, database navigation bars, publication info;
- Related articles, citations, keyword tags, data source descriptions;
- Figure/table captions (e.g., Fig. 1, Table 2) and meaningless section numbers;

2. Automatically merge broken long paragraphs and fix sentence splits caused by webpage formatting, e.g.,
- "Staphylococcus aureus , and\nEscherichia coli." should be merged into a complete sentence;
- Do not allow sentences to be split across different paragraphs.

3. Retain high-quality scientific content, including:
- Research background and significance;
- Key mechanisms like microbial metabolites, neuroinflammation, gut-brain axis;
- Important experimental observations and conclusions;
- Research findings on diseases (e.g., Alzheimer's, Parkinson's, multiple sclerosis).

4. Output format:
- Ensure logical flow and semantic completeness;
- Organize content in natural paragraphs;
- Do not add any explanations or extra titles.

User query:
{query}

Original webpage text:
{raw_text}

Please return only the **cleaned** content.
"""}
    ]
    try:
        cleaned = gpt41_llm.chat(prompt).strip()
        return cleaned
    except Exception as e:
        print(f"❌ Cleaning failed: {e}")
        return raw_text  # Return original content if cleaning fails
  
from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_MODEL = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device=str(device))


def dispatch_tools(subtasks, gene=None, organism=None, uniprot_id=None):
    results = {}
    all_docs = []
    subanswers = []
    kept_literature_count = 0
    kept_web_count = 0

    for subtask in subtasks:
        query = subtask["subquery"]
        subanswer = {}

        print(f"\n📥 Processing subtask: {query}")

        keywords = extract_keywords_via_llm(query)
        print(f"🔑 Extracted keywords: {keywords}")
        sorted_keywords = build_keyword_dag(query, keywords)
        print(f"📐 Sorted keywords: {sorted_keywords}")

        tools = assign_tools_to_keyword_group(query, sorted_keywords)
        print(f"🛠️ Assigned tools: {tools}")

        query_embedding = EMBED_MODEL.encode(query, convert_to_tensor=True)
        group_string = " ".join(sorted_keywords)

        def run_tool(tool):
            try:
                if tool == "literature":
                    _, all_raw_docs = aggregate_literature_results_from_subtasks([
                        {"tool": "literature", "query": group_string}
                    ])
                    if not all_raw_docs:
                        return []

                    corpus = []
                    corpus_meta = []
                    for doc in all_raw_docs:
                        title = doc.get("title", "")
                        abstract = clean_abstract(doc.get("abstract", ""))
                        if title or abstract:
                            text = f"{title}. {abstract}"
                            corpus.append(text)
                            corpus_meta.append(doc)

                    if not corpus:
                        return []

                    doc_embeddings = EMBED_MODEL.encode(corpus, convert_to_tensor=True)
                    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
                    top_indices = torch.topk(scores, k=min(10, len(corpus))).indices.tolist()
                    top_docs = [corpus_meta[i] for i in top_indices]

                    subanswer["literature"] = top_docs
                    return top_docs

                elif tool == "webSearch":
                    ddg_result = web_search(group_string, gpt41_llm)
                    all_web_docs = []

                    for res in ddg_result:
                        text = res.get("text", "").strip()
                        if not text and res.get("href"):
                            real_text = extract_webpage_text(res["href"])
                            if real_text:
                                text = real_text
                        if text:
                            cleaned = clean_webpage_content_via_llm(text, query)
                            all_web_docs.append({
                                "title": res.get("title", "Web Page"),
                                "abstract": cleaned,
                                "href": res.get("href", "")
                            })

                    corpus = [f"{doc['title']}. {doc['abstract']}" for doc in all_web_docs]
                    if not corpus:
                        return []

                    doc_embeddings = EMBED_MODEL.encode(corpus, convert_to_tensor=True)
                    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
                    top_indices = torch.topk(scores, k=min(10, len(corpus))).indices.tolist()
                    top_docs = [all_web_docs[i] for i in top_indices]

                    subanswer["webSearch"] = top_docs
                    return top_docs

                elif tool == "uniprot" and uniprot_id:
                    uniprot_result = UniProtSearcher().fetch_uniprot_details(uniprot_id, ["function", "interaction", "sequence"])
                    subanswer["uniprot"] = uniprot_result
                    return [{"title": "UniProt Protein Information", "abstract": uniprot_result}]

                elif tool == "alphafold" and uniprot_id:
                    alphafold_result = AlphaFoldSearcher().search_alphafold(uniprot_id)
                    subanswer["alphafold"] = alphafold_result
                    return [{"title": "AlphaFold 3D Structure", "abstract": alphafold_result}]

                return []

            except Exception as e:
                print(f"❌ Retrieval failed for {tool}: {e}")
                return []

        for tool in tools:
            docs = run_tool(tool)
            all_docs.extend(docs)

        if can_answer_subquery(subanswer, query):
            results[query] = subanswer
            subanswers.append(subanswer)
            if "literature" in subanswer:
                kept_literature_count += len(subanswer["literature"])
            if "webSearch" in subanswer:
                kept_web_count += len(subanswer["webSearch"])

    print(f"\n📊 Validated subtasks: {len(results)}")
    print(f"📚 Retained literature count: {kept_literature_count}")
    print(f"🌐 Retained web content count: {kept_web_count}")

    return results, all_docs, subanswers


def save_final_documents_summary(all_docs, dataset_dir):
    literature_docs = []
    web_docs = []

    for doc in all_docs:
        title = doc.get("title", "").strip()
        abstract = str(doc.get("abstract", "")).strip()
        href = doc.get("href", "").strip()

        if "webSearch" in title.lower() or href:
            web_docs.append({
                "title": title,
                "href": href,
                "text": abstract,
            })
        elif title or abstract:
            literature_docs.append({
                "title": title,
                "abstract": abstract,
            })

    lines = []

    lines.append("📚 Final Literature Results\n")
    for i, doc in enumerate(literature_docs, 1):
        lines.append(f"{i}. {doc['title']}")
        if doc["abstract"]:
            lines.append(f"   Abstract: {doc['abstract']}\n")

    lines.append("\n🌐 Final Web Search Results\n")
    for i, doc in enumerate(web_docs, 1):
        lines.append(f"{i}. {doc['title']}")
        if doc["text"]:
            lines.append(f"   Webpage Summary: {doc['text']}\n")

    file_path = os.path.join(dataset_dir, "final_documents_summary.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Summary saved to: {file_path}")


def can_answer_subquery(subanswer, query):
    """
    Use LLM to determine if the subanswer can answer the given query.
    Returns True if it can, False otherwise.
    """
    prompt = {
        "role": "user",
        "content": (
            f"Determine whether the following information can answer the query:\n\n"
            f"Query: {query}\n\n"
            f"Answer: {json.dumps(subanswer, ensure_ascii=False)}\n\n"
            "If yes, return 'yes'; otherwise, return 'no'."
        )
    }

    response = gpt41_llm.chat([{"role": "system", "content": "You are a biomedical information validation expert."}, prompt])

    return response.lower().strip() == "yes"



gene = None
organism = None

def extract_gene_and_organism(user_input: str):
    global gene, organism

    if gene and organism:
        print(f"✅ Using cached information - Gene: {gene}, Organism: {organism}")
        return gene, organism

    print("🧬 Step 2: Extracting gene and organism (using GPT-4o)")
    messages = [{
        "role": "user",
        "content": f"Extract gene name and organism (scientific Latin name) from the following sentence. Return as JSON, e.g., {{\"gene\": \"TP53\", \"organism\": \"Homo sapiens\"}}. If the organism is not mentioned, default to Homo sapiens.\nSentence: {user_input}"
    }]
    response = gpt41_llm.chat(messages)
    print("🔍 LLM Response:", response)
    try:
        result = safe_json_load(response)
        if not result:
            raise ValueError("❌ Failed to parse gene and organism JSON")
        gene = result.get("gene")
        organism = result.get("organism", "Danio rerio")
        print(f"✅ Step 2 Complete - Gene: {gene}, Organism: {organism}")
        return gene, organism
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        return None, "Danio rerio"

def get_uniprot_id(gene: str, organism: str):
    return UniProtSearcher().deepseek_query_uniprot_id(gene, organism)

def translate_to_english(query: str):
    return gpt41_llm.chat([{"role": "user", "content": f"Rearrange the following terms into a logical academic sequence and translate them into academic English. Return only the logically ordered English keywords: {query}"}]).strip()

def select_top_keywords_via_gpt(keywords, query, topk=3):
    prompt = [
        {"role": "system", "content": "You are a keyword selection expert."},
        {"role": "user", "content": f"""
Based on the following subquery and keyword list, select the top 3 keywords most relevant to the query.

Subquery: {query}
Keyword list: {', '.join(keywords)}

Return a JSON array, e.g., ["microbiome", "cognitive function", "metabolites"]
"""}
    ]
    try:
        response = gpt41_llm.chat(prompt).strip()
        cleaned = re.sub(r"^json\n?|$", "", response.strip(), flags=re.I).strip()
        top_keywords = json.loads(cleaned)

        if isinstance(top_keywords, list) and all(isinstance(k, str) for k in top_keywords):
            return top_keywords[:topk]

    except Exception as e:
        print(f"❌ GPT keyword selection failed: {e}\n🔍 Raw response: {repr(response)}")

    return keywords[:topk]

def safe_query_with_retry(func, query, max_retries=5, sleep_base=5):
    for attempt in range(max_retries):
        try:
            return func(query)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = sleep_base * (2 ** attempt)
                print(f"⚠️ Rate limit hit (429), retrying in {wait} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as e:
            if '429' in str(e):
                wait = sleep_base * (2 ** attempt)
                print(f"⚠️ Scopus rate limited, retrying in {wait} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    print(f"❌ Failed after {max_retries} consecutive attempts, skipping.")
    return []

def aggregate_literature_results_from_subtasks(subtasks):
    final_results = {}
    all_docs = []

    pubmed_searcher = PubMedSearcher()
    pmc_searcher = PMCSearcher()

    for task in subtasks:
        if task.get("tool") != "literature":
            continue

        query = task.get("query", "").strip()
        print(f"\n📥 Current subtask query: {query}")

        en_query = translate_to_english(query)
        print(f"🌐 Translated to English: {en_query}")

        pubmed_raw, pmc_raw = concurrent_literature_search(en_query, pubmed_searcher, pmc_searcher)
        print(f"📄 PubMed: {len(pubmed_raw)} | PMC: {len(pmc_raw)}")

        all_raw_docs = pubmed_raw + pmc_raw
        if not all_raw_docs:
            continue

        corpus = []
        meta_docs = []
        for doc in all_raw_docs:
            title = doc.get("title", "")
            abstract = clean_abstract(doc.get("abstract", ""))
            if title or abstract:
                text = f"{title}. {abstract}"
                corpus.append(text)
                meta_docs.append(doc)

        if not corpus:
            continue

        query_embedding = EMBED_MODEL.encode(query, convert_to_tensor=True)
        doc_embeddings = EMBED_MODEL.encode(corpus, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_indices = torch.topk(scores, k=min(50, len(corpus))).indices.tolist()
        top_docs = [meta_docs[i] for i in top_indices]

        final_results[query] = top_docs
        all_docs.extend(top_docs)

        print(f"✅ Retained documents after similarity filtering: {len(top_docs)}")
        for i, doc in enumerate(top_docs, 1):
            print(f"{i}. {doc.get('title', '')[:300]}")

    return final_results, all_docs

def concurrent_literature_search(keyword_query, pubmed_searcher, pmc_searcher):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_pubmed = executor.submit(pubmed_searcher.search_pubmed, keyword_query)
        future_pmc = executor.submit(pmc_searcher.search_pmc, keyword_query)

        pubmed_result = future_pubmed.result()
        pmc_result = future_pmc.result()

    return pubmed_result, pmc_result

def summarize_final_results(results, all_docs):
    """
    Generate summary by deduplicating and cleaning, no new information is added.
    """
    summary_parts = []
    seen = set()
    valid_count = 0
    for doc in all_docs:
        title = str(doc.get('title', '')).strip()
        abstract_raw = doc.get('abstract', '')
        abstract = clean_abstract(abstract_raw)

        key = f"{title}|||{abstract}"
        if not title and not abstract:
            continue
        if key in seen:
            continue
        seen.add(key)
        valid_count += 1
        summary_parts.append(f"Title: {title}\nAbstract: {abstract}")

    combined_summary = "\n\n".join(summary_parts)
    if not combined_summary:
        combined_summary = "No relevant information retrieved."

    print(f"\n📚 Valid document count: {valid_count}")
    return combined_summary



def generate_report_from_results(results: dict, all_docs: list):
    """
    ⚠️ New Version: Retain all original retrieval results, formatting is fully handled by LLM.
    """
    return {
        "summary": "",  # Summary will no longer be concatenated here
        "docs": all_docs  # Return the raw document list
    }

def clean_abstract(abstract_raw):
    """
    Safely clean the abstract field, supporting str, dict, and list types.
    Returns: Cleaned abstract as string.
    """
    if isinstance(abstract_raw, list):
        return " ".join([
            item.get('text', str(item)) if isinstance(item, dict) else str(item)
            for item in abstract_raw
        ]).strip()
    elif isinstance(abstract_raw, dict):
        return json.dumps(abstract_raw, ensure_ascii=False)
    elif isinstance(abstract_raw, str):
        return abstract_raw.strip()
    else:
        return str(abstract_raw).strip()

def safe_json_load(text):
    """
    Extract JSON from LLM response text.
    - Prioritize parsing complete JSON arrays.
    - Supports markdown-wrapped or prefixed formats.
    """
    text = re.sub(r"^(\s*json|\s*|json\n?)", "", text.strip(), flags=re.I).strip(" \n")

    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        pass

    matches = re.findall(r'\[[\s\S]*?\]', text)
    for match in matches:
        try:
            parsed = json.loads(match)
            return parsed
        except Exception:
            continue

    matches = re.findall(r'\{[\s\S]*?\}', text)
    objects = []
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            continue
    if objects:
        return objects  # ✅ Return as list of objects

    return None

def txt_to_markdown(txt_path, md_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    md_lines = []
    section = None
    idx = 0

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("📚 Final Literature Results"):
            section = "literature"
            idx = 0
            md_lines.append("# 📚 Literature Results\n")
            continue
        elif line.startswith("🌐 Final Web Search Results"):
            section = "web"
            idx = 0
            md_lines.append("# 🌐 Web Search Results\n")
            continue

        match = re.match(r"^(\d+)\.\s+(.*)", line)
        if match:
            idx += 1
            title = match.group(2)
            md_lines.append(f"## {idx}. {title}")
            continue

        if line.startswith("摘要:") or line.startswith("正文摘要:"):
            abstract = line.split(":", 1)[1].strip()
            md_lines.append(f"{abstract}\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_lines))

    print(f"✅ Markdown file saved: {md_path}")

def read_topic():
    topic_path = os.path.join(DATASET_DIR, "topic_level1.txt")
    if not os.path.exists(topic_path):
        print("❌ Topic file not found. Please run construct_dataset.py first.")
        exit(1)
    with open(topic_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    user_input = read_topic()
    print(f"🧠 Automatically read biomedical query: {user_input}")

    if not user_input:
        print("❌ Input cannot be empty!")
        return

    # Step 1: Parse subtasks
    print("🚀 Step 1: Parsing subtasks")
    subtasks = parse_subtasks(user_input)
    print("✅ Subtasks parsed:", json.dumps(subtasks, ensure_ascii=False, indent=2))

    # Step 2: Check if UniProt query is needed
    needs_uniprot = any("uniprot" in t.get("tools", []) or "alphafold" in t.get("tools", []) for t in subtasks)
    gene = organism = uniprot_id = None

    if needs_uniprot:
        gene, organism = extract_gene_and_organism(user_input)
        uniprot_id = get_uniprot_id(gene, organism)
        print(f"✅ UniProt ID: {uniprot_id}")

    # Step 3: Execute all subtasks (dispatch_tools handles keyword extraction, tool assignment, DAG sorting)
    results, all_docs, subanswers = run_dispatch_in_parallel(subtasks, gene, organism, uniprot_id)

    # Step 4: Save literature and web summaries
    save_final_documents_summary(all_docs, DATASET_DIR)

    # Step 5: Generate and display summary content
    summary = generate_report_from_results(results, all_docs)

    print("\n============== ✅ Retrieval Complete ==============")
    print(f"🧬 Gene: {gene or 'Not Identified'} | Organism: {organism or 'Homo sapiens'}")
    if uniprot_id:
        print(f"🔗 UniProt ID: {uniprot_id}")
    print(f"✅ Total documents prepared for display (unfiltered): {len(summary['docs'])}")

    print("📝 Generating summary...")
    final_summary_path = os.path.join(DATASET_DIR, "final_nokw_summary.txt")
    with open(final_summary_path, encoding="utf-8") as f:
        final_report = f.read()
        txt_file = final_summary_path
        md_file = os.path.join(DATASET_DIR, "final_summary.md")
        txt_to_markdown(txt_file, md_file)

    for level in range(1, 4):
        print(f"\n==== 📘 Level {level} Multiple-Choice Question Evaluation Start ====")

        qfile = os.path.join(DATASET_DIR, f"questions_level{level}_final_balanced_1000.txt")
        afile = os.path.join(DATASET_DIR, f"answers_level{level}_final_balanced_1000.txt")

        if not os.path.exists(qfile) or not os.path.exists(afile):
            print(f"⚠️ Missing question or answer file for Level {level}, skipping.")
            continue

        questions = parse_question_file(qfile)[:10]
        answer_map = {}
        with open(afile, "r", encoding="utf-8") as fa:
            for line in fa:
                qid, ans = parse_answer_line(line)
                if qid and ans:
                    answer_map[qid] = ans

        answers = []
        for q in questions:
            qid = q["qid"]
            if qid not in answer_map:
                print(f"⚠️ Standard answer not found for QID {qid}, marking as Unanswered.")
                answers.append("Unanswered")
            else:
                answers.append(answer_map[qid])

        if len(questions) != len(answers):
            print(f"❌ Mismatch in number of questions and answers for Level {level}, skipping.")
            continue

        model_name = getattr(gpt41_llm, "model_name", "deepseek_llm").replace("/", "-")

        result_file = os.path.join(
            DATASET_DIR,
            f"gpt41_llm{level}_BioMedAnswerLog.csv"
        )

        evaluate_mcq_with_semantic_assist(
            report_md_path=os.path.join(DATASET_DIR, "literature_max.md"),
            question_items=questions,
            answer_items=answers,
            result_file=result_file,
            level_name=f"Level{level}"
        )

if __name__ == "__main__":
    main()



