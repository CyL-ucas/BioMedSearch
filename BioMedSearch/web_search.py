import time
import requests
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException
import os
import sys
from bs4 import BeautifulSoup
import random

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from models import gpt41_orgin_llm

# Brave Search Configuration
BRAVE_API_KEY = ""
# Google Search Configuration
API_KEY = ""
CX = ""

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "num": num_results
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "href": item.get("link"),
            "body": item.get("snippet"),
            "source": "google"
        })
    return results

def brave_search(query, max_results=5):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {
        "q": query,
        "count": max_results
    }
    try:
        time.sleep(1)
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title"),
                "href": item.get("url"),
                "body": item.get("description"),
                "source": "brave"
            })
        return results
    except Exception as e:
        print(f"❌ Brave search failed: {e}")
        return []

def duckduckgo_search(query, max_results=5, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}
    results = []

    try:
        resp = requests.post(url, data=data, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", class_="result__a")

        for a in links[:max_results]:
            href = a.get("href")
            title = a.get_text(strip=True)
            snippet = a.find_parent("div", class_="result").find("a", class_="result__snippet")
            body = snippet.get_text(strip=True) if snippet else ""
            results.append({
                "title": title,
                "href": href,
                "body": body,
                "source": "duckduckgo"
            })
        time.sleep(random.uniform(delay, delay + 2))
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {e}")

    return results

def filter_by_gpt(query, docs, model):
    filtered = []
    for doc in docs:
        prompt = [
            {"role": "system", "content": "You are an expert in biomedical information filtering. Determine whether the following title and content can help answer the research question."},
            {"role": "user", "content": f"Question: {query}\nSource: {doc.get('source')}\nTitle: {doc.get('title')}\nContent: {doc.get('body')}\nPlease answer 'Yes' or 'No' only."}
        ]
        try:
            reply = model.chat(prompt).strip()
            if reply.startswith("Yes"):
                filtered.append(doc)
        except Exception as e:
            print(f"❌ GPT filtering failed: {e}")
    return filtered

def web_search(query, model, max_results=10):
    google_results = google_search(query, num_results=20)
    brave_results = brave_search(query, max_results=20)
    ddg_results = duckduckgo_search(query, max_results=20)

    for r in google_results:
        print(f"{r['title']} - {r['href']}\n{r['body']}\n")

    print(f"\n📥 Raw results returned from search engines:")
    print(f"🔹 Google: {len(google_results)} items")
    print(f"🔹 Brave : {len(brave_results)} items")
    print(f"🔹 DuckDuckGo: {len(ddg_results)} items")

    combined = google_results + brave_results + ddg_results
    filtered = []

    for doc in combined:
        prompt = [
            {"role": "system", "content": "You are an expert in biomedical information filtering. Determine whether the following content can directly help answer the research question."},
            {"role": "user", "content": f"Question: {query}\nSource: {doc.get('source')}\nTitle: {doc.get('title')}\nContent: {doc.get('body')}\nPlease answer 'Yes' or 'No' only."}
        ]
        try:
            reply = model.chat(prompt)
            if reply is None:
                raise ValueError("Empty response from model")
            if isinstance(reply, dict) and "content" in reply:
                reply = reply["content"]
            reply = reply.strip()
            if reply.startswith("Yes"):
                filtered.append(doc)
        except Exception as e:
            print(f"❌ GPT filtering failed: {e}")

    seen = set()
    final_results = []
    source_count = {}

    for doc in filtered:
        url = doc.get("href")
        if url and url not in seen:
            seen.add(url)
            final_results.append(doc)
            src = doc.get("source", "unknown")
            source_count[src] = source_count.get(src, 0) + 1
        if len(final_results) >= max_results:
            break

    print("\n📊 Source contribution after GPT filtering:")
    for src, count in source_count.items():
        print(f"🔹 {src}: {count} items")

    return final_results

# Example call
if __name__ == "__main__":
    query = input("🔍 Please enter your query: ")
    results = web_search(query, gpt41_orgin_llm)

    print("\n📄 Final Search Results:")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] 🔗 {res['title']} ({res['source']})")
        print(f"URL: {res['href']}")
        print(f"Snippet: {res['body']}")
