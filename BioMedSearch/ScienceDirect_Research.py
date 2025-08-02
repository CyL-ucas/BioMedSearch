import requests
import urllib.parse
import logging
import time
from requests.exceptions import RequestException, ConnectionError

API_KEY = ""

class ScopusSearcher:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = "https://api.elsevier.com/content/search/scopus"

    def search(self, query, max_results=100, max_retries=3):
        logging.info(f"🚀 Starting Scopus search: {query}")
        start_time = time.time()

        formatted_query = f"TITLE-ABS-KEY({query})"
        params = {
            "query": formatted_query,
            "count": 25,      # ✅ Ensure compliance with Scopus limit
            "start": 0
        }
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }

        results = []
        total_retrieved = 0

        while total_retrieved < max_results:
            # Retry logic for network errors
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        self.base_url,
                        headers=headers,
                        params=params,
                        timeout=10
                    )
                    if response.status_code != 200:
                        logging.error(f"❌ Scopus request failed: {response.status_code} {response.text}")
                        return results  # ❗ API limit or error, return current results
                    break  # Success
                except (ConnectionError, RequestException) as e:
                    wait_time = 2 * (attempt + 1)
                    logging.warning(f"⚠️ Scopus network error ({e}), retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            else:
                logging.error("❌ Scopus failed after multiple network retries, terminating search.")
                return results

            try:
                data = response.json()
            except Exception as e:
                logging.error(f"❌ Failed to parse JSON response: {e}")
                break

            entries = data.get("search-results", {}).get("entry", [])
            if not entries:
                logging.info("🔚 No more results to retrieve.")
                break

            for entry in entries:
                title = entry.get("dc:title", "No Title")
                doi = entry.get("prism:doi", "No DOI")
                abstract = entry.get("dc:description", "")

                if not abstract.strip():
                    continue  # Skip articles with no abstract

                results.append({
                    "title": title.strip(),
                    "doi": doi.strip(),
                    "abstract": abstract.strip(),
                    "source": "Scopus"
                })

                total_retrieved += 1
                if total_retrieved >= max_results:
                    break

            if total_retrieved >= max_results:
                break

            params["start"] += params["count"]  # Pagination

        elapsed = time.time() - start_time
        logging.info(f"✅ Scopus search completed. Total results: {len(results)}. Time taken: {elapsed:.2f} seconds.")
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    searcher = ScopusSearcher()
    results = searcher.search("zebrafish gene expression", max_results=75)

    for idx, r in enumerate(results, 1):
        print(f"{idx}. {r['title']} ({r['doi']})\nAbstract: {r['abstract'][:100]}...\n")
