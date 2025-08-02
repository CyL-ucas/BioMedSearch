import os
import time
import urllib.error
import socket
import ssl
from dotenv import load_dotenv
from Bio import Entrez
from Bio import Medline

# ✅ Load environment variables
load_dotenv()
PMC_API_KEY = os.getenv("PUBMED_API_KEY", "")
EMAIL = os.getenv("ENTREZ_EMAIL", "")

class PMCSearcher:
    BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/"

    def __init__(self, api_key=PMC_API_KEY, email=EMAIL):
        self.api_key = api_key
        self.email = email
        Entrez.email = self.email
        Entrez.api_key = self.api_key

    def search_pmc_url(self, query: str) -> str:
        return f"{self.BASE_URL}?term={query}"

    def search_pmc(self, query: str, max_results: int = 100, retries: int = 3, delay: int = 5):
        for attempt in range(retries):
            try:
                print(f"PMC search URL: {self.search_pmc_url(query)}")

                with Entrez.esearch(db="pmc", term=query, retmax=max_results) as handle:
                    record = Entrez.read(handle)
                    ids = record.get("IdList", [])

                if not ids:
                    print("No PMC articles found.")
                    return []

                # Fetch full abstracts using efetch in medline format
                with Entrez.efetch(db="pmc", id=",".join(ids), rettype="medline", retmode="text") as fetch_handle:
                    records = Medline.parse(fetch_handle)
                    records = list(records)

                results = []
                for rec in records:
                    title = rec.get("TI", "No Title")
                    abstract = rec.get("AB", "")  # Abstract field in medline format is AB
                    pmid = rec.get("PMID", "")
                    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}" if pmid else "No URL"

                    if not abstract.strip():
                        continue  # Skip articles with no abstract

                    results.append({
                        "title": title,
                        "abstract": abstract.strip(),
                        "url": url,
                        "source": "PMC"
                    })

                return results

            except Exception as e:
                print(f"PMC search failed (Attempt {attempt+1}): {e}")
                time.sleep(delay * (attempt + 1))

        print("❌ PMC search failed after maximum retries.")
        return []

if __name__ == "__main__":
    searcher = PMCSearcher()
    results = searcher.search_pmc("gut microbiota metabolites Alzheimer's disease")
    for r in results[:3]:
        print(f"{r['title']} - {r['url']}")
