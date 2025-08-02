import os
import urllib.parse
from dotenv import load_dotenv
from Bio import Entrez

# ✅ Load environment variables
load_dotenv()
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
EMAIL = os.getenv("ENTREZ_EMAIL", "")  # Required by Entrez

class PubMedSearcher:
    BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"

    def __init__(self, api_key=PUBMED_API_KEY, email=EMAIL):
        self.api_key = api_key
        self.email = email
        Entrez.email = self.email
        Entrez.api_key = self.api_key

    def search_pubmed_url(self, query: str) -> str:
        encoded_query = urllib.parse.quote(query)
        return f"{self.BASE_URL}?term={encoded_query}"

    def search_pubmed(self, query: str, max_results: int = 100):
        """
        Query PubMed using Entrez API and return structured article entries.

        Each returned entry includes: title, authors, pubdate, doi, abstract.
        """
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            ids = record["IdList"]
            handle.close()

            if not ids:
                return []

            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="medline", retmode="xml")
            articles = Entrez.read(fetch_handle)["PubmedArticle"]
            fetch_handle.close()

            results = []
            for article in articles:
                info = article["MedlineCitation"]["Article"]
                title = info.get("ArticleTitle", "")
                abstract = info.get("Abstract", {}).get("AbstractText", [""])[0]
                authors = [
                    f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                    for a in info.get("AuthorList", [])
                    if "LastName" in a
                ]
                pubdate = article["MedlineCitation"].get("DateCompleted", {})
                pubyear = pubdate.get("Year", "Unknown")

                # Extract DOI
                doi = ""
                for id_item in info.get("ELocationID", []):
                    if id_item.attributes.get("EIdType") == "doi":
                        doi = str(id_item)

                results.append({
                    "title": str(title),
                    "abstract": str(abstract),
                    "authors": authors,
                    "pubdate": pubyear,
                    "doi": doi,
                    "source": "PubMed"
                })

            return results

        except Exception as e:
            print(f"❌ PubMed search failed: {e}")
            return []

# ✅ Example
if __name__ == "__main__":
    searcher = PubMedSearcher()
    query = "cyp17a1 zebrafish knockout"
    articles = searcher.search_pubmed(query, max_results=3)
    for i, article in enumerate(articles):
        print(f"\n📄 Article {i+1}")
        for k, v in article.items():
            print(f"{k}: {v}")
