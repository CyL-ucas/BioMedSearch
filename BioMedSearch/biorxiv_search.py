import requests
from urllib.parse import quote

def generate_biorxiv_search_url(keywords: str, num_results: int = 1):
    base_url = "https://www.biorxiv.org/search/"
    search_url = f"{base_url}{quote(keywords)}%20jcode%3Abiorxiv%20numresults%3A{num_results}%20sort%3Arelevance-rank%20format_result%3Astandard"
    return search_url

def scrape_biorxiv_results(search_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }
    
    try:
        # Perform the GET request
        response = requests.get(search_url, headers=headers)
        
        # Check if we received HTML instead of JSON
        if response.status_code == 200:
            if 'application/json' in response.headers.get('Content-Type', ''):
                return response.json()  # Successfully returned JSON
            else:
                print("Error: Received HTML content instead of JSON")
                print(response.text)  # Print HTML content for debugging
                return None
        else:
            print(f"Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    key_words = "cyp17a1 zebrafish"
    search_url = generate_biorxiv_search_url(keywords=key_words, num_results=1)
    print(f"Generated URL: {search_url}")
    
    articles = scrape_biorxiv_results(search_url)
    if articles:
        print(articles)
