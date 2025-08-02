import logging
import requests
import json
import sys
import os
from typing import Union, Dict, List

# Set project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models import gpt41_orgin_llm

UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

class UniProtSearcher:
    def __init__(self):
        pass

    def fetch_uniprot_details(self, uniprot_id: str, fields: List[str]) -> Dict[str, str]:
        """Fetch information from UniProt based on uniprot_id and specified fields"""
        url = UNIPROT_BASE_URL.format(uniprot_id=uniprot_id)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logging.error(f"UniProt query failed: {e}")
            return {}

        result = {}
        if "Function" in fields:
            result["Function"] = self.extract_comment(data, "FUNCTION")
        if "Interaction" in fields:
            result["Interaction"] = self.extract_comment(data, "INTERACTION")
        if "Sequence" in fields:
            result["Sequence"] = self.extract_sequence(data)
        return result

    def extract_comment(self, data, comment_type: str) -> str:
        comments = data.get("comments", [])
        extracted = [
            text.get("value", "")
            for comment in comments
            if comment.get("commentType") == comment_type
            for text in comment.get("texts", [])
        ]
        return "\n".join(extracted) if extracted else f"No {comment_type} information available"

    def extract_sequence(self, data) -> str:
        return data.get("sequence", {}).get("value", "No sequence information available")

    def deepseek_query_uniprot_id(self, gene_name: str, organism: str = "Homo sapiens") -> Union[str, None]:
        messages = [{
            "role": "user",
            "content": f"Please find the UniProt ID using gene name: {gene_name} and organism name: {organism}. Only return the UniProt ID itself."
        }]
        response = gpt41_orgin_llm.chat(messages)
        if not response:
            logging.error("❌ DeepSeek query failed")
            return None
        return response.strip()

    def extract_fields_from_query(self, query: str) -> List[str]:
        """
        Use LLM to determine if the query contains fields to be searched: Function, Interaction, Sequence, Domain
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Determine if the following sentence requires searching these UniProt fields: Function, Interaction, Sequence, Domain. "
                    f"If mentioned, return a JSON array like [\"Function\", \"Interaction\"]. Otherwise, return [].\n"
                    f"Sentence: {query}"
                )
            }
        ]
        response = gpt41_orgin_llm.chat(messages)
        try:
            fields = json.loads(response.strip())
            if "Domain" in fields:
                fields.extend(["Function", "Sequence"])
                fields = list(set(fields) - {"Domain"})
            return fields
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse fields from GPT response: {response}")
            return []

    def search_by_query_intent(self, query: str, uniprot_id: str) -> Dict[str, str]:
        if not uniprot_id:
            return {"error": "UniProt ID is missing"}
        fields = self.extract_fields_from_query(query)
        if not fields:
            return {"notice": "No Function/Interaction/Sequence/Domain-related fields detected in the query, UniProt search not required"}
        return self.fetch_uniprot_details(uniprot_id, fields)

# Main function test
if __name__ == "__main__":
    searcher = UniProtSearcher()
    query = "interaction, function, sequence of cyp17a1 protein"
    gene_name = "cyp17a1"
    organism = "Homo sapiens"
    # uniprot_id = searcher.deepseek_query_uniprot_id(gene_name, organism)
    uniprot_id = "B3DH80"
    if not uniprot_id:
        print("Failed to retrieve UniProt ID")
        sys.exit(1)

    info = searcher.search_by_query_intent(query, uniprot_id)

    print(f"UniProt ID: {uniprot_id}")
    print("Retrieved information (only displaying valid fields):\n")

    for key, value in info.items():
        if isinstance(value, str) and not value.startswith("No"):
            print(f"{key}:\n{value}\n{'-'*40}")
