import logging
from typing import Union, Dict

# Base URL of AlphaFold structure prediction database
ALPHAFOLD_BASE_URL = "https://alphafold.ebi.ac.uk/entry/"

class AlphaFoldSearcher:
    """
    Query AlphaFold predicted 3D structure information via UniProt ID.
    """
    def __init__(self):
        pass

    def search_alphafold(self, uniprot_id: str) -> Union[str, Dict[str, str]]:
        """
        Query AlphaFold predicted 3D structure using UniProt ID.

        :param uniprot_id: The UniProt ID to query.
        :return: A dictionary containing the AlphaFold 3D structure URL and PDB download link.
        """
        if not uniprot_id:
            logging.warning("⚠️ Unable to query AlphaFold: UniProt ID is missing.")
            return {
                "AlphaFold_URL": "⚠️ Unable to query AlphaFold: UniProt ID is missing.",
                "PDB_Download_URL": "⚠️ Unable to download PDB file: UniProt ID is missing."
            }

        # ✅ Generate AlphaFold page URL and PDB file download URL
        alphafold_url = f"{ALPHAFOLD_BASE_URL}{uniprot_id}"
        pdb_download_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

        return {
            "AlphaFold_URL": alphafold_url,
            "PDB_Download_URL": pdb_download_url
        }

