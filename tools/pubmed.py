import time
from pymed import PubMed
from config import DEFAULT_MAX_PAPERS, DEFAULT_MAX_RETRIES


def query_pubmed(query, max_papers=DEFAULT_MAX_PAPERS, max_retries=DEFAULT_MAX_RETRIES):
    """Query PubMed and return structured paper results."""

    try:
        pubmed = PubMed(tool="AgenticAI", email="your_email@example.com")

        papers = list(pubmed.query(query, max_results=max_papers))

        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            words = query.split()
            simplified_query = " ".join(words[:-retries]) if len(words) > retries else query

            print(f"Retry {retries}, simplified query: {simplified_query}")
            time.sleep(1)

            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        if papers:
            results = []
            for paper in papers:
                results.append({
                    "pubmed_id": getattr(paper, "pubmed_id", None),
                    "title": paper.title or "No title available",
                    "abstract": paper.abstract or "No abstract available",
                    "journal": paper.journal or "No journal available",
                    "doi": getattr(paper, "doi", None),
                    "keywords": getattr(paper, "keywords", None),
                })
            return results

        else:
            return []

    except Exception as e:
        print(f"Error querying PubMed: {e}")
        return []