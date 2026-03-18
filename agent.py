from tools import query_pubmed, summarize_papers_stats, search_clinical_trials
from llm import rerank_papers, summarize_papers, llm_parse_query
from config import DEFAULT_MAX_PAPERS, RERANK_TOP_K


class DemoAgent:

    def __init__(self):
        pass

    # def run(self, query):
    #     # step 1: retrieve candidate papers
    #     candidate_papers = query_pubmed(query, max_papers=DEFAULT_MAX_PAPERS)

    #     # step 2: summary statistics on candidate set
    #     stats = summarize_papers_stats(candidate_papers)

    #     # step 3: rerank and keep top K
    #     top_papers = rerank_papers(query, candidate_papers, top_k=RERANK_TOP_K)

    #     # step 4: summarize selected papers
    #     summary = summarize_papers(query, top_papers, stats)

    #     return {
    #         "candidate_papers": candidate_papers,
    #         "papers": top_papers,
    #         "stats": stats,
    #         "summary": summary
    #     }

    def run_agent(self, user_input: str):

        print("\n[Step 1] Parsing user query with LLM...")
        query = llm_parse_query(user_input)
        print("Parsed query:", query)

        print("\n[Step 2] Searching ClinicalTrials.gov...")
        df = search_clinical_trials(query)

        print(f"\n[Step 3] Found {len(df)} trials")
        print(df.head())

        return df