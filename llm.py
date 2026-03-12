from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME, RERANK_TOP_K

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


def _build_paper_context(papers):
    context = ""
    for i, p in enumerate(papers, 1):
        context += f"""
Paper {i}
PMID: {p.get('pubmed_id')}
Title: {p.get('title')}
Journal: {p.get('journal')}
Abstract: {p.get('abstract')}
"""
    return context


def rerank_papers(query, papers, top_k=RERANK_TOP_K):
    """
    Use LLM to rerank papers by relevance to the user query.
    Return the selected top_k papers.
    """
    if len(papers) <= top_k:
        return papers

    context = _build_paper_context(papers)

    prompt = f"""
User query: {query}

Below are {len(papers)} candidate PubMed papers:

{context}

Select the {top_k} most relevant papers for the user query.

Return ONLY a comma-separated list of paper numbers, for example:
1,3,5,7,8,10

Do not explain anything.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical literature relevance judge."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.strip()

    try:
        selected_indices = []
        for x in text.split(","):
            x = x.strip()
            if x.isdigit():
                idx = int(x)
                if 1 <= idx <= len(papers):
                    selected_indices.append(idx - 1)

        seen = set()
        selected_indices = [i for i in selected_indices if not (i in seen or seen.add(i))]

        selected_papers = [papers[i] for i in selected_indices[:top_k]]
        
        if len(selected_papers) < top_k:
            for p in papers:
                if p not in selected_papers:
                    selected_papers.append(p)
                if len(selected_papers) == top_k:
                    break

        return selected_papers

    except Exception:
        return papers[:top_k]


def summarize_papers(query, papers, stats):
    context = _build_paper_context(papers)

    prompt = f"""
User query: {query}

Summary statistics on the full candidate set:
- Number of candidate papers: {stats['n_papers']}
- Missing abstracts: {stats['missing_abstract_count']}
- Average abstract length: {stats['avg_abstract_length']}
- Top journals: {stats['top_journals']}
- Top keywords: {stats['top_keywords']}

Below are the top reranked PubMed papers:

{context}

Please do the following:
1. Summarize the main research themes.
2. Mention any obvious patterns from the candidate-set summary statistics.
3. Briefly note whether the selected papers seem focused or heterogeneous.
4. Keep the answer concise but informative.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical research assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content