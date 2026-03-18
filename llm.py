from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME, RERANK_TOP_K
import json
import re

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

import re

def _clean_llm_json(content: str) -> str:
    # remove ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

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


def llm_parse_query(user_input: str) -> dict:
    """
    Use LLM to convert natural language into structured query
    """

    prompt = f"""
You are a clinical research assistant.

Convert the following user request into a structured JSON query for ClinicalTrials.gov search.

User request:
{user_input}

Return ONLY JSON with the following fields:
- disease: string
- drugs: list of strings
- phase: one of PHASE1, PHASE2, PHASE3, PHASE4 (uppercase)
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD

Rules:
- Normalize drug names (e.g., "pembro" → "pembrolizumab")
- Normalize disease (e.g., "NSCLC" → "lung cancer")
- Phase II → PHASE2
- If date is given as year, convert to full range
- Be precise and do not hallucinate

Example output:
{{
  "disease": "lung cancer",
  "drugs": ["pembrolizumab", "ipilimumab"],
  "phase": "PHASE2",
  "start_date": "2016-01-01",
  "end_date": "2020-12-31"
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        return json.loads(content)
    except:
        raise ValueError("LLM did not return valid JSON:\n" + content)
    

# Code for LLM to extract distinguished semantic terms


def _clean_llm_json(content: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def llm_extract_trial_semantic_terms(fields: dict) -> dict:
    """
    Use LLM to extract trial-specific semantic retrieval terms.
    Return JSON like:
    {
      "disease_terms": [...],
      "drug_terms": [...],
      "setting_terms": [...],
      "other_terms": [...]
    }
    """
    prompt = f"""
You are helping construct a PubMed retrieval query for one specific clinical trial.

Trial information:
- NCT ID: {fields.get('nct_id')}
- Brief title: {fields.get('brief_title')}
- Official title: {fields.get('official_title')}
- Brief summary: {fields.get('brief_summary')}
- Disease: {fields.get('disease')}
- Drugs: {fields.get('drugs')}
- Phase: {fields.get('phase')}

Goal:
Extract the most discriminative semantic terms that would help retrieve papers specifically about THIS trial, not just the general topic.

Rules:
- Prefer concrete disease subtype, treatment setting, regimen pattern, and special descriptors
- Keep terms concise
- Avoid generic words like "study", "patients", "trial"
- Do not hallucinate facts not supported by the input
- Return ONLY JSON

Return JSON with keys:
- disease_terms: list of strings
- drug_terms: list of strings
- setting_terms: list of strings
- other_terms: list of strings
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical information retrieval assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        return json.loads(content)
    except Exception:
        raise ValueError("LLM did not return valid JSON for trial semantic terms:\n" + content)
    


# code for llm to judge the papers retrived by a,b,c mode.
def llm_judge_trial_papers(trial_fields: dict, candidate_papers: list) -> dict:
    """
    Judge which candidate paper is most likely linked to the given trial.
    Return structured JSON.
    """
    if not candidate_papers:
        return {
            "match_found": False,
            "selected_pubmed_id": None,
            "selected_title": None,
            "label": "no_candidate",
            "confidence": "low",
            "reason": "No candidate papers were retrieved."
        }

    paper_blocks = []
    for i, p in enumerate(candidate_papers, 1):
        paper_blocks.append(f"""
Candidate {i}
PMID: {p.get("pubmed_id")}
Title: {p.get("title")}
Journal: {p.get("journal")}
Abstract: {p.get("abstract")}
""")

    papers_text = "\n".join(paper_blocks)

    prompt = f"""
You are linking one ClinicalTrials.gov trial to PubMed papers.

Trial:
- NCT ID: {trial_fields.get("nct_id")}
- Brief title: {trial_fields.get("brief_title")}
- Official title: {trial_fields.get("official_title")}
- Brief summary: {trial_fields.get("brief_summary")}
- Disease: {trial_fields.get("disease")}
- Drugs: {trial_fields.get("drugs")}
- Phase: {trial_fields.get("phase")}

Candidate PubMed papers:
{papers_text}

Task:
Decide whether one of the candidate papers is likely reporting results for THIS specific trial.

Use these labels:
- primary_results
- secondary_or_followup
- protocol_or_review
- not_a_match

Rules:
- Prefer papers that match the disease, drug(s), phase, and trial setting.
- If an NCT ID is explicitly consistent, that is strong evidence.
- Do not force a match if the candidates are only generally related.
- If no candidate is convincing, return match_found=false and selected index null.

Return ONLY JSON in this format:
{{
  "match_found": true,
  "selected_index": 1,
  "selected_pubmed_id": "12345678",
  "selected_title": "paper title",
  "label": "primary_results",
  "confidence": "high",
  "reason": "brief explanation"
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical trial-to-paper linkage judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        result = json.loads(content)
        return result
    except Exception:
        raise ValueError("LLM did not return valid JSON for trial-paper judge:\n" + content)
    

