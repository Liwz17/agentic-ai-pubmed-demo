from typing import Dict, Any, List, Optional
import pandas as pd
from tools import query_pubmed
from llm import rerank_papers, llm_extract_trial_semantic_terms, llm_judge_trial_papers, llm_extract_survival_from_text
import re
from tools import (
    extract_trial_retrieval_fields,
    build_query_A,
    build_query_B_llm,
    build_query_C,
    _run_pubmed_query_once,
    fetch_pubmed_abstract,
)


def _safe_lower(x):
    return x.lower().strip() if isinstance(x, str) else ""


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", _safe_lower(a)))
    b_tokens = set(re.findall(r"\w+", _safe_lower(b)))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _paper_mentions_nct(paper: Dict[str, Any], nct_id: str) -> bool:
    if not nct_id:
        return False
    text = " ".join([
        str(paper.get("title", "")),
        str(paper.get("abstract", "")),
        str(paper.get("keywords", "")),
    ])
    return nct_id.lower() in text.lower()


def _build_pubmed_queries_for_trial(trial_row):
    precise_queries = []
    fallback_queries = []

    nct_id = trial_row.get("nct_id")
    brief_title = trial_row.get("brief_title")
    official_title = trial_row.get("official_title")
    conditions = trial_row.get("conditions", []) or []
    interventions = trial_row.get("interventions", []) or []

    if nct_id:
        precise_queries.append(nct_id)

    if brief_title:
        precise_queries.append(brief_title)

    if official_title and official_title != brief_title:
        precise_queries.append(official_title)

    if conditions and interventions:
        fallback_queries.append(f"{conditions[0]} {interventions[0]} trial results")

    if conditions:
        fallback_queries.append(f"{conditions[0]} clinical trial results")

    return {
        "precise": precise_queries,
        "fallback": fallback_queries
    }


def _score_trial_paper_match(trial_row: pd.Series, paper: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    reasons = []

    nct_id = trial_row.get("nct_id")
    brief_title = trial_row.get("brief_title", "")
    official_title = trial_row.get("official_title", "")
    conditions = trial_row.get("conditions", []) or []
    interventions = trial_row.get("interventions", []) or []
    phases = trial_row.get("phases", []) or []

    paper_title = paper.get("title", "") or ""
    paper_abstract = paper.get("abstract", "") or ""
    full_text = f"{paper_title} {paper_abstract}".lower()

    if nct_id and _paper_mentions_nct(paper, nct_id):
        score += 10
        reasons.append("Exact NCT ID mentioned")

    title_sim_1 = _token_overlap_score(brief_title, paper_title)
    title_sim_2 = _token_overlap_score(official_title, paper_title)
    title_sim = max(title_sim_1, title_sim_2)

    if title_sim >= 0.5:
        score += 4
        reasons.append(f"High title overlap ({title_sim:.2f})")
    elif title_sim >= 0.25:
        score += 2
        reasons.append(f"Moderate title overlap ({title_sim:.2f})")

    matched_conditions = [c for c in conditions if c.lower() in full_text]
    if matched_conditions:
        score += min(2, len(matched_conditions))
        reasons.append(f"Condition matched: {matched_conditions[:2]}")

    matched_interventions = [d for d in interventions if d.lower() in full_text]
    if matched_interventions:
        score += min(3, len(matched_interventions))
        reasons.append(f"Intervention matched: {matched_interventions[:2]}")

    matched_phases = [p for p in phases if p.lower().replace("phase", "phase ") in full_text or p.lower() in full_text]
    if matched_phases:
        score += 1
        reasons.append(f"Phase matched: {matched_phases[:1]}")

    negative_terms = ["review", "protocol", "design", "editorial", "commentary"]
    if any(term in full_text for term in negative_terms):
        score -= 3
        reasons.append("Looks like non-results publication")

    if "results" in full_text or "efficacy" in full_text or "safety" in full_text:
        score += 1
        reasons.append("Results-like wording found")

    if score >= 10:
        label = "high"
    elif score >= 5:
        label = "medium"
    else:
        label = "low"

    return {
        "rule_score": score,
        "rule_confidence": label,
        "rule_reasons": reasons,
    }


def find_pubmed_candidates_for_trial(trial_row: pd.Series, max_papers_per_query: int = 5) -> List[Dict[str, Any]]:
    queries = _build_pubmed_queries_for_trial(trial_row)

    seen_pmids = set()
    candidates = []

    for q in queries:
        papers = query_pubmed(q, max_papers=max_papers_per_query)
        for p in papers:
            pmid = p.get("pubmed_id")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                candidates.append(p)

    return candidates


def link_one_trial_to_pubmed(trial_row: pd.Series, top_k_rerank: int = 5) -> Dict[str, Any]:
    candidates = find_pubmed_candidates_for_trial(trial_row)

    if not candidates:
        return {
            "nct_id": trial_row.get("nct_id"),
            "matched_pmid": None,
            "matched_title": None,
            "match_status": "no_candidate",
            "match_confidence": "low",
            "match_reason": "No PubMed candidates found",
            "n_candidates": 0,
        }

    scored = []
    for p in candidates:
        s = _score_trial_paper_match(trial_row, p)
        scored.append({
            **p,
            **s
        })

    scored = sorted(scored, key=lambda x: x["rule_score"], reverse=True)

    top_candidates = scored[:10]
    reranked = rerank_papers(
        query=f"Find the paper most likely reporting results for trial {trial_row.get('nct_id')} {trial_row.get('brief_title')}",
        papers=top_candidates,
        top_k=min(top_k_rerank, len(top_candidates))
    )

    best = reranked[0] if reranked else top_candidates[0]

    best_score = best.get("rule_score", 0)
    if best_score >= 10:
        status = "matched"
        confidence = "high"
    elif best_score >= 5:
        status = "possible_match"
        confidence = "medium"
    else:
        status = "weak_match"
        confidence = "low"

    return {
        "nct_id": trial_row.get("nct_id"),
        "brief_title": trial_row.get("brief_title"),
        "matched_pmid": best.get("pubmed_id"),
        "matched_title": best.get("title"),
        "match_status": status,
        "match_confidence": confidence,
        "match_reason": "; ".join(best.get("rule_reasons", [])),
        "rule_score": best.get("rule_score"),
        "n_candidates": len(candidates),
    }


def link_trials_to_pubmed(df_trials: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    if limit is not None:
        df_trials = df_trials.head(limit).copy()

    rows = []
    for _, trial_row in df_trials.iterrows():
        result = link_one_trial_to_pubmed(trial_row)
        rows.append(result)

    return pd.DataFrame(rows)

def _dedup_papers_by_pmid(papers: list) -> list:
    seen = set()
    unique = []

    for p in papers:
        pmid = p.get("pubmed_id")
        if not pmid:
            continue
        if pmid not in seen:
            seen.add(pmid)
            unique.append(p)

    return unique


def link_one_trial(
    trial_row: dict,
    mode: str = "hybrid",
    use_llm: bool = True,
    max_papers_per_query: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Link one trial to PubMed papers.

    mode:
    - "nct_only": only use Query A
    - "hybrid": use A + B + C

    use_llm:
    - True: run LLM judge
    - False: only return candidates
    """
    if mode not in {"nct_only", "hybrid"}:
        raise ValueError("mode must be 'nct_only' or 'hybrid'")

    fields = extract_trial_retrieval_fields(trial_row)

    if verbose:
        print("\n=== Trial fields ===")
        print(fields)

    papers_A = []
    papers_B = []
    papers_C = []
    semantic_terms = None
    query_A = None
    query_B = None
    query_C = None

    # A always available if nct_id exists
    query_A = build_query_A(fields)
    if query_A:
        if verbose:
            print("\n=== Query A ===")
            print(query_A)
        papers_A = _run_pubmed_query_once(query_A, max_papers=max_papers_per_query)

    if mode == "hybrid":
        # B
        semantic_terms = llm_extract_trial_semantic_terms(fields)
        query_B = build_query_B_llm(fields, semantic_terms)

        if verbose:
            print("\n=== Query B ===")
            print(query_B)
        papers_B = _run_pubmed_query_once(query_B, max_papers=max_papers_per_query)

        # C
        query_C = build_query_C(fields)
        if verbose:
            print("\n=== Query C ===")
            print(query_C)
        papers_C = _run_pubmed_query_once(query_C, max_papers=max_papers_per_query)

    all_candidates = _dedup_papers_by_pmid(papers_A + papers_B + papers_C)

    if verbose:
        print(f"\nMode: {mode}")
        print(f"A retrieved: {len(papers_A)}")
        print(f"B retrieved: {len(papers_B)}")
        print(f"C retrieved: {len(papers_C)}")
        print(f"Unique candidates: {len(all_candidates)}")

        for p in all_candidates[:5]:
            print("Candidate:", p["pubmed_id"], p["title"])

    judge_result = None
    if use_llm:
        judge_result = llm_judge_trial_papers(fields, all_candidates)

        if verbose:
            print("\n=== LLM Judge Result ===")
            print(judge_result)

    return {
        "mode": mode,
        "trial_fields": fields,
        "semantic_terms": semantic_terms,
        "query_A": query_A,
        "query_B": query_B,
        "query_C": query_C,
        "papers_A": papers_A,
        "papers_B": papers_B,
        "papers_C": papers_C,
        "all_candidates": all_candidates,
        "judge_result": judge_result,
    }


def extract_survival_from_link_result(link_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use selected PubMed paper from link_result and extract median OS by arm.
    """
    judge_result = link_result.get("judge_result", {})
    pubmed_id = judge_result.get("selected_pubmed_id")

    if not pubmed_id:
        return {
            "status": "error",
            "message": "No selected_pubmed_id found in link_result['judge_result']",
            "pubmed_record": None,
            "survival_extraction": None,
        }

    pubmed_record = fetch_pubmed_abstract(pubmed_id)

    if not pubmed_record.get("abstract"):
        return {
            "status": "warning",
            "message": f"Abstract not found for PMID={pubmed_id}",
            "pubmed_record": pubmed_record,
            "survival_extraction": {
                "paper_id": pubmed_id,
                "outcome_found": False,
                "outcome_type": "overall_survival",
                "source_used": "abstract",
                "arms": [],
                "notes": "No abstract available from PubMed.",
            },
        }

    extraction = llm_extract_survival_from_text(pubmed_record)

    return {
        "status": "ok",
        "message": "Survival extraction completed.",
        "pubmed_record": pubmed_record,
        "survival_extraction": extraction,
    }