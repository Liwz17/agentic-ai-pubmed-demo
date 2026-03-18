from collections import Counter
import pandas as pd

def summarize_papers_stats(papers):
    n_papers = len(papers)

    journal_counter = Counter()
    keyword_counter = Counter()
    missing_abstract_count = 0
    abstract_lengths = []

    for p in papers:
        journal = p.get("journal")
        if journal:
            journal_counter[journal] += 1

        keywords = p.get("keywords") or []
        for kw in keywords:
            if kw:
                keyword_counter[kw] += 1

        abstract = p.get("abstract")
        if not abstract or abstract == "No abstract available":
            missing_abstract_count += 1
        else:
            abstract_lengths.append(len(abstract))

    avg_abstract_length = (
        sum(abstract_lengths) / len(abstract_lengths)
        if abstract_lengths else 0
    )

    return {
        "n_papers": n_papers,
        "top_journals": journal_counter.most_common(5),
        "top_keywords": keyword_counter.most_common(10),
        "missing_abstract_count": missing_abstract_count,
        "avg_abstract_length": round(avg_abstract_length, 2),
    }

def summarize_trial_linking_results(df_links: pd.DataFrame) -> dict:
    if df_links.empty:
        return {
            "n_trials": 0,
            "n_matched": 0,
            "n_possible": 0,
            "n_unmatched": 0,
            "match_rate": 0.0,
        }

    n_trials = len(df_links)
    n_matched = (df_links["match_status"] == "matched").sum()
    n_possible = (df_links["match_status"] == "possible_match").sum()
    n_unmatched = n_trials - n_matched - n_possible

    return {
        "n_trials": n_trials,
        "n_matched": int(n_matched),
        "n_possible": int(n_possible),
        "n_unmatched": int(n_unmatched),
        "match_rate": round(n_matched / n_trials, 3),
        "possible_rate": round(n_possible / n_trials, 3),
        "avg_candidates": round(df_links["n_candidates"].mean(), 2) if "n_candidates" in df_links.columns else None,
    }