import pprint
from tools import search_clinical_trials
from llm import llm_parse_query, llm_extract_trial_semantic_terms
from tools import (
    extract_trial_retrieval_fields,
    build_query_A,
    build_query_B_llm,
    build_query_C,
    _run_pubmed_query_once,
)


def test_one_trial_abc(df_trials, idx=0, max_papers=5):
    trial_row = df_trials.iloc[idx].to_dict()

    print("\n=== Raw trial row ===")
    pprint.pprint(trial_row)

    fields = extract_trial_retrieval_fields(trial_row)

    print("\n=== Extracted retrieval fields ===")
    pprint.pprint(fields)

    # A
    query_A = build_query_A(fields)
    print("\n=== Query A ===")
    print(query_A)

    papers_A = _run_pubmed_query_once(query_A, max_papers=max_papers) if query_A else []
    print(f"A retrieved: {len(papers_A)}")
    for p in papers_A[:3]:
        print("A:", p["pubmed_id"], p["title"])

    # B
    semantic_terms = llm_extract_trial_semantic_terms(fields)
    print("\n=== B semantic terms ===")
    pprint.pprint(semantic_terms)

    query_B = build_query_B_llm(fields, semantic_terms)
    print("\n=== Query B ===")
    print(query_B)

    papers_B = _run_pubmed_query_once(query_B, max_papers=max_papers)
    print(f"B retrieved: {len(papers_B)}")
    for p in papers_B[:3]:
        print("B:", p["pubmed_id"], p["title"])

    # C
    query_C = build_query_C(fields)
    print("\n=== Query C ===")
    print(query_C)

    papers_C = _run_pubmed_query_once(query_C, max_papers=max_papers)
    print(f"C retrieved: {len(papers_C)}")
    for p in papers_C[:3]:
        print("C:", p["pubmed_id"], p["title"])

    return {
        "trial_row": trial_row,
        "fields": fields,
        "semantic_terms": semantic_terms,
        "query_A": query_A,
        "query_B": query_B,
        "query_C": query_C,
        "papers_A": papers_A,
        "papers_B": papers_B,
        "papers_C": papers_C,
    }


if __name__ == "__main__":
    user_input = "Find phase II lung cancer trials with pembrolizumab or ipilimumab started between 2016 and 2020."
    structured_query = llm_parse_query(user_input)
    df_trials = search_clinical_trials(structured_query)

    print(f"Found {len(df_trials)} trials")

    result = test_one_trial_abc(df_trials, idx=2, max_papers=5)