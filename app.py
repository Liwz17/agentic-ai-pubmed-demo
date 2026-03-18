from llm import llm_parse_query
from tools import search_clinical_trials, summarize_trial_linking_results
from linker import link_trials_to_pubmed
import pandas as pd


def run_agentic_app(user_input: str, mode: str = "hybrid"):
    print("\n[Step 1] Parsing natural language query...")
    structured_query = llm_parse_query(user_input)
    print(structured_query)

    print("\n[Step 2] Searching ClinicalTrials.gov...")
    df_trials = search_clinical_trials(structured_query)
    print(f"Found {len(df_trials)} trials")

    if df_trials.empty:
        print("No trials found.")
        return {"query": structured_query, "trials": df_trials, "links": pd.DataFrame(), "stats": {}}

    print("\n[Step 3] Sample trials:")
    cols = [c for c in ["nct_id", "brief_title", "start_date", "phases"] if c in df_trials.columns]
    print(df_trials[cols].head())

    from linker import link_one_trial

    print("\n[Step 4] Linking ONE trial to PubMed papers...")

    trial_row = df_trials.iloc[1].to_dict()

    result = link_one_trial(
        trial_row,
        mode=mode,
        use_llm=True,
        max_papers_per_query=5,
        verbose=True
    )

    print("\n[Step 5] Final judge result:")
    print(result["judge_result"])

    return {
        "query": structured_query,
        "trials": df_trials,
        "link_result": result,
    }


if __name__ == "__main__":
    user_input = "Find phase II lung cancer trials with pembrolizumab or ipilimumab started between 2016 and 2020, and check whether their results were published in PubMed."

    results = run_agentic_app(user_input, mode="nct_only")