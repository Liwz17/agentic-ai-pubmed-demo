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
        return {
            "query": structured_query,
            "trials": df_trials,
            "link_result": None,
        }

    print("\n[Step 3] Candidate trials:")
    cols = [c for c in ["nct_id", "brief_title", "start_date", "phases"] if c in df_trials.columns]
    display_df = df_trials[cols].head(10).copy()
    display_df = display_df.reset_index(drop=True)
    print(display_df)

    while True:
        try:
            choice = input("\nEnter trial index to link to PubMed (0-9): ").strip()
            trial_idx = int(choice)

            if 0 <= trial_idx < len(display_df):
                break
            else:
                print("Index out of range. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    from linker import link_one_trial

    print(f"\n[Step 4] Linking selected trial #{trial_idx} to PubMed papers with mode = {mode} ...")
    trial_row = df_trials.iloc[trial_idx].to_dict()

    result = link_one_trial(
        trial_row,
        mode=mode,
        use_llm=True,
        max_papers_per_query=5,
        verbose=True
    )

    print("\n[Step 5] Final result:")
    print(result["judge_result"])

    return {
        "query": structured_query,
        "trials": df_trials,
        "selected_trial_index": trial_idx,
        "link_result": result,
    }

if __name__ == "__main__":
    user_input = input("Enter your trial search request: ").strip()
    mode = input("Choose mode ('nct_only' or 'hybrid') [default: hybrid]: ").strip() or "hybrid"

    results = run_agentic_app(user_input, mode=mode)