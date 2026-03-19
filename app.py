from llm import llm_parse_query
from tools import search_clinical_trials, summarize_trial_linking_results, draw_single_trial_forest_plot
from linker import link_trials_to_pubmed, link_one_trial, extract_survival_from_link_result
import pandas as pd
import json



def _select_trial_interactively(df_trials: pd.DataFrame, page_size: int = 10) -> int | None:
    """
    Let user browse candidate trials page by page and select one trial index.

    Returns
    -------
    trial_idx : int | None
        Global row index in df_trials, or None if user quits.
    """
    cols = [c for c in ["nct_id", "brief_title", "start_date", "phases"] if c in df_trials.columns]

    total = len(df_trials)
    current_page = 0

    while True:
        start = current_page * page_size
        end = min(start + page_size, total)

        display_df = df_trials.iloc[start:end][cols].copy()
        display_df = display_df.reset_index(drop=True)

        print(f"\n[Step 3] Candidate trials (showing {start}–{end - 1} of {total - 1})")
        print(display_df)

        print("\nOptions:")
        print("  [0-9]  Select a trial index on the current page")
        print("  n      Next page")
        print("  p      Previous page")
        print("  g NUM  Jump to global index NUM (example: g 25)")
        print("  q      Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "n":
            if end < total:
                current_page += 1
            else:
                print("Already at the last page.")
            continue

        if choice == "p":
            if current_page > 0:
                current_page -= 1
            else:
                print("Already at the first page.")
            continue

        if choice == "q":
            return None

        if choice.startswith("g "):
            parts = choice.split()
            if len(parts) == 2 and parts[1].isdigit():
                global_idx = int(parts[1])
                if 0 <= global_idx < total:
                    return global_idx
                else:
                    print("Global index out of range.")
            else:
                print("Invalid jump command. Use format: g 25")
            continue

        try:
            idx_in_page = int(choice)
            if 0 <= idx_in_page < len(display_df):
                trial_idx = start + idx_in_page
                return trial_idx
            else:
                print("Index out of range on the current page.")
        except ValueError:
            print("Invalid input. Please enter 0-9, n, p, g NUM, or q.")


def run_agentic_app(user_input: str, mode: str = "hybrid", page_size: int = 10):
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
            "selected_trial_index": None,
            "link_result": None,
            "survival_result": None,
        }

    selected_trial_idx = _select_trial_interactively(df_trials, page_size=page_size)

    if selected_trial_idx is None:
        print("No trial selected. Exiting.")
        return {
            "query": structured_query,
            "trials": df_trials,
            "selected_trial_index": None,
            "link_result": None,
            "survival_result": None,
        }

    trial_row = df_trials.iloc[selected_trial_idx].to_dict()

    print(f"\n[Step 4] Linking selected trial #{selected_trial_idx} to PubMed papers with mode = {mode} ...")
    print(f"Selected NCT ID: {trial_row.get('nct_id')}")
    print(f"Selected trial title: {trial_row.get('brief_title')}")

    result = link_one_trial(
        trial_row,
        mode=mode,
        use_llm=True,
        max_papers_per_query=5,
        verbose=True,
    )

    print("\n[Step 5] Final link result:")
    print(result["judge_result"])

    print("\n[Step 6] Extracting median OS and 95% CI by treatment arm...")
    survival_result = extract_survival_from_link_result(result)

    print("\n=== Survival Extraction Result ===")
    print(json.dumps(survival_result["survival_extraction"], indent=2, ensure_ascii=False))


    # ✅ Step 7: draw forest plot (single trial)
    plot_rows = survival_result["survival_extraction"].get("plot_rows", [])

    if plot_rows:
        print("\n[Step 7] Drawing forest plot for selected trial...")

        trial_label = trial_row.get("brief_title")  

        draw_single_trial_forest_plot(
            plot_rows,
            trial_label=trial_label
        )
    else:
        print("\n[Step 7] No plot data available.")

if __name__ == "__main__":
    user_input = input("Enter your trial search request: ").strip()

    mode = input("Choose mode ('nct_only' or 'hybrid') [default: hybrid]: ").strip().lower()
    if not mode:
        mode = "hybrid"
    if mode not in {"nct_only", "hybrid"}:
        print("Invalid mode. Falling back to 'hybrid'.")
        mode = "hybrid"

    results = run_agentic_app(user_input, mode=mode, page_size=10)