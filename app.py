from llm import llm_parse_query
from tools import search_clinical_trials, summarize_trial_linking_results, draw_single_trial_forest_plot
from linker import link_trials_to_pubmed, link_one_trial, extract_survival_from_link_result
import pandas as pd
import json



def _select_trials_interactively(df_trials: pd.DataFrame, page_size: int = 10) -> list[int]:
    """
    Let user browse candidate trials page by page and select MULTIPLE trials.

    Returns
    -------
    selected_trial_indices : list[int]
        List of global row indices in df_trials.
    """
    cols = [c for c in ["nct_id", "brief_title", "start_date", "phases"] if c in df_trials.columns]

    total = len(df_trials)
    current_page = 0
    selected_indices = set()

    while True:
        start = current_page * page_size
        end = min(start + page_size, total)

        display_df = df_trials.iloc[start:end][cols].copy()
        display_df = display_df.reset_index(drop=True)

        print(f"\n[Step 3] Candidate trials (showing {start}–{end - 1} of {total - 1})")
        print(display_df)

        print("\nCurrently selected indices:", sorted(selected_indices) if selected_indices else "None")

        print("\nOptions:")
        print("  [0-9]      Add trial index from current page")
        print("  r NUM      Remove global index NUM (example: r 25)")
        print("  g NUM      Add global index NUM (example: g 25)")
        print("  n          Next page")
        print("  p          Previous page")
        print("  d          Done (finish selection)")
        print("  q          Quit (return empty list)")

        choice = input("\nYour choice: ").strip().lower()

        # ---------- navigation ----------
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

        # ---------- finish ----------
        if choice == "d":
            return sorted(selected_indices)

        if choice == "q":
            return []

        # ---------- add via global ----------
        if choice.startswith("g "):
            parts = choice.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < total:
                    selected_indices.add(idx)
                    print(f"Added trial {idx}")
                else:
                    print("Global index out of range.")
            else:
                print("Invalid format. Use: g 25")
            continue

        # ---------- remove ----------
        if choice.startswith("r "):
            parts = choice.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx in selected_indices:
                    selected_indices.remove(idx)
                    print(f"Removed trial {idx}")
                else:
                    print("Index not in selection.")
            else:
                print("Invalid format. Use: r 25")
            continue

        # ---------- add from page ----------
        try:
            idx_in_page = int(choice)
            if 0 <= idx_in_page < len(display_df):
                global_idx = start + idx_in_page
                selected_indices.add(global_idx)
                print(f"Added trial {global_idx}")
            else:
                print("Index out of range on current page.")
        except ValueError:
            print("Invalid input.")


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

    selected_trial_indices = _select_trials_interactively(df_trials, page_size=page_size)

    if selected_trial_indices is None:
        print("No trial selected. Exiting.")
        return {
            "query": structured_query,
            "trials": df_trials,
            "selected_trial_index": None,
            "link_result": None,
            "survival_result": None,
        }

    per_trial_results = []

    for idx in selected_trial_indices:
        trial_row = df_trials.iloc[idx].to_dict()

        print(f"\n[Step 4] Linking selected trial #{idx} to PubMed papers with mode = {mode} ...")
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
        survival_result = extract_survival_from_link_result(result, trial_row=trial_row)

        print("\n=== Survival Extraction Result ===")
        print(json.dumps(survival_result["survival_extraction"], indent=2, ensure_ascii=False))

        # Step 7: draw forest plot for this single selected trial
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

        per_trial_results.append({
            "selected_trial_index": idx,
            "trial_row": trial_row,
            "link_result": result,
            "survival_result": survival_result,
        })
    return {
        "query": structured_query,
        "trials": df_trials,
        "selected_trial_indices": selected_trial_indices,
        "per_trial_results": per_trial_results,
    }

if __name__ == "__main__":
    user_input = input("Enter your trial search request: ").strip()

    mode = input("Choose mode ('nct_only' or 'hybrid') [default: hybrid]: ").strip().lower()
    if not mode:
        mode = "hybrid"
    if mode not in {"nct_only", "hybrid"}:
        print("Invalid mode. Falling back to 'hybrid'.")
        mode = "hybrid"

    results = run_agentic_app(user_input, mode=mode, page_size=10)