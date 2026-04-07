from llm import llm_parse_query
from tools import search_clinical_trials, draw_single_trial_forest_plot, draw_multi_trial_forest_plot, select_trials_interactively
from linker import link_one_trial, extract_survival_from_link_result
import pandas as pd
import json
from Coordinator import AgentCoordinator



def run_agentic_app(
    user_input: str,
    mode: str = "hybrid",
    page_size: int = 10,
    selection_mode: str = "interactive",
    max_auto_trials: int = 5,
    draw_plots: bool = True,
):
    app = AgentCoordinator(
        mode=mode,
        page_size=page_size,
        selection_mode=selection_mode,
        max_auto_trials=max_auto_trials,
        draw_plots=draw_plots,
    )

    results = app.run(user_input)

    if results.get("survival_table") is not None:
        print("\n=== Survival Summary Table ===")
        print(results["survival_table"])

    return results


if __name__ == "__main__":
    user_input = input("Enter your trial search request: ").strip()

    mode = input("Choose PubMed mode ('nct_only' or 'hybrid') [default: hybrid]: ").strip().lower()
    if not mode:
        mode = "hybrid"
    if mode not in {"nct_only", "hybrid"}:
        print("Invalid PubMed mode. Falling back to 'hybrid'.")
        mode = "hybrid"

    selection_mode = input("Choose selection mode ('interactive' or 'auto') [default: interactive]: ").strip().lower()
    if not selection_mode:
        selection_mode = "interactive"
    if selection_mode not in {"interactive", "auto"}:
        print("Invalid selection mode. Falling back to 'interactive'.")
        selection_mode = "interactive"

    results = run_agentic_app(
        user_input=user_input,
        mode=mode,
        page_size=10,
        selection_mode=selection_mode,
        max_auto_trials=5,
        draw_plots=True,
    )
