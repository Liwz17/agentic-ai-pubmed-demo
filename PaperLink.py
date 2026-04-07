import json
import pandas as pd

from linker import (
    llm_extract_trial_semantic_terms,
    link_one_trial,
    extract_survival_from_link_result,
)
from tools import draw_single_trial_forest_plot, draw_multi_trial_forest_plot


class TrialPaperLinkingAgent:
    """
    Agent responsible for:
    - linking trials to PubMed papers
    - judging match quality
    - extracting survival info
    """

    def __init__(self, mode="hybrid", draw_plots=True):
        self.mode = mode
        self.draw_plots = draw_plots

    def get_semantic_terms(self, trial_row):
        return llm_extract_trial_semantic_terms(trial_row)

    def link_trial(self, trial_row):
        return link_one_trial(
            trial_row,
            mode=self.mode,
            use_llm=True,
            max_papers_per_query=5,
            verbose=True,
        )

    def extract_survival(self, link_result, trial_row):
        return extract_survival_from_link_result(link_result, trial_row=trial_row)

    def run_one(self, trial_row, idx=None):
        print(f"\n[PubMedAgent] Processing trial {idx}...")

        semantic_terms = self.get_semantic_terms(trial_row)
        link_result = self.link_trial(trial_row)
        survival_result = self.extract_survival(link_result, trial_row)

        plot_rows = survival_result.get("survival_extraction", {}).get("plot_rows", [])

        # if self.draw_plots and plot_rows:
        #     draw_single_trial_forest_plot(
        #         plot_rows,
        #         trial_label=trial_row.get("brief_title")
        #     )

        return {
            "trial": trial_row,
            "semantic_terms": semantic_terms,
            "link_result": link_result,
            "survival_result": survival_result,
            "plot_rows": plot_rows,
        }

    def run_batch(self, trials, indices):
        results = []
        all_plot_rows = []

        for trial, idx in zip(trials, indices):
            res = self.run_one(trial, idx)
            results.append(res)

            if res["plot_rows"]:
                all_plot_rows.extend(res["plot_rows"])

        if self.draw_plots and all_plot_rows:
            draw_multi_trial_forest_plot(all_plot_rows)

        return {
            "per_trial_results": results,
            "all_plot_rows": all_plot_rows,
        }

    import pandas as pd

    def build_survival_table(self, per_trial_results):
        """
        Build a trial-level summary table for audit / debugging.

        One row per trial.
        Focus on retrieval / extraction status rather than arm-level survival values.
        """
        rows = []

        for item in per_trial_results:
            trial = item.get("trial", {}) or {}
            link_result = item.get("link_result", {}) or {}
            survival_result = item.get("survival_result", {}) or {}

            judge_result = link_result.get("judge_result", {}) or {}
            extraction = survival_result.get("survival_extraction", {}) or {}
            eligibility = survival_result.get("eligibility_judgment", {}) or {}

            matched_pmid = judge_result.get("selected_pubmed_id")
            matched_title = judge_result.get("selected_title")
            match_found = judge_result.get("match_found")

            rows.append({
                "nct_id": trial.get("nct_id"),
                "trial_title": trial.get("brief_title"),
                "pubmed_match_found": match_found,
                "pubmed_id": matched_pmid,
                "paper_title": matched_title,
                "source_used": extraction.get("source_used"),
                "paper_type": eligibility.get("paper_type"),
                "extraction_status": survival_result.get("status"),
                "outcome_found": extraction.get("outcome_found"),
                "n_arms_extracted": len(extraction.get("arms", []) or []),
                "notes": extraction.get("notes"),
            })

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values(by=["nct_id"]).reset_index(drop=True)

        return df