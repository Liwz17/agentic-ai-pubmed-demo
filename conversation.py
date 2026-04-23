"""
ConversationCoordinator (v2) — wires together the new agent implementations.

Agents used:
- TrialRetrievalAgent  (trial_retrieval_agent.py)  — owns query parsing LLM call
- PaperLinkAgent       (paper_link_agent.py)        — owns all 4 linking/extraction LLM calls
- InspectorAgent       (inspector.py)               — independent QC layer on top

Differences from conversation.py:
- Uses the proper BaseAgent subclasses instead of legacy wrapper classes.
- InspectorAgent acts as a genuine second opinion: PaperLinkAgent makes its
  own judgments, then InspectorAgent independently reviews the same evidence.
  Callers can compare the two to detect disagreements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trial_retrieval_agent import TrialRetrievalAgent
from paper_link_agent import PaperLinkAgent
from inspector import InspectorAgent
from tools import PubMedFilter
from tools.outcomes import build_outcome_summary_table


class ConversationCoordinator:
    """
    Thin coordination layer that runs the three agents in sequence:
      1. TrialRetrievalAgent  — parse query, search, select trials
      2. PaperLinkAgent       — link each trial to a paper, extract survival
      3. InspectorAgent       — independently review paper matches and eligibility
    """

    def __init__(
        self,
        mode: str = "hybrid",
        page_size: int = 10,
        selection_mode: str = "interactive",
        max_auto_trials: int = 5,
        draw_plots: bool = True,
        pubmed_filter: Optional[PubMedFilter] = None,
        outcomes_raw: Optional[str] = None,
    ) -> None:
        self.trial_agent = TrialRetrievalAgent(
            page_size=page_size,
            selection_mode=selection_mode,
            max_auto_trials=max_auto_trials,
        )
        self.paper_agent = PaperLinkAgent(
            mode=mode,
            draw_plots=draw_plots,
            pubmed_filter=pubmed_filter,
            outcomes_raw=outcomes_raw,
        )
        self.inspector = InspectorAgent()

    def run(self, user_input: str) -> Dict[str, Any]:
        # --- Step 1: trial retrieval ---
        trial_packet = self.trial_agent.run(user_input)

        if trial_packet["status"] != "success":
            return {
                "status": trial_packet["status"],
                "query": trial_packet.get("query"),
                "trials": trial_packet.get("trials"),
                "selected_trial_indices": [],
                "per_trial_results": [],
                "all_plot_rows": [],
                "survival_table": None,
                "inspection_results": [],
            }

        # --- Step 2: paper linking + survival extraction ---
        paper_result = self.paper_agent.run_batch(
            trial_packet["selected_trials"],
            trial_packet["selected_indices"],
        )

        per_trial_results = paper_result["per_trial_results"]

        # --- Step 3: independent inspector review ---
        inspection_results = self._inspect_batch(per_trial_results)

        if self.paper_agent.is_multi_outcome:
            survival_table = build_outcome_summary_table(per_trial_results)
        else:
            survival_table = self.paper_agent.build_survival_table(per_trial_results)

        return {
            "status": "success",
            "query": trial_packet["query"],
            "trials": trial_packet["trials"],
            "selected_trial_indices": trial_packet["selected_indices"],
            "selected_trials": trial_packet["selected_trials"],
            "per_trial_results": per_trial_results,
            "all_plot_rows": paper_result["all_plot_rows"],
            "survival_table": survival_table,
            "inspection_results": inspection_results,
        }

    def _inspect_batch(
        self, per_trial_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run InspectorAgent over each trial result as an independent QC pass.

        For each trial, the inspector re-judges:
        - trial-paper match quality (same candidates, fresh eyes)
        - survival extraction eligibility (same pubmed record, fresh eyes)

        The output includes both the inspector's verdict and the
        PaperLinkAgent's internal verdict so callers can compare them.
        """
        results = []

        for item in per_trial_results:
            trial = item.get("trial", {}) or {}
            link_result = item.get("link_result", {}) or {}
            survival_result = item.get("survival_result", {}) or {}

            trial_fields = link_result.get("trial_fields") or trial
            candidate_papers = link_result.get("all_candidates", []) or []
            pubmed_record = survival_result.get("pubmed_record")

            # inspector: trial-paper match
            trial_paper_review = self.inspector.judge_trial_papers(
                trial_fields=trial_fields,
                candidate_papers=candidate_papers,
            )

            # inspector: survival eligibility (only if we have a paper)
            survival_eligibility_review: Optional[Dict[str, Any]] = None
            if pubmed_record:
                survival_eligibility_review = self.inspector.judge_survival_plot_eligibility(
                    pubmed_record=pubmed_record,
                )

            results.append({
                "nct_id": trial.get("nct_id"),
                "trial_title": trial.get("brief_title"),
                # inspector verdicts
                "trial_paper_review": trial_paper_review,
                "survival_eligibility_review": survival_eligibility_review,
                # paper_agent's own verdicts (for comparison)
                "agent_judge_result": link_result.get("judge_result"),
                "agent_eligibility_judgment": survival_result.get("eligibility_judgment"),
            })

        return results


def run_conversation(
    user_input: str,
    mode: str = "hybrid",
    page_size: int = 10,
    selection_mode: str = "interactive",
    max_auto_trials: int = 5,
    draw_plots: bool = True,
    pubmed_filter: Optional[PubMedFilter] = None,
    outcomes_raw: Optional[str] = None,
) -> Dict[str, Any]:
    app = ConversationCoordinator(
        mode=mode,
        page_size=page_size,
        selection_mode=selection_mode,
        max_auto_trials=max_auto_trials,
        draw_plots=draw_plots,
        pubmed_filter=pubmed_filter,
        outcomes_raw=outcomes_raw,
    )

    results = app.run(user_input)

    if results.get("survival_table") is not None:
        print("\n=== Survival Summary Table ===")
        print(results["survival_table"])

    if results.get("inspection_results"):
        print("\n=== Inspector QC Review ===")
        for item in results["inspection_results"]:
            nct_id = item.get("nct_id")
            agent_judge = item.get("agent_judge_result", {}) or {}
            inspector_judge = item.get("trial_paper_review", {}) or {}
            eligibility = item.get("survival_eligibility_review", {}) or {}

            print(f"\nTrial: {nct_id}")

            # show both verdicts side by side
            print(
                f"  paper match  | agent={agent_judge.get('match_found')} "
                f"inspector={inspector_judge.get('match_found')} "
                f"| inspector confidence={inspector_judge.get('confidence')} "
                f"| reason={inspector_judge.get('reason')}"
            )

            if eligibility:
                agent_elig = item.get("agent_eligibility_judgment", {}) or {}
                print(
                    f"  eligibility  | agent={agent_elig.get('eligible')} "
                    f"inspector={eligibility.get('eligible')} "
                    f"| paper_type={eligibility.get('paper_type')}"
                )

    return results


if __name__ == "__main__":
    user_input = input("Enter your trial search request: ").strip()

    mode = input("PubMed mode ('nct_only' or 'hybrid') [default: hybrid]: ").strip().lower()
    if not mode:
        mode = "hybrid"
    if mode not in {"nct_only", "hybrid"}:
        print("Invalid mode, falling back to 'hybrid'.")
        mode = "hybrid"

    selection_mode = input(
        "Selection mode ('interactive' or 'auto') [default: interactive]: "
    ).strip().lower()
    if not selection_mode:
        selection_mode = "interactive"
    if selection_mode not in {"interactive", "auto"}:
        print("Invalid selection mode, falling back to 'interactive'.")
        selection_mode = "interactive"

    # --- outcome selection ---
    print("Outcomes to extract (free text, or Enter for OS only):")
    print("  e.g.: 'OS, PFS, response rate'  or  'overall survival and grade 3/4 adverse events'")
    outcomes_raw = input("  > ").strip() or None

    # --- PubMed filters (optional) ---
    pub_date_start = input("PubMed date start (YYYY/MM/DD, or Enter to skip): ").strip() or None
    pub_date_end = input("PubMed date end   (YYYY/MM/DD, or Enter to skip): ").strip() or None

    print("PubMed publication type filters (comma-separated, or Enter to skip):")
    print("  options: clinical_trial, rct, meta_analysis, systematic_review, final_report")
    pt_raw = input("  > ").strip()
    publication_types = [x.strip() for x in pt_raw.split(",") if x.strip()] if pt_raw else []

    pubmed_filter = None
    if pub_date_start or pub_date_end or publication_types:
        pubmed_filter = PubMedFilter(
            pub_date_start=pub_date_start,
            pub_date_end=pub_date_end,
            publication_types=publication_types,
        )

    run_conversation(
        user_input=user_input,
        mode=mode,
        page_size=10,
        selection_mode=selection_mode,
        max_auto_trials=5,
        draw_plots=True,
        pubmed_filter=pubmed_filter,
        outcomes_raw=outcomes_raw,
    )
