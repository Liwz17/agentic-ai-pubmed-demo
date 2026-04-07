from TrialRetrieval import TrialRetrievalAgent
from PaperLink import TrialPaperLinkingAgent


class AgentCoordinator:
    def __init__(
        self,
        mode="hybrid",
        page_size=10,
        selection_mode="interactive",
        max_auto_trials=5,
        draw_plots=True,
    ):
        self.trial_agent = TrialRetrievalAgent(
            page_size=page_size,
            selection_mode=selection_mode,
            max_auto_trials=max_auto_trials,
        )
        self.pubmed_agent = TrialPaperLinkingAgent(
            mode=mode,
            draw_plots=draw_plots,
        )

    def run(self, user_input):
        trial_packet = self.trial_agent.run(user_input)

        if trial_packet["status"] != "success":
            return {
                "query": trial_packet.get("query"),
                "trials": trial_packet.get("trials"),
                "selected_trial_indices": [],
                "per_trial_results": [],
                "all_plot_rows": [],
                "survival_table": None,
            }

        pubmed_result = self.pubmed_agent.run_batch(
            trial_packet["selected_trials"],
            trial_packet["selected_indices"],
        )

        survival_table = self.pubmed_agent.build_survival_table(
            pubmed_result["per_trial_results"]
        )

        return {
            "query": trial_packet["query"],
            "trials": trial_packet["trials"],
            "selected_trial_indices": trial_packet["selected_indices"],
            "per_trial_results": pubmed_result["per_trial_results"],
            "all_plot_rows": pubmed_result["all_plot_rows"],
            "survival_table": survival_table,
        }