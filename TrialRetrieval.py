from llm import llm_parse_query
from tools import search_clinical_trials, select_trials_interactively


class TrialRetrievalAgent:
    """
    Agent responsible for:
    - understanding user query
    - retrieving clinical trials
    - selecting relevant trials
    """

    def __init__(self, page_size=10, selection_mode="interactive", max_auto_trials=5):
        self.page_size = page_size
        self.selection_mode = selection_mode
        self.max_auto_trials = max_auto_trials

    def parse_query(self, user_input):
        return llm_parse_query(user_input)

    def search_trials(self, structured_query):
        return search_clinical_trials(structured_query)

    def select_trials_interactive(self, df_trials):
        return select_trials_interactively(df_trials, page_size=self.page_size)

    def select_trials_auto(self, df_trials):
        n = min(len(df_trials), self.max_auto_trials)
        selected_indices = list(range(n))

        print(f"\n[TrialAgent] Auto-selecting top {n} trial(s):")
        cols_to_show = ["nct_id", "brief_title", "phase", "disease", "drugs", "study_first_post_date"]

        for rank, idx in enumerate(selected_indices, start=1):
            row = df_trials.iloc[idx]
            print(f"\n--- Auto-selected trial #{rank} (df index={idx}) ---")
            for c in cols_to_show:
                if c in df_trials.columns:
                    print(f"{c}: {row.get(c)}")

        return selected_indices

    def select_trials(self, df_trials):
        if self.selection_mode == "interactive":
            return self.select_trials_interactive(df_trials)
        elif self.selection_mode == "auto":
            return self.select_trials_auto(df_trials)
        else:
            raise ValueError("selection_mode must be 'interactive' or 'auto'")

    def run(self, user_input):
        print("\n[TrialAgent] Parsing query...")
        structured_query = self.parse_query(user_input)
        print(structured_query)
        print("\n[TrialAgent] Searching trials...")
        df_trials = self.search_trials(structured_query)

        if df_trials.empty:
            return {
                "status": "no_trials",
                "query": structured_query,
                "trials": df_trials,
                "selected_indices": [],
                "selected_trials": [],
            }

        print(f"\n[TrialAgent] Selecting trials with mode = {self.selection_mode} ...")
        selected_indices = self.select_trials(df_trials)

        if not selected_indices:
            return {
                "status": "no_selection",
                "query": structured_query,
                "trials": df_trials,
                "selected_indices": [],
                "selected_trials": [],
            }

        selected_trials = [df_trials.iloc[i].to_dict() for i in selected_indices]

        return {
            "status": "success",
            "query": structured_query,
            "trials": df_trials,
            "selected_indices": selected_indices,
            "selected_trials": selected_trials,
        }