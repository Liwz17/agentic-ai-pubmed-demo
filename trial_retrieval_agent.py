"""
TrialRetrievalAgent — a proper BaseAgent subclass for trial retrieval.

Key differences from TrialRetrieval_legacy.py / TrialRetrieval.py:
- Inherits BaseAgent: owns its own OpenAI client and conversation history.
- Both LLM steps (query parsing, semantic term extraction) go through
  self._call_chat_model() — no delegation to llm.py LLM functions.
- Non-LLM work (ClinicalTrials.gov search, user selection) is still
  delegated to the existing tools utilities.
- Trivial one-liner wrapper methods removed.
- Output shape is identical to the legacy agents for downstream
  compatibility with conversation.py and Coordinator_legacy.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from base_agent import BaseAgent
from prompts import (
    TRIAL_RETRIEVAL_SYSTEM_PROMPT,
    build_trial_query_parsing_prompt,
    build_trial_semantic_expansion_prompt,
)
from tools import search_clinical_trials, select_trials_interactively


class TrialRetrievalAgent(BaseAgent):
    """
    Agent for converting a user request into a structured ClinicalTrials.gov
    query, retrieving trials, and selecting relevant ones.

    LLM responsibilities (owned by this agent):
    1. Parse a natural-language user request into a structured query.
    2. Extract discriminative semantic terms for a selected trial
       (used downstream by PaperLinkAgent for PubMed Query B).

    Non-LLM work (delegated to tools):
    - ClinicalTrials.gov API search  →  tools.search_clinical_trials
    - Interactive trial selection    →  tools.select_trials_interactively
    - Auto selection                 →  local logic, no LLM needed
    """

    def __init__(
        self,
        page_size: int = 10,
        selection_mode: str = "interactive",
        max_auto_trials: int = 5,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(system_prompt=TRIAL_RETRIEVAL_SYSTEM_PROMPT, model=model)
        self.page_size = page_size
        self.selection_mode = selection_mode
        self.max_auto_trials = max_auto_trials

    # ------------------------------------------------------------------
    # LLM-driven steps — the agent owns these
    # ------------------------------------------------------------------

    def parse_query(self, user_input: str) -> Dict[str, Any]:
        """
        Ask the LLM to convert a natural-language request into a structured
        ClinicalTrials.gov query (disease, drugs, phase, date range).
        """
        prompt = build_trial_query_parsing_prompt(user_input)
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            return json.loads(content)
        except Exception as exc:
            raise ValueError(
                f"TrialRetrievalAgent.parse_query: invalid JSON:\n{content[:300]}"
            ) from exc

    def extract_semantic_terms(self, trial_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the LLM to extract discriminative PubMed search terms for one
        trial. Used downstream by PaperLinkAgent to build Query B.
        """
        prompt = build_trial_semantic_expansion_prompt(trial_fields)
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            return json.loads(content)
        except Exception as exc:
            raise ValueError(
                f"TrialRetrievalAgent.extract_semantic_terms: invalid JSON:\n{content[:300]}"
            ) from exc

    # ------------------------------------------------------------------
    # Non-LLM steps — delegates to tools
    # ------------------------------------------------------------------

    def _search_trials(self, structured_query: Dict[str, Any], date_field: str = "start_date") -> pd.DataFrame:
        structured_query = dict(structured_query)
        structured_query["date_field"] = date_field
        return search_clinical_trials(structured_query)

    def _select_trials_interactive(self, df_trials: pd.DataFrame) -> List[int]:
        return select_trials_interactively(df_trials, page_size=self.page_size)

    def _select_trials_auto(self, df_trials: pd.DataFrame) -> List[int]:
        n = min(len(df_trials), self.max_auto_trials)
        selected_indices = list(range(n))

        cols = ["nct_id", "brief_title", "phase", "disease", "drugs", "study_first_post_date"]
        print(f"\n[TrialRetrievalAgent] Auto-selecting top {n} trial(s):")
        for rank, idx in enumerate(selected_indices, start=1):
            row = df_trials.iloc[idx]
            print(f"\n--- Auto-selected #{rank} (index={idx}) ---")
            for c in cols:
                if c in df_trials.columns:
                    print(f"  {c}: {row.get(c)}")

        return selected_indices

    def _select_trials(self, df_trials: pd.DataFrame) -> List[int]:
        if self.selection_mode == "interactive":
            return self._select_trials_interactive(df_trials)
        elif self.selection_mode == "auto":
            return self._select_trials_auto(df_trials)
        else:
            raise ValueError(
                f"selection_mode must be 'interactive' or 'auto', got {self.selection_mode!r}"
            )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, user_input: str, date_field: str = "start_date") -> Dict[str, Any]:
        """
        End-to-end flow:
          1. Parse user request into structured query  (LLM)
          2. Search ClinicalTrials.gov
          3. Select relevant trials (interactive or auto)

        date_field: which ClinicalTrials.gov date to filter on.
          "start_date" | "primary_completion_date" | "completion_date"
          | "results_first_posted_date" | "first_posted_date"
        """
        self.clear()  # each run is independent — reset conversation history
        print("\n[TrialRetrievalAgent] Parsing query...")
        structured_query = self.parse_query(user_input)
        print(f"  {structured_query}")
        print(f"  Date field: {date_field}")

        print("\n[TrialRetrievalAgent] Searching trials...")
        df_trials = self._search_trials(structured_query, date_field=date_field)

        if df_trials.empty:
            return {
                "status": "no_trials",
                "query": structured_query,
                "trials": df_trials,
                "selected_indices": [],
                "selected_trials": [],
            }

        print(f"\n[TrialRetrievalAgent] Selecting trials (mode={self.selection_mode})...")
        selected_indices = self._select_trials(df_trials)

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json(content: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()
