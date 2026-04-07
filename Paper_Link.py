import json
import openai
from typing import Dict, Any, List, Optional
import pandas as pd
from tools import (
    extract_trial_retrieval_fields,
    build_query_A,
    build_query_B_llm,
    build_query_C,
    _run_pubmed_query_once,
)
from linker import dedup_papers_by_pmid
import re

class TrialPaperLinkingAgent:
    """
    Agent responsible for:
    - linking trials to PubMed papers
    - judging match quality
    - extracting survival info
    """

    def __init__(
        self,
        api_key,
        model="gpt-4o-mini",
        base_url=None,
        mode="hybrid",
        max_papers_per_query=5,
        verbose=True,
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.mode = mode
        self.max_papers_per_query = max_papers_per_query
        self.verbose = verbose

        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical trial-to-paper linking agent. "
                    "Your job is to retrieve candidate PubMed papers for a clinical trial, "
                    "judge which paper best matches the trial, and extract survival evidence."
                )
            }
        ]
    def _clean_llm_json(content: str) -> str:
        # remove ```json ... ```
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        content = self._clean_llm_json(content)
        return json.loads(content)

    def get_semantic_terms(self, fields: dict) -> dict:
        user_prompt = f"""
You are helping construct a PubMed retrieval query for one specific clinical trial.

Trial information:
- NCT ID: {fields.get('nct_id')}
- Brief title: {fields.get('brief_title')}
- Official title: {fields.get('official_title')}
- Brief summary: {fields.get('brief_summary')}
- Disease: {fields.get('disease')}
- Drugs: {fields.get('drugs')}
- Phase: {fields.get('phase')}

Goal:
Extract the most discriminative semantic terms that would help retrieve papers specifically about THIS trial, not just the general topic.

Rules:
- Prefer concrete disease subtype, treatment setting, regimen pattern, and special descriptors
- Keep terms concise
- Avoid generic words like "study", "patients", "trial"
- Do not hallucinate facts not supported by the input
- Return ONLY JSON

Return JSON with keys:
- disease_terms: list of strings
- drug_terms: list of strings
- setting_terms: list of strings
- other_terms: list of strings
"""
        return self._call_llm_json(
            system_prompt="You are a biomedical information retrieval assistant.",
            user_prompt=user_prompt
        )
    
    def judge_candidates(self, fields: dict, candidates: list) -> dict:
        candidate_text = []
        for i, p in enumerate(candidates, start=1):
            candidate_text.append(
                f"[{i}] PMID: {p.get('pubmed_id')}\n"
                f"Title: {p.get('title')}\n"
                f"Abstract: {p.get('abstract', '')}\n"
            )

        user_prompt = f"""
You are judging which PubMed paper best matches a specific clinical trial.

Trial:
- NCT ID: {fields.get('nct_id')}
- Brief title: {fields.get('brief_title')}
- Official title: {fields.get('official_title')}
- Disease: {fields.get('disease')}
- Drugs: {fields.get('drugs')}
- Phase: {fields.get('phase')}

Candidate papers:
{chr(10).join(candidate_text)}

Task:
Select the paper most likely reporting results for this trial.

Return ONLY JSON with keys:
- match_found: true/false
- selected_pubmed_id: string or null
- selected_title: string or null
- confidence: high/medium/low
- reasoning: short explanation
"""
        return self._call_llm_json(
            system_prompt="You are a biomedical trial-paper matching assistant.",
            user_prompt=user_prompt
        )
    def retrieve_candidates(self, fields: dict):
        papers_A, papers_B, papers_C = [], [], []
        semantic_terms = None
        query_A = query_B = query_C = None

        query_A = build_query_A(fields)
        if query_A:
            if self.verbose:
                print("\n=== Query A ===")
                print(query_A)
            papers_A = _run_pubmed_query_once(query_A, max_papers=self.max_papers_per_query)

        if self.mode == "hybrid":
            semantic_terms = self.get_semantic_terms(fields)

            query_B = build_query_B_llm(fields, semantic_terms)
            if self.verbose:
                print("\n=== Query B ===")
                print(query_B)
            papers_B = _run_pubmed_query_once(query_B, max_papers=self.max_papers_per_query)

            query_C = build_query_C(fields)
            if self.verbose:
                print("\n=== Query C ===")
                print(query_C)
            papers_C = _run_pubmed_query_once(query_C, max_papers=self.max_papers_per_query)

        all_candidates = dedup_papers_by_pmid(papers_A + papers_B + papers_C)

        return {
            "semantic_terms": semantic_terms,
            "query_A": query_A,
            "query_B": query_B,
            "query_C": query_C,
            "papers_A": papers_A,
            "papers_B": papers_B,
            "papers_C": papers_C,
            "all_candidates": all_candidates,
        }

    def run_one_trial(self, trial_row: dict) -> dict:
        fields = extract_trial_retrieval_fields(trial_row)

        if self.verbose:
            print("\n=== Trial fields ===")
            print(fields)

        retrieval_result = self.retrieve_candidates(fields)
        all_candidates = retrieval_result["all_candidates"]

        if self.verbose:
            print(f"\nUnique candidates: {len(all_candidates)}")
            for p in all_candidates[:5]:
                print("Candidate:", p.get("pubmed_id"), p.get("title"))

        judge_result = self.judge_candidates(fields, all_candidates)

        if self.verbose:
            print("\n=== Agent Judge Result ===")
            print(judge_result)

        link_result = {
            "mode": self.mode,
            "trial_fields": fields,
            **retrieval_result,
            "judge_result": judge_result,
        }

        survival_result = self.extract_survival(link_result, trial_row)

        return {
            "trial_row": trial_row,
            "link_result": link_result,
            "survival_result": survival_result,
        }