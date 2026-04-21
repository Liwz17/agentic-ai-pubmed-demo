from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional

from base_agent import BaseAgent
from prompts import (
    INSPECTOR_SYSTEM_PROMPT,
    build_inspector_survival_eligibility_prompt,
    build_inspector_trial_paper_judging_prompt,
)
from llm import get_best_text_for_extraction


class InspectorAgent(BaseAgent):
    """
    Focused inspection agent for two tasks only:
    1. judge whether a candidate PubMed paper matches a trial
    2. judge whether a paper is eligible for arm-level survival extraction
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_text_chars: int = 12000,
    ) -> None:
        super().__init__(
            system_prompt=INSPECTOR_SYSTEM_PROMPT,
            model=model,
        )
        self.max_text_chars = max_text_chars

    def add_functions(self, functions: Iterable[Any]) -> None:
        super().add_functions(functions)

    def add_message(self, role: str, content: str) -> None:
        super().add_message(role, content)

    def _call_chat_model(
        self,
        user_message: Optional[str] = None,
        temperature: float = 0.0,
        extra_messages: Optional[list[dict[str, str]]] = None,
        **completion_kwargs: Any,
    ) -> Any:
        return super()._call_chat_model(
            user_message=user_message,
            temperature=temperature,
            extra_messages=extra_messages,
            **completion_kwargs,
        )

    def _call_chat_model_streaming(
        self,
        user_message: str,
        temperature: float = 0.0,
        extra_messages: Optional[list[dict[str, str]]] = None,
        **completion_kwargs: Any,
    ):
        return super()._call_chat_model_streaming(
            user_message=user_message,
            temperature=temperature,
            extra_messages=extra_messages,
            **completion_kwargs,
        )

    def clear(self) -> None:
        super().clear()

    def judge_trial_papers(
        self,
        trial_fields: Dict[str, Any],
        candidate_papers: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Judge which candidate paper is most likely linked to the given trial.
        """
        if not candidate_papers:
            return {
                "match_found": False,
                "selected_pubmed_id": None,
                "selected_title": None,
                "label": "no_candidate",
                "confidence": "low",
                "reason": "No candidate papers were retrieved.",
            }

        paper_blocks = []
        for i, paper in enumerate(candidate_papers, start=1):
            paper_blocks.append(
                "\n".join(
                    [
                        f"Candidate {i}",
                        f"PMID: {paper.get('pubmed_id')}",
                        f"Title: {paper.get('title')}",
                        f"Journal: {paper.get('journal')}",
                        f"Abstract: {paper.get('abstract')}",
                    ]
                )
            )

        prompt = build_inspector_trial_paper_judging_prompt(
            trial_fields=trial_fields,
            papers_text="\n\n".join(paper_blocks),
        )

        response = self._call_chat_model(user_message=prompt, temperature=0.0)
        content = response.choices[0].message.content or ""
        content = self._clean_llm_json(content)

        try:
            return json.loads(content)
        except Exception as exc:
            raise ValueError(
                "InspectorAgent did not return valid JSON for trial-paper judgment:\n"
                f"{content}"
            ) from exc

    def judge_survival_plot_eligibility(
        self,
        pubmed_record: Dict[str, Any],
        full_text: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Judge whether a paper is suitable for arm-level survival extraction.
        """
        if full_text and str(full_text).strip():
            source_text = str(full_text).strip()
            source_used = "user_pdf"
            pmcid = None
        else:
            source_text, source_used, pmcid = get_best_text_for_extraction(
                pubmed_record,
                email=email,
            )

        if not source_text or not str(source_text).strip():
            return {
                "eligible": False,
                "reason": "No usable text available.",
                "paper_type": "unknown",
                "source_used": source_used,
                "pmcid": pmcid,
            }

        prompt = build_inspector_survival_eligibility_prompt(
            pubmed_record=pubmed_record,
            source_text=source_text[: self.max_text_chars],
            pmcid=pmcid,
        )

        response = self._call_chat_model(user_message=prompt, temperature=0.0)
        content = response.choices[0].message.content or ""
        content = self._clean_llm_json(content)

        try:
            result = json.loads(content)
        except Exception:
            result = {
                "eligible": True,
                "paper_type": "unknown",
                "reason": "Eligibility JSON parse failed; defaulting to extraction.",
            }

        result["source_used"] = source_used
        result["pmcid"] = pmcid
        return result

    @staticmethod
    def _clean_llm_json(content: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()
