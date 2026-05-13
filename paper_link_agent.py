"""
PaperLinkAgent — a proper BaseAgent subclass for trial-to-paper linking.

Key differences from PaperLink_legacy.py / PaperLink.py:
- Inherits BaseAgent: owns its own OpenAI client and conversation history.
- All four LLM decisions go through self._call_chat_model() — no opaque
  black-box delegation to linker.py or llm.py LLM functions.
- Non-LLM work (PubMed queries, dedup, text fetching, plot building) is
  still delegated to the existing utility modules.
- self.clear() resets the message history before each independent trial so
  that runs don't bleed context into one another.
- Output shapes are identical to the legacy agents for downstream
  compatibility with conversation.py and Coordinator_legacy.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from base_agent import BaseAgent
from linker import dedup_papers_by_pmid
from tools.pmc import get_best_text_for_extraction
from tools.pdf import read_pdf_text
from prompts import (
    PAPER_LINK_SYSTEM_PROMPT,
    build_paper_link_semantic_terms_prompt,
    build_outcome_parse_prompt,
    build_outcome_extraction_prompt,
    build_primary_endpoint_identification_prompt,
    build_survival_eligibility_prompt,
    build_trial_paper_judging_prompt,
)
from tools.outcomes import (
    OutcomeSpec,
    DEFAULT_OUTCOME_SPECS,
    build_multi_outcome_plot_rows,
    draw_all_outcome_plots,
)
from tools import (
    PubMedFilter,
    _run_pubmed_query_once,
    build_query_A,
    build_query_B_llm,
    build_query_C,
    extract_trial_retrieval_fields,
    fetch_pubmed_abstract,
)


class PaperLinkAgent(BaseAgent):
    """
    Agent for linking clinical trials to PubMed papers and extracting
    arm-level survival outcomes.

    LLM responsibilities (owned by this agent):
    1. Extract discriminative PubMed search terms for a trial.
    2. Judge which candidate paper best matches the trial.
    3. Decide whether a paper is eligible for arm-level survival extraction.
    4. Extract median OS by treatment arm.

    Non-LLM work (delegated to utility modules):
    - PubMed query building and execution  →  tools/
    - Candidate deduplication              →  linker.dedup_papers_by_pmid
    - PMC / abstract text fetching         →  llm.get_best_text_for_extraction
    - Plot-row construction                →  tools/outcomes.build_multi_outcome_plot_rows
    """

    def __init__(
        self,
        mode: str = "nct_only",
        draw_plots: bool = True,
        max_papers_per_query: int = 5,
        max_text_chars: int = 12000,
        pubmed_filter: Optional[PubMedFilter] = None,
        outcomes_raw: Optional[str] = None,
        enable_pdf_prompt: bool = True,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(system_prompt=PAPER_LINK_SYSTEM_PROMPT, model=model)
        self.mode = mode
        self.draw_plots = draw_plots
        self.max_papers_per_query = max_papers_per_query
        self.max_text_chars = max_text_chars
        self.pubmed_filter = pubmed_filter
        self._outcomes_raw = outcomes_raw
        self.enable_pdf_prompt = enable_pdf_prompt
        # parsed lazily on first run so __init__ makes no LLM calls
        self.outcome_specs: Optional[List[OutcomeSpec]] = None
        # True when user requests per-paper primary endpoint discovery
        self.auto_primary: bool = False

    def _ensure_outcome_specs(self) -> None:
        """Parse outcome_specs from raw user input on first call (lazy)."""
        if self.outcome_specs is not None:
            return
        if self._outcomes_raw:
            self.outcome_specs = self.parse_outcome_request(self._outcomes_raw)
        else:
            self.outcome_specs = list(DEFAULT_OUTCOME_SPECS)

    # ------------------------------------------------------------------
    # LLM-driven steps — the agent owns these
    # ------------------------------------------------------------------

    def parse_outcome_request(self, raw_input: str) -> List[OutcomeSpec]:
        """
        One LLM call: parse the user's free-text outcome request into a list
        of OutcomeSpec objects that drive all downstream extraction and plotting.
        Called once per run_batch(), not once per trial.

        Special case: if the LLM returns the sentinel key "__auto_primary__",
        sets self.auto_primary = True and returns [] — specs are determined
        per paper by identify_primary_endpoint().
        """
        prompt = build_outcome_parse_prompt(raw_input)
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            items = json.loads(content)
            if any(item.get("key") == "__auto_primary__" for item in items):
                print("[PaperLinkAgent] Auto-primary mode: primary endpoint will be identified per paper.")
                self.auto_primary = True
                return []
            specs = [
                OutcomeSpec(
                    key=item["key"],
                    display=item["display"],
                    plot_type=item.get("plot_type", "table_only"),
                )
                for item in items
                if "key" in item and "display" in item
            ]
            return specs or list(DEFAULT_OUTCOME_SPECS)
        except Exception:
            print("[PaperLinkAgent] outcome parse failed, falling back to OS only.")
            return list(DEFAULT_OUTCOME_SPECS)

    def identify_primary_endpoint(
        self,
        pubmed_record: Dict[str, Any],
        source_text: str,
        pmcid: Optional[str] = None,
    ) -> tuple:
        """
        Per-paper LLM call: read the paper and return the primary endpoint(s).
        Returns (List[OutcomeSpec], raw_result_dict).
        raw_result_dict contains the full LLM response including evidence quotes.
        Returns ([], {}) if the primary endpoint cannot be identified.
        """
        prompt = build_primary_endpoint_identification_prompt(
            pubmed_record, source_text[: self.max_text_chars], pmcid=pmcid
        )
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            raw = json.loads(content)
        except Exception:
            print(f"[PaperLinkAgent] Primary endpoint identification JSON parse failed for PMID={pubmed_record.get('pubmed_id')}.")
            return [], {}

        if not raw.get("found"):
            print(f"  [auto-primary] Primary endpoint not found: {raw.get('notes', '')}")
            return [], raw

        specs = []
        for item in raw.get("primary_endpoints", []):
            if "key" not in item or "display" not in item:
                continue
            specs.append(OutcomeSpec(
                key=item["key"],
                display=item["display"],
                plot_type=item.get("plot_type", "table_only"),
            ))
            print(f"  [auto-primary] Identified: {item['display']} | evidence: {item.get('evidence', '')[:80]}")
        return specs, raw

    def get_semantic_terms(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the LLM to extract discriminative PubMed retrieval terms for
        this trial (used to build Query B in hybrid mode).
        """
        prompt = build_paper_link_semantic_terms_prompt(fields)
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            return json.loads(content)
        except Exception as exc:
            raise ValueError(
                f"PaperLinkAgent.get_semantic_terms: invalid JSON:\n{content}"
            ) from exc

    def judge_trial_papers(
        self,
        trial_fields: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to select the best-matching paper from the candidate list.
        """
        if not candidates:
            return {
                "match_found": False,
                "selected_pubmed_id": None,
                "selected_title": None,
                "label": "no_candidate",
                "confidence": "low",
                "reason": "No candidate papers were retrieved.",
            }

        paper_blocks = [
            "\n".join([
                f"Candidate {i}",
                f"PMID: {p.get('pubmed_id')}",
                f"Title: {p.get('title')}",
                f"Journal: {p.get('journal')}",
                f"Abstract: {p.get('abstract')}",
            ])
            for i, p in enumerate(candidates, 1)
        ]
        prompt = build_trial_paper_judging_prompt(
            trial_fields, "\n\n".join(paper_blocks)
        )
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            return json.loads(content)
        except Exception as exc:
            raise ValueError(
                f"PaperLinkAgent.judge_trial_papers: invalid JSON:\n{content}"
            ) from exc

    def judge_survival_eligibility(
        self,
        pubmed_record: Dict[str, Any],
        source_text: str,
        pmcid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask the LLM whether this paper is eligible for arm-level efficacy extraction.
        """
        prompt = build_survival_eligibility_prompt(
            pubmed_record,
            source_text[: self.max_text_chars],
            pmcid=pmcid,
        )
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            return json.loads(content)
        except Exception:
            # conservative fallback: don't block extraction on a parse failure
            return {
                "eligible": True,
                "paper_type": "unknown",
                "reason": "Eligibility JSON parse failed; defaulting to extraction.",
            }

    def extract_outcomes_from_text(
        self,
        pubmed_record: Dict[str, Any],
        source_text: str,
        source_used: str,
        pmcid: Optional[str] = None,
        trial_id: Optional[str] = None,
        trial_label: Optional[str] = None,
        specs: Optional[List[OutcomeSpec]] = None,
    ) -> Dict[str, Any]:
        """
        Extract outcomes per arm from paper text.
        specs: which outcomes to extract. Defaults to self.outcome_specs.
               Pass a per-paper list when self.auto_primary is True.
        """
        effective_specs = specs if specs is not None else (self.outcome_specs or DEFAULT_OUTCOME_SPECS)
        prompt = build_outcome_extraction_prompt(
            pubmed_record,
            source_text[: self.max_text_chars],
            source_used,
            effective_specs,
            pmcid=pmcid,
        )
        response = self._call_chat_model(user_message=prompt)
        content = self._clean_json(response.choices[0].message.content or "")
        try:
            result = json.loads(content)
        except Exception as exc:
            raise ValueError(
                f"PaperLinkAgent.extract_outcomes_from_text: invalid JSON:\n{content}"
            ) from exc

        pubmed_id = str(pubmed_record.get("pubmed_id", ""))
        result.setdefault("paper_id", pubmed_id)
        result.setdefault("source_used", source_used)
        result["pmcid"] = pmcid
        result.setdefault("arms", [])
        result.setdefault("notes", "")
        arm_meta = {"arm_name", "arm_sample_size", "arm_sample_size_raw"}
        result["outcome_found"] = any(
            isinstance(v, dict) and v.get("found")
            for arm in result["arms"]
            for k, v in arm.items()
            if k not in arm_meta
        )

        result["plot_rows"] = build_multi_outcome_plot_rows(
            result, effective_specs,
            trial_id=trial_id, trial_label=trial_label,
        )
        return result

    # ------------------------------------------------------------------
    # Non-LLM retrieval — delegates to tools/linker utilities
    # ------------------------------------------------------------------

    def _retrieve_candidates(
        self,
        fields: Dict[str, Any],
        semantic_terms: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run PubMed queries A / B / C and return deduplicated candidates.
        Query B is only built in hybrid mode and uses the LLM-generated terms.
        """
        papers_A, papers_B, papers_C = [], [], []
        query_A = query_B = query_C = None

        query_A = build_query_A(fields, pubmed_filter=self.pubmed_filter)
        if query_A:
            print(f"  [Query A] {query_A}")
            papers_A = _run_pubmed_query_once(
                query_A, max_papers=self.max_papers_per_query
            )

        if self.mode == "hybrid":
            query_B = build_query_B_llm(
                fields, semantic_terms, pubmed_filter=self.pubmed_filter
            )
            print(f"  [Query B] {query_B}")
            papers_B = _run_pubmed_query_once(
                query_B, max_papers=self.max_papers_per_query
            )

            query_C = build_query_C(fields, pubmed_filter=self.pubmed_filter)
            print(f"  [Query C] {query_C}")
            papers_C = _run_pubmed_query_once(
                query_C, max_papers=self.max_papers_per_query
            )

        all_candidates = dedup_papers_by_pmid(papers_A + papers_B + papers_C)

        return {
            "query_A": query_A,
            "query_B": query_B,
            "query_C": query_C,
            "papers_A": papers_A,
            "papers_B": papers_B,
            "papers_C": papers_C,
            "all_candidates": all_candidates,
        }

    # ------------------------------------------------------------------
    # Survival pipeline helper
    # ------------------------------------------------------------------

    def _run_survival_pipeline(
        self,
        judge_result: Dict[str, Any],
        trial_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Given the paper-judging result, fetch the matched paper, screen it
        for eligibility, and extract survival data.

        Returns a dict with the same shape as the legacy
        extract_survival_from_link_result() for downstream compatibility.
        """
        pubmed_id = judge_result.get("selected_pubmed_id")
        trial_id = trial_row.get("nct_id") if trial_row else None
        trial_label = trial_row.get("brief_title") if trial_row else None

        empty_extraction = {
            "paper_id": pubmed_id,
            "outcome_found": False,
            "outcome_type": "overall_survival",
            "source_used": "abstract",
            "arms": [],
            "plot_rows": [],
            "notes": "",
        }

        if not pubmed_id:
            empty_extraction["notes"] = "No paper selected by judge."
            return {
                "status": "error",
                "message": "No selected_pubmed_id from judge.",
                "pubmed_record": None,
                "survival_extraction": empty_extraction,
            }

        try:
            pubmed_record = fetch_pubmed_abstract(pubmed_id)
        except Exception as exc:
            empty_extraction["notes"] = f"Failed to fetch PubMed record: {exc}"
            return {
                "status": "error",
                "message": f"Failed to fetch PubMed abstract for PMID={pubmed_id}",
                "pubmed_record": None,
                "survival_extraction": empty_extraction,
            }

        source_text, source_used, pmcid = get_best_text_for_extraction(pubmed_record)
        print(f"  Source: {source_used} | PMCID: {pmcid}")

        if not source_text or not str(source_text).strip():
            empty_extraction["notes"] = "No usable text available."
            return {
                "status": "error",
                "message": "No usable text for extraction.",
                "pubmed_record": pubmed_record,
                "survival_extraction": empty_extraction,
            }

        eligibility = self.judge_survival_eligibility(
            pubmed_record, source_text, pmcid=pmcid
        )
        eligibility["source_used"] = source_used
        eligibility["pmcid"] = pmcid

        if not eligibility.get("eligible", True):
            empty_extraction["source_used"] = source_used
            empty_extraction["notes"] = (
                f"Paper not eligible. "
                f"paper_type={eligibility.get('paper_type')}; "
                f"reason={eligibility.get('reason')}"
            )
            return {
                "status": "skipped",
                "message": "Paper not eligible for arm-level survival extraction.",
                "pubmed_record": pubmed_record,
                "eligibility_judgment": eligibility,
                "survival_extraction": empty_extraction,
            }

        # In auto_primary mode, identify the primary endpoint from this paper first
        extraction_specs: Optional[List[OutcomeSpec]] = None
        primary_endpoint_identification: Optional[Dict[str, Any]] = None
        if self.auto_primary:
            print(f"  [auto-primary] Identifying primary endpoint for PMID={pubmed_id}...")
            identified, id_raw = self.identify_primary_endpoint(pubmed_record, source_text, pmcid=pmcid)
            primary_endpoint_identification = id_raw
            if not identified:
                empty_extraction["source_used"] = source_used
                empty_extraction["notes"] = "Auto-primary: could not identify primary endpoint from paper text."
                return {
                    "status": "skipped",
                    "message": "Auto-primary mode: primary endpoint not identified.",
                    "pubmed_record": pubmed_record,
                    "eligibility_judgment": eligibility,
                    "primary_endpoint_identification": primary_endpoint_identification,
                    "survival_extraction": empty_extraction,
                }
            extraction_specs = identified

        try:
            extraction = self.extract_outcomes_from_text(
                pubmed_record, source_text, source_used,
                pmcid=pmcid, trial_id=trial_id, trial_label=trial_label,
                specs=extraction_specs,
            )
        except Exception as exc:
            empty_extraction["notes"] = f"Extraction failed: {exc}"
            return {
                "status": "error",
                "message": f"Survival extraction failed for PMID={pubmed_id}",
                "pubmed_record": pubmed_record,
                "eligibility_judgment": eligibility,
                "survival_extraction": empty_extraction,
            }

        if extraction.get("source_used") == "abstract":
            extraction = self._try_pdf_fallback(
                extraction, pubmed_record, pubmed_id, trial_id, trial_label
            )

        return {
            "status": "ok",
            "message": "Survival extraction completed.",
            "pubmed_record": pubmed_record,
            "eligibility_judgment": eligibility,
            "primary_endpoint_identification": primary_endpoint_identification,
            "survival_extraction": extraction,
            "source_text": source_text,
        }

    def _try_pdf_fallback(
        self,
        extraction: Dict[str, Any],
        pubmed_record: Dict[str, Any],
        pubmed_id: str,
        trial_id: Optional[str],
        trial_label: Optional[str],
    ) -> Dict[str, Any]:
        """
        Offer the user a chance to supply a local PDF when only the abstract
        was used, then re-run extraction if a valid path is given.
        """
        if not self.enable_pdf_prompt:
            return extraction

        print(f"\n[PaperLinkAgent] Abstract-only extraction for PMID {pubmed_id}.")
        print(f"  Title: {pubmed_record.get('title', '')}")
        choice = input("Provide a local PDF for a second-pass extraction? [y/N]: ").strip().lower()

        if choice not in {"y", "yes"}:
            extraction["notes"] = (
                extraction.get("notes", "") + " | User skipped PDF upload."
            )
            return extraction

        pdf_path = input("PDF path (or Enter to skip): ").strip()
        if not pdf_path:
            extraction["notes"] = (
                extraction.get("notes", "") + " | No PDF path provided."
            )
            return extraction

        pdf_text = read_pdf_text(pdf_path)
        if not pdf_text:
            extraction["notes"] = (
                extraction.get("notes", "") + " | PDF had no readable text."
            )
            return extraction

        print("[PaperLinkAgent] Re-running extraction with user PDF...")
        try:
            extraction = self.extract_outcomes_from_text(
                pubmed_record, pdf_text, "user_pdf",
                pmcid=None, trial_id=trial_id, trial_label=trial_label,
            )
            extraction["user_pdf_path"] = pdf_path
        except Exception as exc:
            extraction["notes"] = (
                extraction.get("notes", "") + f" | PDF retry failed: {exc}"
            )
        return extraction

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def find_papers_for_trial(
        self, trial_row: Dict[str, Any], idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Steps 1-4 of run_one: retrieve candidates and judge the best paper.
        Does NOT run eligibility screening or outcome extraction.
        """
        self._ensure_outcome_specs()
        self.clear()
        print(f"\n[PaperLinkAgent] Finding papers for trial {idx}...")

        fields = extract_trial_retrieval_fields(trial_row)
        semantic_terms = self.get_semantic_terms(fields)
        retrieval = self._retrieve_candidates(fields, semantic_terms)
        all_candidates = retrieval["all_candidates"]
        print(
            f"  Candidates: "
            f"A={len(retrieval['papers_A'])} "
            f"B={len(retrieval['papers_B'])} "
            f"C={len(retrieval['papers_C'])} "
            f"unique={len(all_candidates)}"
        )

        judge_result = self.judge_trial_papers(fields, all_candidates)
        print(
            f"  AI recommendation: match={judge_result.get('match_found')} | "
            f"PMID={judge_result.get('selected_pubmed_id')}"
        )

        link_result = {
            "mode": self.mode,
            "trial_fields": fields,
            "semantic_terms": semantic_terms,
            "query_A": retrieval["query_A"],
            "query_B": retrieval["query_B"],
            "query_C": retrieval["query_C"],
            "papers_A": retrieval["papers_A"],
            "papers_B": retrieval["papers_B"],
            "papers_C": retrieval["papers_C"],
            "all_candidates": all_candidates,
            "judge_result": judge_result,
        }
        return {"trial": trial_row, "semantic_terms": semantic_terms, "link_result": link_result}

    def extract_for_trial(
        self,
        find_result: Dict[str, Any],
        selected_pmid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Steps 5-6 of run_one: eligibility screen + outcome extraction.
        selected_pmid: if provided and different from the AI pick, overrides the judge result.
        """
        trial_row = find_result["trial"]
        link_result = find_result["link_result"]
        judge_result = dict(link_result["judge_result"])

        if selected_pmid and str(selected_pmid) != str(judge_result.get("selected_pubmed_id") or ""):
            candidates = link_result.get("all_candidates", [])
            matched = next((c for c in candidates if str(c.get("pubmed_id")) == str(selected_pmid)), None)
            judge_result["selected_pubmed_id"] = selected_pmid
            judge_result["match_found"] = True
            judge_result["confidence"] = "user_selected"
            judge_result["reason"] = "User manually selected this paper."
            if matched:
                judge_result["selected_title"] = matched.get("title")

        self.clear()
        survival_result = self._run_survival_pipeline(judge_result, trial_row)
        plot_rows = survival_result.get("survival_extraction", {}).get("plot_rows", [])

        return {
            "trial": trial_row,
            "semantic_terms": find_result.get("semantic_terms"),
            "link_result": {**link_result, "judge_result": judge_result},
            "survival_result": survival_result,
            "plot_rows": plot_rows,
        }

    def run_one(
        self, trial_row: Dict[str, Any], idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process one trial end-to-end:
          1. Extract structured trial fields
          2. Get semantic search terms          (LLM)
          3. Retrieve PubMed candidates
          4. Judge which paper matches          (LLM)
          5. Fetch paper, screen eligibility   (LLM)
          6. Extract arm-level survival data   (LLM)
        """
        self._ensure_outcome_specs()  # parse outcomes on first call if not yet done
        self.clear()  # each trial is independent — reset conversation history
        print(f"\n[PaperLinkAgent] Processing trial {idx}...")

        fields = extract_trial_retrieval_fields(trial_row)

        semantic_terms = self.get_semantic_terms(fields)

        retrieval = self._retrieve_candidates(fields, semantic_terms)
        all_candidates = retrieval["all_candidates"]
        print(
            f"  Candidates: "
            f"A={len(retrieval['papers_A'])} "
            f"B={len(retrieval['papers_B'])} "
            f"C={len(retrieval['papers_C'])} "
            f"unique={len(all_candidates)}"
        )

        judge_result = self.judge_trial_papers(fields, all_candidates)
        print(
            f"  Match: {judge_result.get('match_found')} | "
            f"PMID: {judge_result.get('selected_pubmed_id')}"
        )

        # build link_result in the same shape as the legacy pipeline
        link_result = {
            "mode": self.mode,
            "trial_fields": fields,
            "semantic_terms": semantic_terms,
            "query_A": retrieval["query_A"],
            "query_B": retrieval["query_B"],
            "query_C": retrieval["query_C"],
            "papers_A": retrieval["papers_A"],
            "papers_B": retrieval["papers_B"],
            "papers_C": retrieval["papers_C"],
            "all_candidates": all_candidates,
            "judge_result": judge_result,
        }

        survival_result = self._run_survival_pipeline(judge_result, trial_row)
        plot_rows = survival_result.get("survival_extraction", {}).get("plot_rows", [])

        return {
            "trial": trial_row,
            "semantic_terms": semantic_terms,
            "link_result": link_result,
            "survival_result": survival_result,
            "plot_rows": plot_rows,
        }

    def run_batch(
        self,
        trials: List[Dict[str, Any]],
        indices: List[int],
    ) -> Dict[str, Any]:
        # parse outcome specs once before iterating trials (one LLM call total)
        self._ensure_outcome_specs()
        results = []
        all_plot_rows = []

        for trial, idx in zip(trials, indices):
            res = self.run_one(trial, idx)
            results.append(res)
            if res["plot_rows"]:
                all_plot_rows.extend(res["plot_rows"])

        if self.draw_plots and all_plot_rows:
            draw_all_outcome_plots(all_plot_rows, self.outcome_specs or DEFAULT_OUTCOME_SPECS)  # unit_logs discarded in CLI mode

        return {
            "per_trial_results": results,
            "all_plot_rows": all_plot_rows,
        }

    def build_survival_table(
        self, per_trial_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Build a trial-level summary table (one row per trial) for audit
        and debugging.
        """
        rows = []
        for item in per_trial_results:
            trial = item.get("trial", {}) or {}
            link_result = item.get("link_result", {}) or {}
            survival_result = item.get("survival_result", {}) or {}

            judge_result = link_result.get("judge_result", {}) or {}
            extraction = survival_result.get("survival_extraction", {}) or {}
            eligibility = survival_result.get("eligibility_judgment", {}) or {}

            rows.append({
                "nct_id": trial.get("nct_id"),
                "trial_title": trial.get("brief_title"),
                "pubmed_match_found": judge_result.get("match_found"),
                "pubmed_id": judge_result.get("selected_pubmed_id"),
                "paper_title": judge_result.get("selected_title"),
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json(content: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()
