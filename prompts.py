"""
Central prompt layer for the clinical-trial-to-paper pipeline.

Design:
- one system prompt per agent
- multiple task-specific prompt templates grouped by agent/task

This file is intentionally prompt-only. It does not contain business logic,
API calls, or orchestration code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Agent-level system prompts
# ---------------------------------------------------------------------------

TRIAL_RETRIEVAL_SYSTEM_PROMPT = """
You are TrialRetrievalAgent for a clinical-trial-to-paper pipeline.

Your LLM responsibilities are limited to two trial-retrieval-side tasks:
- convert a user's natural-language request into a structured ClinicalTrials.gov search query
- extract discriminative semantic terms for a specific trial to support downstream paper retrieval

The surrounding workflow, such as API calls, candidate retrieval, filtering,
selection, and orchestration, is handled outside the LLM. Your job is to return
clean, accurate structured outputs that downstream code can use directly.

General rules:
- Be precise and conservative.
- Do not hallucinate clinical details that are not supported by the user request.
- Normalize disease names, drug names, phase labels, and dates when possible.
- When extracting semantic terms, prefer concrete retrieval cues rather than generic wording.
- If information is missing or ambiguous, make the safest reasonable interpretation.
- Prefer compact structured outputs over explanations.
""".strip()


PAPER_LINK_SYSTEM_PROMPT = """
You are PaperLinkAgent, a biomedical trial-to-paper linking assistant.

Your responsibilities include:
- linking clinical trials to likely PubMed result papers
- evaluating whether a paper is a true trial match
- extracting key outcome information from paper text
- preparing structured outputs for downstream audit and plotting

General rules:
- Prefer evidence-backed matches over aggressive matching.
- Do not force a link when evidence is weak.
- Extract only what is explicitly supported by the text.
- Preserve uncertainty in structured outputs.
""".strip()


INSPECTOR_SYSTEM_PROMPT = """
You are InspectorAgent, a focused quality-control assistant for the
clinical-trial-to-paper pipeline.

Your responsibilities are limited to two inspection tasks:
- judging whether a candidate PubMed paper is likely a true match for a trial
- judging whether a paper is eligible for arm-level survival extraction

General rules:
- Be conservative and evidence-driven.
- Do not force positive judgments when evidence is weak.
- Prefer precise structured outputs over narrative discussion.
- Focus only on match quality and survival-extraction eligibility.
""".strip()


# Backward-compatible aliases if later code expects these shorter names.
TRIAL_RETRIEVAL_PROMPT = TRIAL_RETRIEVAL_SYSTEM_PROMPT
PAPER_LINK_PROMPT = PAPER_LINK_SYSTEM_PROMPT
INSPECTOR_PROMPT = INSPECTOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Shared task prompts
# ---------------------------------------------------------------------------

TRIAL_TASK_PROMPTS: Dict[str, str] = {
    "parse_query": """
You are a clinical research assistant.

Convert the following user request into a structured JSON query for
ClinicalTrials.gov search.

User request:
{user_input}

Return ONLY JSON with the following fields:
- disease: string
- drugs: list of strings
- phase: one of PHASE1, PHASE2, PHASE3, PHASE4 (uppercase)
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD

Rules:
- Normalize drug names (for example, "pembro" -> "pembrolizumab")
- Normalize disease names when needed
- Normalize phase labels into PHASE1/2/3/4 format
- If date is given as year, convert it into a full date range
- Be precise and do not hallucinate

Example output:
{{
  "disease": "lung cancer",
  "drugs": ["pembrolizumab", "ipilimumab"],
  "phase": "PHASE2",
  "start_date": "2016-01-01",
  "end_date": "2020-12-31"
}}
""".strip(),
    "expand_semantic_terms": """
You are helping construct a PubMed retrieval query for one specific clinical trial.

Trial information:
- NCT ID: {nct_id}
- Brief title: {brief_title}
- Official title: {official_title}
- Brief summary: {brief_summary}
- Disease: {disease}
- Drugs: {drugs}
- Phase: {phase}

Goal:
Extract the most discriminative semantic terms that would help retrieve papers
specifically about THIS trial, not just the general topic.

Return ONLY JSON with keys:
- disease_terms: list of strings
- drug_terms: list of strings
- setting_terms: list of strings
- other_terms: list of strings

Rules:
- Prefer concrete disease subtype, treatment setting, regimen pattern, and special descriptors
- Keep terms concise
- Avoid generic words like "study", "patients", "trial"
- Do not hallucinate facts not supported by the input
- Return ONLY JSON
""".strip(),
    "filter_relevance": """
You are reviewing candidate ClinicalTrials.gov studies for relevance to a user request.

User request:
{user_input}

Structured query:
{structured_query_json}

Candidate trials:
{candidate_trials_text}

Task:
Assess whether each trial is relevant to the user's request and note the main reason.

Return ONLY JSON in this format:
{{
  "relevance_results": [
    {{
      "trial_index": 0,
      "is_relevant": true,
      "confidence": "high",
      "reason": "brief explanation"
    }}
  ]
}}
""".strip(),
    "rank_trials": """
You are ranking candidate ClinicalTrials.gov studies for downstream review.

User request:
{user_input}

Structured query:
{structured_query_json}

Candidate trials:
{candidate_trials_text}

Task:
Rank the most relevant trials for the user's intent.

Return ONLY JSON in this format:
{{
  "ranked_trials": [
    {{
      "trial_index": 0,
      "rank": 1,
      "confidence": "high",
      "reason": "brief explanation"
    }}
  ]
}}

Rules:
- Prefer disease, intervention, phase, and date consistency
- Do not force confidence when the match is weak
- Keep the rationale short and practical
""".strip(),
}


def build_trial_query_parsing_prompt(user_input: str) -> str:
    return TRIAL_TASK_PROMPTS["parse_query"].format(user_input=user_input)


def build_trial_semantic_expansion_prompt(
    trial_fields: Dict[str, Any],
) -> str:
    return TRIAL_TASK_PROMPTS["expand_semantic_terms"].format(
        nct_id=trial_fields.get("nct_id"),
        brief_title=trial_fields.get("brief_title"),
        official_title=trial_fields.get("official_title"),
        brief_summary=trial_fields.get("brief_summary"),
        disease=trial_fields.get("disease"),
        drugs=trial_fields.get("drugs"),
        phase=trial_fields.get("phase"),
    )


def build_trial_relevance_filter_prompt(
    user_input: str,
    structured_query_json: str,
    candidate_trials_text: str,
) -> str:
    return TRIAL_TASK_PROMPTS["filter_relevance"].format(
        user_input=user_input,
        structured_query_json=structured_query_json,
        candidate_trials_text=candidate_trials_text,
    )


def build_trial_ranking_prompt(
    user_input: str,
    structured_query_json: str,
    candidate_trials_text: str,
) -> str:
    return TRIAL_TASK_PROMPTS["rank_trials"].format(
        user_input=user_input,
        structured_query_json=structured_query_json,
        candidate_trials_text=candidate_trials_text,
    )


LINKING_TASK_PROMPTS: Dict[str, str] = {
    "extract_trial_semantic_terms": """
You are helping construct a PubMed retrieval query for one specific clinical trial.

Trial information:
- NCT ID: {nct_id}
- Brief title: {brief_title}
- Official title: {official_title}
- Brief summary: {brief_summary}
- Disease: {disease}
- Drugs: {drugs}
- Phase: {phase}

Goal:
Extract the most discriminative semantic terms that would help retrieve papers
specifically about THIS trial, not just the general topic.

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
""".strip(),
    "judge_trial_papers": """
You are helping evaluate whether candidate PubMed papers truly match one ClinicalTrials.gov trial.

Trial:
- NCT ID: {nct_id}
- Brief title: {brief_title}
- Official title: {official_title}
- Brief summary: {brief_summary}
- Disease: {disease}
- Drugs: {drugs}
- Phase: {phase}

Candidate PubMed papers:
{papers_text}

Task:
Decide whether one of the candidate papers is likely reporting results for THIS specific trial.

Use these labels:
- primary_results
- secondary_or_followup
- protocol_or_review
- not_a_match

Rules:
- Prefer papers that match the disease, drug(s), phase, and trial setting
- If an NCT ID is explicitly consistent, that is strong evidence
- Do not force a match if the candidates are only generally related
- If no candidate is convincing, return match_found=false and selected index null

Return ONLY JSON in this format:
{{
  "match_found": true,
  "selected_index": 1,
  "selected_pubmed_id": "12345678",
  "selected_title": "paper title",
  "label": "primary_results",
  "confidence": "high",
  "reason": "brief explanation"
}}
""".strip(),
}


SURVIVAL_TASK_PROMPTS: Dict[str, str] = {
    "judge_survival_plot_eligibility": """
You are an expert biomedical trial-reading assistant.

Task:
Determine whether this paper is appropriate for ARM-LEVEL overall survival extraction
for a forest-plot style summary across trials.

The paper is ELIGIBLE only if it primarily reports study-level or treatment-arm-level
overall survival outcomes, for example median OS by arm, with or without confidence intervals.

The paper is NOT ELIGIBLE if it is mainly:
- a case report
- a case series
- patient-by-patient narrative reporting
- a review, commentary, or non-original trial report
- a biomarker or correlative paper without treatment-arm-level survival extraction target
- a paper that reports only individual patient outcomes rather than arm-level trial outcomes

Return JSON only in this format:
{{
  "eligible": true,
  "paper_type": "trial_report | case_series | case_report | biomarker_study | review | unknown",
  "reason": "short explanation"
}}

Paper metadata:
- PubMed ID: {pubmed_id}
- Title: {title}
- Journal: {journal}
- Year: {year}
- PMCID: {pmcid}

Text:
\"\"\"
{source_text}
\"\"\"
""".strip(),
    "extract_survival_from_text": """
You are an expert biomedical information extraction assistant.

Paper metadata:
- PubMed ID: {pubmed_id}
- PMCID: {pmcid}
- Title: {title}
- Journal: {journal}
- Year: {year}

Task:
Extract the median OVERALL SURVIVAL (OS), its 95% confidence interval (CI),
and the sample size for each treatment arm from the text below.

Important rules:
1. Extract ONLY OVERALL SURVIVAL (OS), not progression-free survival (PFS), disease-free survival (DFS), duration of response (DOR), overall response rate (ORR), or other endpoints.
2. If this is a single-arm study, return one arm only.
3. If OS is reported for subgroups only, extract only if those subgroups are clearly the treatment arms of the study.
4. If OS is not explicitly reported, set "outcome_found" to false and return an empty arms list.
5. If the median OS is reported as "not reached" or "NR", preserve that exactly in the raw field. For parsed numeric fields, use null when a bound is not numeric.
6. Do not infer or calculate anything that is not explicitly stated in the text.
7. Preserve reported values faithfully. Do not convert years to months. Do not perform arithmetic.
8. Include the exact supporting sentence or phrase in "evidence".
9. If arm names are not explicitly stated next to the OS result, use the clearest study-arm label supported by the text.
10. Extract arm-level sample size only when explicitly stated or clearly attributable to that arm.
11. If only total study sample size is given and cannot be assigned to a specific arm, set arm_sample_size to null and mention that in notes.
12. Output JSON only. No markdown, no explanation.

Return ONLY JSON in exactly this format:
{{
  "paper_id": "{pubmed_id}",
  "outcome_found": true,
  "outcome_type": "overall_survival",
  "source_used": "{source_used}",
  "arms": [
    {{
      "arm_name": "string",
      "arm_sample_size_raw": "string or null",
      "arm_sample_size": "string or null",
      "median_os_raw": "string",
      "median_os_value": "string or null",
      "median_os_unit": "string or null",
      "ci_95_raw": "string",
      "ci_lower": "string or null",
      "ci_upper": "string or null",
      "ci_unit": "string or null",
      "evidence": "string"
    }}
  ],
  "notes": "string"
}}

Text:
\"\"\"
{source_text}
\"\"\"
""".strip(),
    "summarize_plot_rows": """
You are summarizing extracted trial-level survival outputs for downstream review.

Trial:
{trial_text}

Linked paper:
{paper_text}

Extracted plot rows:
{plot_rows_text}

Task:
Produce a concise structured summary of what was extracted and whether the rows
look ready for plotting.

Return ONLY JSON in this format:
{{
  "plot_ready": true,
  "n_rows": 2,
  "summary": "brief summary",
  "warnings": ["warning 1", "warning 2"]
}}
""".strip(),
}


def build_paper_link_semantic_terms_prompt(fields: Dict[str, Any]) -> str:
    return LINKING_TASK_PROMPTS["extract_trial_semantic_terms"].format(
        nct_id=fields.get("nct_id"),
        brief_title=fields.get("brief_title"),
        official_title=fields.get("official_title"),
        brief_summary=fields.get("brief_summary"),
        disease=fields.get("disease"),
        drugs=fields.get("drugs"),
        phase=fields.get("phase"),
    )


def build_trial_paper_judging_prompt(
    trial_fields: Dict[str, Any],
    papers_text: str,
) -> str:
    return LINKING_TASK_PROMPTS["judge_trial_papers"].format(
        nct_id=trial_fields.get("nct_id"),
        brief_title=trial_fields.get("brief_title"),
        official_title=trial_fields.get("official_title"),
        brief_summary=trial_fields.get("brief_summary"),
        disease=trial_fields.get("disease"),
        drugs=trial_fields.get("drugs"),
        phase=trial_fields.get("phase"),
        papers_text=papers_text,
    )


def build_survival_eligibility_prompt(
    pubmed_record: Dict[str, Any],
    source_text: str,
    pmcid: Optional[str] = None,
) -> str:
    return SURVIVAL_TASK_PROMPTS["judge_survival_plot_eligibility"].format(
        pubmed_id=pubmed_record.get("pubmed_id", ""),
        title=pubmed_record.get("title", ""),
        journal=pubmed_record.get("journal", ""),
        year=pubmed_record.get("year", ""),
        pmcid=pmcid or "",
        source_text=source_text,
    )


def build_survival_extraction_prompt(
    pubmed_record: Dict[str, Any],
    source_text: str,
    source_used: str,
    pmcid: Optional[str] = None,
) -> str:
    return SURVIVAL_TASK_PROMPTS["extract_survival_from_text"].format(
        pubmed_id=pubmed_record.get("pubmed_id", ""),
        title=pubmed_record.get("title", ""),
        journal=pubmed_record.get("journal", ""),
        year=pubmed_record.get("year", ""),
        pmcid=pmcid or "",
        source_used=source_used,
        source_text=source_text,
    )


def build_plot_row_summary_prompt(
    trial_text: str,
    paper_text: str,
    plot_rows_text: str,
) -> str:
    return SURVIVAL_TASK_PROMPTS["summarize_plot_rows"].format(
        trial_text=trial_text,
        paper_text=paper_text,
        plot_rows_text=plot_rows_text,
    )


# ---------------------------------------------------------------------------
# InspectorAgent task prompts
# ---------------------------------------------------------------------------

INSPECTOR_PROMPTS: Dict[str, str] = {
    "judge_trial_papers": LINKING_TASK_PROMPTS["judge_trial_papers"],
    "judge_survival_plot_eligibility": SURVIVAL_TASK_PROMPTS["judge_survival_plot_eligibility"],
}


def build_inspector_trial_paper_judging_prompt(
    trial_fields: Dict[str, Any],
    papers_text: str,
) -> str:
    return INSPECTOR_PROMPTS["judge_trial_papers"].format(
        nct_id=trial_fields.get("nct_id"),
        brief_title=trial_fields.get("brief_title"),
        official_title=trial_fields.get("official_title"),
        brief_summary=trial_fields.get("brief_summary"),
        disease=trial_fields.get("disease"),
        drugs=trial_fields.get("drugs"),
        phase=trial_fields.get("phase"),
        papers_text=papers_text,
    )


def build_inspector_survival_eligibility_prompt(
    pubmed_record: Dict[str, Any],
    source_text: str,
    pmcid: Optional[str] = None,
) -> str:
    return INSPECTOR_PROMPTS["judge_survival_plot_eligibility"].format(
        pubmed_id=pubmed_record.get("pubmed_id", ""),
        title=pubmed_record.get("title", ""),
        journal=pubmed_record.get("journal", ""),
        year=pubmed_record.get("year", ""),
        pmcid=pmcid or "",
        source_text=source_text,
    )


# ---------------------------------------------------------------------------
# Optional grouped exports for later coordinator / agent wiring
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "trial_retrieval": TRIAL_RETRIEVAL_SYSTEM_PROMPT,
    "paper_link": PAPER_LINK_SYSTEM_PROMPT,
    "inspector": INSPECTOR_SYSTEM_PROMPT,
}


AGENT_TASK_PROMPTS: Dict[str, Dict[str, str]] = {
    "trial_retrieval": TRIAL_TASK_PROMPTS,
    "paper_link": {
        **LINKING_TASK_PROMPTS,
        **SURVIVAL_TASK_PROMPTS,
    },
    "inspector": INSPECTOR_PROMPTS,
}


TASK_PROMPT_GROUPS: Dict[str, Dict[str, str]] = {
    "trial": TRIAL_TASK_PROMPTS,
    "linking": LINKING_TASK_PROMPTS,
    "survival": SURVIVAL_TASK_PROMPTS,
}
