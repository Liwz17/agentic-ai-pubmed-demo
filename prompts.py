"""
Central prompt layer for the clinical-trial-to-paper pipeline.

Design:
- one system prompt per agent
- multiple task-specific prompt templates grouped by agent/task

This file is intentionally prompt-only. It does not contain business logic,
API calls, or orchestration code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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
Determine whether this paper reports arm-level primary efficacy outcomes (such as OS, PFS,
or response rate) suitable for quantitative extraction across treatment arms.

The paper is ELIGIBLE only if ALL of the following are true:
1. It is an interventional clinical trial (phase 1, 2, 3, or 4; or explicitly described
   as a randomized controlled trial, open-label trial, single-arm trial, or similar).
2. It reports efficacy outcomes at the treatment-arm level — meaning aggregate summary
   statistics (e.g. median OS, median PFS, ORR%) for a defined treatment group,
   not just individual patient-level narratives.

The paper is NOT ELIGIBLE if any of the following apply:
- It is a case report or case series (no formal trial design)
- It reports outcomes patient-by-patient without arm-level aggregate statistics
- It is a review, meta-analysis, commentary, editorial, or protocol paper
- It is a biomarker, correlative, or translational study without arm-level efficacy data
- It is a retrospective chart review without a pre-specified trial design
- The only "outcomes" reported are toxicity or safety without any efficacy endpoint

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


def build_outcome_parse_prompt(raw_input: str) -> str:
    """
    Ask LLM to parse the user's free-text outcome request into structured OutcomeSpec-compatible dicts.
    If the user wants per-paper primary endpoint discovery, returns the sentinel __auto_primary__.
    """
    return f"""You are a clinical research outcomes parsing assistant.

The user wants to extract the following outcomes from clinical trial papers:
"{raw_input}"

SPECIAL CASE — Auto primary-endpoint discovery:
If the user is asking for the primary endpoint(s) of each paper to be automatically identified
(e.g. "primary endpoint", "primary outcome", "main endpoint", "每篇paper的primary endpoint",
"主要终点", "the paper's own primary endpoint", etc.), return ONLY this single-item array:
[{{"key": "__auto_primary__", "display": "Primary Endpoint (auto-discover)", "plot_type": "auto"}}]

Otherwise, parse each requested outcome into a structured list:
- key: a concise snake_case identifier (e.g. "overall_survival", "objective_response_rate")
- display: a clear human-readable label with abbreviation (e.g. "Overall Survival (OS)")
- plot_type:
    "forest"     — time-to-event outcome with a median (OS, PFS, DOR, EFS, etc.)
    "bar"        — rate, proportion, or percentage (ORR, DCR, AE rate, CR rate, etc.)
    "table_only" — anything else (scores, counts, biomarker levels, etc.)

Rules:
- Normalize common abbreviations (OS → overall_survival, PFS → progression_free_survival, etc.)
- Do not duplicate; if the user asks for "OS and overall survival" treat as one outcome
- Return ONLY a JSON array, no explanation

Example output:
[
  {{"key": "overall_survival", "display": "Overall Survival (OS)", "plot_type": "forest"}},
  {{"key": "objective_response_rate", "display": "Objective Response Rate (ORR)", "plot_type": "bar"}}
]

User input: "{raw_input}"
""".strip()


def build_outcome_extraction_prompt(
    pubmed_record: Dict[str, Any],
    source_text: str,
    source_used: str,
    specs: List[Any],   # List[OutcomeSpec] — imported lazily to avoid circular dep
    pmcid: Optional[str] = None,
) -> str:
    """
    Dynamically build an extraction prompt from a list of OutcomeSpec objects.
    Each spec carries key, display, and plot_type — the schema per outcome is
    determined by plot_type.
    """
    outcome_lines = []
    schema_blocks = []

    for spec in specs:
        key = spec.key
        label = spec.display
        if spec.plot_type == "forest":
            outcome_lines.append(
                f"- {label}: report median value, unit (months/years), "
                f"95% CI lower and upper, and exact supporting quote"
            )
            schema_blocks.append(f'''      "{key}": {{
        "found": true,
        "value": "numeric string or null",
        "unit": "months | years | null",
        "ci_lower": "numeric string or null",
        "ci_upper": "numeric string or null",
        "ci_unit": "months | years | null",
        "value_raw": "verbatim reported value",
        "ci_raw": "verbatim reported CI",
        "evidence": "exact supporting sentence"
      }}''')
        elif spec.plot_type == "bar":
            outcome_lines.append(
                f"- {label}: report rate as numeric %, CI lower and upper if available, "
                f"responder count if available, and exact supporting quote"
            )
            schema_blocks.append(f'''      "{key}": {{
        "found": true,
        "value": "numeric percentage string or null",
        "unit": "%",
        "ci_lower": "numeric percentage string or null",
        "ci_upper": "numeric percentage string or null",
        "ci_unit": "%",
        "responders": "count string or null",
        "value_raw": "verbatim reported value",
        "ci_raw": "verbatim CI if any",
        "evidence": "exact supporting sentence"
      }}''')
        else:
            outcome_lines.append(
                f"- {label}: report value, unit, and exact supporting quote"
            )
            schema_blocks.append(f'''      "{key}": {{
        "found": true,
        "value": "string or null",
        "unit": "string or null",
        "value_raw": "verbatim",
        "evidence": "exact supporting sentence"
      }}''')

    outcomes_description = "\n".join(outcome_lines)
    schema_content = ",\n".join(schema_blocks)
    pubmed_id = pubmed_record.get("pubmed_id", "")

    return f"""You are an expert biomedical information extraction assistant.

Paper metadata:
- PubMed ID: {pubmed_id}
- PMCID: {pmcid or ""}
- Title: {pubmed_record.get("title", "")}
- Journal: {pubmed_record.get("journal", "")}
- Year: {pubmed_record.get("year", "")}
- Source used: {source_used}

Task:
Extract the following outcome(s) per treatment arm from the paper text below:
{outcomes_description}

Rules:
1. Extract ONLY what is explicitly stated in the text. Do not infer or calculate.
2. If a value is "not reached" or "NR", set "value" to null and preserve the raw text.
3. Set "found": false for any outcome not reported in the paper.
4. For single-arm studies, return one arm only.
5. Include the exact supporting sentence or phrase in "evidence".
6. Preserve reported numeric values exactly. Do not convert units.
7. Output JSON only — no markdown, no explanation.

Return ONLY JSON in this format:
{{
  "paper_id": "{pubmed_id}",
  "arms": [
    {{
      "arm_name": "string",
      "arm_sample_size": number or null,
      "arm_sample_size_raw": "string or null",
{schema_content}
    }}
  ],
  "notes": "string"
}}

Text:
\"\"\"
{source_text}
\"\"\"
""".strip()


def build_primary_endpoint_identification_prompt(
    pubmed_record: Dict[str, Any],
    source_text: str,
    pmcid: Optional[str] = None,
) -> str:
    return f"""You are an expert clinical research reading assistant.

Paper metadata:
- PubMed ID: {pubmed_record.get("pubmed_id", "")}
- PMCID: {pmcid or ""}
- Title: {pubmed_record.get("title", "")}
- Journal: {pubmed_record.get("journal", "")}
- Year: {pubmed_record.get("year", "")}

Task:
Identify the PRIMARY ENDPOINT of this clinical trial paper using the two-step approach below.

Step 1 — Explicit detection (preferred):
Search for explicit statements such as:
"primary endpoint", "primary outcome", "primary efficacy endpoint",
"co-primary endpoints", "primary end point", "the study's primary measure", etc.
If found, set "source": "explicit" and quote the exact sentence in "evidence".

Step 2 — Inference (fallback, only if Step 1 finds nothing):
If no explicit primary endpoint language exists, infer from the paper's structure and emphasis.
Strong inference signals:
- The outcome reported first and most prominently in the Results section
- The outcome used for sample size / power calculation in the Methods
- The outcome on which the trial's main conclusion is based
- The outcome with the most statistical detail (p-value, HR, primary analysis label)
If inferred, set "source": "inferred" and explain the reasoning in "evidence".

Rules:
1. Always try Step 1 first. Only use Step 2 if Step 1 yields nothing.
2. If there are co-primary endpoints, return all of them.
3. If neither step yields a confident answer, set "found": false.
4. Classify plot_type:
   - "forest" for time-to-event outcomes with a median (OS, PFS, DFS, EFS, DOR, etc.)
   - "bar" for rates or proportions (ORR, DCR, CR rate, AE rate, etc.)
   - "table_only" for anything else
5. Return ONLY JSON, no explanation.

Return ONLY JSON in this format:
{{
  "found": true,
  "primary_endpoints": [
    {{
      "key": "overall_survival",
      "display": "Overall Survival (OS)",
      "plot_type": "forest",
      "source": "explicit",
      "evidence": "exact quote or inference reasoning"
    }}
  ],
  "notes": "optional note"
}}

Text:
\"\"\"
{source_text}
\"\"\"
""".strip()


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
