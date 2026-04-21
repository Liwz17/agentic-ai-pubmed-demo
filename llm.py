"""
LLM utility functions (legacy).

Non-LLM utilities have been moved to tools/:
  - PDF reading          -> tools/pdf.py
  - PMC text fetching    -> tools/pmc.py
  - Plot row building    -> tools/stats.py

Re-exported here so existing imports (linker.py, PaperLink_legacy.py, etc.)
continue to work without changes.
"""

import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from config import MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# re-exports for backward compatibility
from tools.pdf import read_pdf_text  # noqa: F401 — re-exported for legacy imports
from tools.pmc import get_best_text_for_extraction
from tools.stats import build_plot_rows_from_extraction

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def _clean_llm_json(content: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


# ---------------------------------------------------------------------------
# LLM functions — kept for legacy compatibility (linker.py, PaperLink_legacy)
# New code should use agent methods instead.
# ---------------------------------------------------------------------------

def llm_parse_query(user_input: str) -> dict:
    """Convert natural-language user request to structured ClinicalTrials.gov query."""
    prompt = f"""
You are a clinical research assistant.

Convert the following user request into a structured JSON query for ClinicalTrials.gov search.

User request:
{user_input}

Return ONLY JSON with the following fields:
- disease: string
- drugs: list of strings
- phase: one of PHASE1, PHASE2, PHASE3, PHASE4 (uppercase)
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD

Rules:
- Normalize drug names (e.g., "pembro" -> "pembrolizumab")
- Normalize disease (e.g., "NSCLC" -> "lung cancer")
- Phase II -> PHASE2
- If date is given as year, convert to full range
- Be precise and do not hallucinate

Example output:
{{
  "disease": "lung cancer",
  "drugs": ["pembrolizumab", "ipilimumab"],
  "phase": "PHASE2",
  "start_date": "2016-01-01",
  "end_date": "2020-12-31"
}}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = _clean_llm_json(response.choices[0].message.content.strip())
    try:
        return json.loads(content)
    except Exception:
        raise ValueError("LLM did not return valid JSON:\n" + content)


def llm_extract_trial_semantic_terms(fields: dict) -> dict:
    """Extract discriminative PubMed retrieval terms for a trial."""
    prompt = f"""
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
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical information retrieval assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = _clean_llm_json(response.choices[0].message.content.strip())
    try:
        return json.loads(content)
    except Exception:
        raise ValueError("LLM did not return valid JSON for trial semantic terms:\n" + content)


def llm_judge_trial_papers(trial_fields: dict, candidate_papers: list) -> dict:
    """Judge which candidate paper best matches the given trial."""
    if not candidate_papers:
        return {
            "match_found": False,
            "selected_pubmed_id": None,
            "selected_title": None,
            "label": "no_candidate",
            "confidence": "low",
            "reason": "No candidate papers were retrieved.",
        }

    paper_blocks = [
        f"Candidate {i}\nPMID: {p.get('pubmed_id')}\nTitle: {p.get('title')}\n"
        f"Journal: {p.get('journal')}\nAbstract: {p.get('abstract')}"
        for i, p in enumerate(candidate_papers, 1)
    ]

    prompt = f"""
You are linking one ClinicalTrials.gov trial to PubMed papers.

Trial:
- NCT ID: {trial_fields.get("nct_id")}
- Brief title: {trial_fields.get("brief_title")}
- Official title: {trial_fields.get("official_title")}
- Brief summary: {trial_fields.get("brief_summary")}
- Disease: {trial_fields.get("disease")}
- Drugs: {trial_fields.get("drugs")}
- Phase: {trial_fields.get("phase")}

Candidate PubMed papers:
{chr(10).join(paper_blocks)}

Task:
Decide whether one of the candidate papers is likely reporting results for THIS specific trial.

Labels: primary_results | secondary_or_followup | protocol_or_review | not_a_match

Rules:
- Prefer papers that match the disease, drug(s), phase, and trial setting.
- If an NCT ID is explicitly consistent, that is strong evidence.
- Do not force a match if the candidates are only generally related.
- If no candidate is convincing, return match_found=false and selected index null.

Return ONLY JSON:
{{
  "match_found": true,
  "selected_index": 1,
  "selected_pubmed_id": "12345678",
  "selected_title": "paper title",
  "label": "primary_results",
  "confidence": "high",
  "reason": "brief explanation"
}}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical trial-to-paper linkage judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = _clean_llm_json(response.choices[0].message.content.strip())
    try:
        return json.loads(content)
    except Exception:
        raise ValueError("LLM did not return valid JSON for trial-paper judge:\n" + content)


def llm_judge_survival_plot_eligibility(
    pubmed_record: Dict[str, Any],
    full_text: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """Judge whether a paper is eligible for arm-level survival extraction."""
    if full_text and str(full_text).strip():
        source_text, source_used, pmcid = full_text, "user_pdf", None
    else:
        source_text, source_used, pmcid = get_best_text_for_extraction(pubmed_record, email=email)

    if not source_text or not str(source_text).strip():
        return {
            "eligible": False,
            "reason": "No usable text available.",
            "paper_type": "unknown",
            "source_used": source_used,
            "pmcid": pmcid,
        }

    prompt = f"""
You are an expert biomedical trial-reading assistant.

Determine whether this paper is appropriate for ARM-LEVEL overall survival extraction
for a forest-plot style summary across trials.

ELIGIBLE only if it reports study-level or treatment-arm-level OS outcomes (median OS by arm).
NOT ELIGIBLE if mainly a case report, case series, review, commentary, or biomarker study.

Return JSON only:
{{
  "eligible": true,
  "paper_type": "trial_report | case_series | case_report | biomarker_study | review | unknown",
  "reason": "short explanation"
}}

Paper metadata:
- PubMed ID: {pubmed_record.get("pubmed_id", "")}
- Title: {pubmed_record.get("title", "")}
- Journal: {pubmed_record.get("journal", "")}
- Year: {pubmed_record.get("year", "")}
- PMCID: {pmcid or ""}

Text:
\"\"\"
{source_text[:12000]}
\"\"\"
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise biomedical paper triage assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = _clean_llm_json(response.choices[0].message.content.strip())
    try:
        result = json.loads(content)
    except Exception:
        result = {"eligible": True, "paper_type": "unknown", "reason": "JSON parse failed."}

    result["source_used"] = source_used
    result["pmcid"] = pmcid
    return result


def llm_extract_survival_from_text(
    pubmed_record: Dict[str, Any],
    full_text: Optional[str] = None,
    trial_id: Optional[str] = None,
    trial_label: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract median OS by treatment arm from a paper."""
    if full_text and str(full_text).strip():
        source_text, source_used, pmcid = full_text, "full_text", None
    else:
        source_text, source_used, pmcid = get_best_text_for_extraction(pubmed_record, email=email)

    print(f"[Step] Source: {source_used} | PMID: {pubmed_record.get('pubmed_id')}")
    if pmcid:
        print(f"  PMCID: {pmcid}")

    if not source_text or not str(source_text).strip():
        return {
            "paper_id": str(pubmed_record.get("pubmed_id", "")),
            "pmcid": pmcid,
            "outcome_found": False,
            "outcome_type": "overall_survival",
            "source_used": source_used,
            "arms": [],
            "plot_rows": [],
            "notes": f"No usable text. source_used={source_used}",
        }

    prompt = f"""
You are an expert biomedical information extraction assistant.

Paper metadata:
- PubMed ID: {pubmed_record.get("pubmed_id", "")}
- PMCID: {pmcid or ""}
- Title: {pubmed_record.get("title", "")}
- Journal: {pubmed_record.get("journal", "")}
- Year: {pubmed_record.get("year", "")}

Extract the median OVERALL SURVIVAL (OS), 95% CI, and sample size per arm.

Rules:
1. OS only — not PFS, DFS, DOR, ORR.
2. Single-arm study -> one arm only.
3. OS not reported -> outcome_found=false, empty arms.
4. "NR"/"not reached" -> preserve in raw field, null for numeric.
5. Do not infer or calculate. Preserve values exactly.
6. Include exact supporting quote in "evidence".
7. Output JSON only.

Return ONLY JSON:
{{
  "paper_id": "{pubmed_record.get("pubmed_id", "")}",
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
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise biomedical survival-outcome extraction assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = _clean_llm_json(response.choices[0].message.content.strip())

    try:
        result = json.loads(content)
    except Exception:
        raise ValueError("LLM did not return valid JSON for survival extraction:\n" + content)

    pubmed_id = str(pubmed_record.get("pubmed_id", ""))
    result.setdefault("paper_id", pubmed_id)
    result.setdefault("outcome_found", False)
    result.setdefault("outcome_type", "overall_survival")
    result.setdefault("source_used", source_used)
    result["pmcid"] = pmcid
    if result.get("arms") is None:
        result["arms"] = []

    normalized_arms = []
    for arm in result["arms"]:
        arm_n = arm.get("arm_sample_size")
        if arm_n in ["", "null", "None", None]:
            arm_n = None
        else:
            try:
                arm_n = int(float(arm_n))
            except Exception:
                arm_n = None
        normalized_arms.append({
            "arm_name": arm.get("arm_name", "unknown arm"),
            "arm_sample_size_raw": arm.get("arm_sample_size_raw"),
            "arm_sample_size": arm_n,
            "median_os_raw": arm.get("median_os_raw"),
            "median_os_value": arm.get("median_os_value"),
            "median_os_unit": arm.get("median_os_unit"),
            "ci_95_raw": arm.get("ci_95_raw"),
            "ci_lower": arm.get("ci_lower"),
            "ci_upper": arm.get("ci_upper"),
            "ci_unit": arm.get("ci_unit"),
            "evidence": arm.get("evidence", ""),
        })
    result["arms"] = normalized_arms
    result.setdefault("notes", "")
    result["plot_rows"] = build_plot_rows_from_extraction(result, trial_id=trial_id, trial_label=trial_label)

    return result
