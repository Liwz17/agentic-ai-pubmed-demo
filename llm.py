from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME, RERANK_TOP_K
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import requests
import xml.etree.ElementTree as ET
import os
import fitz  

def read_pdf_text(pdf_path: str) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    text_chunks = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            txt = page.get_text("text")
            if txt:
                text_chunks.append(txt)
    finally:
        doc.close()

    return "\n\n".join(text_chunks).strip()


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

import re

def _clean_llm_json(content: str) -> str:
    # remove ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

def _build_paper_context(papers):
    context = ""
    for i, p in enumerate(papers, 1):
        context += f"""
Paper {i}
PMID: {p.get('pubmed_id')}
Title: {p.get('title')}
Journal: {p.get('journal')}
Abstract: {p.get('abstract')}
"""
    return context




def summarize_papers(query, papers, stats):
    context = _build_paper_context(papers)

    prompt = f"""
User query: {query}

Summary statistics on the full candidate set:
- Number of candidate papers: {stats['n_papers']}
- Missing abstracts: {stats['missing_abstract_count']}
- Average abstract length: {stats['avg_abstract_length']}
- Top journals: {stats['top_journals']}
- Top keywords: {stats['top_keywords']}

Below are the top reranked PubMed papers:

{context}

Please do the following:
1. Summarize the main research themes.
2. Mention any obvious patterns from the candidate-set summary statistics.
3. Briefly note whether the selected papers seem focused or heterogeneous.
4. Keep the answer concise but informative.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical research assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# code to convert user input to structured trial search request, clinical trials agent.
def llm_parse_query(user_input: str) -> dict:
    """
    Use LLM to convert natural language into structured query
    """

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
- Normalize drug names (e.g., "pembro" → "pembrolizumab")
- Normalize disease (e.g., "NSCLC" → "lung cancer")
- Phase II → PHASE2
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
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        return json.loads(content)
    except:
        raise ValueError("LLM did not return valid JSON:\n" + content)
    

# Code for LLM to extract distinguished semantic terms


def _clean_llm_json(content: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()



def _to_float_or_none(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.lower() in {"nr", "not reached", "na", "n/a", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    u = str(unit).strip().lower()
    if u in {"month", "months", "mo", "mos"}:
        return "months"
    if u in {"year", "years", "yr", "yrs"}:
        return "years"
    return u if u else None









# generate semantic information for retriving papers(case B)
def llm_extract_trial_semantic_terms(fields: dict) -> dict:
    """
    Use LLM to extract trial-specific semantic retrieval terms.
    Return JSON like:
    {
      "disease_terms": [...],
      "drug_terms": [...],
      "setting_terms": [...],
      "other_terms": [...]
    }
    """
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical information retrieval assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        return json.loads(content)
    except Exception:
        raise ValueError("LLM did not return valid JSON for trial semantic terms:\n" + content)
    


# def rerank_papers(query, papers, top_k=RERANK_TOP_K):
#     """
#     Use LLM to rerank papers by relevance to the user query.
#     Return the selected top_k papers.
#     """
#     if len(papers) <= top_k:
#         return papers

#     context = _build_paper_context(papers)

#     prompt = f"""
# User query: {query}

# Below are {len(papers)} candidate PubMed papers:

# {context}

# Select the {top_k} most relevant papers for the user query.

# Return ONLY a comma-separated list of paper numbers, for example:
# 1,3,5,7,8,10

# Do not explain anything.
# """

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a biomedical literature relevance judge."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     text = response.choices[0].message.content.strip()

#     try:
#         selected_indices = []
#         for x in text.split(","):
#             x = x.strip()
#             if x.isdigit():
#                 idx = int(x)
#                 if 1 <= idx <= len(papers):
#                     selected_indices.append(idx - 1)

#         seen = set()
#         selected_indices = [i for i in selected_indices if not (i in seen or seen.add(i))]

#         selected_papers = [papers[i] for i in selected_indices[:top_k]]
        
#         if len(selected_papers) < top_k:
#             for p in papers:
#                 if p not in selected_papers:
#                     selected_papers.append(p)
#                 if len(selected_papers) == top_k:
#                     break

#         return selected_papers

#     except Exception:
#         return papers[:top_k]

# code for llm to judge the papers retrived by a,b,c mode.
def llm_judge_trial_papers(trial_fields: dict, candidate_papers: list) -> dict:
    """
    Judge which candidate paper is most likely linked to the given trial.
    Return structured JSON.
    """
    if not candidate_papers:
        return {
            "match_found": False,
            "selected_pubmed_id": None,
            "selected_title": None,
            "label": "no_candidate",
            "confidence": "low",
            "reason": "No candidate papers were retrieved."
        }

    paper_blocks = []
    for i, p in enumerate(candidate_papers, 1):
        paper_blocks.append(f"""
Candidate {i}
PMID: {p.get("pubmed_id")}
Title: {p.get("title")}
Journal: {p.get("journal")}
Abstract: {p.get("abstract")}
""")

    papers_text = "\n".join(paper_blocks)

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
{papers_text}

Task:
Decide whether one of the candidate papers is likely reporting results for THIS specific trial.

Use these labels:
- primary_results
- secondary_or_followup
- protocol_or_review
- not_a_match

Rules:
- Prefer papers that match the disease, drug(s), phase, and trial setting.
- If an NCT ID is explicitly consistent, that is strong evidence.
- Do not force a match if the candidates are only generally related.
- If no candidate is convincing, return match_found=false and selected index null.

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
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a biomedical trial-to-paper linkage judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        result = json.loads(content)
        return result
    except Exception:
        raise ValueError("LLM did not return valid JSON for trial-paper judge:\n" + content)
    

NCBI_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def get_pmcid_from_pmid(pmid: str, email: Optional[str] = None) -> Optional[str]:
    """
    Convert PMID -> PMCID using NCBI idconv API.
    Returns PMCID like 'PMC1234567' if available, else None.
    """
    if not pmid:
        return None

    params = {
        "tool": "pubmed-survival-agent",
        "format": "json",
        "ids": str(pmid),
    }
    if email:
        params["email"] = email

    try:
        resp = requests.get(NCBI_IDCONV_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("records", [])
        if not records:
            return None

        pmcid = records[0].get("pmcid")
        return pmcid if pmcid else None

    except Exception:
        return None


def fetch_pmc_full_text_xml(pmcid: str, email: Optional[str] = None) -> Optional[str]:
    """
    Fetch PMC full text in XML via EFetch.
    pmcid can be 'PMC1234567' or numeric part; EFetch db=pmc works with numeric id.
    """
    if not pmcid:
        return None

    pmc_numeric = str(pmcid).replace("PMC", "").strip()

    params = {
        "db": "pmc",
        "id": pmc_numeric,
        "retmode": "xml",
        "tool": "pubmed-survival-agent",
    }
    if email:
        params["email"] = email

    try:
        resp = requests.get(NCBI_EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        xml_text = resp.text.strip()
        return xml_text if xml_text else None
    except Exception:
        return None


def _safe_join_text(elements) -> str:
    chunks = []
    for el in elements:
        txt = " ".join(el.itertext()).strip()
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)


def parse_pmc_xml_to_text(xml_text: str) -> str:
    """
    Parse PMC XML and return a reasonably clean full-text string.
    Prefer abstract + body.
    """
    if not xml_text or not xml_text.strip():
        return ""

    try:
        root = ET.fromstring(xml_text)

        abstract_nodes = root.findall(".//abstract")
        body_nodes = root.findall(".//body")

        abstract_text = _safe_join_text(abstract_nodes)
        body_text = _safe_join_text(body_nodes)

        combined = "\n\n".join(
            x for x in [abstract_text, body_text] if x and x.strip()
        ).strip()

        return combined

    except Exception:
        return ""


def get_best_text_for_extraction(
    pubmed_record: Dict[str, Any],
    email: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    PMC + fallback:
    1. Try PMID -> PMCID
    2. If PMCID exists, fetch PMC full text XML and parse it
    3. Else fallback to abstract

    Returns:
        source_text, source_used, pmcid
    where source_used is one of:
        'pmc_full_text', 'abstract', 'none'
    """
    pmid = str(pubmed_record.get("pubmed_id", "")).strip()
    abstract = str(pubmed_record.get("abstract", "") or "").strip()

    pmcid = get_pmcid_from_pmid(pmid, email=email)
    if pmcid:
        xml_text = fetch_pmc_full_text_xml(pmcid, email=email)
        full_text = parse_pmc_xml_to_text(xml_text) if xml_text else ""

        if full_text.strip():
            return full_text, "pmc_full_text", pmcid

    if abstract:
        return abstract, "abstract", pmcid

    return "", "none", pmcid

def llm_judge_survival_plot_eligibility(
    pubmed_record: Dict[str, Any],
    full_text: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask LLM whether this paper is suitable for arm-level survival extraction
    for forest-plot style summarization.
    """

    # 和 extraction 一样，先决定看什么文本
    if full_text and str(full_text).strip():
        source_text = full_text
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

    prompt = f"""
You are an expert biomedical trial-reading assistant.

Task:
Determine whether this paper is appropriate for ARM-LEVEL overall survival extraction
for a forest-plot style summary across trials.

The paper is ELIGIBLE only if it primarily reports study-level or treatment-arm-level
overall survival outcomes (for example median OS by arm, with or without CI).

The paper is NOT ELIGIBLE if it is mainly:
- a case report
- a case series
- patient-by-patient narrative reporting
- a review, commentary, or non-original trial report
- a biomarker/correlative paper without treatment-arm-level survival extraction target
- a paper that reports only individual patient outcomes rather than arm-level trial outcomes

Return JSON only in this format:
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
            {
                "role": "system",
                "content": "You are a precise biomedical paper triage assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        result = json.loads(content)
    except Exception:
        result = {
            "eligible": True,   # conservative fallback: don't block extraction if JSON parsing fails
            "paper_type": "unknown",
            "reason": "Eligibility JSON parse failed; defaulting to extraction."
        }

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
    """
    Use LLM to extract median overall survival (OS), 95% CI, and sample size by treatment arm.

    Retrieval strategy:
    1. If full_text is explicitly provided, use it.
    2. Else try PMC full text from PMID.
    3. If PMC full text is unavailable, fallback to abstract.
    """

    # Priority 1: externally provided full text
    if full_text and str(full_text).strip():
        source_text = full_text
        source_used = "full_text"
        pmcid = None
    else:
        source_text, source_used, pmcid = get_best_text_for_extraction(
            pubmed_record,
            email=email,
        )

    # ===== Agent logging: retrieval step =====
    print("[Step] Retrieving text for extraction...")

    print(f"  PMID: {pubmed_record.get('pubmed_id')}")
    print(f"  Source used: {source_used}")

    if pmcid:
        print(f"  PMCID: {pmcid}")

    if not source_text or not str(source_text).strip():
        empty_result = {
            "paper_id": str(pubmed_record.get("pubmed_id", "")),
            "pmcid": pmcid,
            "outcome_found": False,
            "outcome_type": "overall_survival",
            "source_used": source_used,
            "arms": [],
            "plot_rows": [],
            "notes": f"No usable text available. source_used={source_used}"
        }
        return empty_result

    prompt = f"""
You are an expert biomedical information extraction assistant.

Paper metadata:
- PubMed ID: {pubmed_record.get("pubmed_id", "")}
- PMCID: {pmcid or ""}
- Title: {pubmed_record.get("title", "")}
- Journal: {pubmed_record.get("journal", "")}
- Year: {pubmed_record.get("year", "")}

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
            {
                "role": "system",
                "content": "You are a precise biomedical survival-outcome extraction assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    content = _clean_llm_json(content)

    try:
        result = json.loads(content)

        if "paper_id" not in result:
            result["paper_id"] = str(pubmed_record.get("pubmed_id", ""))

        result["pmcid"] = pmcid

        if "outcome_found" not in result:
            result["outcome_found"] = False

        if "outcome_type" not in result:
            result["outcome_type"] = "overall_survival"

        if "source_used" not in result:
            result["source_used"] = source_used

        if "arms" not in result or result["arms"] is None:
            result["arms"] = []

        if "notes" not in result:
            result["notes"] = ""

        normalized_arms = []
        for arm in result["arms"]:
            arm_sample_size = arm.get("arm_sample_size")

            if arm_sample_size in ["", "null", "None"]:
                arm_sample_size = None

            try:
                if arm_sample_size is not None:
                    arm_sample_size = int(float(arm_sample_size))
            except Exception:
                arm_sample_size = None

            normalized_arms.append({
                "arm_name": arm.get("arm_name", "unknown arm"),
                "arm_sample_size_raw": arm.get("arm_sample_size_raw"),
                "arm_sample_size": arm_sample_size,
                "median_os_raw": arm.get("median_os_raw"),
                "median_os_value": arm.get("median_os_value"),
                "median_os_unit": arm.get("median_os_unit"),
                "ci_95_raw": arm.get("ci_95_raw"),
                "ci_lower": arm.get("ci_lower"),
                "ci_upper": arm.get("ci_upper"),
                "ci_unit": arm.get("ci_unit"),
                "evidence": arm.get("evidence", "")
            })

        result["arms"] = normalized_arms

        result["plot_rows"] = build_plot_rows_from_extraction(
            result,
            trial_id=trial_id,
            trial_label=trial_label,
        )

        return result

    except Exception:
        raise ValueError(
            "LLM did not return valid JSON for survival extraction:\n" + content
        )


def build_plot_rows_from_extraction(
    extraction_result: Dict[str, Any],
    trial_id: Optional[str] = None,
    trial_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert extraction-rich result into plot-ready rows (NO unit conversion).

    Each returned row corresponds to one treatment arm.
    Units are preserved exactly as extracted.
    """
    rows: List[Dict[str, Any]] = []

    if not extraction_result.get("outcome_found", False):
        return rows

    paper_id = extraction_result.get("paper_id")
    source_used = extraction_result.get("source_used")
    arms = extraction_result.get("arms", [])

    for arm in arms:
        arm_name = arm.get("arm_name", "unknown arm")

        median_os_value = _to_float_or_none(arm.get("median_os_value"))
        median_os_unit = _normalize_unit(arm.get("median_os_unit"))

        ci_lower = _to_float_or_none(arm.get("ci_lower"))
        ci_upper = _to_float_or_none(arm.get("ci_upper"))
        ci_unit = _normalize_unit(arm.get("ci_unit")) or median_os_unit

        # sample size
        arm_sample_size = arm.get("arm_sample_size")
        try:
            if arm_sample_size not in [None, "", "null", "None"]:
                arm_sample_size = int(float(arm_sample_size))
            else:
                arm_sample_size = None
        except Exception:
            arm_sample_size = None

        plot_eligible = median_os_value is not None

        rows.append({
            "trial_id": trial_id,
            "trial_label": trial_label,
            "paper_id": paper_id,
            "source_used": source_used,

            "arm_name": arm_name,
            "display_label": f"{trial_label} | {arm_name}" if trial_label else arm_name,

            "median_os_raw": arm.get("median_os_raw"),
            "ci_95_raw": arm.get("ci_95_raw"),
            "median_os_value": median_os_value,
            "median_os_unit": median_os_unit,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_unit": ci_unit,

            # sample size
            "sample_size": arm_sample_size,
            "arm_sample_size": arm_sample_size,
            "arm_sample_size_raw": arm.get("arm_sample_size_raw"),

            # metadata
            "plot_eligible": plot_eligible,
            "evidence": arm.get("evidence", "")
        })

    return rows

