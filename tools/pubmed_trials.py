import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pymed import PubMed
import requests
import xml.etree.ElementTree as ET

DEFAULT_MAX_PAPERS = 10
PUBMED_SLEEP_SECONDS = 0.5

PUBMED_CACHE = {}

# ---------------------------------------------------------------------------
# PubMedFilter — optional per-query publication filters
# ---------------------------------------------------------------------------

_PUB_TYPE_MAP: Dict[str, str] = {
    "clinical_trial": "Clinical Trial[Publication Type]",
    "rct": "Randomized Controlled Trial[Publication Type]",
    "meta_analysis": "Meta-Analysis[Publication Type]",
    "systematic_review": "Systematic Review[Publication Type]",
}

# PubMed has no "Final Report" publication type; approximate via title/abstract.
_FINAL_REPORT_CLAUSE = (
    '("final results"[Title/Abstract] OR "primary results"[Title/Abstract])'
)


@dataclass
class PubMedFilter:
    """
    Optional filters to narrow PubMed queries by date and publication type.

    publication_types — list of any of:
        "clinical_trial", "rct", "meta_analysis", "systematic_review",
        "final_report"   (approximated via title/abstract keywords)
    """
    pub_date_start: Optional[str] = None   # YYYY/MM/DD  (or YYYY)
    pub_date_end: Optional[str] = None     # YYYY/MM/DD  (or YYYY)
    publication_types: List[str] = field(default_factory=list)


def _make_pubmed_client():
    return PubMed(tool="AgenticAI", email="wli13@mdanderson.org")


def _clean_pubmed_id(raw_id):
    if raw_id is None:
        return None

    text = str(raw_id)

    return text.split("\n")[0].strip()

def _normalize_pubmed_results(papers) -> List[Dict[str, Any]]:
    results = []
    for paper in papers:
        raw_pubmed_id = getattr(paper, "pubmed_id", None)
        results.append({
            "pubmed_id": _clean_pubmed_id(raw_pubmed_id),
            "title": paper.title or "No title available",
            "abstract": paper.abstract or "No abstract available",
            "journal": paper.journal or "No journal available",
            "doi": getattr(paper, "doi", None),
            "keywords": getattr(paper, "keywords", None),
        })
    return results


def _run_pubmed_query_once(
    query: str,
    max_papers: int = DEFAULT_MAX_PAPERS,
    sleep_seconds: float = PUBMED_SLEEP_SECONDS,
) -> List[Dict[str, Any]]:
    cache_key = ("trial_query_once", query, max_papers)
    if cache_key in PUBMED_CACHE:
        return PUBMED_CACHE[cache_key]

    try:
        pubmed = _make_pubmed_client()
        time.sleep(sleep_seconds)
        papers = list(pubmed.query(query, max_results=max_papers))
        results = _normalize_pubmed_results(papers)
        PUBMED_CACHE[cache_key] = results
        return results
    except Exception as e:
        if "429" in str(e):
            print("PubMed rate limit hit. Sleeping for 5 seconds...")
            time.sleep(5)
        print(f"Error querying PubMed: {e}")
        return []


# def _sanitize_user_query(user_query: str) -> str:
#     return " ".join(user_query.strip().split())


# def _phase_to_pubmed_clause(phase: Optional[str]) -> str:
#     """
#     Convert structured phase label into a PubMed text clause.
#     """
#     if not phase:
#         return ""

#     phase_map = {
#         "PHASE1": '(("phase 1"[Title/Abstract]) OR ("phase i"[Title/Abstract]))',
#         "PHASE2": '(("phase 2"[Title/Abstract]) OR ("phase ii"[Title/Abstract]))',
#         "PHASE3": '(("phase 3"[Title/Abstract]) OR ("phase iii"[Title/Abstract]))',
#         "PHASE4": '(("phase 4"[Title/Abstract]) OR ("phase iv"[Title/Abstract]))',
#     }

#     return phase_map.get(phase.upper(), "")


# def _build_pubmed_trial_query(
#     user_query: str,
#     mode: str = "clinical_trial_filter",
#     nct_id: Optional[str] = None,
#     phase: Optional[str] = None,
# ) -> str:
#     """
#     Build a PubMed query string based on a user query and retrieval mode.
#     """
#     q = _sanitize_user_query(user_query)
#     phase_clause = _phase_to_pubmed_clause(phase)

#     if mode == "basic":
#         query = q

#     elif mode == "title_abstract":
#         query = f"({q}[Title/Abstract])"

#     elif mode == "title_abstract_trial":
#         query = f"({q}[Title/Abstract]) AND (trial[Title/Abstract])"

#     elif mode == "clinical_trial_filter":
#         query = f"({q}[Title/Abstract]) AND (Clinical Trial[Publication Type])"

#     elif mode == "nct_semantic":
#         if not nct_id:
#             raise ValueError("mode='nct_semantic' requires nct_id")
#         query = f"{nct_id} OR (({q}[Title/Abstract]) AND (Clinical Trial[Publication Type]))"

#     else:
#         raise ValueError(
#             "Unsupported mode. Choose from: "
#             "'basic', 'title_abstract', 'title_abstract_trial', "
#             "'clinical_trial_filter', 'nct_semantic'"
#         )

#     if phase_clause:
#         query = f"({query}) AND {phase_clause}"

#     return query


def query_pubmed_trial(
    disease: str,
    drugs: List[str],
    phase: Optional[str] = None,
    max_papers: int = DEFAULT_MAX_PAPERS,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Trial-oriented PubMed search using structured query builder.
    """

    final_query = build_structured_pubmed_query(
        disease=disease,
        drugs=drugs,
        phase=phase,
        require_clinical_trial=True,
    )

    if verbose:
        print("[PubMed mode] structured_trial_search")
        print(f"[PubMed query] {final_query}")

    results = _run_pubmed_query_once(
        query=final_query,
        max_papers=max_papers,
    )

    if verbose:
        print(f"[PubMed results] {len(results)} papers")

    return results


def _date_filter_clause(
    start: Optional[str], end: Optional[str]
) -> str:
    """Build a PubMed publication-date range clause."""
    if not start and not end:
        return ""
    s = start or "1900/01/01"
    e = end or "3000/12/31"
    return f'("{s}"[Date - Publication] : "{e}"[Date - Publication])'


def _pub_type_clause(types: List[str]) -> str:
    """
    Build an OR clause for publication type filters.
    "final_report" is approximated as a title/abstract keyword match.
    """
    clauses: List[str] = []
    for t in types:
        if t == "final_report":
            clauses.append(_FINAL_REPORT_CLAUSE)
        elif t in _PUB_TYPE_MAP:
            clauses.append(_PUB_TYPE_MAP[t])
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


def _term_clause(term: str, field: str = "Title/Abstract") -> str:
    """
    Convert a term into PubMed field query.
    """
    term = term.strip()
    return f'("{term}"[{field}])'


def _and_clause(clauses: List[str]) -> str:
    return " AND ".join([c for c in clauses if c])


def _or_clause(clauses: List[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


def _phase_to_clause(phase: Optional[str]) -> str:
    if not phase:
        return ""

    phase_map = {
        "PHASE1": ['"phase 1"', '"phase i"'],
        "PHASE2": ['"phase 2"', '"phase ii"'],
        "PHASE3": ['"phase 3"', '"phase iii"'],
        "PHASE4": ['"phase 4"', '"phase iv"'],
    }

    terms = phase_map.get(phase.upper(), [])
    return _or_clause([f"{t}[Title/Abstract]" for t in terms])


def build_structured_pubmed_query(
    disease: Optional[str] = None,
    drugs: Optional[List[str]] = None,
    phase: Optional[str] = None,
    require_clinical_trial: bool = True,
    pubmed_filter: Optional["PubMedFilter"] = None,
) -> str:
    """
    Build structured PubMed query:
    (disease) AND (drug1 OR drug2) AND (phase) AND (clinical trial)
    Optionally apply date-range and publication-type filters from pubmed_filter.
    """

    clauses = []

    # 1. disease
    if disease:
        clauses.append(_term_clause(disease))

    # 2. drugs (OR)
    if drugs:
        drug_clauses = [_term_clause(d) for d in drugs if d]
        clauses.append(_or_clause(drug_clauses))

    # 3. phase
    phase_clause = _phase_to_clause(phase)
    if phase_clause:
        clauses.append(phase_clause)

    # 4. clinical trial filter
    if require_clinical_trial:
        clauses.append("(Clinical Trial[Publication Type])")

    # 5. optional filters from PubMedFilter
    if pubmed_filter:
        date_clause = _date_filter_clause(
            pubmed_filter.pub_date_start, pubmed_filter.pub_date_end
        )
        if date_clause:
            clauses.append(date_clause)
        pt_clause = _pub_type_clause(pubmed_filter.publication_types)
        if pt_clause:
            clauses.append(pt_clause)

    # 6. combine
    return _and_clause(clauses)




def extract_trial_retrieval_fields(trial_row: dict) -> dict:
    nct_id = trial_row.get("nct_id")
    brief_title = trial_row.get("brief_title")
    official_title = trial_row.get("official_title")
    brief_summary = trial_row.get("brief_summary")

    conditions = trial_row.get("conditions", []) or []
    interventions = trial_row.get("interventions", []) or []
    phases = trial_row.get("phases", []) or []

    # pick one disease
    disease = conditions[0] if conditions else None

    # clean interventions
    bad_terms = {
        "placebo", "chemotherapy", "platinum doublet chemotherapy",
        "standard therapy", "best supportive care"
    }

    clean_drugs = []
    for x in interventions:
        if not x:
            continue
        x_lower = x.lower().strip()
        if x_lower not in bad_terms:
            clean_drugs.append(x)

    # pick phase
    phase = None
    if "PHASE2" in phases:
        phase = "PHASE2"
    elif phases:
        phase = phases[0]

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "brief_summary": brief_summary,
        "disease": disease,
        "drugs": clean_drugs,
        "phase": phase,
        "raw_conditions": conditions,
        "raw_interventions": interventions,
        "raw_phases": phases,
    }



def build_query_B_llm(
    fields: dict,
    semantic_terms: dict,
    pubmed_filter: Optional["PubMedFilter"] = None,
) -> str:
    clauses = []

    disease_terms = semantic_terms.get("disease_terms", []) or []
    drug_terms = semantic_terms.get("drug_terms", []) or []
    setting_terms = semantic_terms.get("setting_terms", []) or []

    # 1. disease: keep at most 1-2
    if disease_terms:
        clauses.append(_or_clause([_term_clause(x) for x in disease_terms[:2]]))
    elif fields.get("disease"):
        clauses.append(_term_clause(fields["disease"]))

    # 2. drugs: keep at most 2-3
    if drug_terms:
        clauses.append(_or_clause([_term_clause(x) for x in drug_terms[:3]]))
    elif fields.get("drugs"):
        clauses.append(_or_clause([_term_clause(x) for x in fields["drugs"][:3]]))

    # 3. setting: keep ONLY the first one
    if setting_terms:
        clauses.append(_term_clause(setting_terms[0]))

    # 4. phase: optional, but keep
    phase_clause = _phase_to_clause(fields.get("phase"))
    if phase_clause:
        clauses.append(phase_clause)

    # 5. publication type — only apply if user explicitly set a filter
    if pubmed_filter and pubmed_filter.publication_types:
        pt = _pub_type_clause(pubmed_filter.publication_types)
        if pt:
            clauses.append(pt)

    # 6. date range
    if pubmed_filter:
        date_clause = _date_filter_clause(
            pubmed_filter.pub_date_start, pubmed_filter.pub_date_end
        )
        if date_clause:
            clauses.append(date_clause)

    return _and_clause(clauses)

def build_query_A(
    fields: dict,
    pubmed_filter: Optional["PubMedFilter"] = None,
) -> Optional[str]:
    nct_id = fields.get("nct_id")
    if not nct_id:
        return None
    clauses = [f"{nct_id}[All Fields]"]
    if pubmed_filter:
        date_clause = _date_filter_clause(
            pubmed_filter.pub_date_start, pubmed_filter.pub_date_end
        )
        if date_clause:
            clauses.append(date_clause)
    return _and_clause(clauses)


def build_query_C(
    fields: dict,
    pubmed_filter: Optional["PubMedFilter"] = None,
) -> str:
    clauses = []

    disease = fields.get("disease")
    drugs = fields.get("drugs", [])
    phase = fields.get("phase")

    if disease:
        clauses.append(_term_clause(disease))

    if drugs:
        drug_clauses = [_term_clause(d) for d in drugs if d]
        clauses.append(_or_clause(drug_clauses))

    phase_clause = _phase_to_clause(phase)
    if phase_clause:
        clauses.append(phase_clause)

    # publication type — only apply if user explicitly set a filter
    if pubmed_filter and pubmed_filter.publication_types:
        pt = _pub_type_clause(pubmed_filter.publication_types)
        if pt:
            clauses.append(pt)

    # date range
    if pubmed_filter:
        date_clause = _date_filter_clause(
            pubmed_filter.pub_date_start, pubmed_filter.pub_date_end
        )
        if date_clause:
            clauses.append(date_clause)

    return _and_clause(clauses)


def fetch_pubmed_abstract(pubmed_id: str) -> Dict[str, Any]:
    """
    Fetch title + abstract from PubMed using E-utilities.
    Returns:
        {
            "pubmed_id": ...,
            "title": ...,
            "abstract": ...,
            "journal": ...,
            "year": ...
        }
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pubmed_id,
        "retmode": "xml",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    article = root.find(".//PubmedArticle")
    if article is None:
        raise ValueError(f"No PubMed article found for PMID={pubmed_id}")

    title_node = article.find(".//ArticleTitle")
    title = "".join(title_node.itertext()).strip() if title_node is not None else ""

    abstract_nodes = article.findall(".//Abstract/AbstractText")
    abstract_parts = []
    for node in abstract_nodes:
        label = node.attrib.get("Label", "").strip()
        section_text = "".join(node.itertext()).strip()
        if label:
            abstract_parts.append(f"{label}: {section_text}")
        else:
            abstract_parts.append(section_text)

    abstract = "\n".join([x for x in abstract_parts if x])

    journal_node = article.find(".//Journal/Title")
    journal = "".join(journal_node.itertext()).strip() if journal_node is not None else ""

    year = ""
    year_node = article.find(".//PubDate/Year")
    if year_node is not None and year_node.text:
        year = year_node.text.strip()

    return {
        "pubmed_id": pubmed_id,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
    }