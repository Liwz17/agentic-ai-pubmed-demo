import requests
import pandas as pd
from typing import Any, Dict, List, Optional

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 100)

CTG_API_BASE = "https://clinicaltrials.gov/api/v2/studies"


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    """
    Safely navigate nested dictionaries.
    """
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _normalize_trial_record(study: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw ClinicalTrials.gov study record into a flatter structure.
    """
    protocol = study.get("protocolSection", {})

    nct_id = _safe_get(protocol, ["identificationModule", "nctId"])
    brief_title = _safe_get(protocol, ["identificationModule", "briefTitle"])
    official_title = _safe_get(protocol, ["identificationModule", "officialTitle"])

    conditions = _safe_get(protocol, ["conditionsModule", "conditions"], default=[]) or []

    intervention_list = _safe_get(
        protocol, ["armsInterventionsModule", "interventions"], default=[]
    ) or []
    interventions = []
    for item in intervention_list:
        name = item.get("name")
        if name:
            interventions.append(name)

    phases = _safe_get(protocol, ["designModule", "phases"], default=[]) or []
    overall_status = _safe_get(protocol, ["statusModule", "overallStatus"])
    start_date = _safe_get(
        protocol, ["statusModule", "startDateStruct", "date"]
    )
    sponsor = _safe_get(protocol, ["contactsLocationsModule", "centralContacts"], default=[])
    brief_summary = _safe_get(protocol, ["descriptionModule", "briefSummary"])

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "conditions": conditions,
        "interventions": interventions,
        "phases": phases,
        "overall_status": overall_status,
        "start_date": start_date,
        "sponsor": sponsor,
        "brief_summary": brief_summary,
    }


def _build_search_terms(query: Dict[str, Any]) -> List[str]:
    """
    Build one or more query.term strings.
    We do one term per drug so that 'OR' behavior is explicit and easy to control.
    """
    disease = query["disease"].strip()
    drugs = query.get("drugs", [])

    if not drugs:
        return [disease]

    terms = []
    for drug in drugs:
        drug = drug.strip()
        if drug:
            terms.append(f"{disease} {drug}")

    return terms


def _fetch_studies_one_term(
    term: str,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch all pages for one search term from the ClinicalTrials.gov API v2.
    """
    studies: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while True:
        params = {
            "query.term": term,
            "pageSize": page_size,
            "format": "json",
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(CTG_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("studies", [])
        studies.extend(batch)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return studies


def search_clinical_trials(query: Dict[str, Any]) -> pd.DataFrame:
    """
    Part (a): identify trials that satisfy a user-specified query.

    Expected query format:
    {
        "disease": "lung cancer",
        "drugs": ["pembrolizumab", "ipilimumab"],
        "phase": "PHASE2",
        "start_date": "2016-01-01",
        "end_date": "2020-12-31"
    }
    """
    required_fields = ["disease", "phase", "start_date", "end_date"]
    missing = [k for k in required_fields if k not in query]
    if missing:
        raise ValueError(f"Missing required query fields: {missing}")

    search_terms = _build_search_terms(query)

    seen_nct_ids = set()
    all_raw_studies = []

    for term in search_terms:
        raw = _fetch_studies_one_term(
            term=term,
            page_size=100,
        )
        
        for study in raw:
            nct_id = _safe_get(
                study,
                ["protocolSection", "identificationModule", "nctId"]
            )
            
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                all_raw_studies.append(study)

    # Normalize
    rows = [_normalize_trial_record(study) for study in all_raw_studies]
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Drop rows without NCT ID and deduplicate by NCT ID
    df = df[df["nct_id"].notna()].copy()
    df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)

    # Local filter: phase
    target_phase = query["phase"]
    df = df[
        df["phases"].apply(
            lambda x: target_phase in x if isinstance(x, list) else False
        )
    ].copy()

    df["start_date_dt"] = pd.to_datetime(df["start_date"], errors="coerce")
    start_dt = pd.to_datetime(query["start_date"])
    end_dt = pd.to_datetime(query["end_date"])

    df = df[
        (df["start_date_dt"] >= start_dt) &
        (df["start_date_dt"] <= end_dt)
    ].copy()

    # Optional: sort by first posted date
    df = df.sort_values(by="start_date_dt").reset_index(drop=True)

    return df