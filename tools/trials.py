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
    completion_date = _safe_get(
        protocol, ["statusModule", "completionDateStruct", "date"]
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
        "completion_date": completion_date,
        "sponsor": sponsor,
        "brief_summary": brief_summary,
    }


def _build_search_terms(query):
    disease = query["disease"].strip()
    drugs = query.get("drugs", [])

    if not drugs:
        return [disease]

    all_terms = " ".join([d.strip() for d in drugs if d.strip()])
    return [f"{disease} {all_terms}"]


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

def _contains_all_drugs(drug_field, target_drugs):
    """
    Check whether drug_field contains ALL drugs in target_drugs.
    drug_field may be a list or string.
    """

    if not target_drugs:
        return True

    if isinstance(drug_field, list):
        text = " ".join([str(x).lower() for x in drug_field])
    else:
        text = str(drug_field).lower()

    return all(drug.lower() in text for drug in target_drugs)

# search trials from .gov based on structured query
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

    # # --- Local filter: drugs (AND logic: must contain ALL drugs) ---
    # target_drugs = query.get("drugs", [])

    # if target_drugs:
    #     df = df[
    #         df["interventions"].apply(lambda x: _contains_all_drugs(x, target_drugs))
    #     ].copy()

    # print("columns before sort:", df.columns.tolist())
    # print(df.head())

    # Optional: sort by first posted date
    df = df.sort_values(by="start_date_dt").reset_index(drop=True)

    return df

def select_trials_interactively(df_trials: pd.DataFrame, page_size: int = 10) -> list[int]:
    """
    Let user browse candidate trials page by page and select MULTIPLE trials.

    Returns
    -------
    selected_trial_indices : list[int]
        List of global row indices in df_trials.
    """
    cols = [c for c in ["nct_id", "brief_title", "start_date", "phases"] if c in df_trials.columns]

    total = len(df_trials)
    current_page = 0
    selected_indices = set()

    while True:
        start = current_page * page_size
        end = min(start + page_size, total)

        display_df = df_trials.iloc[start:end][cols].copy()
        display_df = display_df.reset_index(drop=True)

        print(f"\n[Step 3] Candidate trials (showing {start}–{end - 1} of {total - 1})")
        print(display_df)

        print("\nCurrently selected indices:", sorted(selected_indices) if selected_indices else "None")

        print("\nOptions:")
        print("  [0-9]      Add trial index from current page")
        print("  r NUM      Remove global index NUM (example: r 25)")
        print("  g NUM      Add global index NUM (example: g 25)")
        print("  n          Next page")
        print("  p          Previous page")
        print("  d          Done (finish selection)")
        print("  q          Quit (return empty list)")

        choice = input("\nYour choice: ").strip().lower()

        # ---------- navigation ----------
        if choice == "n":
            if end < total:
                current_page += 1
            else:
                print("Already at the last page.")
            continue

        if choice == "p":
            if current_page > 0:
                current_page -= 1
            else:
                print("Already at the first page.")
            continue

        # ---------- finish ----------
        if choice == "d":
            return sorted(selected_indices)

        if choice == "q":
            return []

        # ---------- add via global ----------
        if choice.startswith("g "):
            parts = choice.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < total:
                    selected_indices.add(idx)
                    print(f"Added trial {idx}")
                else:
                    print("Global index out of range.")
            else:
                print("Invalid format. Use: g 25")
            continue

        # ---------- remove ----------
        if choice.startswith("r "):
            parts = choice.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx in selected_indices:
                    selected_indices.remove(idx)
                    print(f"Removed trial {idx}")
                else:
                    print("Index not in selection.")
            else:
                print("Invalid format. Use: r 25")
            continue

        # ---------- add from page ----------
        try:
            idx_in_page = int(choice)
            if 0 <= idx_in_page < len(display_df):
                global_idx = start + idx_in_page
                selected_indices.add(global_idx)
                print(f"Added trial {global_idx}")
            else:
                print("Index out of range on current page.")
        except ValueError:
            print("Invalid input.")

