"""
NCBI PMC utilities — fetching full text via PubMed Central APIs.
Nothing here calls an LLM.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple

import requests

NCBI_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def get_pmcid_from_pmid(pmid: str, email: Optional[str] = None) -> Optional[str]:
    """Convert PMID -> PMCID via NCBI idconv. Returns 'PMC1234567' or None."""
    if not pmid:
        return None

    params = {"tool": "pubmed-survival-agent", "format": "json", "ids": str(pmid)}
    if email:
        params["email"] = email

    try:
        resp = requests.get(NCBI_IDCONV_URL, params=params, timeout=20)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if not records:
            return None
        pmcid = records[0].get("pmcid")
        return pmcid if pmcid else None
    except Exception:
        return None


def fetch_pmc_full_text_xml(pmcid: str, email: Optional[str] = None) -> Optional[str]:
    """Fetch PMC full text XML via EFetch. pmcid may be 'PMC1234567' or numeric."""
    if not pmcid:
        return None

    pmc_numeric = str(pmcid).replace("PMC", "").strip()
    params = {"db": "pmc", "id": pmc_numeric, "retmode": "xml", "tool": "pubmed-survival-agent"}
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
    """Parse PMC XML and return abstract + body as plain text."""
    if not xml_text or not xml_text.strip():
        return ""
    try:
        root = ET.fromstring(xml_text)
        abstract_text = _safe_join_text(root.findall(".//abstract"))
        body_text = _safe_join_text(root.findall(".//body"))
        return "\n\n".join(x for x in [abstract_text, body_text] if x.strip()).strip()
    except Exception:
        return ""


def get_best_text_for_extraction(
    pubmed_record: Dict[str, Any],
    email: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Return (source_text, source_used, pmcid).
    Priority: PMC full text > abstract > empty string.
    source_used is one of: 'pmc_full_text', 'abstract', 'none'.
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
