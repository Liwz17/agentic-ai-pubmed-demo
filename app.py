"""
Streamlit web interface for the Clinical Trial Literature Agent.

Run with:
    streamlit run app.py
"""

import io
import json
import base64
import tempfile
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import streamlit as st
import pandas as pd

from trial_retrieval_agent import TrialRetrievalAgent
from paper_link_agent import PaperLinkAgent
from inspector import InspectorAgent
from tools import PubMedFilter
from tools.outcomes import (
    build_outcome_summary_table,
    collect_specs_from_plot_rows,
    draw_all_outcome_plots,
    draw_outcome_forest_plot,
    draw_outcome_bar_chart,
    DEFAULT_OUTCOME_SPECS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Trial Literature Agent",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Chat tool definitions
# ---------------------------------------------------------------------------

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plot_outcome",
            "description": (
                "Plot extracted outcome data as a forest or bar chart. "
                "Use when the user asks to visualise, chart, or plot specific outcomes, trials, or arms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome_key": {
                        "type": "string",
                        "description": (
                            "Snake-case outcome key, e.g. 'overall_survival', "
                            "'progression_free_survival', 'objective_response_rate'. "
                            "Omit or set to null to plot all available outcomes."
                        ),
                    },
                    "trial_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "NCT IDs to include. Omit or set to null for all trials.",
                    },
                    "arm_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arm names to include. Omit or set to null for all arms.",
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["forest", "bar", "auto"],
                        "description": "'forest', 'bar', or 'auto' (let the spec decide). Default 'auto'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_table",
            "description": (
                "Show a summary table of extracted outcomes. "
                "Use when the user asks for a table, summary, or spreadsheet view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trial_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "NCT IDs to include. Omit or set to null for all trials.",
                    },
                    "outcome_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Outcome keys to include. Omit or set to null for all outcomes.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Chat tool execution helpers
# ---------------------------------------------------------------------------

def _execute_plot_outcome(args: dict, pr: dict):
    """
    Filter plot rows according to tool args and draw the requested figure.
    Returns a matplotlib Figure or None if nothing matches.
    """
    outcome_key = args.get("outcome_key") or None
    trial_ids = args.get("trial_ids") or None
    arm_names = args.get("arm_names") or None
    plot_type_override = args.get("plot_type") or "auto"

    all_rows = pr.get("all_plot_rows", [])
    specs = pr.get("specs", [])

    # Filter to eligible rows first
    rows = [r for r in all_rows if r.get("plot_eligible")]

    if outcome_key:
        rows = [r for r in rows if r.get("outcome_key") == outcome_key]
    if trial_ids:
        tids_lower = [t.lower() for t in trial_ids]
        rows = [r for r in rows if (r.get("trial_id") or "").lower() in tids_lower]
    if arm_names:
        arms_lower = [a.lower() for a in arm_names]
        rows = [r for r in rows if (r.get("arm_name") or "").lower() in arms_lower]

    if not rows:
        return None

    # Determine which outcome keys are present
    keys_present = list(dict.fromkeys(r["outcome_key"] for r in rows))
    spec_map = {s.key: s for s in specs}

    figs = []
    for key in keys_present:
        key_rows = [r for r in rows if r["outcome_key"] == key]
        spec = spec_map.get(key)
        if spec is None:
            continue

        # Resolve plot type
        if plot_type_override == "forest" or (plot_type_override == "auto" and spec.plot_type == "forest"):
            fig, _ = draw_outcome_forest_plot(key_rows, spec, show=False)
        elif plot_type_override == "bar" or (plot_type_override == "auto" and spec.plot_type == "bar"):
            fig, _ = draw_outcome_bar_chart(key_rows, spec, show=False)
        else:
            fig = None

        if fig:
            figs.append(fig)

    # Return single fig or the first one (caller can iterate for multi-outcome)
    return figs if figs else None


def _execute_show_table(args: dict, pr: dict):
    """
    Build the outcome summary table and optionally filter by trial_ids / outcome_keys.
    Returns a DataFrame or None.
    """
    trial_ids = args.get("trial_ids") or None
    outcome_keys = args.get("outcome_keys") or None

    per_trial_results = pr.get("per_trial_results", [])
    df = build_outcome_summary_table(per_trial_results)

    if df is None or df.empty:
        return df

    if trial_ids:
        tids_lower = [t.lower() for t in trial_ids]
        # The summary table has a column for trial/NCT ID — find it
        nct_col = next(
            (c for c in df.columns if "nct" in c.lower() or "trial" in c.lower()),
            None,
        )
        if nct_col:
            df = df[df[nct_col].str.lower().isin(tids_lower)]

    if outcome_keys:
        # Filter columns: keep identifier columns + any column matching an outcome key
        id_cols = [c for c in df.columns if not any(ok in c.lower() for ok in outcome_keys)]
        outcome_cols = [c for c in df.columns if any(ok in c.lower() for ok in outcome_keys)]
        keep = id_cols[:2] + outcome_cols  # keep a couple of ID cols + matched outcome cols
        df = df[[c for c in keep if c in df.columns]]

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_pmc_pdf_bytes(pmcid: str) -> Optional[bytes]:
    """Try to download the open-access PDF for a PMC article. Returns None on failure."""
    import requests
    numeric = str(pmcid).replace("PMC", "").strip()
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{numeric}/pdf/"
    try:
        resp = requests.get(
            url, timeout=30, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-tool/1.0)"},
        )
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            return resp.content
    except Exception:
        pass
    return None


_CHAT_TEXT_LIMIT = 15000  # chars of paper text per trial passed to the chat agent

def _build_chat_context(per_trial_results: list) -> str:
    lines = [
        "You are a research assistant helping a user review clinical trial literature extraction results.",
        "For each trial you have the full paper text that was used for extraction, plus extraction metadata.",
        "Answer questions by reading the paper text directly. Be precise and quote from the text when relevant.",
        "If the answer cannot be determined from the available text, say so clearly.",
        "",
        "You have two tools available:",
        "- plot_outcome: call this whenever the user asks to visualise, chart, plot, or graph outcomes.",
        "- show_table: call this whenever the user asks for a table, summary, or spreadsheet view of outcomes.",
        "Prefer calling a tool over describing data in prose when the user's intent is clearly visual.",
        "",
    ]
    for item in per_trial_results:
        trial = item.get("trial", {}) or {}
        sr = item.get("survival_result", {}) or {}
        ex = sr.get("survival_extraction", {}) or {}
        lr = item.get("link_result", {}) or {}
        jr = lr.get("judge_result", {}) or {}
        source_text = sr.get("source_text") or ""

        nct = trial.get("nct_id", "unknown")
        pmid = jr.get("selected_pubmed_id", "unknown")
        source_used = ex.get("source_used", "unknown")

        lines.append(f"{'='*60}")
        lines.append(f"Trial: {nct} — {(trial.get('brief_title') or '')[:120]}")
        lines.append(f"Paper: PMID {pmid} | source_used={source_used} | outcome_found={ex.get('outcome_found')}")
        lines.append(f"Extraction notes: {ex.get('notes', '(none)')}")
        arms_found = [a.get('arm_name') for a in ex.get('arms', [])]
        lines.append(f"Arms extracted: {arms_found if arms_found else 'none'}")
        lines.append("")
        if source_text.strip():
            lines.append("--- Paper text (used for extraction) ---")
            lines.append(source_text[:_CHAT_TEXT_LIMIT])
            if len(source_text) > _CHAT_TEXT_LIMIT:
                lines.append(f"[... text truncated at {_CHAT_TEXT_LIMIT} chars ...]")
        else:
            lines.append("--- Paper text: not available ---")
        lines.append("")

    return "\n".join(lines)


def _merge_extractions(old_ex: dict, new_ex: dict) -> dict:
    """
    Merge PDF re-extraction (new_ex) with the previous extraction (old_ex).
    Arms in new_ex take priority; arms in old_ex that have no name match in
    new_ex are appended so previously found data is never silently discarded.
    """
    new_names = {
        (a.get("arm_name") or "").strip().lower()
        for a in new_ex.get("arms", [])
    }
    kept_old = [
        a for a in old_ex.get("arms", [])
        if (a.get("arm_name") or "").strip().lower() not in new_names
    ]
    merged = dict(new_ex)
    merged["arms"] = new_ex.get("arms", []) + kept_old
    if kept_old:
        merged["notes"] = (
            (new_ex.get("notes") or "") +
            f" | {len(kept_old)} arm(s) retained from prior extraction: "
            + ", ".join(a.get("arm_name", "?") for a in kept_old)
        ).strip(" |")
    return merged


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    pubmed_mode = st.selectbox(
        "PubMed search mode",
        ["nct_only", "hybrid"],
        help="nct_only: NCT ID only (fast, precise). hybrid: NCT ID + LLM semantic terms + structured fields.",
    )

    trial_selection_mode = st.radio(
        "Trial selection mode",
        ["Auto", "Manual"],
        horizontal=True,
        help="Auto: agent pre-selects top N trials. Manual: browse all results and pick yourself.",
    )
    max_auto_trials = st.slider("Max trials (Auto mode)", 1, 20, 5)
    max_papers = st.slider("Max papers per query", 3, 20, 5)

    st.divider()
    st.subheader("PubMed filters (optional)")
    col1, col2 = st.columns(2)
    with col1:
        pub_date_start = st.text_input("Date start", placeholder="YYYY/MM/DD")
    with col2:
        pub_date_end = st.text_input("Date end", placeholder="YYYY/MM/DD")

    pub_types = st.multiselect(
        "Publication types",
        ["clinical_trial", "rct", "meta_analysis", "systematic_review", "final_report"],
        default=[],
    )

    st.divider()
    st.subheader("Outcomes to extract")
    outcomes_raw = st.text_input(
        "Outcomes (free text)",
        placeholder="e.g. OS, PFS, response rate",
        help="Leave blank to extract OS, PFS, and ORR (default). The LLM will parse any custom request.",
    )

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _reset_downstream(keep_trials: bool = False):
    for key in ["paper_results", "inspection_results", "figures", "custom_fig", "custom_fig_name",
                "paper_finding_results", "paper_agent_config"]:
        st.session_state.pop(key, None)
    if not keep_trials:
        st.session_state.pop("trial_packet", None)
        st.session_state.pop("selected_indices", None)


def _build_pubmed_filter():
    start = pub_date_start.strip() or None
    end = pub_date_end.strip() or None
    types = list(pub_types)
    if start or end or types:
        return PubMedFilter(pub_date_start=start, pub_date_end=end, publication_types=types)
    return None


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🔬 Clinical Trial Literature Agent")
st.caption(
    "Find clinical trials → link to PubMed papers → extract outcomes → visualise."
)

# ── Step 1: query input ──────────────────────────────────────────────────────

st.subheader("Step 1 — Search Trials")

query = st.text_area(
    "Trial search request",
    placeholder="e.g. Phase 3 lung cancer pembrolizumab trials started between 2018 and 2022",
    height=80,
)

if st.button("Search ClinicalTrials.gov", type="primary"):
    if not query.strip():
        st.warning("Please enter a search request.")
    else:
        _reset_downstream()
        with st.spinner("Searching ClinicalTrials.gov…"):
            agent = TrialRetrievalAgent(
                selection_mode="auto",
                max_auto_trials=max_auto_trials,
            )
            packet = agent.run(query.strip())
        st.session_state["trial_packet"] = packet

# ── Step 2: trial selection ──────────────────────────────────────────────────

if "trial_packet" in st.session_state:
    packet = st.session_state["trial_packet"]

    if packet["status"] != "success":
        st.error(f"Trial search failed: {packet.get('status')}")
    else:
        df_trials = packet["trials"]
        # DataFrame → list of dicts for easy access
        trials = df_trials.to_dict(orient="records") if hasattr(df_trials, "to_dict") else list(df_trials)
        st.success(f"Found {len(trials)} trials.")

        st.session_state["trials_list"] = trials

        auto_sel = packet.get("selected_indices", list(range(min(max_auto_trials, len(trials)))))

        if trial_selection_mode == "Auto":
            trial_labels = {
                i: f"[{t.get('nct_id', '?')}] {t.get('brief_title', 'No title')}"
                for i, t in enumerate(trials)
            }
            selected_labels = st.multiselect(
                "Select trials to analyse",
                options=list(trial_labels.keys()),
                default=auto_sel,
                format_func=lambda i: trial_labels[i],
            )
            st.session_state["selected_indices"] = selected_labels

        else:  # Manual mode
            st.markdown("**Select trials to analyse** (check the rows you want, then click Run)")
            sel_rows = []
            for i, t in enumerate(trials):
                phases = t.get("phases") or []
                sel_rows.append({
                    "Select": i in auto_sel,
                    "_idx": i,
                    "NCT ID": t.get("nct_id", "?"),
                    "Title": (t.get("brief_title") or "")[:100],
                    "Phase": ", ".join(phases) if isinstance(phases, list) else str(phases or ""),
                    "Status": t.get("overall_status", ""),
                    "Start": t.get("start_date", ""),
                })
            edited = st.data_editor(
                pd.DataFrame(sel_rows),
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False),
                    "_idx": None,
                },
                use_container_width=True,
                hide_index=True,
                key="trial_selector",
            )
            selected_labels = edited[edited["Select"] == True]["_idx"].tolist()
            st.session_state["selected_indices"] = selected_labels
            if selected_labels:
                st.caption(f"{len(selected_labels)} trial(s) selected.")

        st.subheader("Step 2 — Find Papers")

        if st.button("Find Papers for Selected Trials", type="primary"):
            if not selected_labels:
                st.warning("Please select at least one trial.")
            else:
                _reset_downstream(keep_trials=True)
                trials = st.session_state.get("trials_list", [])
                selected_trials = [trials[i] for i in selected_labels]

                pubmed_filter = _build_pubmed_filter()
                paper_agent = PaperLinkAgent(
                    mode=pubmed_mode,
                    draw_plots=False,
                    max_papers_per_query=max_papers,
                    pubmed_filter=pubmed_filter,
                    outcomes_raw=outcomes_raw.strip() or None,
                    enable_pdf_prompt=False,
                )
                paper_agent._ensure_outcome_specs()

                progress = st.progress(0, text="Finding papers…")
                finding_results = []

                for step, (trial, idx) in enumerate(zip(selected_trials, selected_labels)):
                    nct = trial.get("nct_id", f"trial {step+1}")
                    progress.progress(step / len(selected_trials), text=f"Searching papers for {nct}…")
                    res = paper_agent.find_papers_for_trial(trial, idx)
                    finding_results.append(res)

                progress.progress(1.0, text="Paper search complete — review selections below.")
                st.session_state["paper_finding_results"] = finding_results
                st.session_state["paper_agent_config"] = {
                    "mode": pubmed_mode,
                    "max_papers": max_papers,
                    "outcomes_raw": outcomes_raw.strip() or None,
                    "auto_primary": paper_agent.auto_primary,
                    "outcome_specs": paper_agent.outcome_specs,
                }

        if "paper_finding_results" in st.session_state:
            finding_results = st.session_state["paper_finding_results"]

            st.subheader("Step 2b — Select Papers")
            st.caption("The AI has pre-selected a paper for each trial. Override any selection, then click Extract.")

            for find_res in finding_results:
                trial = find_res["trial"]
                nct = trial.get("nct_id", "?")
                link_result = find_res["link_result"]
                candidates = link_result.get("all_candidates", [])
                judge_result = link_result.get("judge_result", {}) or {}
                ai_pmid = str(judge_result.get("selected_pubmed_id") or "")

                with st.expander(f"{nct} — {(trial.get('brief_title') or '')[:80]}", expanded=True):
                    if not candidates:
                        st.warning("No candidate papers found for this trial.")
                        continue

                    option_pmids = [str(c.get("pubmed_id")) for c in candidates]
                    option_labels = {
                        str(c.get("pubmed_id")): (
                            "🤖 " if str(c.get("pubmed_id")) == ai_pmid else ""
                        ) + f"PMID {c.get('pubmed_id')} | {(c.get('title') or '')[:100]}"
                        for c in candidates
                    }
                    default_sel = [ai_pmid] if ai_pmid in option_pmids else []

                    st.multiselect(
                        "Select papers to extract (leave empty to skip this trial)",
                        options=option_pmids,
                        default=default_sel,
                        format_func=lambda pmid: option_labels.get(pmid, pmid),
                        key=f"paper_sel_{nct}",
                    )
                    st.caption(f"AI reason: {judge_result.get('reason', '—')}")

            st.subheader("Step 3 — Extract Outcomes")
            if st.button("Extract Outcomes from Selected Papers", type="primary"):
                cfg = st.session_state.get("paper_agent_config", {})
                pubmed_filter = _build_pubmed_filter()
                paper_agent = PaperLinkAgent(
                    mode=cfg.get("mode", pubmed_mode),
                    draw_plots=False,
                    max_papers_per_query=cfg.get("max_papers", max_papers),
                    pubmed_filter=pubmed_filter,
                    outcomes_raw=cfg.get("outcomes_raw"),
                    enable_pdf_prompt=False,
                )
                paper_agent.auto_primary = cfg.get("auto_primary", False)
                paper_agent.outcome_specs = cfg.get("outcome_specs")
                inspector = InspectorAgent()

                progress = st.progress(0, text="Extracting outcomes…")
                per_trial_results = []
                all_plot_rows = []

                for step, find_res in enumerate(finding_results):
                    trial = find_res["trial"]
                    nct = trial.get("nct_id", f"trial {step+1}")

                    selected_pmids = st.session_state.get(f"paper_sel_{nct}", [])

                    if not selected_pmids:
                        progress.progress(step / len(finding_results), text=f"Skipping {nct} (no paper selected)…")
                        continue

                    for pmid in selected_pmids:
                        progress.progress(step / len(finding_results), text=f"Extracting {nct} PMID {pmid}…")
                        res = paper_agent.extract_for_trial(find_res, selected_pmid=pmid)
                        per_trial_results.append(res)
                        all_plot_rows.extend(res.get("plot_rows", []))

                progress.progress(1.0, text="Extraction complete.")

                inspection_results = []
                for item in per_trial_results:
                    trial_fields = item["link_result"].get("trial_fields") or item["trial"]
                    candidates = item["link_result"].get("all_candidates", [])
                    pubmed_record = item["survival_result"].get("pubmed_record")
                    review = inspector.judge_trial_papers(trial_fields, candidates)
                    elig_review = (
                        inspector.judge_survival_plot_eligibility(pubmed_record)
                        if pubmed_record else None
                    )
                    inspection_results.append({
                        "nct_id": item["trial"].get("nct_id"),
                        "trial_title": item["trial"].get("brief_title"),
                        "trial_paper_review": review,
                        "survival_eligibility_review": elig_review,
                        "agent_judge_result": item["link_result"].get("judge_result"),
                        "agent_eligibility_judgment": item["survival_result"].get("eligibility_judgment"),
                    })

                if paper_agent.auto_primary and all_plot_rows:
                    specs = collect_specs_from_plot_rows(all_plot_rows)
                else:
                    specs = paper_agent.outcome_specs or DEFAULT_OUTCOME_SPECS

                figures, unit_logs = [], []
                if all_plot_rows:
                    figures, unit_logs = draw_all_outcome_plots(all_plot_rows, specs, show=False)

                st.session_state["paper_results"] = {
                    "per_trial_results": per_trial_results,
                    "all_plot_rows": all_plot_rows,
                    "specs": specs,
                    "auto_primary": paper_agent.auto_primary,
                }
                st.session_state["inspection_results"] = inspection_results
                st.session_state["figures"] = figures
                st.session_state["unit_logs"] = unit_logs
                st.session_state["chat_history"] = []

# ── Step 3: results ──────────────────────────────────────────────────────────

if "paper_results" in st.session_state:
    pr = st.session_state["paper_results"]
    per_trial_results = pr["per_trial_results"]
    all_plot_rows = pr["all_plot_rows"]
    specs = pr["specs"]
    figures = st.session_state.get("figures", [])
    unit_logs = st.session_state.get("unit_logs", [])
    inspection_results = st.session_state.get("inspection_results", [])

    st.divider()
    st.subheader("Step 3 — Results")

    auto_primary = pr.get("auto_primary", False)

    tab_summary, tab_plots, tab_custom, tab_ai_plot, tab_qc, tab_pdf, tab_debug, tab_ask = st.tabs([
        "Summary Table", "Plots", "Custom Plot", "AI Plot", "QC Review", "Upload PDFs", "Debug", "Ask"
    ])

    # ── Summary Table ────────────────────────────────────────────────────────
    with tab_summary:
        df = build_outcome_summary_table(per_trial_results)

        if df.empty:
            st.info("No outcomes extracted.")
        else:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "outcomes.csv", "text/csv")

        # ── Auto-primary: show identified primary endpoint per paper ─────────
        if auto_primary:
            st.divider()
            st.markdown("**Identified Primary Endpoints** *(auto-discovery mode)*")
            any_found = False
            for item in per_trial_results:
                trial = item.get("trial", {}) or {}
                sr = item.get("survival_result", {}) or {}
                id_result = sr.get("primary_endpoint_identification") or {}
                nct = trial.get("nct_id", "?")
                title = (trial.get("brief_title") or "")[:80]

                endpoints = id_result.get("primary_endpoints", [])
                if not endpoints:
                    st.markdown(
                        f"- **{nct}** — *{title}*  \n"
                        f"  ⚠️ Primary endpoint not identified. {id_result.get('notes', '')}"
                    )
                else:
                    any_found = True
                    for ep in endpoints:
                        source = ep.get("source", "unknown")
                        source_badge = "🔵 explicit" if source == "explicit" else "🟡 inferred"
                        evidence = ep.get("evidence", "")
                        st.markdown(
                            f"- **{nct}** — *{title}*  \n"
                            f"  → **{ep.get('display', ep.get('key', '?'))}** "
                            f"(`{ep.get('plot_type', '?')}`) &nbsp; {source_badge}  \n"
                            f"  *\"{evidence}\"*"
                        )
            if not any_found and not per_trial_results:
                st.info("No primary endpoints identified.")

        st.divider()
        with st.expander("Linked papers & downloads", expanded=False):
            for item in per_trial_results:
                trial = item.get("trial", {}) or {}
                sr = item.get("survival_result", {}) or {}
                ex = sr.get("survival_extraction", {}) or {}
                lr = item.get("link_result", {}) or {}
                jr = lr.get("judge_result", {}) or {}
                pubmed_record = sr.get("pubmed_record") or {}

                nct = trial.get("nct_id", "?")
                pmid = jr.get("selected_pubmed_id")
                pmcid = ex.get("pmcid") or sr.get("pmcid")
                doi = pubmed_record.get("doi")
                title = jr.get("selected_title") or pubmed_record.get("title") or "—"
                source_used = ex.get("source_used", "—")

                st.markdown(f"**{nct}** — {title[:120]}")

                links = []
                if pmid:
                    links.append(f"[PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                if pmcid:
                    numeric = str(pmcid).replace("PMC", "").strip()
                    links.append(f"[PMC Full Text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{numeric}/)")
                if doi:
                    links.append(f"[DOI](https://doi.org/{doi})")
                st.markdown("  |  ".join(links) if links else "_No links available_")
                st.caption(f"source used for extraction: {source_used}")

                st.markdown("---")

    # ── Plots ─────────────────────────���──────────────────────────────────────
    with tab_plots:
        if not figures:
            st.info("No plottable data extracted.")
        else:
            for fig in figures:
                st.pyplot(fig)
                img_bytes = _fig_to_bytes(fig)
                st.download_button(
                    "Download this plot (PNG)",
                    img_bytes,
                    file_name="plot.png",
                    mime="image/png",
                    key=f"dl_{id(fig)}",
                )

        if unit_logs:
            with st.expander("Unit conversion log (time-based outcomes)", expanded=False):
                st.caption(
                    "All time-based outcomes (OS, PFS, etc.) are plotted in **months**. "
                    "Years are multiplied by 12. Unsupported units (weeks, days, etc.) are "
                    "excluded from the plot and shown here."
                )
                import pandas as _pd
                log_df = _pd.DataFrame(unit_logs)[
                    ["outcome", "trial", "arm", "original_value", "original_unit",
                     "plotted_value", "plotted_unit", "note"]
                ]
                skipped = log_df[log_df["plotted_unit"] == "—"]
                if not skipped.empty:
                    st.warning(
                        f"{len(skipped)} arm(s) excluded from plot due to unsupported units — "
                        "see rows marked '—' below."
                    )
                st.dataframe(log_df, use_container_width=True)

    # ── Custom Plot ──────────────────────────────────────────────────────────
    with tab_custom:
        st.markdown("Pick specific trials and arms to generate a custom plot and download it.")

        eligible_rows = [r for r in all_plot_rows if r.get("plot_eligible")]

        if not eligible_rows:
            st.info("No plottable data available.")
        else:
            # Step 1 — pick outcome
            outcome_options = {}
            for r in eligible_rows:
                k = r["outcome_key"]
                if k not in outcome_options:
                    outcome_options[k] = r["outcome_display"]

            selected_key = st.selectbox(
                "Outcome",
                options=list(outcome_options.keys()),
                format_func=lambda k: outcome_options[k],
            )

            # Step 2 — pick trial+arm combinations for that outcome
            outcome_eligible = [r for r in eligible_rows if r["outcome_key"] == selected_key]
            combo_options = [
                (r["trial_label"], r["arm_name"])
                for r in outcome_eligible
            ]
            # deduplicate while preserving order
            seen = set()
            combo_options_deduped = []
            for c in combo_options:
                if c not in seen:
                    seen.add(c)
                    combo_options_deduped.append(c)

            selected_combos = st.multiselect(
                "Trials / Arms",
                options=combo_options_deduped,
                default=combo_options_deduped,
                format_func=lambda c: f"{c[0]}  —  {c[1]}",
            )

            if st.button("Generate Custom Plot", type="primary"):
                selected_set = set(selected_combos)
                filtered_rows = [
                    r for r in outcome_eligible
                    if (r["trial_label"], r["arm_name"]) in selected_set
                ]

                spec_map = {s.key: s for s in specs}
                spec = spec_map.get(selected_key)

                if not filtered_rows or spec is None:
                    st.warning("No data for the selected combination.")
                else:
                    if spec.plot_type == "forest":
                        fig, _ = draw_outcome_forest_plot(filtered_rows, spec, show=False)
                    elif spec.plot_type == "bar":
                        fig, _ = draw_outcome_bar_chart(filtered_rows, spec, show=False)
                    else:
                        fig = None
                        st.info(f"{spec.display} is table-only — no plot available.")

                    if fig:
                        st.session_state["custom_fig"] = fig
                        st.session_state["custom_fig_name"] = (
                            f"custom_{selected_key}.png"
                        )

            if "custom_fig" in st.session_state:
                fig = st.session_state["custom_fig"]
                st.pyplot(fig)
                st.download_button(
                    "Download PNG",
                    _fig_to_bytes(fig),
                    file_name=st.session_state.get("custom_fig_name", "custom_plot.png"),
                    mime="image/png",
                )

    # ── AI Plot ──────────────────────────────────────────────────────────────
    with tab_ai_plot:
        st.caption("Describe what you want to plot. AI generates matplotlib code, renders it, and self-reviews.")

        ai_plot_request = st.text_input(
            "What do you want to plot?",
            placeholder="e.g. Forest plot of OS by arm for all trials, blue color palette",
            key="ai_plot_request",
        )

        col_gen, col_clear = st.columns([1, 4])
        with col_gen:
            gen_clicked = st.button("Generate Plot", type="primary", key="ai_plot_gen")
        with col_clear:
            if st.button("Clear", key="ai_plot_clear"):
                for k in ["ai_plot_code", "ai_plot_fig_b64", "ai_plot_review", "ai_plot_fig", "ai_plotter"]:
                    st.session_state.pop(k, None)
                st.rerun()

        if gen_clicked and ai_plot_request.strip():
            eligible_rows = [r for r in all_plot_rows if r.get("plot_eligible")]
            if not eligible_rows:
                st.warning("No plottable data available.")
            else:
                from tools.sandbox_plot import run_sandboxed_plot
                from tools.ai_plotter import AIPlotterAgent

                plotter = AIPlotterAgent()

                with st.spinner("Generating plot code…"):
                    code = plotter.generate_code(eligible_rows, ai_plot_request.strip())
                st.session_state["ai_plot_code"] = code

                df = pd.DataFrame(eligible_rows)
                fig, b64, err = run_sandboxed_plot(code, {"plot_rows": eligible_rows, "df": df})

                if err:
                    st.error(f"Plot execution error:\n```\n{err}\n```")
                else:
                    st.session_state["ai_plot_fig"] = fig
                    st.session_state["ai_plot_fig_b64"] = b64

                    with st.spinner("AI reviewing plot…"):
                        review = plotter.review_plot(b64, ai_plot_request.strip())
                    st.session_state["ai_plot_review"] = review
                    st.session_state["ai_plotter"] = plotter

        # Show current plot
        if "ai_plot_fig" in st.session_state:
            st.pyplot(st.session_state["ai_plot_fig"])
            st.download_button(
                "Download PNG",
                data=base64.b64decode(st.session_state["ai_plot_fig_b64"]),
                file_name="ai_plot.png",
                mime="image/png",
            )

        # Show AI review
        if "ai_plot_review" in st.session_state:
            review_text = st.session_state["ai_plot_review"]
            with st.expander("AI Review", expanded=True):
                if "APPROVED" in review_text.upper():
                    st.success("AI: Plot looks good.")
                else:
                    st.info(review_text[:500])
                    if st.button("Apply AI Refinements", key="ai_plot_refine"):
                        plotter = st.session_state.get("ai_plotter")
                        if plotter:
                            with st.spinner("Refining…"):
                                new_code = plotter.refine_code(review_text)
                            st.session_state["ai_plot_code"] = new_code
                            from tools.sandbox_plot import run_sandboxed_plot
                            eligible_rows = [r for r in all_plot_rows if r.get("plot_eligible")]
                            df2 = pd.DataFrame(eligible_rows)
                            fig2, b64_2, err2 = run_sandboxed_plot(new_code, {"plot_rows": eligible_rows, "df": df2})
                            if err2:
                                st.error(err2)
                            else:
                                st.session_state["ai_plot_fig"] = fig2
                                st.session_state["ai_plot_fig_b64"] = b64_2
                                st.rerun()

        # Show generated code
        if "ai_plot_code" in st.session_state:
            with st.expander("Generated code", expanded=False):
                st.code(st.session_state["ai_plot_code"], language="python")

        # Follow-up refinement input
        if "ai_plot_fig" in st.session_state:
            followup = st.text_input(
                "Ask for further changes",
                placeholder="e.g. make the bars wider, use green colors, add gridlines",
                key="ai_plot_followup",
            )
            if st.button("Apply Changes", key="ai_plot_followup_btn") and followup.strip():
                plotter = st.session_state.get("ai_plotter")
                if plotter:
                    with st.spinner("Applying changes…"):
                        new_code = plotter.refine_code(followup.strip())
                    st.session_state["ai_plot_code"] = new_code
                    from tools.sandbox_plot import run_sandboxed_plot
                    eligible_rows = [r for r in all_plot_rows if r.get("plot_eligible")]
                    df3 = pd.DataFrame(eligible_rows)
                    fig3, b64_3, err3 = run_sandboxed_plot(new_code, {"plot_rows": eligible_rows, "df": df3})
                    if err3:
                        st.error(err3)
                    else:
                        st.session_state["ai_plot_fig"] = fig3
                        st.session_state["ai_plot_fig_b64"] = b64_3
                        st.rerun()

    # ── QC Review ───────────────────────────────────────────────────────────
    with tab_qc:
        for item in inspection_results:
            nct = item.get("nct_id", "?")
            title = item.get("trial_title", "")
            agent_j = item.get("agent_judge_result") or {}
            insp_j = item.get("trial_paper_review") or {}
            agent_e = item.get("agent_eligibility_judgment") or {}
            insp_e = item.get("survival_eligibility_review") or {}

            with st.expander(f"{nct} — {title[:60]}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Agent verdict**")
                    st.write(f"Match: `{agent_j.get('match_found')}` | "
                             f"Confidence: `{agent_j.get('confidence')}`")
                    st.write(agent_j.get("reason", ""))
                    if agent_e:
                        st.write(f"Eligible: `{agent_e.get('eligible')}` | "
                                 f"Type: `{agent_e.get('paper_type')}`")
                with col_b:
                    st.markdown("**Inspector verdict**")
                    st.write(f"Match: `{insp_j.get('match_found')}` | "
                             f"Confidence: `{insp_j.get('confidence')}`")
                    st.write(insp_j.get("reason", ""))
                    if insp_e:
                        st.write(f"Eligible: `{insp_e.get('eligible')}` | "
                                 f"Type: `{insp_e.get('paper_type')}`")

    # ── Upload PDFs ──────────────────────────────────────────────────────────
    with tab_pdf:
        st.markdown(
            "For papers where only the abstract was available, upload the full PDF here "
            "to re-run extraction with the full text. "
            "Results will update the Summary Table and Plots automatically."
        )

        abstract_only = [
            item for item in per_trial_results
            if (item["survival_result"].get("survival_extraction") or {}).get("source_used") == "abstract"
        ]

        if not abstract_only:
            st.info("All papers used full text — no PDF uploads needed.")

        for item in abstract_only:
            t = item["trial"]
            sr = item["survival_result"]
            jr = item["link_result"].get("judge_result", {}) or {}
            pmid = jr.get("selected_pubmed_id", "?")
            nct = t.get("nct_id", "?")

            with st.expander(f"{nct} — PMID {pmid} (abstract only)"):
                uploaded = st.file_uploader(
                    f"Upload PDF for PMID {pmid}",
                    type="pdf",
                    key=f"pdf_{nct}_{pmid}",
                )
                if uploaded and st.button(f"Re-extract {nct}", key=f"rerun_{nct}_{pmid}"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    try:
                        from tools.pdf import read_pdf_text
                        pdf_text = read_pdf_text(tmp_path)
                        if not pdf_text:
                            st.error("Could not extract text from PDF.")
                        else:
                            pubmed_record = sr.get("pubmed_record") or {}
                            _agent = PaperLinkAgent(
                                outcomes_raw=outcomes_raw.strip() or None,
                                enable_pdf_prompt=False,
                            )
                            _agent._ensure_outcome_specs()

                            with st.spinner("Re-extracting from PDF…"):
                                new_ex = _agent.extract_outcomes_from_text(
                                    pubmed_record, pdf_text, "user_pdf",
                                    trial_id=nct, trial_label=t.get("brief_title"),
                                )

                            # ── write back into session state ──────────────
                            # merge new PDF extraction with old to avoid losing arms
                            updated_results = st.session_state["paper_results"]["per_trial_results"]
                            for r in updated_results:
                                if r["trial"].get("nct_id") == nct:
                                    old_ex = r["survival_result"].get("survival_extraction") or {}
                                    merged_ex = _merge_extractions(old_ex, new_ex)
                                    # recompute plot_rows for merged arms
                                    from tools.outcomes import build_multi_outcome_plot_rows
                                    merged_ex["plot_rows"] = build_multi_outcome_plot_rows(
                                        merged_ex,
                                        _agent.outcome_specs or DEFAULT_OUTCOME_SPECS,
                                        trial_id=nct,
                                        trial_label=t.get("brief_title"),
                                    )
                                    r["survival_result"]["survival_extraction"] = merged_ex
                                    r["plot_rows"] = merged_ex["plot_rows"]
                                    new_ex = merged_ex  # use merged for success message
                                    break

                            # rebuild all_plot_rows and figures
                            new_all_plot_rows = []
                            for r in updated_results:
                                new_all_plot_rows.extend(r.get("plot_rows", []))

                            new_figs, new_unit_logs = [], []
                            if new_all_plot_rows:
                                new_figs, new_unit_logs = draw_all_outcome_plots(
                                    new_all_plot_rows, specs, show=False
                                )

                            st.session_state["paper_results"]["per_trial_results"] = updated_results
                            st.session_state["paper_results"]["all_plot_rows"] = new_all_plot_rows
                            st.session_state["figures"] = new_figs
                            st.session_state["unit_logs"] = new_unit_logs

                            n_arms = len(new_ex.get("arms", []))
                            n_plot_rows = len(new_ex.get("plot_rows", []))
                            if n_plot_rows > 0:
                                st.success(
                                    f"Re-extraction complete — {n_arms} arm(s) extracted, "
                                    f"{n_plot_rows} plottable row(s). Summary Table and Plots updated."
                                )
                            else:
                                st.warning(
                                    f"Re-extraction complete — {n_arms} arm(s) found but "
                                    f"no plottable values (outcome may not be reported in this PDF). "
                                    f"Check the raw result below."
                                )
                                st.json(new_ex)
                            st.rerun()
                    finally:
                        os.unlink(tmp_path)

    # ── Debug ────────────────────────────────────────────────────────────────
    with tab_debug:
        for item in per_trial_results:
            nct = item["trial"].get("nct_id", "?")
            with st.expander(f"{nct} — raw link result"):
                lr = item["link_result"]
                st.write(f"**Query A:** `{lr.get('query_A')}`")
                st.write(f"**Query B:** `{lr.get('query_B')}`")
                st.write(f"**Query C:** `{lr.get('query_C')}`")
                st.write(f"**Candidates:** {len(lr.get('all_candidates', []))}")
                jr = lr.get("judge_result", {}) or {}
                st.write(f"**Match:** `{jr.get('match_found')}` | "
                         f"PMID `{jr.get('selected_pubmed_id')}` | "
                         f"label `{jr.get('label')}` | "
                         f"confidence `{jr.get('confidence')}`")
                st.write(f"**Reason:** {jr.get('reason')}")
                sr = item["survival_result"]
                ex = sr.get("survival_extraction", {}) or {}
                st.write(f"**Source used:** `{ex.get('source_used')}` | "
                         f"outcome_found `{ex.get('outcome_found')}`")
                st.write(f"**Notes:** {ex.get('notes')}")

    # ── Ask ──────────────────────────────────────────────────────────────────
    with tab_ask:
        st.caption(
            "Ask anything about the papers or extraction results. "
            "The agent reads the actual paper text retrieved during the pipeline run — "
            "e.g. \"Why was only one arm found for NCT123?\", "
            "\"What does the paper say about PFS in the control arm?\", "
            "\"Is there a subgroup analysis reported?\""
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Clear button
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

        # Render existing history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # New user input
        if question := st.chat_input("Ask about the results…"):
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state["chat_history"].append({"role": "user", "content": question})

            # Build agent with current extraction context, replay history
            from base_agent import BaseAgent
            context = _build_chat_context(per_trial_results)
            _chat_agent = BaseAgent(system_prompt=context)
            for msg in st.session_state["chat_history"][:-1]:
                if msg["role"] in ("user", "assistant"):
                    _chat_agent.add_message(msg["role"], msg["content"])

            with st.chat_message("assistant"):
                with st.spinner(""):
                    resp = _chat_agent._call_chat_model(
                        user_message=question,
                        temperature=0.2,
                        tools=CHAT_TOOLS,
                        tool_choice="auto",
                    )
                    msg = resp.choices[0].message

                    reply = ""
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            fn = tc.function.name
                            args = json.loads(tc.function.arguments)

                            if fn == "plot_outcome":
                                figs = _execute_plot_outcome(args, pr)
                                if figs:
                                    for fig in figs:
                                        st.pyplot(fig)
                                    outcome_label = args.get("outcome_key") or "all outcomes"
                                    reply = f"Here is the plot for {outcome_label}."
                                else:
                                    reply = "No plottable data matched your request."

                            elif fn == "show_table":
                                df_chat = _execute_show_table(args, pr)
                                if df_chat is not None and not df_chat.empty:
                                    st.dataframe(df_chat, use_container_width=True)
                                    reply = "Here is the summary table."
                                else:
                                    reply = "No data matched your request."

                            else:
                                reply = f"Unknown tool: {fn}"
                    else:
                        reply = msg.content or ""

                st.markdown(reply)

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
