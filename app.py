"""
Streamlit web interface for the Clinical Trial Literature Agent.

Run with:
    streamlit run app.py
"""

import io
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
    st.subheader("Outcomes to extract")
    outcomes_raw = st.text_input(
        "Outcomes (free text)",
        placeholder="e.g. OS, PFS, response rate",
        help="Leave blank to extract OS, PFS, and ORR (default). The LLM will parse any custom request.",
    )

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

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _reset_downstream(keep_trials: bool = False):
    for key in ["paper_results", "inspection_results", "figures"]:
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

        st.subheader("Step 2 — Find Papers & Extract Outcomes")

        if st.button("Run pipeline on selected trials", type="primary"):
            if not selected_labels:
                st.warning("Please select at least one trial.")
            else:
                _reset_downstream(keep_trials=True)
                trials = st.session_state.get("trials_list", [])
                selected_trials = [trials[i] for i in selected_labels]

                pubmed_filter = _build_pubmed_filter()
                paper_agent = PaperLinkAgent(
                    mode=pubmed_mode,
                    draw_plots=False,          # we draw manually in Streamlit
                    max_papers_per_query=max_papers,
                    pubmed_filter=pubmed_filter,
                    outcomes_raw=outcomes_raw.strip() or None,
                    enable_pdf_prompt=False,   # no blocking input() in Streamlit
                )
                inspector = InspectorAgent()

                progress = st.progress(0, text="Starting…")
                per_trial_results = []
                all_plot_rows = []

                for step, (trial, idx) in enumerate(zip(selected_trials, selected_labels)):
                    nct = trial.get("nct_id", f"trial {step+1}")
                    progress.progress(
                        (step) / len(selected_trials),
                        text=f"Processing {nct}…",
                    )
                    res = paper_agent.run_one(trial, idx)
                    per_trial_results.append(res)
                    all_plot_rows.extend(res.get("plot_rows", []))

                progress.progress(1.0, text="Extraction complete.")

                # inspector QC
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

                # build figures (show=False → returns list)
                # In auto_primary mode outcome_specs is [], so derive specs from the
                # plot_rows themselves — works for both fixed and auto_primary modes.
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
                st.session_state["chat_history"] = []  # reset on new run

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

    tab_summary, tab_plots, tab_qc, tab_pdf, tab_debug, tab_ask = st.tabs([
        "Summary Table", "Plots", "QC Review", "Upload PDFs", "Debug", "Ask"
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
                _chat_agent.add_message(msg["role"], msg["content"])

            with st.chat_message("assistant"):
                with st.spinner(""):
                    resp = _chat_agent._call_chat_model(
                        user_message=question, temperature=0.2
                    )
                    reply = resp.choices[0].message.content or ""
                st.markdown(reply)

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
