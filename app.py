"""
Streamlit web interface for the Clinical Trial Literature Agent.

Run with:
    streamlit run app.py
"""

import io

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
from ui.ask_tab import render_ask_tab
from ui.ai_plot_tab import render_ai_plot_tab
from ui.pdf_tab import render_pdf_tab

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Trial Literature Agent",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

_DATE_FIELD_LABELS = [
    "Study Start Date",
    "Primary Completion Date",
    "Study Completion Date",
    "Results First Posted Date",
    "First Posted Date",
    "Last Update Posted Date",
]
_DATE_FIELD_KEYS = {
    "Study Start Date":          "start_date",
    "Primary Completion Date":   "primary_completion_date",
    "Study Completion Date":     "completion_date",
    "Results First Posted Date": "results_first_posted_date",
    "First Posted Date":         "first_posted_date",
    "Last Update Posted Date":   "last_update_posted_date",
}

with st.sidebar:
    st.header("Configuration")

    trial_selection_mode = st.radio(
        "Trial selection mode",
        ["Auto", "Manual"],
        horizontal=True,
        help="Auto: agent pre-selects top N trials. Manual: browse all results and pick yourself.",
    )
    max_auto_trials = st.slider("Max trials (Auto mode)", 1, 20, 5)
    max_papers = st.slider("Max papers per query", 3, 20, 5)

    st.divider()
    st.subheader("Trial date field")
    st.caption(
        "When your query includes a year range (e.g. '2018–2022'), this controls "
        "which ClinicalTrials.gov date is compared against that range."
    )
    date_field_label = st.radio(
        "Filter trials by",
        _DATE_FIELD_LABELS,
        index=0,
        help=(
            "Study Start Date — when the trial began enrolling (default).  \n"
            "Primary Completion Date — when primary endpoint data collection ended.  \n"
            "Study Completion Date — when all data collection ended.  \n"
            "Results First Posted Date — when results were first posted on ClinicalTrials.gov.  \n"
            "First Posted Date — when the trial record was first registered.  \n"
            "Last Update Posted Date — when the trial record was last updated."
        ),
    )
    date_field = _DATE_FIELD_KEYS[date_field_label]

    st.divider()
    st.subheader("PubMed filters (optional)")
    st.caption(
        "These filters apply to **fallback keyword search only** — when the NCT ID lookup "
        "finds no papers, the pipeline falls back to a drug + disease keyword query and "
        "applies these filters to narrow results. The primary NCT ID search always runs "
        "unfiltered so a valid paper is never missed due to date or type restrictions."
    )
    col1, col2 = st.columns(2)
    with col1:
        pub_date_start = st.text_input(
            "Published from", placeholder="YYYY/MM/DD",
            help=(
                "Earliest **paper publication date** to include in fallback results "
                "(YYYY/MM/DD or YYYY-MM-DD). Not the trial start date."
            ),
        )
    with col2:
        pub_date_end = st.text_input(
            "Published to", placeholder="YYYY/MM/DD",
            help=(
                "Latest **paper publication date** to include in fallback results "
                "(YYYY/MM/DD or YYYY-MM-DD). Not the trial end date."
            ),
        )

    pub_types = st.multiselect(
        "Publication types",
        ["clinical_trial", "rct", "meta_analysis", "systematic_review", "final_report"],
        default=[],
        help="Restricts fallback keyword results to the selected publication types. Has no effect when NCT ID search succeeds.",
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
                "paper_finding_results", "paper_agent_config", "chat_history", "unit_logs",
                "ai_plot_code", "ai_plot_fig_b64", "ai_plot_review", "ai_plot_figs",
                "ai_plotter", "ai_plot_default_figs", "ai_plot_approved", "ai_plot_rows"]:
        st.session_state.pop(key, None)
    if not keep_trials:
        st.session_state.pop("trial_packet", None)
        st.session_state.pop("selected_indices", None)


def _normalize_pubmed_date(raw: str) -> str:
    """Normalize date to YYYY/MM/DD (PubMed format). Accepts YYYY-MM-DD or YYYY/MM/DD or YYYY."""
    raw = raw.strip().replace("-", "/")
    return raw

def _build_pubmed_filter():
    start = _normalize_pubmed_date(pub_date_start.strip()) if pub_date_start.strip() else None
    end = _normalize_pubmed_date(pub_date_end.strip()) if pub_date_end.strip() else None
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
st.caption(
    f"Any year range in your query (e.g. '2018–2022') is compared against "
    f"**{date_field_label}** from ClinicalTrials.gov. "
    f"Change the date field in the sidebar if you want a different comparison."
)

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
            packet = agent.run(query.strip(), date_field=date_field)
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
        st.caption(
            "For each trial the pipeline first searches PubMed by **NCT ID** (precise, unfiltered). "
            "If that returns nothing, it automatically falls back to a **drug + disease keyword search** "
            "using your PubMed filters. Candidates are then judged by the AI — 🤖 marks its top pick."
        )

        if st.button("Find Papers for Selected Trials", type="primary"):
            if not selected_labels:
                st.warning("Please select at least one trial.")
            else:
                _reset_downstream(keep_trials=True)
                trials = st.session_state.get("trials_list", [])
                selected_trials = [trials[i] for i in selected_labels]

                pubmed_filter = _build_pubmed_filter()
                paper_agent = PaperLinkAgent(
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
                        st.warning(
                            f"No candidate papers found for {nct}. "
                            "The NCT ID search and both fallback levels returned nothing. "
                            "Try: (1) loosening or clearing PubMed filters, or "
                            "(2) uploading a PDF directly in the Upload PDFs tab after extraction."
                        )
                        continue

                    option_pmids = [str(c.get("pubmed_id")) for c in candidates]
                    option_labels = {
                        str(c.get("pubmed_id")): (
                            "🤖 " if str(c.get("pubmed_id")) == ai_pmid else ""
                        ) + f"PMID {c.get('pubmed_id')} | {(c.get('title') or '')[:100]}"
                        for c in candidates
                    }
                    default_sel = [ai_pmid] if ai_pmid in option_pmids else []

                    # If the LLM didn't pick a paper but Query A (NCT ID search) found one,
                    # pre-select the first NCT ID result — it's nearly certain to be correct.
                    if not default_sel and not link_result.get("fallback_used"):
                        papers_A_pmids = [str(p.get("pubmed_id")) for p in link_result.get("papers_A", [])]
                        for pmid in papers_A_pmids:
                            if pmid in option_pmids:
                                default_sel = [pmid]
                                break

                    st.multiselect(
                        "Select papers to extract (leave empty to skip this trial)",
                        options=option_pmids,
                        default=default_sel,
                        format_func=lambda pmid: option_labels.get(pmid, pmid),
                        key=f"paper_sel_{nct}",
                    )
                    if link_result.get("fallback_used"):
                        st.caption(
                            "⚠️ NCT ID search found no papers — candidates below are from a "
                            "fallback drug + disease keyword search and may be less precise. "
                            "Review the AI's reason carefully before extracting."
                        )
                    st.caption(f"AI reason: {judge_result.get('reason', '—')}")

            st.subheader("Step 3 — Extract Outcomes")
            if st.button("Extract Outcomes from Selected Papers", type="primary"):
                cfg = st.session_state.get("paper_agent_config", {})
                pubmed_filter = _build_pubmed_filter()
                paper_agent = PaperLinkAgent(
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
                    agent_judge = item["link_result"].get("judge_result") or {}

                    review = inspector.judge_trial_papers(trial_fields, candidates)
                    elig_review = (
                        inspector.judge_survival_plot_eligibility(pubmed_record)
                        if pubmed_record else None
                    )

                    # Fix #2: tiebreaker when agent and inspector disagree on match
                    tiebreak_result = None
                    agent_match = bool(agent_judge.get("match_found"))
                    insp_match = bool(review.get("match_found"))
                    if agent_match != insp_match:
                        nct = item["trial"].get("nct_id", "?")
                        print(f"  [tiebreaker] {nct}: agent={agent_match} vs inspector={insp_match} — running tiebreaker…")
                        try:
                            tiebreak_result = inspector.tiebreak_paper_match(
                                trial_fields, candidates, agent_judge, review
                            )
                            print(f"  [tiebreaker] Final: match={tiebreak_result.get('match_found')} "
                                  f"confidence={tiebreak_result.get('confidence')}")
                        except Exception as tb_exc:
                            print(f"  [tiebreaker] Failed: {tb_exc}")

                    inspection_results.append({
                        "nct_id": item["trial"].get("nct_id"),
                        "trial_title": item["trial"].get("brief_title"),
                        "trial_paper_review": review,
                        "survival_eligibility_review": elig_review,
                        "agent_judge_result": agent_judge,
                        "agent_eligibility_judgment": item["survival_result"].get("eligibility_judgment"),
                        "tiebreak_result": tiebreak_result,
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
            if "is_subgroup" not in df.columns:
                df["is_subgroup"] = False

            has_subgroups = bool(df["is_subgroup"].any())
            show_subgroups = False
            if has_subgroups:
                show_subgroups = st.checkbox(
                    "Show subgroup rows",
                    value=False,
                    help="Pre-specified biomarker/mutation-defined subgroup analyses, indented under their parent arm.",
                )
            display_df = df[~df["is_subgroup"]].copy() if (has_subgroups and not show_subgroups) else df.copy()

            # Build a display column that marks subgroup rows (leading spaces stripped by st.dataframe)
            def _arm_display(r):
                if r.get("is_subgroup"):
                    return f"[subgroup] {r['arm_name']} › {r.get('subgroup_name', '')}"
                return r["arm_name"]

            display_df["arm"] = display_df.apply(_arm_display, axis=1)
            base_cols = ["nct_id", "trial_title", "pubmed_id", "arm", "arm_n"]
            if has_subgroups and show_subgroups and "subgroup_n" in display_df.columns:
                base_cols.append("subgroup_n")
            base_cols += ["outcome", "value", "unit", "ci_lower", "ci_upper", "evidence"]
            display_df = display_df[[c for c in base_cols if c in display_df.columns]]

            st.dataframe(display_df, use_container_width=True)
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
    render_ai_plot_tab(tab_ai_plot, all_plot_rows, specs)

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

                tb = item.get("tiebreak_result")
                if tb:
                    st.divider()
                    st.markdown("**⚖️ Tiebreaker verdict** *(agents disagreed — third-pass resolution)*")
                    tb_match = tb.get("match_found")
                    tb_conf = tb.get("confidence", "?")
                    if tb_match:
                        st.success(f"Match: `True` | Confidence: `{tb_conf}`")
                    else:
                        st.warning(f"Match: `False` | Confidence: `{tb_conf}`")
                    st.write(tb.get("reason", ""))

    # ── Upload PDFs ──────────────────────────────────────────────────────────
    render_pdf_tab(tab_pdf, per_trial_results, outcomes_raw, specs)

    # ── Debug ────────────────────────────────────────────────────────────────
    with tab_debug:
        for item in per_trial_results:
            nct = item["trial"].get("nct_id", "?")
            with st.expander(f"{nct} — raw link result"):
                lr = item["link_result"]
                st.write(f"**Query A:** `{lr.get('query_A')}`")
                if lr.get("fallback_used"):
                    st.warning("Fallback search was triggered (Query A returned 0 results).")
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
    render_ask_tab(tab_ask, per_trial_results, pr)
