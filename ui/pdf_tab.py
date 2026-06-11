"""Upload PDFs tab: PDF re-extraction and session state write-back."""

import os
import tempfile
import streamlit as st

from paper_link_agent import PaperLinkAgent
from tools.outcomes import (
    build_multi_outcome_plot_rows,
    draw_all_outcome_plots,
    DEFAULT_OUTCOME_SPECS,
)


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


def render_pdf_tab(tab, per_trial_results: list, outcomes_raw: str, specs: list) -> None:
    with tab:
        st.markdown(
            "For papers where only the abstract was available, upload the full PDF here "
            "to re-run extraction with the full text. "
            "Re-extraction starts automatically on upload — no button needed. "
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

                processed_key = f"pdf_processed_{nct}_{pmid}"
                file_sig = (uploaded.name, uploaded.size) if uploaded else None

                if uploaded and file_sig != st.session_state.get(processed_key):
                    st.session_state[processed_key] = file_sig
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

                            updated_results = st.session_state["paper_results"]["per_trial_results"]
                            for r in updated_results:
                                if r["trial"].get("nct_id") == nct:
                                    old_ex = r["survival_result"].get("survival_extraction") or {}
                                    merged_ex = _merge_extractions(old_ex, new_ex)
                                    merged_ex["plot_rows"] = build_multi_outcome_plot_rows(
                                        merged_ex,
                                        _agent.outcome_specs or DEFAULT_OUTCOME_SPECS,
                                        trial_id=nct,
                                        trial_label=t.get("brief_title"),
                                    )
                                    r["survival_result"]["survival_extraction"] = merged_ex
                                    r["plot_rows"] = merged_ex["plot_rows"]
                                    r["survival_result"]["source_text"] = pdf_text
                                    new_ex = merged_ex
                                    break

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
                            st.session_state.pop("ai_plot_rows", None)

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
