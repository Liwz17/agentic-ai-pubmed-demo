"""AI Plot tab: code generation, sandbox execution, and review/refine loop."""

import base64
import streamlit as st
import pandas as pd

from tools.outcomes import draw_all_outcome_plots


def render_ai_plot_tab(tab, all_plot_rows: list, specs: list) -> None:
    with tab:
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
                for k in ["ai_plot_code", "ai_plot_fig_b64", "ai_plot_review", "ai_plot_figs", "ai_plotter"]:
                    st.session_state.pop(k, None)
                st.rerun()

        if gen_clicked:
            eligible_rows = [r for r in all_plot_rows if r.get("plot_eligible")]
            ai_rows = [dict(r) for r in all_plot_rows]
            st.session_state["ai_plot_rows"] = ai_rows

            if not ai_rows:
                st.warning("No plottable data available.")
            elif not ai_plot_request.strip():
                default_figs, _ = draw_all_outcome_plots(eligible_rows, specs, show=False)
                if not default_figs:
                    st.info("No plottable outcomes found.")
                else:
                    for k in ["ai_plot_code", "ai_plot_fig_b64", "ai_plot_review", "ai_plot_figs", "ai_plotter", "ai_plot_default_figs"]:
                        st.session_state.pop(k, None)
                    st.session_state["ai_plot_default_figs"] = default_figs
            else:
                from tools.sandbox_plot import run_sandboxed_plot
                from tools.ai_plotter import AIPlotterAgent

                plotter = AIPlotterAgent(model="openai/gpt-4o")
                _df = pd.DataFrame(ai_rows)
                _data = {"plot_rows": ai_rows, "df": _df}
                MAX_RETRIES = 5
                MAX_REVIEW_ROUNDS = 5

                with st.status("Generating plot…", expanded=True) as status:
                    st.write("Generating code…")
                    code = plotter.generate_code(ai_rows, ai_plot_request.strip())
                    st.session_state["ai_plot_code"] = code

                    figs, b64, err = None, None, None
                    for attempt in range(1, MAX_RETRIES + 1):
                        st.write(f"Running code (attempt {attempt}/{MAX_RETRIES})…")
                        figs, b64, err = run_sandboxed_plot(code, _data)
                        if not err:
                            break
                        st.write(f"⚠️ Error: `{err.splitlines()[-1]}`")
                        if attempt < MAX_RETRIES:
                            st.write("Fixing error…")
                            code = plotter.fix_code(code, err)
                            st.session_state["ai_plot_code"] = code
                        else:
                            st.write("Max retries reached.")

                    if err:
                        status.update(label="Failed after retries", state="error")
                        st.error(f"Could not render plot:\n```\n{err}\n```")
                    else:
                        review = ""
                        approved = False
                        failed_attempts: list = []
                        for round_n in range(1, MAX_REVIEW_ROUNDS + 1):
                            st.write(f"Reviewing plot (round {round_n}/{MAX_REVIEW_ROUNDS})…")
                            review = plotter.review_plot(b64, ai_plot_request.strip())
                            last_line = review.strip().splitlines()[-1].strip().upper() if review.strip() else ""
                            if "APPROVED" in last_line and "ISSUES" not in last_line:
                                st.write("✅ Plot approved.")
                                approved = True
                                break
                            issues_start = review.upper().find("ISSUES:")
                            feedback = review[issues_start:].strip() if issues_start != -1 else review.strip()
                            st.write(f"Issues found — refining… ({feedback[:120].strip()}…)")
                            refined_code = plotter.refine_code(
                                code, feedback, fig_base64=b64,
                                failed_attempts=failed_attempts or None,
                            )
                            figs2, b64_2, err2 = run_sandboxed_plot(refined_code, _data)
                            if err2:
                                st.write(f"⚠️ Refinement error: `{err2.splitlines()[-1]}`")
                                break
                            code, figs, b64 = refined_code, figs2, b64_2
                            st.session_state["ai_plot_code"] = code
                            failed_attempts.append(f"Round {round_n}: {feedback[:200].strip()}")

                        if not approved:
                            st.write("⚠️ Max review rounds reached — plot shown with known issues.")

                        st.session_state["ai_plot_figs"] = figs
                        st.session_state["ai_plot_fig_b64"] = b64
                        st.session_state["ai_plot_review"] = review
                        st.session_state["ai_plot_approved"] = approved
                        st.session_state["ai_plotter"] = plotter
                        st.session_state.pop("ai_plot_default_figs", None)
                        status.update(
                            label="Plot ready ✅" if approved else "Plot shown (not fully approved) ⚠️",
                            state="complete" if approved else "error",
                        )

        if "ai_plot_default_figs" in st.session_state:
            for dfig in st.session_state["ai_plot_default_figs"]:
                st.pyplot(dfig)

        if "ai_plot_figs" in st.session_state:
            figs_to_show = st.session_state["ai_plot_figs"] or []
            for f in figs_to_show:
                st.pyplot(f)
            if figs_to_show and st.session_state.get("ai_plot_fig_b64"):
                st.download_button(
                    "Download first plot as PNG",
                    data=base64.b64decode(st.session_state["ai_plot_fig_b64"]),
                    file_name="ai_plot.png",
                    mime="image/png",
                )

        if "ai_plot_review" in st.session_state:
            review_text = st.session_state["ai_plot_review"]
            approved = st.session_state.get("ai_plot_approved", False)
            with st.expander("AI Review", expanded=not approved):
                if approved:
                    st.success("AI: Plot passed all checks.")
                else:
                    st.warning("Plot shown but AI review found unresolved issues:")
                    st.info(review_text[:800])

        if "ai_plot_code" in st.session_state:
            with st.expander("Generated code", expanded=False):
                st.code(st.session_state["ai_plot_code"], language="python")

        if "ai_plot_figs" in st.session_state:
            followup = st.text_input(
                "Ask for further changes",
                placeholder="e.g. make the bars wider, use green colors, add gridlines",
                key="ai_plot_followup",
            )
            if st.button("Apply Changes", key="ai_plot_followup_btn") and followup.strip():
                plotter = st.session_state.get("ai_plotter")
                if plotter:
                    current_code = st.session_state.get("ai_plot_code", "")
                    with st.spinner("Applying changes…"):
                        new_code = plotter.refine_code(current_code, followup.strip())
                    st.session_state["ai_plot_code"] = new_code
                    from tools.sandbox_plot import run_sandboxed_plot
                    ai_rows3 = st.session_state.get("ai_plot_rows") or [dict(r) for r in all_plot_rows]
                    df3 = pd.DataFrame(ai_rows3)
                    figs3, b64_3, err3 = run_sandboxed_plot(new_code, {"plot_rows": ai_rows3, "df": df3})
                    if err3:
                        st.error(err3)
                    else:
                        st.session_state["ai_plot_figs"] = figs3
                        st.session_state["ai_plot_fig_b64"] = b64_3
                        st.rerun()
