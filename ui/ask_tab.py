"""Ask tab: Q&A agent, chat tools, context building, and conversation trimming."""

import json
import streamlit as st
import pandas as pd

from tools.outcomes import (
    build_outcome_summary_table,
    draw_outcome_forest_plot,
    draw_outcome_bar_chart,
)

# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------

_CHAT_TEXT_TOTAL = 300_000      # total chars of paper text across all trials
_CHAT_TEXT_PER_TRIAL_MAX = 60_000  # per-trial ceiling
_CHAT_HISTORY_LIMIT = 40_000    # max chars of conversation history; oldest pairs dropped first

# ---------------------------------------------------------------------------
# Tool definitions
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
                    "use_ai_plot": {
                        "type": "boolean",
                        "description": (
                            "Set to true when the user explicitly requests AI-generated or custom plots "
                            "(says 'AI plot', 'custom chart', asks for specific colors/styles/layout, "
                            "or wants something standard forest/bar charts cannot produce). "
                            "Default false — uses fast hardcoded charts."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_ad_hoc",
            "description": (
                "Plot arbitrary data that you extract directly from the paper text. "
                "Use this whenever the user asks to visualise data that is NOT in the standard "
                "pre-extracted outcomes (e.g. patient demographics, baseline characteristics, "
                "safety events, subgroup counts, any table or paragraph in the paper). "
                "Extract ALL relevant numbers from the text yourself, then call this tool. "
                "Do NOT require a follow-up prompt — extract comprehensively in one pass."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Chart title describing what is being plotted.",
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Y-axis label, e.g. '% of patients', 'Count', 'Months'.",
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["grouped_bar", "horizontal_bar"],
                        "description": (
                            "grouped_bar: side-by-side bars per category, groups colour-coded — "
                            "best when comparing two arms across several metrics. "
                            "horizontal_bar: horizontal bars — best when there are many categories "
                            "with long names."
                        ),
                    },
                    "series": {
                        "type": "array",
                        "description": (
                            "One entry per (group, category) data point. "
                            "Extract ALL data points from the relevant text — do not omit any."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "group": {
                                    "type": "string",
                                    "description": "Arm or group name, e.g. 'NI arm', 'N-CT arm', 'Overall'.",
                                },
                                "category": {
                                    "type": "string",
                                    "description": "Category label shown on the axis, e.g. 'Male', 'Brain metastases', 'PD-L1 ≥1%'.",
                                },
                                "value": {
                                    "type": "number",
                                    "description": "Numeric value (e.g. 84.4 for 84.4%).",
                                },
                                "unit": {
                                    "type": "string",
                                    "description": "Unit string appended to the value label, e.g. '%', ' pts'. Omit if dimensionless.",
                                },
                            },
                            "required": ["group", "category", "value"],
                        },
                    },
                    "use_ai_plot": {
                        "type": "boolean",
                        "description": (
                            "Set to true when the user explicitly requests AI-generated or custom plots "
                            "(says 'AI plot', 'custom chart', wants specific colors/styles/layout). "
                            "Default false — uses fast hardcoded charts."
                        ),
                    },
                },
                "required": ["title", "series"],
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
# Tool execution helpers
# ---------------------------------------------------------------------------

def _execute_ai_plot(user_request: str, pr: dict, write_fn=None) -> list | None:
    """Run the AIPlotterAgent pipeline: generate → sandbox → one review/refine round.

    write_fn: optional callable(str) used to stream progress (e.g. st.write).
    """
    from tools.ai_plotter import AIPlotterAgent
    from tools.sandbox_plot import run_sandboxed_plot

    def _w(msg):
        if write_fn:
            write_fn(msg)

    all_rows = pr.get("all_plot_rows", [])
    if not all_rows:
        return None

    ai_rows = [dict(r) for r in all_rows]
    plotter = AIPlotterAgent(model="openai/gpt-4o")

    _w("Generating code…")
    code = plotter.generate_code(ai_rows, user_request)

    data = {"plot_rows": ai_rows, "df": pd.DataFrame(ai_rows)}
    figs, b64, err = None, None, None
    for attempt in range(1, 4):
        _w(f"Running code (attempt {attempt}/3)…")
        figs, b64, err = run_sandboxed_plot(code, data)
        if not err:
            break
        _w(f"⚠️ Error: `{err.splitlines()[-1]}` — fixing…")
        code = plotter.fix_code(code, err)

    if err or not figs:
        _w("❌ Could not render plot after retries.")
        return None

    _w("Reviewing plot…")
    review = plotter.review_plot(b64, user_request)
    last_line = review.strip().splitlines()[-1].strip().upper() if review.strip() else ""
    if "APPROVED" in last_line and "ISSUES" not in last_line:
        _w("✅ Plot approved.")
    else:
        issues_start = review.upper().find("ISSUES:")
        feedback = review[issues_start:].strip() if issues_start != -1 else review.strip()
        _w(f"Issues found — refining… ({feedback[:120].strip()}…)")
        refined_code = plotter.refine_code(code, feedback, fig_base64=b64)
        figs2, _, err2 = run_sandboxed_plot(refined_code, data)
        if not err2 and figs2:
            figs = figs2
            _w("✅ Refinement applied.")
        else:
            _w("⚠️ Refinement failed — showing original.")

    return figs or None


def _execute_ai_plot_adhoc(args: dict, write_fn=None) -> list | None:
    """AI-plot path for plot_ad_hoc: generates matplotlib code for agent-extracted series data."""
    import re as _re
    from tools.ai_plotter import AIPlotterAgent
    from tools.sandbox_plot import run_sandboxed_plot

    def _w(msg):
        if write_fn:
            write_fn(msg)

    series = args.get("series", [])
    title = args.get("title", "")
    y_label = args.get("y_label", "")
    chart_type = args.get("chart_type", "grouped_bar")

    if not series:
        return None

    groups = list(dict.fromkeys(s["group"] for s in series))
    categories = list(dict.fromkeys(s["category"] for s in series))
    series_literal = json.dumps(series, indent=2)

    prompt = f"""Generate matplotlib code to create a {chart_type.replace('_', ' ')} chart.

Title: {title!r}
Y-axis label: {y_label!r}

Data (available as `series` list in the sandbox):
{series_literal}

Each series item: {{"group": str, "category": str, "value": float, "unit": str}}.
Groups (colour-code by group): {groups}
Categories (axis): {categories}

ENVIRONMENT: NO imports. Pre-injected globals: plt, np, pd, re, textwrap, series (list of dicts above).

Rules:
- Colour-code bars by 'group'. Show categories on the axis.
- Add numeric value labels on/above each bar including unit if present.
- Add title, legend, axis labels. Call plt.tight_layout(pad=1.5).
- Handle missing (group, category) combinations gracefully (use 0 or skip).
- Set figs = [fig]; fig = figs[0] at the end.
- NEVER raise exceptions. Empty data → draw ax.text(0.5, 0.5, 'No data', transform=ax.transAxes).

Output ONLY a Python code block — no explanation, no imports.
```python
# no imports — plt, np, pd, re, textwrap, series are pre-injected
...
figs = [fig]
fig = figs[0]
```"""

    plotter = AIPlotterAgent(model="openai/gpt-4o")
    plotter._last_data_desc = f"series = {series_literal}"

    _w("Generating code…")
    resp = plotter._call_chat_model(user_message=prompt, temperature=0.2)
    content = resp.choices[0].message.content or ""
    match = _re.search(r"```python\s*(.*?)```", content, _re.DOTALL)
    code = match.group(1).strip() if match else content.strip()

    sandbox_data = {"series": series, "plot_rows": [], "df": pd.DataFrame()}
    figs, b64, err = None, None, None
    for attempt in range(1, 4):
        _w(f"Running code (attempt {attempt}/3)…")
        figs, b64, err = run_sandboxed_plot(code, sandbox_data)
        if not err:
            break
        _w(f"⚠️ Error: `{err.splitlines()[-1]}` — fixing…")
        code = plotter.fix_adhoc_code(code, err, series_literal=series_literal)

    if err or not figs:
        _w("❌ Could not render plot after retries.")
        return None

    _w("Reviewing plot…")
    review = plotter.review_plot(b64, title)
    last_line = review.strip().splitlines()[-1].strip().upper() if review.strip() else ""
    if "APPROVED" in last_line and "ISSUES" not in last_line:
        _w("✅ Plot approved.")
    else:
        issues_start = review.upper().find("ISSUES:")
        feedback = review[issues_start:].strip() if issues_start != -1 else review.strip()
        _w(f"Issues found — refining… ({feedback[:120].strip()}…)")
        refined_code = plotter.refine_code(code, feedback, fig_base64=b64)
        figs2, _, err2 = run_sandboxed_plot(refined_code, sandbox_data)
        if not err2 and figs2:
            figs = figs2
            _w("✅ Refinement applied.")
        else:
            _w("⚠️ Refinement failed — showing original.")

    return figs or None


def _execute_plot_outcome(args: dict, pr: dict, user_request: str = "") -> list | None:
    if args.get("use_ai_plot"):
        return _execute_ai_plot(user_request or "Plot the extracted outcomes.", pr)

    outcome_key = args.get("outcome_key") or None
    trial_ids = args.get("trial_ids") or None
    arm_names = args.get("arm_names") or None
    plot_type_override = args.get("plot_type") or "auto"

    all_rows = pr.get("all_plot_rows", [])
    specs = pr.get("specs", [])

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

    keys_present = list(dict.fromkeys(r["outcome_key"] for r in rows))
    spec_map = {s.key: s for s in specs}

    figs = []
    for key in keys_present:
        key_rows = [r for r in rows if r["outcome_key"] == key]
        spec = spec_map.get(key)
        if spec is None:
            continue
        if plot_type_override == "forest" or (plot_type_override == "auto" and spec.plot_type == "forest"):
            fig, _ = draw_outcome_forest_plot(key_rows, spec, show=False)
        elif plot_type_override == "bar" or (plot_type_override == "auto" and spec.plot_type == "bar"):
            fig, _ = draw_outcome_bar_chart(key_rows, spec, show=False)
        else:
            fig = None
        if fig:
            figs.append(fig)

    return figs if figs else None


def _execute_plot_ad_hoc(args: dict) -> list | None:
    """Render arbitrary structured data provided directly by the agent."""
    import matplotlib.pyplot as plt
    import numpy as np

    title = args.get("title", "")
    y_label = args.get("y_label", "")
    chart_type = args.get("chart_type", "grouped_bar")
    series = args.get("series", [])

    if not series:
        return None

    groups = list(dict.fromkeys(s["group"] for s in series))
    categories = list(dict.fromkeys(s["category"] for s in series))
    value_map = {
        (s["group"], s["category"]): (float(s["value"]), s.get("unit", ""))
        for s in series
    }
    colors = [f"C{i}" for i in range(len(groups))]

    if chart_type == "horizontal_bar":
        gap = 0.3
        block = len(groups) * gap + 0.3
        fig, ax = plt.subplots(figsize=(9, max(4, len(categories) * block + 1)))
        for ci, cat in enumerate(categories):
            for gi, grp in enumerate(groups):
                val, unit = value_map.get((grp, cat), (0.0, ""))
                y = ci * block + gi * gap
                ax.barh(y, val, gap * 0.85, color=colors[gi],
                        label=grp if ci == 0 else "")
                ax.text(val + max(val * 0.01, 0.3), y, f"{val}{unit}",
                        va="center", fontsize=8)
        center_ys = [ci * block + (len(groups) - 1) * gap / 2
                     for ci in range(len(categories))]
        ax.set_yticks(center_ys)
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlabel(y_label or "Value")
    else:  # grouped_bar (default)
        x = np.arange(len(categories))
        w = 0.7 / max(len(groups), 1)
        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.4 + 2), 5))
        for gi, grp in enumerate(groups):
            vals = [value_map.get((grp, cat), (0.0, ""))[0] for cat in categories]
            bars = ax.bar(x + (gi - len(groups) / 2 + 0.5) * w, vals,
                          w * 0.9, color=colors[gi], label=grp)
            for bar, cat, v in zip(bars, categories, vals):
                _, unit = value_map.get((grp, cat), (0.0, ""))
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(max(vals) * 0.01, 0.3),
                        f"{v}{unit}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(y_label or "Value")

    ax.set_title(title)
    if groups:
        ax.legend()
    plt.tight_layout(pad=1.5)
    return [fig]


def _execute_show_table(args: dict, pr: dict):
    trial_ids = args.get("trial_ids") or None
    outcome_keys = args.get("outcome_keys") or None

    per_trial_results = pr.get("per_trial_results", [])
    df = build_outcome_summary_table(per_trial_results)

    if df is None or df.empty:
        return df

    if trial_ids:
        tids_lower = [t.lower() for t in trial_ids]
        nct_col = next(
            (c for c in df.columns if "nct" in c.lower() or "trial" in c.lower()), None
        )
        if nct_col:
            df = df[df[nct_col].str.lower().isin(tids_lower)]

    if outcome_keys:
        id_cols = [c for c in df.columns if not any(ok in c.lower() for ok in outcome_keys)]
        outcome_cols = [c for c in df.columns if any(ok in c.lower() for ok in outcome_keys)]
        keep = id_cols[:2] + outcome_cols
        df = df[[c for c in keep if c in df.columns]]

    return df

# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _trim_chat_history(history: list, max_chars: int) -> list:
    """Drop oldest user/assistant pairs until history fits within max_chars."""
    total = sum(len(m.get("content") or "") for m in history)
    if total <= max_chars:
        return history
    trimmed = list(history)
    while trimmed and total > max_chars:
        dropped = trimmed.pop(0)
        total -= len(dropped.get("content") or "")
    return trimmed


def _build_chat_context(per_trial_results: list) -> str:
    lines = [
        "You are a meticulous clinical research assistant. You have the full paper text for each trial.",
        "",
        "════ COMPREHENSIVENESS — HIGHEST PRIORITY RULE ════",
        "Your default mode is EXHAUSTIVE extraction. When answering any question about a section,",
        "paragraph, table, or topic you MUST report every data point present — every number,",
        "percentage, p-value, confidence interval, subgroup, footnote, and comparison.",
        "NEVER decide that something is 'unimportant' and omit it.",
        "NEVER summarise into vague phrases like 'similar between arms' or 'no significant difference'",
        "— always give the exact numbers for each arm.",
        "NEVER stop partway through a list or table — if there are 12 adverse events, report all 12.",
        "If the user has to ask a follow-up to get more data from the same section, you have failed.",
        "",
        "════ FORMAT RULES ════",
        "* Use a markdown TABLE whenever data has multiple variables across multiple arms/groups.",
        "  Example: baseline characteristics → one row per variable, one column per arm.",
        "* Use bullet points only for single-variable lists.",
        "* After a table or list, add a brief plain-English summary if it aids interpretation.",
        "* Quote exact text from the paper to support every number you report.",
        "",
        "════ TOOL RULES ════",
        "You have three tools:",
        "- plot_outcome: plot pre-extracted pipeline outcomes (OS, PFS, ORR, etc.).",
        "  Set use_ai_plot=true if user says 'AI plot', 'custom chart', or requests a specific style.",
        "- plot_ad_hoc: plot ANY data from the paper text (demographics, safety, any table/paragraph).",
        "  Set use_ai_plot=true same as above.",
        "  When calling plot_ad_hoc, extract ALL data points from the relevant section in one pass.",
        "- show_table: show a summary table of pre-extracted pipeline outcomes.",
        "ONLY call a plot tool when the user explicitly requests a visualisation — words like 'plot',",
        "'chart', 'graph', 'visualize', 'draw'. Words like 'extract', 'tell me', 'summarize',",
        "'give me the info' mean TEXT reply only — apply the comprehensiveness rules above.",
        "",
    ]
    n = max(1, len(per_trial_results))
    per_trial_limit = min(_CHAT_TEXT_PER_TRIAL_MAX, _CHAT_TEXT_TOTAL // n)

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
            lines.append(source_text[:per_trial_limit])
            if len(source_text) > per_trial_limit:
                lines.append(f"[... text truncated at {per_trial_limit:,} chars ...]")
        else:
            lines.append("--- Paper text: not available ---")
        lines.append("")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Tab renderer
# ---------------------------------------------------------------------------

def render_ask_tab(tab, per_trial_results: list, pr: dict) -> None:
    with tab:
        st.caption(
            "Ask anything about the papers or extraction results. "
            "The agent reads the actual paper text retrieved during the pipeline run — "
            "e.g. \"Why was only one arm found for NCT123?\", "
            "\"What does the paper say about PFS in the control arm?\", "
            "\"Is there a subgroup analysis reported?\"  \n"
            "**Plots:** by default, standard forest/bar charts are used (fast). "
            "Say **'AI plot'** or describe a custom style to use the AI Plot engine instead "
            "(generates matplotlib code, slower but fully flexible)."
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

        for hist_msg in st.session_state["chat_history"]:
            with st.chat_message(hist_msg["role"]):
                st.markdown(hist_msg["content"])

        if question := st.chat_input("Ask about the results…"):
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state["chat_history"].append({"role": "user", "content": question})

            from base_agent import BaseAgent
            context = _build_chat_context(per_trial_results)
            _chat_agent = BaseAgent(system_prompt=context, model="openai/gpt-4o")
            prior = _trim_chat_history(st.session_state["chat_history"][:-1], _CHAT_HISTORY_LIMIT)
            for hist_msg in prior:
                if hist_msg["role"] in ("user", "assistant"):
                    _chat_agent.add_message(hist_msg["role"], hist_msg["content"])

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    resp = _chat_agent._call_chat_model(
                        user_message=question,
                        temperature=0.2,
                        tools=CHAT_TOOLS,
                        tool_choice="auto",
                    )
                    llm_msg = resp.choices[0].message

                reply_parts: list[str] = []
                if llm_msg.tool_calls:
                    for tc in llm_msg.tool_calls:
                        fn = tc.function.name
                        args = json.loads(tc.function.arguments)

                        if fn == "plot_outcome":
                            if args.get("use_ai_plot"):
                                with st.status("Generating AI plot…", expanded=True) as ai_status:
                                    figs = _execute_ai_plot(
                                        question, pr, write_fn=st.write
                                    )
                                    ai_status.update(
                                        label="AI plot ready ✅" if figs else "AI plot failed ❌",
                                        state="complete" if figs else "error",
                                    )
                                if figs:
                                    for fig in figs:
                                        st.pyplot(fig)
                                    st.caption("AI-generated chart — matplotlib code produced by the AI Plot engine, with review/refine pass.")
                                    outcome_label = args.get("outcome_key") or "all outcomes"
                                    reply_parts.append(f"Here is the AI-generated plot for {outcome_label}.")
                                else:
                                    reply_parts.append("Could not generate an AI plot for this request.")
                            else:
                                figs = _execute_plot_outcome(args, pr, user_request=question)
                                if figs:
                                    for fig in figs:
                                        st.pyplot(fig)
                                    st.caption("Standard chart — hardcoded forest/bar plot from pre-extracted pipeline outcomes.")
                                    outcome_label = args.get("outcome_key") or "all outcomes"
                                    reply_parts.append(f"Here is the plot for {outcome_label}.")
                                else:
                                    reply_parts.append("No plottable data matched your request.")

                        elif fn == "plot_ad_hoc":
                            if args.get("use_ai_plot"):
                                with st.status("Generating AI plot…", expanded=True) as ai_status:
                                    figs = _execute_ai_plot_adhoc(args, write_fn=st.write)
                                    ai_status.update(
                                        label="AI plot ready ✅" if figs else "AI plot failed ❌",
                                        state="complete" if figs else "error",
                                    )
                                if figs:
                                    for fig in figs:
                                        st.pyplot(fig)
                                    st.caption("AI-generated chart — matplotlib code produced by the AI Plot engine, with review/refine pass.")
                                    reply_parts.append(f"Here is the AI-generated plot: {args.get('title', '')}.")
                                else:
                                    reply_parts.append("Could not generate an AI plot for this request.")
                            else:
                                figs = _execute_plot_ad_hoc(args)
                                if figs:
                                    for fig in figs:
                                        st.pyplot(fig)
                                    st.caption("Standard chart — data extracted from paper text by the agent, rendered with hardcoded matplotlib.")
                                    reply_parts.append(f"Here is the plot: {args.get('title', '')}.")
                                else:
                                    reply_parts.append("Could not render the plot — no data provided.")

                        elif fn == "show_table":
                            df_chat = _execute_show_table(args, pr)
                            if df_chat is not None and not df_chat.empty:
                                st.dataframe(df_chat, use_container_width=True)
                                reply_parts.append("Here is the summary table.")
                            else:
                                reply_parts.append("No data matched your request.")

                        else:
                            reply_parts.append(f"Unknown tool: {fn}")
                else:
                    reply_parts.append(llm_msg.content or "")

                reply = "  \n".join(reply_parts)
                st.markdown(reply)

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
