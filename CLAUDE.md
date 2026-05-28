# CLAUDE.md

## Project

Streamlit app for clinical trial literature review. Pipeline: search ClinicalTrials.gov → link trials to PubMed papers → extract arm-level outcomes → visualize.

## Architecture

- `app.py` — Streamlit frontend. All UI state lives in `st.session_state`.
- `base_agent.py` — Base class for all agents. Owns OpenAI client and message history. Pass extra API params via `**completion_kwargs` in `_call_chat_model`.
- `paper_link_agent.py` — Main agent: finds papers, judges matches, extracts outcomes. `find_papers_for_trial()` and `extract_for_trial()` are the split-step entry points.
- `prompts.py` — All prompt templates. No business logic here.
- `tools/` — Utility modules: PubMed queries, PMC text fetching, outcome plotting, PDF reading, sandbox exec, AI plotter.
- `config.py` — Model name and API settings (OpenRouter).

## Environment

```bash
conda activate agentic_env
streamlit run app.py
```

## Session State Keys

`trial_packet`, `trials_list`, `selected_indices`, `paper_finding_results`, `paper_agent_config`, `paper_results`, `inspection_results`, `figures`, `unit_logs`, `chat_history`, `ai_plot_*`

`_reset_downstream()` clears downstream keys — add any new keys there when extending the pipeline.

## AI Plot pipeline

`AIPlotterAgent` (`tools/ai_plotter.py`) orchestrates three sequential LLM calls per plot request:

1. **`summarize_labels()`** — Reads all unique `trial_label` and `arm_name` strings and returns `trial_label_map` / `arm_label_map` dicts with short (3–5 word) clinically meaningful aliases. These are injected into the codegen prompt as literal Python dict definitions the AI copies verbatim.
2. **`generate_code()`** — Produces the full matplotlib code. The codegen prompt contains strict skeleton patterns for bar charts and forest plots; both skeletons use `trial_label_map.get(t, t[:12])` and `arm_label_map.get(a, a)` on all axes and legend entries.
3. **`review_plot()` / `refine_code()` loop** — The reviewer checks the rendered PNG against a calibrated checklist (short alias labels are explicitly marked as intentional, not flagged as truncated). `refine_code` receives the accumulated `failed_attempts` list from prior rounds so the model avoids repeating the same fix.

### Key invariants to maintain

- `resolve_plot_rows()` in `tools/sandbox_plot.py` is the single authority on `resolved_value` and `plot_eligible`. It runs inside the sandbox after AI code executes, recalculates `plot_eligible`, and strips NaN. Never trust `plot_eligible` from the pipeline build time alone.
- `all_plot_rows` (built by `build_multi_outcome_plot_rows`) is the canonical data source for AI Plot. Do not rebuild plot rows from the Summary Table DataFrame — pandas `None→NaN` conversion is lossy.
- `refine_code()` resets `self.messages` every call (stateless). Context continuity across rounds is provided by the `failed_attempts` list injected into the prompt, not by conversation history.
- The sandbox does **no** label post-processing. `tight_layout(pad=1.5)` is the only sandbox-applied change. Label content, rotation, and font size are entirely owned by AI-generated code.

## Subgroup extraction

Arms in the extraction JSON may include a `subgroups` list: `[{subgroup_name, subgroup_n, ...outcome fields}]`. Both `build_multi_outcome_plot_rows` and `build_outcome_summary_table` process subgroups, adding `is_subgroup=True`, `subgroup_name`, and `subgroup_of` fields. The Summary Table shows subgroup rows indented when "Show subgroup rows" is checked.
