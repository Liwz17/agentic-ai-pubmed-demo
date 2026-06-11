# Clinical Trial Literature Agent

An agentic pipeline that searches ClinicalTrials.gov, links each trial to its published PubMed paper, and extracts arm-level efficacy outcomes for visualisation.

---

## Quick start

**Requirements:** Python 3.10+, an [OpenRouter](https://openrouter.ai) API key.

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Launch
streamlit run app.py
```

---

## How to use

### 1. Search for trials

Type a natural-language query in the search box, e.g.:

```
Phase 3 lung cancer pembrolizumab trials 2018–2022
EGFR-mutant NSCLC osimertinib phase 3
```

The agent parses your query, searches ClinicalTrials.gov, and returns matching trials. Use the sidebar to control how many trials are selected automatically (Auto mode) or browse and pick yourself (Manual mode).

### 2. Select papers

For each trial the pipeline searches PubMed by NCT ID. All candidate papers are shown — the AI pre-selects the best match. Review the AI's reasoning, then confirm or adjust the selection before extraction runs.

### 3. Extract outcomes

Choose what to extract:

| Input | Behaviour |
|---|---|
| `OS, PFS, ORR` | Extracts those specific outcomes from every paper |
| `overall survival and grade 3/4 AEs` | Free-text, the agent interprets it |
| `primary endpoint` | Auto-discovers each paper's own primary endpoint |

Click **Extract Outcomes**. The pipeline fetches full text from PMC where available, extracts arm-level values with confidence intervals, and runs a QC pass.

### 4. Explore results

Results appear across several tabs:

| Tab | What it shows |
|---|---|
| **Summary Table** | All extracted outcomes per trial and arm |
| **Plots** | Forest plots (time-to-event) and bar charts (rates) |
| **Custom Plot** | Filter by outcome, trial, or arm to build a specific chart |
| **AI Plot** | Describe any chart in plain text — the agent writes and runs the code |
| **QC Review** | Agent vs Inspector verdicts; tiebreaker when they disagree |
| **Upload PDFs** | Supply a local PDF for any paper with abstract-only text; re-extraction runs automatically |
| **Ask** | Chat with the agent about the extracted papers — supports plot and table tool calls |

### Tips

- **No paper found?** Try loosening or clearing the PubMed filters (sidebar), or upload a PDF manually in the Upload PDFs tab.
- **Abstract-only extraction?** PMC full text is attempted first. If the paper is not open-access, upload the PDF for a more complete extraction.
- **AI Plot:** describe what you want naturally — "bar chart of ORR by arm", "forest plot for OS only in NCT12345678". Say "AI plot" to use the AI engine instead of the default chart style.
- **Ask tab:** ask anything about the papers — "what were the grade 3/4 AEs?", "show me a table of PFS across all trials". The agent reads the full paper text.

---

## Sidebar configuration

| Setting | Description |
|---|---|
| Trial selection mode | Auto: agent picks top N. Manual: you browse and select. |
| Max trials (Auto) | How many trials to select automatically (1–20). |
| Max papers per query | How many PubMed candidates to retrieve per trial (3–20). |
| Filter trials by | Which ClinicalTrials.gov date field to apply when your query includes a year range. |
| Published from / to | Filters the **fallback** keyword search by paper publication date (not trial start date). Has no effect on the primary NCT ID search. |

---

## How the pipeline works

```
User query
  └─ TrialRetrievalAgent  →  ClinicalTrials.gov search  →  trial list
       └─ PaperLinkAgent   →  PubMed (NCT ID search, fallback if needed)
            └─ judge match  →  extract outcomes  →  quality check + retry
                 └─ InspectorAgent  →  independent re-judgment  →  tiebreaker if disagreed
```

**PubMed search:** Query A (NCT ID, always unfiltered) runs first. If it returns nothing, the pipeline tries two automatic fallbacks: drug + disease + phase, then drug only. Both fallbacks apply the PubMed filters you set in the sidebar.

**QC layer:** An independent `InspectorAgent` re-judges each paper match without seeing the primary agent's reasoning. If they disagree, a third LLM call resolves it. All verdicts are shown in the QC Review tab.

**Extraction quality:** After the first pass, a checker flags incomplete results (outcome not found despite eligible paper; only one arm found for a multi-arm trial type) and triggers a targeted retry. The retry is only adopted if it improves the result.

---

## Project structure

```
app.py                    Streamlit web interface (orchestration only)
conversation.py           CLI entry point
config.py                 API key, model, and global settings
base_agent.py             BaseAgent — shared LLM call + message history
trial_retrieval_agent.py  Searches and selects ClinicalTrials.gov trials
paper_link_agent.py       Links trials to papers, extracts outcomes
inspector.py              Independent QC agent
prompts.py                All LLM prompts
linker.py                 LLM utilities and query helpers (legacy)
ui/
  ask_tab.py              Ask tab: chat agent, tool calling, context budgeting
  ai_plot_tab.py          AI Plot tab: code generation, sandbox, review/refine loop
  pdf_tab.py              Upload PDFs tab: re-extraction from local PDFs
tools/
  pubmed_trials.py        ClinicalTrials.gov API + PubMed query builders
  pmc.py                  PMC full-text fetching
  pdf.py                  Local PDF text extraction (PyMuPDF)
  outcomes.py             OutcomeSpec, plot-row building, forest/bar plots
  sandbox_plot.py         Restricted sandbox executor for AI-generated matplotlib code
  ai_plotter.py           AIPlotterAgent: label aliasing, codegen, review/refine
  stats.py                Statistical utilities
```

---

## Notes

- Full-text extraction via PMC is attempted first; abstract-only extraction is less reliable for multi-arm outcomes.
- The PubMed date and publication-type filters only affect the fallback keyword search. The primary NCT ID search always runs unfiltered — a valid paper is never excluded by date or type settings.
- The `†` symbol on a plot means the numeric value was `None` in the extraction output but was recovered from the raw evidence text by pattern matching. Worth cross-checking against the paper.

---

## Recent improvements

- **Paper selection step** — After finding papers, the UI shows all candidate papers per trial as a multiselect. The AI pre-selects the best match; the user can add, remove, or skip any paper before extraction runs. Each selected paper is extracted independently.
- **Multi-paper extraction** — Multiple papers from the same trial can be extracted in a single run, each producing its own result row.
- **Stronger arm completeness** — The extraction prompt explicitly requires all treatment arms (experimental + control + subgroups), not just the first one found.
- **AI Plot tab** — Describe a plot in plain text; the agent generates matplotlib code, executes it in a sandboxed environment, self-reviews the rendered image, and accepts follow-up refinements.
- **Ask tab tool calling** — The Q&A chat agent can call `plot_outcome` and `show_table` tools directly, rendering figures and dataframes in the chat.
- **Ask tab dynamic context budget** — Paper text is capped at 300K total chars (60K per trial max), scaled by the number of papers loaded. Conversation history is trimmed oldest-first when it exceeds 40K chars.
- **Ask tab ad-hoc plotting** — A `plot_ad_hoc` tool lets the chat agent plot arbitrary data read directly from the paper text (demographics, safety events, any table) without requiring pre-extraction. The agent extracts values itself and passes them as structured `{group, category, value, unit}` series.
- **Ask tab dual-mode plotting** — Plot requests in the Ask tab run in standard mode (fast hardcoded charts) by default. Saying "AI plot" or describing a custom style switches to the AI Plot engine with one review/refine round.
- **Ask tab exhaustive extraction** — The system prompt has hard constraints: report every data point in a queried section, use markdown tables for multi-variable data, quote exact paper text, never summarise with vague phrases.
- **PDF auto-reextraction** — Uploading a PDF triggers re-extraction immediately with no button click. The file signature `(name, size)` is tracked so extraction runs exactly once per file.
- **UI module extraction** — Ask, AI Plot, and Upload PDFs tabs moved into a dedicated `ui/` package; `app.py` is now orchestration-only.
- **Adaptive fallback search** — Hybrid mode (which ran extra LLM calls on every trial) replaced by a smarter fallback that only fires when the NCT ID search returns nothing.
- **Sidebar reorder** — PubMed filters moved above Outcomes for a more logical configuration flow.
- **Inspector tiebreaker** — When the main agent and Inspector disagree on paper match, a third LLM call resolves it. The verdict is shown in the QC tab with an ⚖️ header.
- **Extraction quality check + retry** — After the first extraction pass, a checker triggers a targeted second pass if the result looks incomplete. The retry is only adopted if it improves the result.
- **† dagger annotation** — Data points marked with † on a plot were recovered from raw evidence text by pattern matching rather than clean LLM extraction. A figure footnote explains this.
- **Date format normalisation** — PubMed date inputs accept both `YYYY/MM/DD` and `YYYY-MM-DD`; dashes are auto-converted.
- **Chat history cleared on new search** — Starting a new trial search clears the Ask tab conversation so stale exchanges from a prior run are not shown.

---

## Supplement — implementation notes

This section documents internal design decisions for contributors.

### AI Plot pipeline

`AIPlotterAgent` (`tools/ai_plotter.py`) runs three sequential LLM calls per request:

1. **`summarize_labels()`** — produces short (3–5 word) clinically meaningful aliases for every unique trial label and arm name. These are injected into the codegen prompt as literal Python dicts the AI copies verbatim.
2. **`generate_code()`** — produces full matplotlib code. The codegen prompt contains strict skeleton patterns for bar charts and forest plots; both use `trial_label_map.get(t, t[:12])` and `arm_label_map.get(a, a)` on all axes.
3. **`review_plot()` / `refine_code()` loop** — reviewer checks the rendered PNG against a calibrated checklist. Short alias labels are explicitly marked as intentional. `refine_code` receives the accumulated `failed_attempts` list so the model avoids repeating the same fix.

Key invariants:
- `resolve_plot_rows()` in `tools/sandbox_plot.py` is the single authority on `resolved_value` and `plot_eligible`. It recalculates `plot_eligible` and strips NaN inside the sandbox after AI code executes.
- `all_plot_rows` (from `build_multi_outcome_plot_rows`) is the canonical data source for AI Plot. Do not rebuild from the Summary Table DataFrame — pandas `None→NaN` conversion is lossy.
- `refine_code()` resets `self.messages` every call (stateless). Context continuity is provided by the `failed_attempts` list, not conversation history.

### Session state keys

`trial_packet`, `trials_list`, `selected_indices`, `paper_finding_results`, `paper_agent_config`, `paper_results`, `inspection_results`, `figures`, `unit_logs`, `chat_history`, `ai_plot_*`

`_reset_downstream()` clears all downstream keys when a new search starts.

### Subgroup extraction

Arms in the extraction JSON may include a `subgroups` list: `[{subgroup_name, subgroup_n, ...outcome fields}]`. Both `build_multi_outcome_plot_rows` and `build_outcome_summary_table` process subgroups, adding `is_subgroup=True`, `subgroup_name`, and `subgroup_of`. The Summary Table shows subgroup rows indented when "Show subgroup rows" is checked.
