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
