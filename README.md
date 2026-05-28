# Clinical Trial Literature Agent

An agentic pipeline that searches ClinicalTrials.gov, links each trial to its published PubMed paper, and extracts arm-level efficacy outcomes for visualisation.

## What it does

1. **Trial retrieval** — Parses a natural-language query (e.g. "Phase 3 lung cancer pembrolizumab trials 2018–2022") into a ClinicalTrials.gov search and returns matching trials.
2. **Paper linking** — For each trial, searches PubMed using the NCT ID (and optionally LLM-generated semantic terms) to find the primary results paper.
3. **Eligibility screening** — Judges whether the linked paper is an interventional trial report with arm-level efficacy data (filters out case reports, reviews, etc.).
4. **Outcome extraction** — Extracts user-specified outcomes (e.g. OS, PFS, ORR) or auto-discovers each paper's primary endpoint, per treatment arm with confidence intervals.
5. **Visualisation** — Generates forest plots (time-to-event outcomes) and bar charts (rates) across trials.

---

## Setup

### Requirements

- Python 3.10 or later
- An [OpenRouter](https://openrouter.ai) API key (used to call the LLM)

### Install

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure API key

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

The model and endpoint are set in `config.py` (default: `openai/gpt-4o-mini` via OpenRouter).

---

## Running

### Web app (recommended)

```bash
streamlit run app.py
```

Opens a browser UI with step-by-step trial search, paper linking, outcome extraction, QC review, and an interactive Q&A tab.

### Command line

```bash
python conversation.py
```

Runs the full pipeline interactively in the terminal. Prompts for search query, PubMed mode, outcomes, and optional date filters.

---

## Key features

### PubMed search modes

| Mode | Queries run | Use when |
|---|---|---|
| `nct_only` (default) | NCT ID only | Fast, precise — trial has a known NCT ID |
| `hybrid` | NCT ID + LLM semantic terms + structured fields | Trial ID not in paper, or low recall |

### Outcome extraction modes

**Fixed outcomes** — specify what you want:
```
OS, PFS, ORR
overall survival and grade 3/4 adverse events
```

**Auto-primary** — let the agent discover each paper's own primary endpoint:
```
primary endpoint
every paper's primary endpoint
```

In auto-primary mode, the agent first looks for explicit statements ("the primary endpoint was…") and falls back to inferring from the paper's structure if none are found. Results in the Summary Table are labelled 🔵 explicit or 🟡 inferred.

### Eligibility screening

Papers are screened before extraction. A paper is eligible only if it is:
1. An interventional clinical trial (phase 1–4, RCT, single-arm trial, etc.)
2. Reporting arm-level aggregate efficacy outcomes (not patient-by-patient narratives)

Case reports, case series, reviews, and biomarker studies are excluded.

### QC layer

An independent `InspectorAgent` re-judges each paper match and eligibility without seeing the primary agent's reasoning, providing a second opinion displayed in the QC tab.

---

## Project structure

```
app.py                  Streamlit web interface
conversation.py         CLI entry point and ConversationCoordinator
config.py               API key, model, and global settings
base_agent.py           BaseAgent — shared LLM call + message history logic
trial_retrieval_agent.py  Searches and selects ClinicalTrials.gov trials
paper_link_agent.py     Links trials to papers, extracts outcomes
inspector.py            Independent QC agent
prompts.py              All LLM prompts (system + task)
linker.py               PubMed query deduplication utilities
tools/
  pubmed_trials.py      ClinicalTrials.gov API + PubMed query builders
  pmc.py                PMC full-text fetching
  pdf.py                Local PDF text extraction (PyMuPDF)
  outcomes.py           OutcomeSpec, plot-row building, forest/bar plots
  stats.py              Legacy statistical utilities
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key from openrouter.ai |

---

## Recent improvements

- **Paper selection step** — After finding papers, the UI now shows all candidate papers per trial as a multiselect. The AI pre-selects the best match; the user can add, remove, or skip any paper before extraction runs. Each selected paper is extracted independently.
- **Multi-paper extraction** — Multiple papers from the same trial can be extracted in a single run, each producing its own result row.
- **Stronger arm completeness** — The extraction prompt explicitly requires all treatment arms (experimental + control + subgroups), not just the first one found.
- **AI Plot tab** — A new tab lets you describe a plot in plain text; the agent generates matplotlib code, executes it in a sandboxed environment, self-reviews the rendered image, and accepts follow-up refinements.
- **Ask tab tool calling** — The Q&A chat agent can now call `plot_outcome` and `show_table` tools directly, rendering figures and dataframes in the chat when the user asks for visuals.
- **Sidebar reorder** — PubMed filters moved above Outcomes in the sidebar for a more logical configuration flow.

### AI Plot reliability overhaul

- **Subgroup extraction** — Arm-level JSON now supports a `subgroups` array. `build_multi_outcome_plot_rows` and `build_outcome_summary_table` both process subgroup entries, displaying them indented under their parent arm in the Summary Table (toggle with "Show subgroup rows").
- **Resilient JSON parsing** — `extract_outcomes_from_text` no longer raises on malformed LLM output; a `_try_parse_json` fallback recovers the largest `{...}` block before gracefully returning an empty result, so one bad paper does not abort the batch.
- **NaN-safe data pipeline** — `resolve_plot_rows` in the sandbox explicitly converts `float('nan')` to `None` at three levels (_to_float, post-conversion cleanup, and `plot_eligible` recalculation), eliminating invisible bars caused by pandas converting `None` to `numpy.nan`.
- **AI Plot uses normalised pipeline rows directly** — The AI Plot tab now passes `all_plot_rows` (already normalised by `build_multi_outcome_plot_rows`) straight to the sandbox instead of rebuilding from the Summary Table DataFrame, removing a lossy conversion step that introduced NaN.
- **ZeroDivisionError guard** — The bar-chart code skeleton in the codegen prompt includes an explicit `if not arms or not trials` guard before `w = 0.7 / len(arms)`, and the refine loop no longer suggests raising exceptions as a fix.
- **Failed-attempt memory in refine loop** — Each review-refine round appends the reviewer's unresolved feedback to a `failed_attempts` list. The next `refine_code` call receives this list as context so the model knows which approaches have already been tried and must try something different.
- **Meaningful short display names** — Before generating code, `AIPlotterAgent.summarize_labels()` makes one dedicated LLM call to produce short (3–5 word), clinically meaningful aliases for every unique trial label and arm name. These aliases are injected into the codegen prompt as literal `trial_label_map` / `arm_label_map` dicts that the AI copies verbatim into its code, ensuring all axes and legend entries use readable names rather than truncated paper titles.
- **Calibrated reviewer** — The review prompt now explicitly tells the reviewer that all axis labels are pre-computed aliases; short labels should be treated as intentional, not flagged as truncated. Only text physically clipped by the figure border counts as a failure.

---

## Notes

- Full-text extraction via PMC is attempted first; falls back to abstract if unavailable. Abstract-only extraction is less reliable for multi-arm outcomes.
- In the web app, the **Upload PDFs** tab lets you supply a local PDF for any paper that only had abstract-level text, triggering a re-extraction pass.
- The `hybrid` PubMed mode makes additional LLM calls per trial (semantic term extraction) and is slower but improves recall for trials whose NCT ID does not appear in the paper.
