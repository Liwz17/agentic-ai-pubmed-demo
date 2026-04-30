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
每篇paper的primary endpoint
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

## Notes

- Full-text extraction via PMC is attempted first; falls back to abstract if unavailable. Abstract-only extraction is less reliable for multi-arm outcomes.
- In the web app, the **Upload PDFs** tab lets you supply a local PDF for any paper that only had abstract-level text, triggering a re-extraction pass.
- The `hybrid` PubMed mode makes additional LLM calls per trial (semantic term extraction) and is slower but improves recall for trials whose NCT ID does not appear in the paper.
