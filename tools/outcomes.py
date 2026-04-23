"""
Outcome type definitions, plot row building, and figure generation
for multi-outcome extraction.

Outcome specs are now produced by an LLM parse step (see PaperLinkAgent.parse_outcome_request)
rather than a hardcoded alias dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

# ---------------------------------------------------------------------------
# OutcomeSpec — the unit of outcome description throughout the pipeline
# ---------------------------------------------------------------------------

@dataclass
class OutcomeSpec:
    key: str        # snake_case identifier, e.g. "overall_survival"
    display: str    # human-readable label, e.g. "Overall Survival (OS)"
    plot_type: str  # "forest" | "bar" | "table_only"


DEFAULT_OUTCOME_SPECS: List[OutcomeSpec] = [
    OutcomeSpec(key="overall_survival", display="Overall Survival (OS)", plot_type="forest"),
]

# kept for the legacy OS-only extraction path in build_plot_rows_from_extraction (stats.py)
TIME_BASED_OUTCOMES = {"overall_survival", "progression_free_survival", "duration_of_response"}
RATE_OUTCOMES = {"objective_response_rate", "disease_control_rate", "complete_response_rate"}


# ---------------------------------------------------------------------------
# Extraction result → plot rows
# ---------------------------------------------------------------------------

def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nr", "not reached", "na", "n/a", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _normalize_unit(unit: Any) -> Optional[str]:
    if unit is None:
        return None
    u = str(unit).strip().lower()
    if u in {"month", "months", "mo", "mos"}:
        return "months"
    if u in {"year", "years", "yr", "yrs"}:
        return "years"
    if u in {"%", "percent", "percentage"}:
        return "%"
    return u if u else None


def build_multi_outcome_plot_rows(
    extraction: Dict[str, Any],
    specs: List[OutcomeSpec],
    trial_id: Optional[str] = None,
    trial_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a multi-outcome extraction result into plot-ready rows.
    One row per (arm, outcome spec) combination that has a valid value.
    Uses specs — not hardcoded sets — to determine which outcomes to look for.
    """
    rows: List[Dict[str, Any]] = []
    paper_id = extraction.get("paper_id")
    source_used = extraction.get("source_used")

    for arm in extraction.get("arms", []):
        arm_name = arm.get("arm_name", "unknown arm")
        arm_n_raw = arm.get("arm_sample_size")
        try:
            arm_n = None if arm_n_raw in [None, "", "null", "None"] else int(float(arm_n_raw))
        except Exception:
            arm_n = None

        for spec in specs:
            data = arm.get(spec.key)
            if not data or not data.get("found"):
                continue

            value = _to_float_or_none(data.get("value"))
            unit = _normalize_unit(data.get("unit"))
            ci_lower = _to_float_or_none(data.get("ci_lower"))
            ci_upper = _to_float_or_none(data.get("ci_upper"))
            ci_unit = _normalize_unit(data.get("ci_unit")) or unit

            rows.append({
                "trial_id": trial_id,
                "trial_label": trial_label,
                "paper_id": paper_id,
                "source_used": source_used,
                "arm_name": arm_name,
                "outcome_key": spec.key,
                "outcome_display": spec.display,
                "plot_type": spec.plot_type,
                "value": value,
                "unit": unit,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_unit": ci_unit,
                "value_raw": data.get("value_raw") or data.get("raw"),
                "ci_raw": data.get("ci_raw"),
                "evidence": data.get("evidence", ""),
                "sample_size": arm_n,
                "plot_eligible": value is not None,
            })

    return rows


# ---------------------------------------------------------------------------
# Forest plot — one time-based outcome across trials
# ---------------------------------------------------------------------------

def draw_outcome_forest_plot(
    plot_rows: List[Dict[str, Any]],
    spec: OutcomeSpec,
) -> None:
    rows = [r for r in plot_rows if r.get("outcome_key") == spec.key and r.get("plot_eligible")]
    if not rows:
        print(f"No eligible rows for {spec.display} forest plot.")
        return

    df = pd.DataFrame(rows).copy()
    for col in ["value", "ci_lower", "ci_upper"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["value"].notna()].copy()
    if df.empty:
        return

    if "trial_label" not in df.columns:
        df["trial_label"] = "Unknown Trial"
    df["trial_label"] = df["trial_label"].fillna("Unknown Trial").astype(str)
    df = df.sort_values(["trial_label", "value"], ascending=[True, True]).reset_index(drop=True)

    all_vals = [v for col in ["value", "ci_lower", "ci_upper"]
                for v in df[col].tolist() if pd.notna(v)]
    x_min = min(all_vals) if all_vals else 0.0
    x_max = max(all_vals) if all_vals else 10.0
    x_range = max(x_max - x_min, 1.0)
    open_ext = 0.08 * x_range
    r_offset = 0.03 * x_range

    entries = []
    for trial_label, g in df.groupby("trial_label", sort=False):
        entries.append({"type": "header", "label": _wrap(trial_label, 36, 3),
                        "value": None, "ci_lower": None, "ci_upper": None, "sample_size": None})
        for _, row in g.iterrows():
            entries.append({"type": "arm", "label": _wrap(row["arm_name"], 32, 2),
                            "value": row["value"], "ci_lower": row["ci_lower"],
                            "ci_upper": row["ci_upper"], "sample_size": row.get("sample_size")})

    n = len(entries)
    plt.figure(figsize=(13, max(6, 0.65 * n + 2)))
    ax = plt.gca()

    for i, e in enumerate(entries):
        if e["type"] == "header":
            continue
        est, lo, hi, n_val = e["value"], e["ci_lower"], e["ci_upper"], e["sample_size"]
        ax.plot(est, i, "o", color="steelblue")
        anchor = est
        if pd.notna(lo) and pd.notna(hi):
            ax.hlines(i, lo, hi, color="steelblue")
            anchor = hi
        elif pd.notna(lo):
            ax.hlines(i, lo, est, color="steelblue")
            ax.hlines(i, est, est + open_ext, linestyles="dashed", color="steelblue")
            anchor = est + open_ext
            ax.text(anchor + r_offset, i, "upper NR", va="center", fontsize=8)
        elif pd.notna(hi):
            ax.hlines(i, est, hi, color="steelblue")
            ax.hlines(i, est - open_ext, est, linestyles="dashed", color="steelblue")
            anchor = hi
        else:
            ax.hlines(i, est - open_ext / 2, est + open_ext / 2, linestyles="dashed", color="steelblue")
            anchor = est + open_ext / 2
            ax.text(anchor + r_offset, i, "no CI", va="center", fontsize=8)
        if pd.notna(n_val):
            try:
                ax.text(anchor + r_offset, i, f"  n={int(n_val)}", va="center", fontsize=9)
            except Exception:
                pass

    units = [r for r in df["unit"].dropna().unique() if r]
    unit_label = units[0] if len(units) == 1 else "mixed units"
    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries])
    for tick, e in zip(ax.get_yticklabels(), entries):
        tick.set_fontweight("bold" if e["type"] == "header" else "normal")
        tick.set_fontsize(10 if e["type"] == "header" else 9)
    ax.set_xlabel(f"{spec.display} ({unit_label})")
    ax.set_title(f"Forest Plot — {spec.display}")
    ax.invert_yaxis()
    ax.set_xlim(x_min - 0.02 * x_range, x_max + 0.28 * x_range)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Bar chart — one rate outcome across trials
# ---------------------------------------------------------------------------

def draw_outcome_bar_chart(
    plot_rows: List[Dict[str, Any]],
    spec: OutcomeSpec,
) -> None:
    rows = [r for r in plot_rows if r.get("outcome_key") == spec.key and r.get("plot_eligible")]
    if not rows:
        print(f"No eligible rows for {spec.display} bar chart.")
        return

    df = pd.DataFrame(rows).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()
    if df.empty:
        return

    if "trial_label" not in df.columns:
        df["trial_label"] = "Unknown Trial"
    df["trial_label"] = df["trial_label"].fillna("Unknown Trial").astype(str)
    df["bar_label"] = df.apply(
        lambda r: f"{_wrap(r['trial_label'], 24, 2)}\n{r['arm_name']}", axis=1
    )

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    trials = df["trial_label"].unique()
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(trials)}
    colors = df["trial_label"].map(color_map).tolist()

    _, ax = plt.subplots(figsize=(max(6, 1.2 * len(df)), 5))
    x = range(len(df))
    bars = ax.bar(x, df["value"].tolist(), color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, df["value"].tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["bar_label"].tolist(), ha="center", fontsize=8)
    ax.set_ylabel(f"{spec.display} (%)")
    ax.set_title(f"Bar Chart — {spec.display}")
    ax.set_ylim(0, min(115, df["value"].max() + 15))
    ax.grid(True, axis="y", alpha=0.3)
    legend_handles = [mpatches.Patch(color=color_map[t], label=t) for t in trials]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Dispatcher — draw all plots driven by plot_type in each row
# ---------------------------------------------------------------------------

def draw_all_outcome_plots(
    plot_rows: List[Dict[str, Any]],
    specs: List[OutcomeSpec],
) -> None:
    """Route to forest plot or bar chart based on spec.plot_type."""
    spec_map = {s.key: s for s in specs}
    seen_keys: set = set()
    for row in plot_rows:
        key = row.get("outcome_key")
        if key in seen_keys or not row.get("plot_eligible"):
            continue
        seen_keys.add(key)
        spec = spec_map.get(key)
        if spec is None:
            continue
        if spec.plot_type == "forest":
            draw_outcome_forest_plot(plot_rows, spec)
        elif spec.plot_type == "bar":
            draw_outcome_bar_chart(plot_rows, spec)
        # "table_only" → skip figures, data is in the summary table


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_outcome_summary_table(per_trial_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per (trial, arm, outcome) for display and audit."""
    arm_meta_fields = {"arm_name", "arm_sample_size", "arm_sample_size_raw"}
    rows = []
    for item in per_trial_results:
        trial = item.get("trial", {}) or {}
        survival_result = item.get("survival_result", {}) or {}
        extraction = survival_result.get("survival_extraction", {}) or {}

        nct_id = trial.get("nct_id")
        trial_title = trial.get("brief_title")
        paper_id = extraction.get("paper_id")

        for arm in extraction.get("arms", []):
            arm_name = arm.get("arm_name", "unknown arm")
            arm_n = arm.get("arm_sample_size")
            for key, data in arm.items():
                if key in arm_meta_fields or not isinstance(data, dict):
                    continue
                if not data.get("found"):
                    continue
                rows.append({
                    "nct_id": nct_id,
                    "trial_title": trial_title,
                    "pubmed_id": paper_id,
                    "arm_name": arm_name,
                    "arm_n": arm_n,
                    "outcome": key,
                    "value": data.get("value"),
                    "unit": data.get("unit"),
                    "ci_lower": data.get("ci_lower"),
                    "ci_upper": data.get("ci_upper"),
                    "evidence": data.get("evidence", ""),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["nct_id", "arm_name", "outcome"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 36, max_lines: int = 3) -> str:
    if not text or not str(text).strip():
        return "Unknown"
    wrapped = textwrap.wrap(str(text).strip(), width=width)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][:-3] + "..." if len(wrapped[-1]) >= 3 else wrapped[-1] + "..."
    return "\n".join(wrapped)
