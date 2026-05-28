"""
Outcome type definitions, plot row building, and figure generation
for multi-outcome extraction.

Outcome specs are now produced by an LLM parse step (see PaperLinkAgent.parse_outcome_request)
rather than a hardcoded alias dict.
"""

from __future__ import annotations

import re
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
    OutcomeSpec(key="progression_free_survival", display="Progression-Free Survival (PFS)", plot_type="forest"),
    OutcomeSpec(key="objective_response_rate", display="Objective Response Rate (ORR)", plot_type="bar"),
]


# ---------------------------------------------------------------------------
# Extraction result → plot rows
# ---------------------------------------------------------------------------

def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nr", "not reached", "na", "n/a", "none", "null"}:
        return None
    # strip unit suffixes and % signs
    s = re.sub(r"\s*(months?|years?|mo|yrs?|%)\s*$", "", s, flags=re.IGNORECASE).strip()
    try:
        return float(s)
    except Exception:
        pass
    # Handle mixed ranges like "NR–19.4" or "6.3 to NR": extract any number present
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else None


def _to_int_or_none(x: Any) -> Optional[int]:
    """Parse arm sample size; handles ranges like '100–150' by taking first number."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nr", "na", "n/a", "none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None


def _to_percent_or_none(x: Any) -> Optional[float]:
    """Parse a percentage value from LLM output.

    Handles: "62.5", "62.5%", "62.5% (45/72)", "45/72", "0.625" (proportion).
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nr", "not reached", "na", "n/a", "none", "null"}:
        return None
    # strip parenthetical e.g. "62.5% (45/72)" → "62.5%"
    s = re.sub(r"\s*\(.*?\)", "", s).strip()
    # strip % sign
    s = s.replace("%", "").strip()
    # try plain float
    try:
        val = float(s)
        # if looks like a proportion (0 < val <= 1), convert to percentage
        if 0 < val <= 1:
            val = round(val * 100, 2)
        return val
    except Exception:
        pass
    # try fraction "N/M"
    m = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den > 0:
            return round(num / den * 100, 2)
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
        arm_n = _to_int_or_none(arm_n_raw)

        for spec in specs:
            data = arm.get(spec.key)
            if not data or not data.get("found"):
                continue

            unit = _normalize_unit(data.get("unit"))
            # If a forest-type outcome is reported as a rate (%), split it into a separate key
            # so months and % rows never share the same outcome_key (causes mixed-axis confusion)
            is_rate_format = spec.plot_type == "forest" and unit == "%"
            effective_plot_type = "bar" if is_rate_format else spec.plot_type
            effective_key = (spec.key + "_rate") if is_rate_format else spec.key
            effective_display = (spec.display + " (rate %)") if is_rate_format else spec.display
            parse = _to_percent_or_none if effective_plot_type == "bar" else _to_float_or_none
            value = parse(data.get("value"))
            ci_lower = parse(data.get("ci_lower"))
            ci_upper = parse(data.get("ci_upper"))
            ci_unit = _normalize_unit(data.get("ci_unit")) or unit

            rows.append({
                "trial_id": trial_id,
                "trial_label": trial_label,
                "paper_id": paper_id,
                "source_used": source_used,
                "arm_name": arm_name,
                "outcome_key": effective_key,
                "outcome_display": effective_display,
                "plot_type": effective_plot_type,
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
                "is_subgroup": False,
                "subgroup_name": None,
                "subgroup_of": None,
            })

        # Process pre-specified subgroups within this arm
        for sg in arm.get("subgroups") or []:
            sg_name = sg.get("subgroup_name", "Unknown subgroup")
            sg_n = _to_int_or_none(sg.get("subgroup_n"))

            for spec in specs:
                data = sg.get(spec.key)
                if not data or not data.get("found"):
                    continue

                unit = _normalize_unit(data.get("unit"))
                is_rate_format = spec.plot_type == "forest" and unit == "%"
                effective_plot_type = "bar" if is_rate_format else spec.plot_type
                effective_key = (spec.key + "_rate") if is_rate_format else spec.key
                effective_display = (spec.display + " (rate %)") if is_rate_format else spec.display
                parse = _to_percent_or_none if effective_plot_type == "bar" else _to_float_or_none
                value = parse(data.get("value"))
                ci_lower = parse(data.get("ci_lower"))
                ci_upper = parse(data.get("ci_upper"))
                ci_unit = _normalize_unit(data.get("ci_unit")) or unit

                rows.append({
                    "trial_id": trial_id,
                    "trial_label": trial_label,
                    "paper_id": paper_id,
                    "source_used": source_used,
                    "arm_name": f"{arm_name} › {sg_name}",
                    "outcome_key": effective_key,
                    "outcome_display": effective_display,
                    "plot_type": effective_plot_type,
                    "value": value,
                    "unit": unit,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_unit": ci_unit,
                    "value_raw": data.get("value_raw") or data.get("raw"),
                    "ci_raw": data.get("ci_raw"),
                    "evidence": data.get("evidence", ""),
                    "sample_size": sg_n,
                    "plot_eligible": value is not None,
                    "is_subgroup": True,
                    "subgroup_name": sg_name,
                    "subgroup_of": arm_name,
                })

    return rows


# ---------------------------------------------------------------------------
# Forest plot — one time-based outcome across trials
# ---------------------------------------------------------------------------

def draw_outcome_forest_plot(
    plot_rows: List[Dict[str, Any]],
    spec: OutcomeSpec,
    show: bool = True,
):
    """Returns (fig, unit_log). unit_log is a list of dicts with per-arm unit info."""
    rows = [r for r in plot_rows if r.get("outcome_key") == spec.key and r.get("plot_eligible")]
    if not rows:
        print(f"No eligible rows for {spec.display} forest plot.")
        return None, []

    df = pd.DataFrame(rows).copy()
    for col in ["value", "ci_lower", "ci_upper"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["value"].notna()].copy()
    if df.empty:
        return None, []

    if "trial_label" not in df.columns:
        df["trial_label"] = "Unknown Trial"
    df["trial_label"] = df["trial_label"].fillna("Unknown Trial").astype(str)

    # --- Unit normalization: convert years → months, skip everything else ---
    unit_log: List[Dict[str, Any]] = []
    converted_rows: List[Any] = []

    for _, row in df.iterrows():
        raw_unit = row.get("unit") or ""
        norm = _normalize_unit(raw_unit)
        orig_val = row["value"]
        orig_lo = row["ci_lower"] if pd.notna(row["ci_lower"]) else None
        orig_hi = row["ci_upper"] if pd.notna(row["ci_upper"]) else None

        if norm == "months" or not norm:
            conv_val, conv_lo, conv_hi = orig_val, orig_lo, orig_hi
            note = "no conversion" if norm == "months" else "unit unknown, plotted as-is"
            plotted_unit = "months"
        elif norm == "years":
            conv_val = orig_val * 12
            conv_lo = orig_lo * 12 if orig_lo is not None else None
            conv_hi = orig_hi * 12 if orig_hi is not None else None
            note = "× 12 (years → months)"
            plotted_unit = "months"
        else:
            conv_val = None
            note = f"skipped: unsupported unit ({raw_unit or 'unknown'})"
            plotted_unit = "—"
            print(f"  [unit warning] {spec.display} | {row.get('trial_label')} | "
                  f"{row.get('arm_name')}: {note}")

        unit_log.append({
            "outcome": spec.display,
            "trial": row.get("trial_label", ""),
            "arm": row.get("arm_name", ""),
            "original_value": f"{orig_val} {raw_unit or '?'}".strip(),
            "original_unit": raw_unit or "unknown",
            "plotted_value": f"{conv_val:.2f} months" if conv_val is not None else "—",
            "plotted_unit": plotted_unit,
            "note": note,
        })

        if conv_val is not None:
            r = row.copy()
            r["value"] = conv_val
            r["ci_lower"] = conv_lo
            r["ci_upper"] = conv_hi
            converted_rows.append(r)

    if not converted_rows:
        print(f"No plottable rows after unit normalization for {spec.display}.")
        return None, unit_log

    df = pd.DataFrame(converted_rows)
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
        ci_note = None
        anchor = est
        if pd.notna(lo) and pd.notna(hi):
            ax.hlines(i, lo, hi, color="steelblue")
            anchor = hi
        elif pd.notna(lo):
            ax.hlines(i, lo, est, color="steelblue")
            ax.hlines(i, est, est + open_ext, linestyles="dashed", color="steelblue")
            anchor = est + open_ext
            ci_note = "upper NR"
        elif pd.notna(hi):
            ax.hlines(i, est, hi, color="steelblue")
            ax.hlines(i, est - open_ext, est, linestyles="dashed", color="steelblue")
            anchor = hi
        else:
            ax.hlines(i, est - open_ext / 2, est + open_ext / 2, linestyles="dashed", color="steelblue")
            anchor = est + open_ext / 2
            ci_note = "no CI"
        # combine ci_note and n into one text to avoid overlap
        parts = []
        if ci_note:
            parts.append(ci_note)
        if pd.notna(n_val):
            try:
                parts.append(f"n={int(n_val)}")
            except Exception:
                pass
        if parts:
            ax.text(anchor + r_offset, i, "  ".join(parts), va="center", fontsize=8)

    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries])
    for tick, e in zip(ax.get_yticklabels(), entries):
        tick.set_fontweight("bold" if e["type"] == "header" else "normal")
        tick.set_fontsize(10 if e["type"] == "header" else 9)
    ax.set_xlabel(f"{spec.display} (months)")
    ax.set_title(f"Forest Plot — {spec.display}")
    ax.invert_yaxis()
    ax.set_xlim(x_min - 0.02 * x_range, x_max + 0.28 * x_range)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    if show:
        plt.show()
    return fig, unit_log


# ---------------------------------------------------------------------------
# Bar chart — one rate outcome across trials
# ---------------------------------------------------------------------------

def draw_outcome_bar_chart(
    plot_rows: List[Dict[str, Any]],
    spec: OutcomeSpec,
    show: bool = True,
):
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
        lambda r: f"{_wrap(r['trial_label'], 20, 1)}\n{_wrap(r['arm_name'], 20, 1)}", axis=1
    )

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    trials = df["trial_label"].unique()
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(trials)}
    colors = df["trial_label"].map(color_map).tolist()

    _, ax = plt.subplots(figsize=(max(6, 1.5 * len(df)), 6))
    x = range(len(df))
    bars = ax.bar(x, df["value"].tolist(), color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, df["value"].tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["bar_label"].tolist(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(f"{spec.display} (%)")
    ax.set_title(f"Bar Chart — {spec.display}")
    ax.set_ylim(0, min(115, df["value"].max() + 15))
    ax.grid(True, axis="y", alpha=0.3)
    legend_handles = [mpatches.Patch(color=color_map[t], label=_wrap(t, 40, 2)) for t in trials]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", framealpha=0.85)
    plt.tight_layout()
    fig = plt.gcf()
    if show:
        plt.show()
    return fig, []


# ---------------------------------------------------------------------------
# Dispatcher — draw all plots driven by plot_type in each row
# ---------------------------------------------------------------------------

def draw_all_outcome_plots(
    plot_rows: List[Dict[str, Any]],
    specs: List[OutcomeSpec],
    show: bool = True,
):
    """
    Route to forest/bar based on spec.plot_type.
    Returns (figs, unit_logs) where unit_logs is a list of dicts
    describing per-arm unit conversion for all forest-type outcomes.
    """
    spec_map = {s.key: s for s in specs}
    seen_keys: set = set()
    figs = []
    unit_logs: List[Dict[str, Any]] = []
    for row in plot_rows:
        key = row.get("outcome_key")
        if key in seen_keys or not row.get("plot_eligible"):
            continue
        seen_keys.add(key)
        spec = spec_map.get(key)
        if spec is None:
            continue
        if spec.plot_type == "forest":
            fig, log = draw_outcome_forest_plot(plot_rows, spec, show=show)
            unit_logs.extend(log)
        elif spec.plot_type == "bar":
            fig, _ = draw_outcome_bar_chart(plot_rows, spec, show=show)
        else:
            continue
        if fig is not None:
            figs.append(fig)
    return figs, unit_logs


# ---------------------------------------------------------------------------
# Spec collection — derive specs from plot_rows (used in auto_primary mode)
# ---------------------------------------------------------------------------

def collect_specs_from_plot_rows(plot_rows: List[Dict[str, Any]]) -> List[OutcomeSpec]:
    """
    Reconstruct a deduplicated list of OutcomeSpec objects from plot_rows.
    Works for both fixed-outcome and auto_primary modes, because plot_rows
    always carry outcome_key, outcome_display, and plot_type.
    """
    seen: Dict[str, OutcomeSpec] = {}
    for row in plot_rows:
        key = row.get("outcome_key")
        if key and key not in seen:
            seen[key] = OutcomeSpec(
                key=key,
                display=row.get("outcome_display", key),
                plot_type=row.get("plot_type", "table_only"),
            )
    return list(seen.values())


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_outcome_summary_table(per_trial_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per (trial, arm, outcome) for display and audit."""
    arm_meta_fields = {"arm_name", "arm_sample_size", "arm_sample_size_raw", "subgroups"}
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
                    "is_subgroup": False,
                    "subgroup_name": None,
                    "subgroup_n": None,
                    "outcome": key,
                    "value": data.get("value"),
                    "unit": data.get("unit"),
                    "ci_lower": data.get("ci_lower"),
                    "ci_upper": data.get("ci_upper"),
                    "evidence": data.get("evidence", ""),
                })
            # Process pre-specified subgroups
            for sg in arm.get("subgroups") or []:
                sg_name = sg.get("subgroup_name", "Unknown subgroup")
                sg_n = sg.get("subgroup_n")
                for key, data in sg.items():
                    if key in {"subgroup_name", "subgroup_n"} or not isinstance(data, dict):
                        continue
                    if not data.get("found"):
                        continue
                    rows.append({
                        "nct_id": nct_id,
                        "trial_title": trial_title,
                        "pubmed_id": paper_id,
                        "arm_name": arm_name,
                        "arm_n": arm_n,
                        "is_subgroup": True,
                        "subgroup_name": sg_name,
                        "subgroup_n": sg_n,
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
