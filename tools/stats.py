from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import textwrap

def wrap_text(text, width=36, max_lines=None):
    if not text or not str(text).strip():
        return "Unknown"
    
    wrapped = textwrap.wrap(str(text).strip(), width=width)
    
    if max_lines is not None and len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        if len(wrapped[-1]) >= 3:
            wrapped[-1] = wrapped[-1][:-3] + "..."
        else:
            wrapped[-1] = wrapped[-1] + "..."
    
    return "\n".join(wrapped)

def summarize_papers_stats(papers):
    n_papers = len(papers)

    journal_counter = Counter()
    keyword_counter = Counter()
    missing_abstract_count = 0
    abstract_lengths = []

    for p in papers:
        journal = p.get("journal")
        if journal:
            journal_counter[journal] += 1

        keywords = p.get("keywords") or []
        for kw in keywords:
            if kw:
                keyword_counter[kw] += 1

        abstract = p.get("abstract")
        if not abstract or abstract == "No abstract available":
            missing_abstract_count += 1
        else:
            abstract_lengths.append(len(abstract))

    avg_abstract_length = (
        sum(abstract_lengths) / len(abstract_lengths)
        if abstract_lengths else 0
    )

    return {
        "n_papers": n_papers,
        "top_journals": journal_counter.most_common(5),
        "top_keywords": keyword_counter.most_common(10),
        "missing_abstract_count": missing_abstract_count,
        "avg_abstract_length": round(avg_abstract_length, 2),
    }

def summarize_trial_linking_results(df_links: pd.DataFrame) -> dict:
    if df_links.empty:
        return {
            "n_trials": 0,
            "n_matched": 0,
            "n_possible": 0,
            "n_unmatched": 0,
            "match_rate": 0.0,
        }

    n_trials = len(df_links)
    n_matched = (df_links["match_status"] == "matched").sum()
    n_possible = (df_links["match_status"] == "possible_match").sum()
    n_unmatched = n_trials - n_matched - n_possible

    return {
        "n_trials": n_trials,
        "n_matched": int(n_matched),
        "n_possible": int(n_possible),
        "n_unmatched": int(n_unmatched),
        "match_rate": round(n_matched / n_trials, 3),
        "possible_rate": round(n_possible / n_trials, 3),
        "avg_candidates": round(df_links["n_candidates"].mean(), 2) if "n_candidates" in df_links.columns else None,
    }


import pandas as pd
import matplotlib.pyplot as plt


import textwrap
import pandas as pd
import matplotlib.pyplot as plt


def draw_single_trial_forest_plot(plot_rows, trial_label: str = None, show: bool = True):
    """
    Draw a forest-style plot for one selected trial only.

    Improvements
    ------------
    1. Wrap long trial title in the plot title
    2. Keep y-axis labels short (arm names only)
    3. Put sample size inside plot area, next to each line / estimate
    4. Keep dashed segments for open-ended / missing CI
    """

    if not plot_rows:
        print("No plot rows available for this trial.")
        return

    df = pd.DataFrame(plot_rows).copy()

    if df.empty:
        print("No plot rows available for this trial.")
        return

    # keep only rows with an estimable median OS
    if "plot_eligible" in df.columns:
        df = df[df["plot_eligible"] == True].copy()

    if df.empty:
        print("No eligible treatment arms to plot for this trial.")
        return

    # infer trial label
    if trial_label is not None:
        inferred_trial_label = trial_label
    elif "trial_label" in df.columns and df["trial_label"].notna().any():
        inferred_trial_label = df["trial_label"].dropna().iloc[0]
    else:
        inferred_trial_label = "Selected Trial"

    # wrap long title
    def wrap_trial_label(text, width=48):
        if not text or not str(text).strip():
            return "Selected Trial"
        return "\n".join(textwrap.wrap(str(text).strip(), width=width))

    wrapped_trial_label = wrap_trial_label(inferred_trial_label, width=48)

    # sample size column detection
    n_col = None
    for candidate in ["sample_size", "n", "arm_n", "group_n"]:
        if candidate in df.columns:
            n_col = candidate
            break

    if n_col is not None:
        df[n_col] = pd.to_numeric(df[n_col], errors="coerce")

    # unit handling
    units = []
    if "median_os_unit" in df.columns:
        units = [u for u in df["median_os_unit"].dropna().unique() if str(u).strip() != ""]

    if len(units) == 1:
        x_unit = units[0]
    elif len(units) == 0:
        x_unit = ""
    else:
        x_unit = "mixed units"
        print(f"Warning: multiple units found within this trial: {units}")

    # sort by estimate
    if "median_os_value" in df.columns:
        df["median_os_value"] = pd.to_numeric(df["median_os_value"], errors="coerce")
        df["ci_lower"] = pd.to_numeric(df["ci_lower"], errors="coerce")
        df["ci_upper"] = pd.to_numeric(df["ci_upper"], errors="coerce")
        df = df.sort_values(by="median_os_value", ascending=True).reset_index(drop=True)

    # short labels: arm names only
    if "arm_name" in df.columns:
        labels = df["arm_name"].fillna("unknown arm").astype(str).tolist()
    else:
        labels = [f"arm_{i+1}" for i in range(len(df))]

    estimates = df["median_os_value"].tolist()
    lowers = df["ci_lower"].tolist() if "ci_lower" in df.columns else [None] * len(df)
    uppers = df["ci_upper"].tolist() if "ci_upper" in df.columns else [None] * len(df)
    sample_sizes = df[n_col].tolist() if n_col is not None else [None] * len(df)

    y_pos = list(range(len(df)))

    # compute range for dashed extension + text room
    finite_vals = []
    for x in estimates + lowers + uppers:
        if x is not None and not pd.isna(x):
            finite_vals.append(x)

    if finite_vals:
        x_min = min(finite_vals)
        x_max = max(finite_vals)
        x_range = max(x_max - x_min, 1.0)
    else:
        x_min, x_max, x_range = 0.0, 5.0, 5.0

    open_ci_extension = 0.08 * x_range
    right_text_offset = 0.03 * x_range
    left_pad = 0.02 * x_range
    right_pad = 0.30 * x_range  # room for n text + CI notes

    plt.figure(figsize=(9, max(3.5, 0.85 * len(df) + 2)))

    for i, (est, lo, hi, n_val) in enumerate(zip(estimates, lowers, uppers, sample_sizes)):
        # point estimate
        plt.plot(est, i, "o")

        ci_note_x = None

        if pd.notna(lo) and pd.notna(hi):
            # full CI
            plt.hlines(i, lo, hi)
            anchor_x = hi

        elif pd.notna(lo) and pd.isna(hi):
            # upper missing -> dashed to right
            plt.hlines(i, lo, est)
            plt.hlines(i, est, est + open_ci_extension, linestyles="dashed")
            anchor_x = est + open_ci_extension
            ci_note_x = anchor_x + right_text_offset
            plt.text(ci_note_x, i, "upper NR", va="center", fontsize=8)

        elif pd.isna(lo) and pd.notna(hi):
            # lower missing -> dashed to left
            plt.hlines(i, est, hi)
            plt.hlines(i, est - open_ci_extension, est, linestyles="dashed")
            anchor_x = hi
            plt.text(est - open_ci_extension - right_text_offset, i,
                     "lower missing", va="center", ha="right", fontsize=8)

        else:
            # no CI
            plt.hlines(i, est - open_ci_extension / 2, est + open_ci_extension / 2, linestyles="dashed")
            anchor_x = est + open_ci_extension / 2
            ci_note_x = anchor_x + right_text_offset
            plt.text(ci_note_x, i, "no CI", va="center", fontsize=8)

        # sample size inside plot, to the right of line / point / CI note
        if pd.notna(n_val):
            try:
                n_text_x = anchor_x + right_text_offset
                if ci_note_x is not None:
                    n_text_x = ci_note_x + 0.11 * x_range
                plt.text(n_text_x, i, f"n={int(n_val)}", va="center", fontsize=9)
            except Exception:
                pass

    plt.yticks(y_pos, labels)

    if x_unit and x_unit != "mixed units":
        plt.xlabel(f"Median Overall Survival ({x_unit})")
    else:
        plt.xlabel("Median Overall Survival")

    plt.title(f"Forest Plot by Treatment Arm\n{wrapped_trial_label}")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.3)

    ax = plt.gca()
    ax.set_xlim(x_min - left_pad, x_max + right_pad)

    plt.tight_layout()
    fig = plt.gcf()
    if show:
        plt.show()
    return fig



def _normalize_time_unit(unit):
    """
    Normalize time unit text to 'months' or 'years' when possible.
    """
    if unit is None:
        return None

    u = str(unit).strip().lower()

    if u in {"month", "months", "mo", "mos"}:
        return "months"
    if u in {"year", "years", "yr", "yrs"}:
        return "years"

    return u if u else None


def _convert_time_to_months(value, unit):
    """
    Convert a numeric time value to months.
    If unit is already months or missing, return value unchanged.
    """
    if value is None:
        return None

    unit_norm = _normalize_time_unit(unit)

    if unit_norm == "years":
        return float(value) * 12.0

    # default: leave unchanged for months / unknown / None
    return float(value)


import textwrap
import pandas as pd
import matplotlib.pyplot as plt

def draw_multi_trial_forest_plot(all_plot_rows, show: bool = True):
    """
    Draw one combined forest-style plot across multiple selected trials.

    Hierarchy fix:
    --------------
    1. Add a dedicated header row for each trial
    2. Plot arm rows underneath each header
    3. Keep sample size and CI notes on arm rows only
    """
    if not all_plot_rows:
        print("No multi-trial plot rows available.")
        return

    df = pd.DataFrame(all_plot_rows).copy()

    if df.empty:
        print("No multi-trial plot rows available.")
        return

    # keep only rows eligible for plotting
    if "plot_eligible" in df.columns:
        df = df[df["plot_eligible"] == True].copy()

    if df.empty:
        print("No eligible rows available for combined plotting.")
        return

    # ensure numeric conversion
    df["median_os_value"] = pd.to_numeric(df["median_os_value"], errors="coerce")
    df["ci_lower"] = pd.to_numeric(df["ci_lower"], errors="coerce")
    df["ci_upper"] = pd.to_numeric(df["ci_upper"], errors="coerce")

    # detect sample size column
    n_col = None
    for candidate in ["sample_size", "n", "arm_n", "group_n"]:
        if candidate in df.columns:
            n_col = candidate
            break

    if n_col is not None:
        df[n_col] = pd.to_numeric(df[n_col], errors="coerce")

    # convert to months
    df["median_os_months"] = df.apply(
        lambda r: _convert_time_to_months(r["median_os_value"], r.get("median_os_unit")),
        axis=1
    )
    df["ci_lower_months"] = df.apply(
        lambda r: _convert_time_to_months(r["ci_lower"], r.get("ci_unit")),
        axis=1
    )
    df["ci_upper_months"] = df.apply(
        lambda r: _convert_time_to_months(r["ci_upper"], r.get("ci_unit")),
        axis=1
    )

    # keep rows with non-missing estimate
    df = df[df["median_os_months"].notna()].copy()

    if df.empty:
        print("No valid median OS values available after unit conversion.")
        return

    # fill missing trial labels if needed
    if "trial_label" not in df.columns:
        df["trial_label"] = "Unknown Trial"
    df["trial_label"] = df["trial_label"].fillna("Unknown Trial").astype(str)

    # sort for cleaner grouped display
    df = df.sort_values(["trial_label", "median_os_months"], ascending=[True, True]).reset_index(drop=True)

    # range for plotting and dashed open-ended CI
    finite_vals = []
    for col in ["median_os_months", "ci_lower_months", "ci_upper_months"]:
        finite_vals.extend([x for x in df[col].tolist() if pd.notna(x)])

    if finite_vals:
        x_min = min(finite_vals)
        x_max = max(finite_vals)
        x_range = max(x_max - x_min, 1.0)
    else:
        x_min, x_max, x_range = 0.0, 5.0, 5.0

    open_ci_extension = 0.08 * x_range
    right_text_offset = 0.03 * x_range
    left_pad = 0.02 * x_range
    right_pad = 0.28 * x_range

    def wrap_trial_label(text, width=42):
        if not text or not str(text).strip():
            return "Unknown Trial"
        return "\n".join(textwrap.wrap(str(text).strip(), width=width))

    # --------------------------------------------------
    # Build plotting rows WITH dedicated trial header rows
    # --------------------------------------------------
    plot_entries = []
    unique_trials = df["trial_label"].nunique()

    for trial_label, g in df.groupby("trial_label", sort=False):
        g = g.reset_index(drop=True)

        # dedicated header row
        plot_entries.append({
            "row_type": "header",
            "trial_label": trial_label,
            "display_label": wrap_text(trial_label, width=34, max_lines=3),
            "median_os_months": None,
            "ci_lower_months": None,
            "ci_upper_months": None,
            "sample_size": None,
        })

        # arm rows
        for _, row in g.iterrows():
            arm_name = row.get("arm_name", "unknown arm")
            plot_entries.append({
                "row_type": "arm",
                "trial_label": trial_label,
                "display_label": wrap_text(arm_name, width=32, max_lines=2),
                "median_os_months": row["median_os_months"],
                "ci_lower_months": row["ci_lower_months"],
                "ci_upper_months": row["ci_upper_months"],
                "sample_size": row[n_col] if n_col is not None else None,
            })

    n_rows = len(plot_entries)
    y_pos = list(range(n_rows))
    labels = [entry["display_label"] for entry in plot_entries]

    plt.figure(figsize=(13, max(6, 0.6 * n_rows + 2)))

    for i, entry in enumerate(plot_entries):
        row_type = entry["row_type"]

        # --------------------
        # Header row: no point
        # --------------------
        if row_type == "header":
            continue

        # --------------------
        # Arm row: plot estimate
        # --------------------
        est = entry["median_os_months"]
        lo = entry["ci_lower_months"]
        hi = entry["ci_upper_months"]
        n_val = entry.get("sample_size")

        # point estimate
        plt.plot(est, i, "o")

        # CI segment
        if pd.notna(lo) and pd.notna(hi):
            plt.hlines(i, lo, hi)

        elif pd.notna(lo) and pd.isna(hi):
            plt.hlines(i, lo, est)
            plt.hlines(i, est, est + open_ci_extension, linestyles="dashed")
            hi = est + open_ci_extension
            plt.text(hi + right_text_offset, i, "upper NR", va="center", fontsize=8)

        elif pd.isna(lo) and pd.notna(hi):
            plt.hlines(i, est, hi)
            plt.hlines(i, est - open_ci_extension, est, linestyles="dashed")
            plt.text(
                est - open_ci_extension - right_text_offset,
                i,
                "lower missing",
                va="center",
                ha="right",
                fontsize=8
            )

        else:
            plt.hlines(
                i,
                est - open_ci_extension / 2,
                est + open_ci_extension / 2,
                linestyles="dashed"
            )
            hi = est + open_ci_extension / 2
            plt.text(hi + right_text_offset, i, "no CI", va="center", fontsize=8)

        # sample size
        anchor_x = hi if pd.notna(hi) else est
        if pd.notna(n_val):
            try:
                plt.text(
                    anchor_x + right_text_offset,
                    i,
                    f"n={int(n_val)}",
                    va="center",
                    fontsize=9
                )
            except Exception:
                pass

    plt.yticks(y_pos, labels)
    plt.xlabel("Median Overall Survival (months)")
    plt.title("Forest Plot by Treatment Arm Across Selected Trials")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.3)

    ax = plt.gca()
    ax.set_xlim(x_min - left_pad, x_max + right_pad)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Make header rows visually stronger
    for tick_label, entry in zip(ax.get_yticklabels(), plot_entries):
        if entry["row_type"] == "header":
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(10)
        else:
            tick_label.set_fontsize(9)

    plt.tight_layout()
    fig = plt.gcf()
    if show:
        plt.show()
    return fig



def _to_float_or_none(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.lower() in {"nr", "not reached", "na", "n/a", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    u = str(unit).strip().lower()
    if u in {"month", "months", "mo", "mos"}:
        return "months"
    if u in {"year", "years", "yr", "yrs"}:
        return "years"
    return u if u else None


def build_plot_rows_from_extraction(
    extraction_result: Dict[str, Any],
    trial_id: Optional[str] = None,
    trial_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a survival extraction result into plot-ready rows (one per arm).
    Units are preserved exactly as extracted — no conversion.
    """
    rows: List[Dict[str, Any]] = []

    if not extraction_result.get("outcome_found", False):
        return rows

    paper_id = extraction_result.get("paper_id")
    source_used = extraction_result.get("source_used")

    for arm in extraction_result.get("arms", []):
        arm_name = arm.get("arm_name", "unknown arm")

        median_os_value = _to_float_or_none(arm.get("median_os_value"))
        median_os_unit = _normalize_unit(arm.get("median_os_unit"))
        ci_lower = _to_float_or_none(arm.get("ci_lower"))
        ci_upper = _to_float_or_none(arm.get("ci_upper"))
        ci_unit = _normalize_unit(arm.get("ci_unit")) or median_os_unit

        arm_n = arm.get("arm_sample_size")
        try:
            arm_n = None if arm_n in [None, "", "null", "None"] else int(float(arm_n))
        except Exception:
            arm_n = None

        rows.append({
            "trial_id": trial_id,
            "trial_label": trial_label,
            "paper_id": paper_id,
            "source_used": source_used,
            "arm_name": arm_name,
            "display_label": f"{trial_label} | {arm_name}" if trial_label else arm_name,
            "median_os_raw": arm.get("median_os_raw"),
            "ci_95_raw": arm.get("ci_95_raw"),
            "median_os_value": median_os_value,
            "median_os_unit": median_os_unit,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_unit": ci_unit,
            "sample_size": arm_n,
            "arm_sample_size": arm_n,
            "arm_sample_size_raw": arm.get("arm_sample_size_raw"),
            "plot_eligible": median_os_value is not None,
            "evidence": arm.get("evidence", ""),
        })

    return rows