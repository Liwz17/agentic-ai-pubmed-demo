from collections import Counter
import pandas as pd

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


def draw_single_trial_forest_plot(plot_rows, trial_label: str = None):
    """
    Draw a forest-style plot for one selected trial only.

    Parameters
    ----------
    plot_rows : list[dict]
        Typically from:
        survival_result["survival_extraction"]["plot_rows"]
    trial_label : str, optional
        Trial label to use in title. If None, infer from plot_rows.
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
    inferred_trial_label = None
    if trial_label is not None:
        inferred_trial_label = trial_label
    elif "trial_label" in df.columns and df["trial_label"].notna().any():
        inferred_trial_label = df["trial_label"].dropna().iloc[0]
    else:
        inferred_trial_label = "Selected Trial"

    # check units
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

    # sort by estimate so the plot looks cleaner
    if "median_os_value" in df.columns:
        df = df.sort_values(by="median_os_value", ascending=True).reset_index(drop=True)

    # prepare labels
    if "arm_name" in df.columns:
        labels = df["arm_name"].fillna("unknown arm").astype(str).tolist()
    else:
        labels = [f"arm_{i+1}" for i in range(len(df))]

    estimates = df["median_os_value"].tolist()
    lowers = df["ci_lower"].tolist() if "ci_lower" in df.columns else [None] * len(df)
    uppers = df["ci_upper"].tolist() if "ci_upper" in df.columns else [None] * len(df)

    y_pos = list(range(len(df)))

    plt.figure(figsize=(8, max(3, 0.8 * len(df) + 1.5)))

    for i, (est, lo, hi) in enumerate(zip(estimates, lowers, uppers)):
        # draw point
        plt.plot(est, i, "o")

        # draw CI if available
        if lo is not None and hi is not None:
            plt.hlines(i, lo, hi)
        elif lo is not None and hi is None:
            # only lower bound known
            plt.hlines(i, lo, est)
            plt.text(est, i, "  upper NR/missing", va="center", fontsize=8)
        elif lo is None and hi is not None:
            # only upper bound known
            plt.hlines(i, est, hi)
            plt.text(hi, i, "  lower missing", va="center", fontsize=8)
        else:
            # no CI available
            plt.text(est, i, "  no CI", va="center", fontsize=8)

    plt.yticks(y_pos, labels)

    if x_unit and x_unit != "mixed units":
        plt.xlabel(f"Median Overall Survival ({x_unit})")
    else:
        plt.xlabel("Median Overall Survival")

    plt.title(f"Forest Plot by Treatment Arm\n{inferred_trial_label}")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()