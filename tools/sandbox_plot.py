"""Sandboxed execution of AI-generated matplotlib plotting code."""
import io
import base64
import traceback
from typing import Optional, Tuple

import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_SAFE_BUILTINS = {
    # core builtins
    "print": print, "range": range, "len": len, "zip": zip,
    "enumerate": enumerate, "list": list, "dict": dict, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "None": None, "True": True, "False": False,
    "max": max, "min": min, "sum": sum, "abs": abs, "round": round,
    "sorted": sorted, "reversed": reversed, "next": next,
    "any": any, "all": all, "map": map, "filter": filter,
    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
    "set": set, "frozenset": frozenset, "vars": vars,
    "iter": iter, "open": None,  # open disabled
    # exceptions — needed for try/except in AI-generated code
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ZeroDivisionError": ZeroDivisionError,
    "OverflowError": OverflowError, "NotImplementedError": NotImplementedError,
}


def _to_float(x):
    """Convert string/number to float, return None on failure or NaN/Inf."""
    if x is None:
        return None
    try:
        v = float(str(x).strip().replace("%", "").replace(",", ""))
        return None if v != v else v  # v != v is True only for NaN
    except (ValueError, TypeError):
        return None


def resolve_plot_rows(raw_rows: list) -> list:
    """
    Pre-process plot_rows in Python before handing to AI code.
    - Converts value/ci_lower/ci_upper to float (strips % signs, commas, etc.)
    - Falls back to evidence text when value is None
    - Adds resolved_value, resolved_unit, resolved_parsed fields
    """
    out = []
    for r in raw_rows:
        row = dict(r)
        u = str(row.get("unit") or "")
        v = _to_float(row.get("value"))
        ci_lo = _to_float(row.get("ci_lower"))
        ci_hi = _to_float(row.get("ci_upper"))
        parsed = False

        if v is None:
            text = str(row.get("evidence") or row.get("value_raw") or "")
            m = re.search(r"(\d+\.?\d*)\s*(%|percent)", text)
            if m:
                v, u, parsed = float(m.group(1)), "%", True
            else:
                m = re.search(r"(\d+\.?\d*)\s*(month|mo\b)", text, re.IGNORECASE)
                if m:
                    v, u, parsed = float(m.group(1)), "mo", True
                else:
                    m = re.search(r"(\d+\.?\d*)", text)
                    if m:
                        v, u, parsed = float(m.group(1)), u, True

        # Belt-and-suspenders: catch any stray NaN that slipped through _to_float
        if v is not None and v != v:       ci_lo = None; v = None  # noqa: E702
        if ci_lo is not None and ci_lo != ci_lo: ci_lo = None      # noqa: E702
        if ci_hi is not None and ci_hi != ci_hi: ci_hi = None      # noqa: E702

        # Always overwrite with float/None — AI code never sees raw strings or NaN
        row["value"]    = v
        row["ci_lower"] = ci_lo
        row["ci_upper"] = ci_hi
        row["resolved_value"]  = v
        row["resolved_unit"]   = u
        row["resolved_parsed"] = parsed
        # Recalculate plot_eligible after evidence fallback — original flag may be stale
        row["plot_eligible"] = v is not None
        out.append(row)
    return out


def run_sandboxed_plot(code: str, data: dict) -> Tuple[Optional[list], Optional[str], Optional[str]]:
    """
    Execute AI-generated matplotlib code in a restricted namespace.

    data: dict of variables available to the code (plot_rows, df, etc.)

    Returns: (figs, fig_base64, error_message)
    - figs: list of matplotlib Figure objects, or None on error
    - fig_base64: PNG of the first figure as base64 string, or None
    - error_message: string if error, else None
    """
    plt.close("all")
    # Pre-resolve values so AI code can use r['resolved_value'] / r['resolved_unit'] directly
    raw_rows = data.get("plot_rows", [])
    clean_rows = resolve_plot_rows(raw_rows)
    clean_df = pd.DataFrame(clean_rows)
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "plt": plt,
        "np": np,
        "pd": pd,
        "textwrap": textwrap,
        "re": re,
        **data,
        "plot_rows": clean_rows,   # overwrite with resolved version
        "df": clean_df,            # overwrite with resolved version
    }
    try:
        exec(compile(code, "<ai_plot>", "exec"), namespace)

        # Collect all figures the code produced
        figs: list = []
        raw_fig = namespace.get("fig")
        raw_figs = namespace.get("figs")

        if raw_figs and isinstance(raw_figs, list):
            figs = [f for f in raw_figs if hasattr(f, "axes")]
        if raw_fig is not None:
            if isinstance(raw_fig, list):
                figs = figs or [f for f in raw_fig if hasattr(f, "axes")]
            elif hasattr(raw_fig, "axes") and raw_fig not in figs:
                figs = [raw_fig] + [f for f in figs if f is not raw_fig]

        # Fall back to the current figure if nothing was captured
        if not figs:
            gcf = plt.gcf()
            if gcf.axes:
                figs = [gcf]

        if not figs:
            return None, None, "Code ran but produced no figure (no axes created)."

        # Post-processing: tight_layout only. Label content and style are fully owned by AI code.
        for f in figs:
            try:
                f.tight_layout(pad=1.5)
            except Exception:
                pass

        buf = io.BytesIO()
        figs[0].savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return figs, b64, None
    except Exception:
        return None, None, traceback.format_exc()
