"""Sandboxed execution of AI-generated matplotlib plotting code."""
import io
import base64
import traceback
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_SAFE_BUILTINS = {
    "print": print, "range": range, "len": len, "zip": zip,
    "enumerate": enumerate, "list": list, "dict": dict, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "None": None, "True": True, "False": False,
    "max": max, "min": min, "sum": sum, "abs": abs, "round": round,
    "sorted": sorted, "reversed": reversed, "any": any, "all": all,
    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
}


def run_sandboxed_plot(code: str, data: dict) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    """
    Execute AI-generated matplotlib code in a restricted namespace.

    data: dict of variables available to the code (plot_rows, df, etc.)

    Returns: (fig, fig_base64, error_message)
    - fig: matplotlib Figure or None
    - fig_base64: PNG as base64 string or None
    - error_message: string if error, else None
    """
    plt.close("all")
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "plt": plt,
        "np": np,
        "pd": pd,
        **data,
    }
    try:
        exec(compile(code, "<ai_plot>", "exec"), namespace)
        fig = namespace.get("fig") or plt.gcf()
        if not fig.axes:
            return None, None, "Code ran but produced no figure (no axes created)."
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return fig, b64, None
    except Exception:
        return None, None, traceback.format_exc()
