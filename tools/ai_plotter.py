"""AIPlotterAgent: generates and iteratively refines matplotlib plots."""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from base_agent import BaseAgent
from config import MODEL_NAME


PLOTTER_SYSTEM_PROMPT = """
You are an expert data visualisation assistant specialising in clinical trial outcome data.
You write clean, correct matplotlib code to visualise trial outcome data.
When reviewing a plot image, give specific, actionable feedback on colours, spacing, labels, and layout.
When asked to refine, output only the complete updated Python code block.
""".strip()


def _build_data_description(plot_rows: list, all_outcomes: list) -> str:
    """Summarise the available data for the code generation prompt."""
    if not plot_rows:
        return "No plottable data available."
    trials = sorted({r["trial_label"] for r in plot_rows})
    outcomes = sorted({r["outcome_key"] for r in plot_rows})
    arms = sorted({r["arm_name"] for r in plot_rows})
    lines = [
        f"Trials: {trials}",
        f"Outcomes: {outcomes}",
        f"Arms: {arms}",
        "",
        "plot_rows is a list of dicts with these keys:",
        "  trial_label (str), arm_name (str), outcome_key (str), outcome_display (str),",
        "  plot_type (str: 'forest'|'bar'|'table_only'), plot_eligible (bool),",
        "  value (float|None), unit (str|None), ci_lower (float|None), ci_upper (float|None),",
        "  value_raw (str)",
        "",
        "Available as 'plot_rows' list and 'df' pandas DataFrame in your code.",
    ]
    return "\n".join(lines)


def _build_codegen_prompt(data_desc: str, user_request: str) -> str:
    return f"""You are generating matplotlib Python code to visualise clinical trial outcome data.

Available data:
{data_desc}

User request: {user_request}

Rules:
1. Use ONLY: plt, np, pd, plot_rows, df — no other imports.
2. Assign the final figure to a variable named `fig`.
3. Create clear labels, a legend, and a descriptive title.
4. Use a professional colour palette suitable for medical publications.
5. Output ONLY a Python code block — no explanation.

```python
# your code here
fig, ax = plt.subplots(...)
...
```"""


def _build_review_prompt(user_request: str) -> str:
    return f"""Here is the plot you generated for this request: "{user_request}"

Review it carefully:
1. Does it correctly represent the data?
2. Are colours, fonts, and spacing appropriate for a medical publication?
3. Are labels and legend clear?

If refinements are needed, output the complete updated Python code block.
If the plot looks good, reply with just: APPROVED"""


class AIPlotterAgent(BaseAgent):
    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(system_prompt=PLOTTER_SYSTEM_PROMPT, model=model)

    def generate_code(self, plot_rows: list, user_request: str) -> str:
        """Generate matplotlib code for the given data and request."""
        df_rows = [r for r in plot_rows if r.get("plot_eligible")]
        data_desc = _build_data_description(df_rows, [])
        prompt = _build_codegen_prompt(data_desc, user_request)
        resp = self._call_chat_model(user_message=prompt, temperature=0.2)
        content = resp.choices[0].message.content or ""
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    def review_plot(self, fig_base64: str, user_request: str) -> str:
        """Send the rendered plot image back to the model and get review/refined code."""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{fig_base64}"},
                },
                {"type": "text", "text": _build_review_prompt(user_request)},
            ],
        }]
        resp = self._call_chat_model(temperature=0.2, extra_messages=messages)
        return resp.choices[0].message.content or ""

    def refine_code(self, feedback: str) -> str:
        """Given the review feedback, produce refined code."""
        resp = self._call_chat_model(
            user_message=f"Apply these refinements:\n{feedback}\n\nReturn the complete updated code block.",
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()
