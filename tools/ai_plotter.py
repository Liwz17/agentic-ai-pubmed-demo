"""AIPlotterAgent: generates and iteratively refines matplotlib plots."""
from __future__ import annotations
import json
import re
from typing import Optional
from base_agent import BaseAgent


PLOTTER_SYSTEM_PROMPT = """
You are an expert data visualisation assistant specialising in clinical trial outcome data.
You write clean, correct matplotlib code to visualise trial outcome data.
When reviewing a plot image, give specific, actionable feedback on colours, spacing, labels, and layout.
When asked to refine, output only the complete updated Python code block.
""".strip()


def _build_data_description(plot_rows: list) -> str:
    """Summarise the available data for the code generation prompt."""
    if not plot_rows:
        return "No plottable data available."
    outcomes = sorted({r["outcome_key"] for r in plot_rows if r.get("outcome_key")})
    arms = sorted({r["arm_name"] for r in plot_rows if r.get("arm_name")})
    eligible = [r for r in plot_rows if r.get("plot_eligible")]
    none_value_rows = [r for r in plot_rows if r.get("value") is None and r.get("evidence")]
    trial_id_map = {}
    for r in plot_rows:
        tid = r.get("trial_id") or ""
        if tid and tid not in trial_id_map:
            trial_id_map[tid] = r.get("trial_label", "")

    lines = [
        f"Trials (trial_id -> trial_label): { {k: v[:60] for k, v in trial_id_map.items()} }",
        f"Outcomes: {outcomes}",
        f"Arms: {arms}",
        f"Rows total: {len(plot_rows)}  |  plot_eligible (resolved_value not None): {len(eligible)}",
        "",
        "plot_rows fields (all pre-resolved before your code runs):",
        "  trial_id      str   NCT ID e.g. 'NCT03823625'",
        "  trial_label   str   paper/trial title",
        "  arm_name      str   treatment arm label",
        "  outcome_key   str   e.g. 'overall_survival'",
        "  outcome_display str human label",
        "  plot_type     str   'forest'|'bar'|'table_only'",
        "  plot_eligible bool  True when resolved_value is not None (reliable)",
        "  resolved_value float|None  USE THIS, never r['value']. Guaranteed float or None, NEVER NaN.",
        "  resolved_unit  str         e.g. 'months', '%', 'mo'",
        "  resolved_parsed bool       True if value recovered from evidence text",
        "  ci_lower  float|None",
        "  ci_upper  float|None",
        "  sample_size int|None",
        "  evidence  str",
        "",
        "Abbreviation -> outcome_key (use for filtering):",
        "  OS->overall_survival  PFS->progression_free_survival  ORR->objective_response_rate",
        "  DCR->disease_control_rate  DoR->duration_of_response  HR->hazard_ratio",
        "  (if unsure, use case-insensitive substring match on outcome_key)",
        "",
        "Filtering rules:",
        "  trial: r['trial_id'] == 'NCT...'  (never trial_label)",
        "  outcome: r['outcome_key'] substring match",
        "  arm: case-insensitive substring match on r['arm_name']",
        "  value: only include rows where r['plot_eligible'] is True",
        f"Rows where value=None but evidence has text (sandbox will attempt parse): {len(none_value_rows)}",
    ]
    for r in none_value_rows[:6]:
        lines.append(f"  [{r.get('trial_label','?')[:30]} | {r.get('arm_name','?')} | {r.get('outcome_key','?')}] evidence: {r.get('evidence','')[:100]}")
    lines.append("")
    lines.append("Available as 'plot_rows' list and 'df' DataFrame in your code.")
    return "\n".join(lines)


def _build_alias_prompt(trial_labels: list, arm_names: list) -> str:
    trials_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(trial_labels))
    arms_block   = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(arm_names))
    return f"""Generate short display names for clinical trial labels and arm names for use on plot axes and legends.

Trial labels:
{trials_block}

Arm names:
{arms_block}

Rules:
- Each alias must be UNIQUE and clinically meaningful (3-5 words max).
- Trial aliases: include drug name(s), indication, and phase where possible.
- Arm aliases: capture the key distinguishing info (drug, dose, combination).
- Different entries must produce different aliases — never duplicate.
- Output valid JSON only, no commentary:
{{
  "trial_aliases": {{"<original trial label>": "<short alias>", ...}},
  "arm_aliases":   {{"<original arm name>":   "<short alias>", ...}}
}}"""


def _build_codegen_prompt(data_desc: str, user_request: str,
                          trial_aliases: dict, arm_aliases: dict) -> str:
    alias_block = (
        "Pre-computed display names — copy these VERBATIM at the top of your code "
        "and use them on every axis tick and legend entry:\n"
        f"  trial_label_map = {trial_aliases!r}\n"
        f"  arm_label_map   = {arm_aliases!r}\n"
        "  # usage: trial_label_map.get(t, t[:12])  |  arm_label_map.get(a, a)\n"
    )
    return f"""You are generating matplotlib Python code to visualise clinical trial outcome data.

Available data:
{data_desc}

User request: {user_request}

ENVIRONMENT: NO import statements. Pre-injected globals:
  plt, np, pd, re, textwrap, plot_rows (list of dicts), df (DataFrame of plot_rows)

{alias_block}
REQUIRED PATTERNS:

AXIS LABEL RULE — applies to EVERY axis of EVERY chart, no exceptions:
  Copy trial_label_map and arm_label_map from above. Use them everywhere:
    - axis tick labels: trial_label_map.get(t, t[:12])
    - legend labels:    arm_label_map.get(a, a)
  NEVER place raw trial_label or arm_name strings on any axis or legend.

1. ONE figure per outcome_key. Never mix different-unit outcomes on the same axes.
   Collect figures in list `figs`; also set `fig = figs[0]` at the end.

2. ALWAYS filter to plot_eligible rows first, then by outcome/trial/arm:
     rows = [r for r in plot_rows if r['plot_eligible']]
     # then further filter by outcome_key, trial_id, arm_name as needed

3. Grouped bar chart for bar-type outcomes (plot_type == 'bar'):
   Skeleton:
     trial_label_map = {{...}}   # copy from Pre-computed display names above
     arm_label_map   = {{...}}   # copy from Pre-computed display names above
     arms   = sorted({{r['arm_name'] for r in rows}})
     trials = sorted({{r['trial_label'] for r in rows}})
     if not arms or not trials:
         ax.text(0.5, 0.5, 'No matching data', transform=ax.transAxes, ha='center', va='center', fontsize=12)
     else:
         x = np.arange(len(trials))
         w = 0.7 / len(arms)
         for i, arm in enumerate(arms):
             vals  = [next((r['resolved_value'] for r in rows if r['trial_label']==t and r['arm_name']==arm), None) for t in trials]
             units = [next((r['resolved_unit']  for r in rows if r['trial_label']==t and r['arm_name']==arm), '%')  for t in trials]
             ys = [v if v is not None else 0 for v in vals]
             bars = ax.bar(x + (i - len(arms)/2 + 0.5)*w, ys, w*0.9,
                           label=arm_label_map.get(arm, arm))
             for bar, v, u in zip(bars, vals, units):
                 if v is not None:
                     ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                             f'{{v:.1f}}{{u}}', ha='center', va='bottom', fontsize=8)
         ax.set_xticks(x)
         ax.set_xticklabels([trial_label_map.get(t, t[:12]) for t in trials],
                            rotation=45, ha='right', fontsize=9)

4. Horizontal forest plot for forest-type outcomes (plot_type == 'forest'):
   Arms MUST be vertically separated. Skeleton:
     trial_label_map = {{...}}   # copy from Pre-computed display names above
     arm_label_map   = {{...}}   # copy from Pre-computed display names above
     trials = sorted({{r['trial_label'] for r in rows}})
     arms   = sorted({{r['arm_name'] for r in rows}})
     gap, block = 0.35, len(arms)*0.35 + 0.6
     yticks, ylabels = [], []
     for ti, trial in enumerate(trials):
         for ai, arm in enumerate(arms):
             r = next((x for x in rows if x['trial_label']==trial and x['arm_name']==arm), None)
             if r is None: continue
             v = r['resolved_value']
             if v is None: continue
             y = ti*block + ai*gap
             xerr = [[v-r['ci_lower']], [r['ci_upper']-v]] if (r['ci_lower'] is not None and r['ci_upper'] is not None) else None
             ax.errorbar(v, y, xerr=xerr, fmt='o', capsize=4, color=f'C{{ai}}',
                         label=arm_label_map.get(arm, arm) if ti==0 else '')
             suffix = '†' if r['resolved_parsed'] else ''
             ann_x = v + (r['ci_upper'] or v)*0.03 + 0.5
             ax.text(ann_x, y, f'{{v:.1f}} {{r["resolved_unit"]}}{{suffix}}', va='center', fontsize=8)
             yticks.append(y)
             ylabels.append(f'{{trial_label_map.get(trial, trial[:12])}} | {{arm_label_map.get(arm, arm)}}')
     if not yticks:
         ax.text(0.5, 0.5, 'No data matches the filter', transform=ax.transAxes, ha='center', va='center', fontsize=12)
     else:
         ax.set_yticks(yticks)
         ax.set_yticklabels(ylabels, fontsize=8)
         fig.set_size_inches(12, max(6, len(yticks)*0.7 + 2))

5. FIGURE SIZE — set at subplots() time:
   Forest: fig, ax = plt.subplots(figsize=(12, max(6, estimated_rows*0.7+2)))
   Bar:    fig, ax = plt.subplots(figsize=(max(8, n_trials*1.5+3), 6))
   Always call plt.tight_layout(pad=1.5). Add title, legend, axis labels with units.

6. VALUE ACCESS: ALWAYS use r['resolved_value'] (float or None, never NaN).
   NEVER use r['value'] directly. Skip row when r['resolved_value'] is None.
   When r['resolved_parsed'] is True add dagger suffix and footnote:
   fig.text(0.01, -0.02, "† value parsed from evidence text", fontsize=7, style='italic')

7. AXIS LABELS: see the AXIS LABEL RULE at the top — always use trial_label_map / arm_label_map.
   Raw trial_label or arm_name strings are only allowed in ax.set_title() or ax.text() annotations.

8. NEVER raise exceptions (no raise/assert/sys.exit).
   Empty data -> draw figure with ax.text(0.5,0.5,'No data',transform=ax.transAxes,ha='center',va='center').
   Always guard: check `if not arms` before any division by len(arms).

Output ONLY a Python code block — no explanation, no imports.
```python
# no imports — plt, np, pd, re, textwrap, plot_rows, df are pre-injected
...
fig = figs[0] if figs else plt.figure()
```"""


def _build_review_prompt(user_request: str) -> str:
    return f"""You generated this plot for: "{user_request}"

IMPORTANT CONTEXT: All trial names and arm names on axes and legends are pre-computed SHORT ALIASES
(3-5 words), not truncated originals. A label like "Pembro NSCLC" or "NI vs N-CT" is COMPLETE and
INTENTIONAL — do NOT flag short labels as "cut off" just because they look abbreviated.

Look at the image carefully. For each item below, write one line: PASS or FAIL -- <reason>.

1. Bars/points for different arms are side by side (grouped), not overlapping or stacked.
2. Each bar/point has a visible numeric label showing the actual value with its unit.
3. All text on axes and legend is fully visible within the figure boundary (not clipped by the edge).
   PASS if labels are short aliases — short is intentional. Only FAIL if text is physically cut off
   by the figure border or overlapping other text to the point of being unreadable.
4. Outcomes with different units (e.g. months vs %) are on separate figures, not the same axes.
5. Labels fit within the plot area without overlapping each other or the bars/points.
6. Overall layout is clean enough for a medical publication.
7. Figure height is appropriate -- not squished or cramped. Forest plots need >= 0.7 inch per row.

After the checklist, write either:
- APPROVED  (if every item is PASS)
- ISSUES: <one concise sentence per failing item, describing exactly what to fix>

Do NOT output any code here."""


def _data_context(data_desc: str) -> str:
    if not data_desc:
        return ""
    return f"Available data in plot_rows:\n{data_desc[:1200]}\n\n"


class AIPlotterAgent(BaseAgent):
    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__(system_prompt=PLOTTER_SYSTEM_PROMPT, model=model)
        self._last_data_desc: str = ""

    def summarize_labels(self, trial_labels: list, arm_names: list) -> tuple[dict, dict]:
        """One LLM call: generate short meaningful display names for all trial labels and arm names."""
        if not trial_labels and not arm_names:
            return {}, {}
        prompt = _build_alias_prompt(trial_labels, arm_names)
        resp = self._call_chat_model(user_message=prompt, temperature=0.1)
        content = resp.choices[0].message.content or ""
        m = re.search(r'\{.*\}', content, re.DOTALL)
        try:
            result = json.loads(m.group(0) if m else content)
            return result.get("trial_aliases", {}), result.get("arm_aliases", {})
        except Exception:
            return {}, {}

    def generate_code(self, plot_rows: list, user_request: str) -> str:
        """Generate matplotlib code for the given data and request."""
        df_rows = [r for r in plot_rows if r.get("plot_type") != "table_only"]
        data_desc = _build_data_description(df_rows)
        self._last_data_desc = data_desc

        trial_labels = list(dict.fromkeys(
            r["trial_label"] for r in df_rows if r.get("trial_label")
        ))
        arm_names = list(dict.fromkeys(
            r["arm_name"] for r in df_rows if r.get("arm_name")
        ))
        trial_aliases, arm_aliases = self.summarize_labels(trial_labels, arm_names)

        prompt = _build_codegen_prompt(data_desc, user_request, trial_aliases, arm_aliases)
        resp = self._call_chat_model(user_message=prompt, temperature=0.2)
        content = resp.choices[0].message.content or ""
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    def review_plot(self, fig_base64: str, user_request: str) -> str:
        """Send rendered plot image back and get checklist review."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fig_base64}"}},
                {"type": "text", "text": _build_review_prompt(user_request)},
            ],
        }]
        resp = self._call_chat_model(temperature=0.2, extra_messages=messages)
        return resp.choices[0].message.content or ""

    def fix_code(self, code: str, error: str) -> str:
        """Fix a runtime error in generated code."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        prompt = (
            f"This matplotlib code raised a runtime error in the sandbox:\n\n"
            f"Error:\n```\n{error}\n```\n\n"
            f"{_data_context(self._last_data_desc)}"
            f"Rules:\n"
            f"- NO import statements. plt, np, pd, re, textwrap, plot_rows, df are pre-injected.\n"
            f"- NEVER raise exceptions. Empty data -> draw figure with ax.text message.\n"
            f"- Filter rows: start with `rows = [r for r in plot_rows if r['plot_eligible']]`.\n"
            f"- If 'No arms found' error: filter is too strict, relax it.\n"
            f"- Guard division: `if not arms: ... else: w = 0.7/len(arms)`.\n\n"
            f"Broken code:\n```python\n{code}\n```\n\n"
            f"Return ONLY the complete corrected Python code block."
        )
        resp = self._call_chat_model(user_message=prompt, temperature=0.2)
        content = resp.choices[0].message.content or ""
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    def refine_code(
        self,
        current_code: str,
        feedback: str,
        fig_base64: Optional[str] = None,
        failed_attempts: Optional[list] = None,
    ) -> str:
        """Fix current_code based on review feedback. Stateless — resets history each call."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        failed_note = ""
        if failed_attempts:
            failed_note = (
                "\nPrevious refinement rounds did NOT fix these issues "
                "(do NOT repeat the same approach — use a fundamentally different technique):\n"
                + "\n".join(f"  - {a}" for a in failed_attempts)
                + "\n\n"
            )
        text = (
            f"Fix the matplotlib code below.\n"
            f"Rules: NO import statements. plt, np, pd, re, textwrap, plot_rows, df are pre-injected.\n\n"
            f"{_data_context(self._last_data_desc)}"
            f"Current code:\n```python\n{current_code}\n```\n\n"
            f"Issues found in the rendered plot:\n{feedback}\n\n"
            f"{failed_note}"
            f"If the issue is 'no data displayed' or empty plot:\n"
            f"  - Start filter with `rows = [r for r in plot_rows if r['plot_eligible']]`\n"
            f"  - Then filter by outcome_key / trial_id / arm_name using exact strings from data above\n"
            f"  - Use case-insensitive substring match for arm_name\n"
            f"  - NEVER raise exceptions; draw ax.text message if rows is empty\n\n"
            f"Return ONLY the complete corrected Python code block."
        )
        if fig_base64:
            extra = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fig_base64}"}},
                    {"type": "text", "text": text},
                ],
            }]
            resp = self._call_chat_model(temperature=0.2, extra_messages=extra)
        else:
            resp = self._call_chat_model(user_message=text, temperature=0.2)
        content = resp.choices[0].message.content or ""
        match = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()
