"""
Inspector: screenshot + plan summary -> structured critique (no coordinates).

Output JSON: { "score": 0..1, "issues": [{"part":"<id>","type":"proportion|missing|misplaced|shape","detail":"..."}], "constraints": ["..."] }
Used by the Fixer to produce a minimal DSL Patch.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


INSPECTOR_PROMPT = r"""You are a CAD model inspector. You see a screenshot of a 3D model and a brief plan summary (step names and key dimensions only).

Evaluate how well the model matches the user request. Do NOT output coordinates or build commands.

Return ONLY a JSON object with this structure:
{
  "score": 0.0 to 1.0,
  "issues": [
    {"part": "<step_id or part name>", "type": "proportion|missing|misplaced|shape", "detail": "short description"}
  ],
  "constraints": ["optional list of high-level constraints that should be preserved"]
}

Rules:
- score: 1.0 = matches request well; lower = more issues.
- issues: only list concrete problems (wrong size, missing part, wrong position, wrong shape).
- part: use the step id or name from the plan summary when relevant.
- type: proportion (size/scale wrong), missing (part absent), misplaced (position wrong), shape (geometry wrong).
- Do not invent coordinates or dimensions in "detail".
- If the model looks good, return score near 1.0 and empty or minimal issues.
"""


def inspector_output_schema() -> Dict[str, Any]:
    """Return the expected inspector output schema for documentation."""
    return {
        "score": "number 0..1",
        "issues": [
            {"part": "string", "type": "proportion|missing|misplaced|shape", "detail": "string"}
        ],
        "constraints": ["string"],
    }


def parse_inspector_response(raw: str) -> Dict[str, Any]:
    """Parse LLM/VLM response into inspector output. Returns dict with score, issues, constraints."""
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0.5, "issues": [{"part": "?", "type": "shape", "detail": "Inspector returned invalid JSON"}], "constraints": []}
    score = float(out.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    issues = list(out.get("issues") or [])
    constraints = list(out.get("constraints") or [])
    return {"score": score, "issues": issues, "constraints": constraints}


def plan_summary_for_inspector(plan: Dict[str, Any], max_steps: int = 30) -> str:
    """Build a short plan summary (names + key dimensions) for the inspector. No coordinates."""
    steps = (plan.get("steps") or [])[:max_steps]
    lines = []
    for s in steps:
        if not s:
            continue
        sid = s.get("id", "?")
        prim = s.get("primitive", "?")
        names = s.get("names") or {}
        feat = names.get("feature", "")
        dist = s.get("distance")
        profile = s.get("profile") or {}
        ptype = profile.get("type", "")
        line = f"  {sid}: {prim}"
        if dist is not None:
            line += f" distance={dist}"
        if ptype:
            line += f" profile={ptype}"
        if feat:
            line += f" feature={feat}"
        lines.append(line)
    return "Plan summary (names + key dimensions):\n" + "\n".join(lines) if lines else "No steps."


def run_inspector(
    user_request: str,
    plan_summary: str,
    image_b64: Optional[str],
    call_llm_with_image: Callable[..., str],
) -> Dict[str, Any]:
    """
    Run the inspector: pass user request, plan summary, and screenshot to VLM;
    return parsed { score, issues, constraints }.
    """
    prompt = INSPECTOR_PROMPT + "\n\nUser request: " + (user_request or "") + "\n\n" + plan_summary
    if image_b64:
        raw = call_llm_with_image("You are a CAD inspector. Respond only with JSON.", prompt, image_b64)
    else:
        raw = call_llm_with_image("You are a CAD inspector. Respond only with JSON.", prompt, "")
    return parse_inspector_response(raw)
