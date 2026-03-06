"""
Fixer: Inspector issues + current plan -> minimal DSL Patch (LLM outputs ONLY patch).

Bounded: max 2 fix iterations. Patch rules: preserve step ids/names; only change what is necessary.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .cad_patch import apply_patch
from .cad_validate import validate_plan


FIXER_PROMPT = r"""You are a CAD plan fixer. You receive the current CAD DSL Plan and a list of issues from an inspector (no coordinates).

Your task: output ONLY a minimal CAD DSL Patch to fix the issues. Do NOT output a full plan or coordinates.

Patch format (JSON only):
{
  "patches": [
    {"op": "replace", "path": "steps[<step_id>].distance", "value": <number>},
    {"op": "replace", "path": "steps[<step_id>].profile.pts", "value": [...]},
    {"op": "replace", "path": "steps[<step_id>].profile.w", "value": <number>},
    {"op": "add", "path": "steps", "value": { <full step object> }},
    {"op": "remove", "path": "steps[<step_id>]"}
  ],
  "intent": "minimal_change"
}

Rules:
- Use existing step ids from the plan. path uses steps[<id>] e.g. steps[hull].distance.
- Only add replace/add/remove operations needed to address the issues.
- For proportion: adjust distance, profile.w, profile.h, profile.radius, or profile.pts.
- For missing: add a new step with "op": "add", "path": "steps", "value": { ... }.
- For misplaced: adjust plane offset (e.g. steps[id].plane) or profile center.
- Do not remove or rename steps unless the issue says to.
- Return ONLY the JSON object, no markdown.
"""


def fixer_output_schema() -> Dict[str, Any]:
    return {
        "patches": [
            {"op": "replace|add|remove", "path": "string", "value": "any (for replace/add)"}
        ],
        "intent": "minimal_change",
    }


def parse_fixer_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into a Patch dict. Returns None if invalid."""
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return None
    patches = out.get("patches")
    if patches is None or not isinstance(patches, list):
        return None
    return {"patches": patches, "intent": out.get("intent", "minimal_change")}


def run_fixer(
    current_plan: Dict[str, Any],
    inspector_output: Dict[str, Any],
    call_llm: Callable[[str, str], str],
) -> Optional[Dict[str, Any]]:
    """
    Run the fixer: give current plan + inspector issues; LLM returns only a Patch.
    Returns the Patch dict or None if parse failed.
    """
    issues_text = json.dumps(inspector_output.get("issues", []), indent=2)
    plan_repr = json.dumps(current_plan, indent=2)
    prompt = (
        FIXER_PROMPT
        + "\n\nCurrent plan (full):\n"
        + plan_repr
        + "\n\nInspector issues (fix these with minimal patch):\n"
        + issues_text
    )
    raw = call_llm("You are a CAD fixer. Output ONLY a JSON Patch object.", prompt)
    return parse_fixer_response(raw)


def post_render_fix_loop(
    plan: Dict[str, Any],
    user_request: str,
    take_screenshot: Callable[[], Optional[str]],
    call_llm_with_image: Callable[..., str],
    call_llm: Callable[[str, str], str],
    execute_plan_fn: Callable[[Dict[str, Any]], Any],
    max_iterations: int = 2,
) -> Dict[str, Any]:
    """
    After building plan: take screenshot -> inspector -> fixer -> apply patch -> re-validate -> re-execute.
    Bounded by max_iterations (e.g. 2). Returns final plan.
    """
    from .cad_inspector import plan_summary_for_inspector, run_inspector

    current = dict(plan)
    for it in range(max_iterations):
        image_b64 = take_screenshot()
        summary = plan_summary_for_inspector(current)
        inspector_out = run_inspector(user_request, summary, image_b64, call_llm_with_image)
        if inspector_out.get("score", 0) >= 0.85 and not inspector_out.get("issues"):
            break
        patch = run_fixer(current, inspector_out, call_llm)
        if not patch or not patch.get("patches"):
            break
        current = apply_patch(current, patch)
        vr = validate_plan(current)
        if not vr.valid:
            break
        execute_plan_fn(current)
    return current
