"""
Universal CAD Operator: best-of-N create, validate, compile, execute, optional Inspector+Fixer loop.

- Pre-render: generate N candidate plans, validate each, optional pre-render fix (validator errors -> patch), select best.
- Build: execute only the selected plan (no clear in edit mode).
- Post-render: bounded Inspector -> Patch -> re-execute (max 1-2 iterations).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from .cad_dsl import default_budget
from .cad_validate import validate_plan, ValidationResult
from .cad_compiler import compile_plan
from .cad_executor import CadExecutor, ExecutionResult, ExecutionTraceEvent
from .cad_patch import apply_patch
from .cad_inspector import plan_summary_for_inspector, run_inspector
from .cad_fixer import run_fixer
from .cad_capabilities import CapabilityModel, default_capability_model


def select_best_candidate(
    candidates: List[Dict[str, Any]],
    validation_results: List[ValidationResult],
) -> int:
    """
    Select best valid candidate. Deterministic: first valid; tie-break by fewer warnings then index.
    """
    best_idx = -1
    best_warnings = 1e9
    for i, vr in enumerate(validation_results):
        if not vr.valid:
            continue
        nw = len(vr.warnings)
        if best_idx < 0 or nw < best_warnings:
            best_idx = i
            best_warnings = nw
    return best_idx


def pre_render_fix(
    plan: Dict[str, Any],
    validation_result: ValidationResult,
    call_llm: Callable[[str, str], str],
    max_iterations: int = 1,
    capabilities: Optional[CapabilityModel] = None,
    allow_unsupported: bool = False,
) -> Dict[str, Any]:
    """Optional fixer for run_best_of_n_create: (plan, vr, call_llm) -> fixed plan."""
    """
    Pre-render fix: give LLM validator errors; LLM returns only a Patch. Apply and re-validate.
    """
    current = copy.deepcopy(plan)
    for _ in range(max_iterations):
        if validation_result.valid:
            return current
        errors_text = "\n".join(validation_result.errors)
        prompt = (
            "The following CAD plan failed validation. Output ONLY a minimal JSON Patch to fix it. "
            "Use replace/add/remove on steps[<id>].\n\nErrors:\n" + errors_text
        )
        raw = call_llm("CAD fixer. Output only JSON Patch.", prompt)
        patch = None
        try:
            import json
            out = json.loads(raw)
            patch = out if isinstance(out.get("patches"), list) else None
        except Exception:
            pass
        if not patch or not patch.get("patches"):
            return current
        current = apply_patch(current, patch)
        validation_result = validate_plan(
            current,
            capabilities=capabilities,
            allow_unsupported=allow_unsupported,
        )
    return current


def run_cad_operator(
    client: Any,
    plan: Dict[str, Any],
    *,
    dry_run: bool = False,
    step_delay: float = 0.4,
    edit_mode: bool = False,
    previous_plan: Optional[Dict[str, Any]] = None,
    run_post_render_fix: bool = False,
    user_request: str = "",
    take_screenshot: Optional[Callable[[], Optional[str]]] = None,
    call_llm_with_image: Optional[Callable[..., str]] = None,
    call_llm: Optional[Callable[[str, str], str]] = None,
    max_post_fix_iterations: int = 2,
    capabilities: Optional[CapabilityModel] = None,
) -> ExecutionResult:
    """
    Validate plan, compile, execute. Optionally run post-render Inspector+Fixer loop.
    Never calls clear(). In edit mode, previous_plan can be provided for
    deterministic in-place vs recreate-required classification.
    """
    cap_model = capabilities or default_capability_model()
    vr = validate_plan(plan, capabilities=cap_model, allow_unsupported=dry_run)
    if not vr.valid:
        return ExecutionResult(
            success=False,
            steps_ok=0,
            steps_total=0,
            message="Validation failed: " + "; ".join(vr.errors[:5]),
            trace=[
                ExecutionTraceEvent(
                    step_id="validation",
                    primitive="plan",
                    action="failed",
                    backend_path=[],
                    reason="; ".join(vr.errors[:5]),
                )
            ],
        )
    session = plan.get("session") or "session"
    mode = (plan.get("mode") or "create").strip().lower()
    is_edit = mode == "edit" or edit_mode
    executor = CadExecutor(
        client,
        dry_run=dry_run,
        step_delay=step_delay,
        edit_mode=is_edit,
        session_id=session,
        capabilities=cap_model,
    )
    result = executor.execute_plan(plan, previous_plan=previous_plan)
    if not result.success or not run_post_render_fix or dry_run:
        return result
    if not take_screenshot or not call_llm_with_image or not call_llm:
        return result
    for _ in range(max_post_fix_iterations):
        image_b64 = take_screenshot() if take_screenshot else None
        summary = plan_summary_for_inspector(plan)
        inspector_out = run_inspector(user_request, summary, image_b64, call_llm_with_image)
        if inspector_out.get("score", 0) >= 0.85 and not inspector_out.get("issues"):
            break
        patch = run_fixer(plan, inspector_out, call_llm)
        if not patch or not patch.get("patches"):
            break
        result.trace.append(
            ExecutionTraceEvent(
                step_id="post_render",
                primitive="patch",
                action="updated",
                backend_path=["post_render_patch_application"],
                reason=f"patch_ops={len(patch.get('patches') or [])}",
            )
        )
        plan = apply_patch(plan, patch)
        if not validate_plan(plan, capabilities=cap_model, allow_unsupported=dry_run).valid:
            result.trace.append(
                ExecutionTraceEvent(
                    step_id="post_render",
                    primitive="patch",
                    action="failed",
                    backend_path=["validation"],
                    reason="post-render patch validation failed",
                )
            )
            break
        result = executor.execute_plan(plan, previous_plan=previous_plan)
    return result


def run_best_of_n_create(
    client: Any,
    candidates: List[Dict[str, Any]],
    *,
    dry_run: bool = False,
    step_delay: float = 0.4,
    pre_render_fix_fn: Optional[Callable[..., Dict]] = None,
    call_llm: Optional[Callable[[str, str], str]] = None,
    capabilities: Optional[CapabilityModel] = None,
) -> tuple[int, Dict[str, Any], ExecutionResult]:
    """
    Validate all candidates, optionally pre-render fix, select best, execute only the best.
    Returns (best_index, best_plan, execution_result).
    """
    cap_model = capabilities or default_capability_model()
    validation_results = [validate_plan(p, capabilities=cap_model, allow_unsupported=dry_run) for p in candidates]
    if pre_render_fix_fn and call_llm:
        fixed = []
        for i, p in enumerate(candidates):
            fixed.append(pre_render_fix_fn(p, validation_results[i], call_llm))
        candidates = fixed
        validation_results = [validate_plan(p, capabilities=cap_model, allow_unsupported=dry_run) for p in candidates]
    best_idx = select_best_candidate(candidates, validation_results)
    if best_idx < 0:
        return -1, {}, ExecutionResult(success=False, steps_ok=0, steps_total=0, message="No valid candidate")
    best_plan = candidates[best_idx]
    executor = CadExecutor(
        client,
        dry_run=dry_run,
        step_delay=step_delay,
        edit_mode=False,
        capabilities=cap_model,
    )
    result = executor.execute_plan(best_plan)
    return best_idx, best_plan, result
