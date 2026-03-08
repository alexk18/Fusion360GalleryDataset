"""
Create-mode benchmark harness for iterative CAD agent quality tracking.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

from .cad_agent_loop import IterativeCreateAgent
from .cad_capabilities import CapabilityModel, default_capability_model
from .cad_dsl_synthesizer import synthesize_dsl_candidates_from_structural_spec
from .cad_quality_metrics import compute_quality_metrics, silhouette_plausibility_proxy
from .cad_structural_planner import plan_structural_specs
from .cad_structural_ranker import rank_structural_specs


def default_create_benchmark_cases() -> List[Dict[str, Any]]:
    return [
        {"input_id": "tank_lowpoly", "prompt": "Build a lowpoly tank", "subtype": "tracked_turret_vehicle", "expected_family": "lowpoly_hard_surface_vehicle"},
        {"input_id": "armored_vehicle", "prompt": "Build an armored vehicle", "subtype": "armored_vehicle", "expected_family": "lowpoly_hard_surface_vehicle"},
        {"input_id": "military_truck_simple", "prompt": "Build a simple military truck", "subtype": "truck", "expected_family": "lowpoly_hard_surface_vehicle"},
        {"input_id": "bedside_table", "prompt": "Build a bedside table", "subtype": "boxy_furniture", "expected_family": "furniture_boxy_panel"},
        {"input_id": "chair_straight_frame", "prompt": "Build a simple chair with straight frame", "subtype": "chair_frame", "expected_family": "furniture_boxy_panel"},
        {"input_id": "bed_headboard", "prompt": "Build a bed with headboard", "subtype": "bed_frame", "expected_family": "furniture_boxy_panel"},
        {"input_id": "airplane_lowpoly", "prompt": "Build a lowpoly airplane", "subtype": "airplane", "expected_family": "lowpoly_hard_surface_vehicle"},
        {"input_id": "boat_lowpoly", "prompt": "Build a lowpoly boat", "subtype": "boat", "expected_family": "lowpoly_hard_surface_vehicle"},
        {"input_id": "bottle", "prompt": "Build a bottle", "subtype": "rotational_container", "expected_family": "rotational_bodies"},
        {"input_id": "vase", "prompt": "Build a vase", "subtype": "rotational_container", "expected_family": "rotational_bodies"},
        {"input_id": "mug", "prompt": "Build a mug", "subtype": "rotational_container", "expected_family": "rotational_bodies"},
        {
            "input_id": "smooth_human_face_sculpture",
            "prompt": "Build a smooth human face sculpture",
            "subtype": "organic_freeform",
            "expected_family": "unsupported_smooth_freeform",
        },
    ]


def _select_structural_spec(prompt: str, capabilities: CapabilityModel) -> Dict[str, Any]:
    specs = plan_structural_specs(prompt, images=[], capabilities=capabilities)
    if not specs:
        return {}
    ranked = rank_structural_specs(specs, prompt, capabilities=capabilities)
    if ranked:
        return dict((ranked[0] or {}).get("spec") or {})
    return dict(specs[0] or {})


def _risk_level(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    risk = str(meta.get("risk_level") or "").strip().lower()
    if risk:
        return risk
    primitive = str(step.get("primitive") or "").strip().lower()
    operation = str(step.get("operation") or "").strip()
    if primitive == "cut_extrude" or operation in ("CutFeatureOperation", "IntersectFeatureOperation"):
        return "risky_detail"
    return "core"


def _count_risky(steps: List[Dict[str, Any]]) -> int:
    return sum(1 for s in (steps or []) if isinstance(s, dict) and _risk_level(s) == "risky_detail")


def _trace_summary(loop_trace: List[Dict[str, Any]], message: str) -> str:
    rows = [r for r in (loop_trace or []) if isinstance(r, dict)]
    actions = [str(r.get("action") or "") for r in rows if str(r.get("action") or "")]
    if actions:
        tail = ",".join(actions[-5:])
        return f"{message}; actions_tail={tail}"
    return str(message or "")


def _try_screenshot(client: Any, output_path: str) -> str:
    try:
        if hasattr(client, "screenshot"):
            r = client.screenshot(output_path, width=512, height=512)
            if getattr(r, "status_code", 200) == 200:
                return output_path
        elif hasattr(client, "send_command"):
            r = client.send_command("screenshot", {"file": output_path, "width": 512, "height": 512})
            if getattr(r, "status_code", 200) == 200:
                return output_path
    except Exception:
        return ""
    return ""


def run_create_benchmark(
    client: Any,
    *,
    cases: Optional[List[Dict[str, Any]]] = None,
    capabilities: Optional[CapabilityModel] = None,
    candidate_count: int = 3,
    max_iterations: int = 32,
    step_delay: float = 0.0,
    screenshot_dir: str = "",
    inspect_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    execute_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    cap_model = capabilities or default_capability_model()
    suite = [dict(c) for c in (cases or default_create_benchmark_cases()) if isinstance(c, dict)]
    results: List[Dict[str, Any]] = []
    started_at = int(time.time())

    for case in suite:
        input_id = str(case.get("input_id") or f"case_{len(results)+1}")
        prompt = str(case.get("prompt") or "").strip()
        subtype = str(case.get("subtype") or "")
        expected_family = str(case.get("expected_family") or "")

        spec = _select_structural_spec(prompt, cap_model)
        family = str(spec.get("object_family") or "")
        outcome_target = str(spec.get("build_outcome_target") or "exact")
        family_routing_ok = bool(expected_family) and family == expected_family

        base_row = {
            "input_id": input_id,
            "prompt": prompt,
            "family": family,
            "subtype": subtype,
            "outcome_target": outcome_target,
            "family_routing_ok": family_routing_ok,
            "build_status": "fail",
            "build_success": False,
            "iteration_count": 0,
            "roles_completed": 0,
            "roles_total": 0,
            "disconnected_components_count": 0,
            "floating_parts_count": 0,
            "attachment_plausibility": 0.0,
            "risky_steps_total": 0,
            "risky_steps_executed": 0,
            "risky_steps_skipped": 0,
            "final_trace_summary": "",
            "screenshot_path": "",
            "blocked_honest": False,
            "silhouette_plausibility_proxy": 0.0,
        }

        if outcome_target == "blocked":
            row = dict(base_row)
            row["build_status"] = "blocked"
            row["blocked_honest"] = True
            row["final_trace_summary"] = str(spec.get("blocked_reason") or "blocked by structural planner")
            results.append(row)
            continue

        candidates = synthesize_dsl_candidates_from_structural_spec(
            spec,
            user_request=prompt,
            candidate_count=max(1, int(candidate_count)),
            mode="create",
            capabilities=cap_model,
        )
        if not candidates:
            row = dict(base_row)
            row["final_trace_summary"] = "no executable candidates synthesized"
            results.append(row)
            continue

        agent = IterativeCreateAgent(
            client,
            capabilities=cap_model,
            step_delay=float(step_delay),
            inspect_fn=inspect_fn,
            execute_fn=execute_fn,
        )
        outcome = agent.run(
            user_request=prompt,
            structural_spec=spec,
            candidates=candidates,
            max_iterations=max_iterations,
        )
        selected_idx = int(outcome.best_index if outcome.best_index >= 0 else 0)
        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
        selected_candidate = dict(candidates[selected_idx] or {})
        selected_steps = [s for s in (selected_candidate.get("steps") or []) if isinstance(s, dict)]
        executed_steps = [s for s in (outcome.best_plan.get("steps") or []) if isinstance(s, dict)]

        final_snapshot = dict((outcome.state_snapshots or [])[-1] or {}) if outcome.state_snapshots else {}
        final_eval = final_snapshot.get("assembly_eval") if isinstance(final_snapshot.get("assembly_eval"), dict) else {}
        roles_completed = int(final_eval.get("required_roles_built", 0) or 0)
        roles_total = int(final_eval.get("required_roles_total", 0) or 0)

        risky_total = _count_risky(selected_steps)
        risky_executed = _count_risky(executed_steps)
        risky_skipped = max(0, risky_total - risky_executed)

        row = dict(base_row)
        row.update(
            {
                "build_status": "success" if bool(outcome.success and outcome.result.success) else "fail",
                "build_success": bool(outcome.success and outcome.result.success),
                "iteration_count": len(outcome.loop_trace or []),
                "roles_completed": roles_completed,
                "roles_total": roles_total,
                "disconnected_components_count": int(final_eval.get("disconnected_components", 0) or 0),
                "floating_parts_count": int(final_eval.get("floating_parts_count", 0) or 0),
                "attachment_plausibility": float(final_eval.get("attachment_plausibility", 0.0) or 0.0),
                "risky_steps_total": risky_total,
                "risky_steps_executed": risky_executed,
                "risky_steps_skipped": risky_skipped,
                "final_trace_summary": _trace_summary(list(outcome.loop_trace or []), str(outcome.result.message or "")),
                "blocked_honest": False,
            }
        )

        if screenshot_dir and row["build_success"]:
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(screenshot_dir, f"{input_id}.png")
            row["screenshot_path"] = _try_screenshot(client, screenshot_path)

        row["silhouette_plausibility_proxy"] = round(silhouette_plausibility_proxy(row), 6)
        results.append(row)

    metrics = compute_quality_metrics(results)
    completed_at = int(time.time())
    return {
        "suite_name": "create_mode_geometry_state_v1",
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_sec": max(0, completed_at - started_at),
        "case_count": len(results),
        "cases": results,
        "metrics": metrics,
    }


def benchmark_routing_matrix(cases: Optional[List[Dict[str, Any]]] = None, *, capabilities: Optional[CapabilityModel] = None) -> List[Dict[str, Any]]:
    cap_model = capabilities or default_capability_model()
    rows = []
    for case in (cases or default_create_benchmark_cases()):
        if not isinstance(case, dict):
            continue
        prompt = str(case.get("prompt") or "")
        expected_family = str(case.get("expected_family") or "")
        spec = _select_structural_spec(prompt, cap_model)
        actual_family = str(spec.get("object_family") or "")
        rows.append(
            {
                "input_id": str(case.get("input_id") or ""),
                "prompt": prompt,
                "subtype": str(case.get("subtype") or ""),
                "expected_family": expected_family,
                "actual_family": actual_family,
                "routed_ok": bool(expected_family) and expected_family == actual_family,
                "outcome_target": str(spec.get("build_outcome_target") or ""),
            }
        )
    return rows
