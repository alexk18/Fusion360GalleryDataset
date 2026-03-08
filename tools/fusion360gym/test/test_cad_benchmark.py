"""
Tests for create-mode benchmark harness and quality metrics.
"""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from cad.cad_benchmark import (
    benchmark_routing_matrix,
    default_create_benchmark_cases,
    run_create_benchmark,
)
from cad.cad_executor import ExecutionResult
from cad.cad_quality_metrics import compute_quality_metrics


class _BenchmarkClient:
    def clear(self):
        return None

    def refresh(self):
        return None


def _execute_ok(fragment_plan):
    steps = [s for s in (fragment_plan.get("steps") or []) if isinstance(s, dict)]
    reg = {}
    for step in steps:
        sid = str(step.get("id") or "")
        if sid:
            reg[sid] = {"feature_name": f"{sid}__feat"}
    return ExecutionResult(
        success=True,
        steps_ok=len(steps),
        steps_total=len(steps),
        message="ok",
        registry=reg,
        trace=[],
    )


def _inspect_ok(executed_steps=None, planned_steps=None):
    executed = [s for s in (executed_steps or []) if isinstance(s, dict)]
    ids = [str(s.get("id") or "") for s in executed if str(s.get("id") or "")]
    rels = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            rels.append({"a": a, "b": b, "relation": "touches_estimate"})
    return {
        "executed_step_ids": ids,
        "get_connected_components": [ids] if ids else [],
        "get_step_relations": rels,
        "get_active_construction_context": {"grounded_step_ids": ids[:1], "floating_step_ids": []},
    }


def test_benchmark_harness_smoke():
    suite = [
        c
        for c in default_create_benchmark_cases()
        if c.get("input_id") in ("tank_lowpoly", "bottle")
    ]
    out = run_create_benchmark(
        _BenchmarkClient(),
        cases=suite,
        candidate_count=1,
        max_iterations=10,
        step_delay=0.0,
        inspect_fn=_inspect_ok,
        execute_fn=_execute_ok,
    )
    assert out["case_count"] == 2
    assert "metrics" in out
    assert "cases" in out
    for row in out["cases"]:
        assert "input_id" in row
        assert "family" in row
        assert "subtype" in row
        assert "outcome_target" in row
        assert "build_status" in row
        assert "iteration_count" in row
        assert "roles_completed" in row
        assert "disconnected_components_count" in row
        assert "floating_parts_count" in row
        assert "risky_steps_executed" in row
        assert "risky_steps_skipped" in row
        assert "final_trace_summary" in row
        assert "screenshot_path" in row


def test_blocked_case_honesty():
    blocked_case = [
        c
        for c in default_create_benchmark_cases()
        if c.get("input_id") == "smooth_human_face_sculpture"
    ]
    out = run_create_benchmark(
        _BenchmarkClient(),
        cases=blocked_case,
        candidate_count=1,
        max_iterations=6,
        step_delay=0.0,
        inspect_fn=_inspect_ok,
        execute_fn=_execute_ok,
    )
    assert out["case_count"] == 1
    row = out["cases"][0]
    assert row["outcome_target"] == "blocked"
    assert row["build_status"] == "blocked"
    assert row["blocked_honest"] is True


def test_family_subtype_benchmark_routing_matrix():
    rows = benchmark_routing_matrix(default_create_benchmark_cases())
    by_id = {r["input_id"]: r for r in rows}
    assert by_id["tank_lowpoly"]["expected_family"] == "lowpoly_hard_surface_vehicle"
    assert by_id["tank_lowpoly"]["actual_family"] == "lowpoly_hard_surface_vehicle"
    assert by_id["tank_lowpoly"]["routed_ok"] is True
    assert by_id["bottle"]["expected_family"] == "rotational_bodies"
    assert by_id["bottle"]["actual_family"] == "rotational_bodies"
    assert by_id["bottle"]["routed_ok"] is True
    assert by_id["smooth_human_face_sculpture"]["actual_family"] == "unsupported_smooth_freeform"
    assert by_id["smooth_human_face_sculpture"]["outcome_target"] == "blocked"
    assert by_id["smooth_human_face_sculpture"]["subtype"] == "organic_freeform"


def test_quality_metrics_aggregate_contract():
    rows = [
        {
            "outcome_target": "exact",
            "build_success": True,
            "build_status": "success",
            "disconnected_components_count": 0,
            "floating_parts_count": 0,
            "roles_completed": 4,
            "roles_total": 4,
            "iteration_count": 5,
            "risky_steps_executed": 1,
            "risky_steps_total": 2,
            "attachment_plausibility": 0.9,
        },
        {
            "outcome_target": "blocked",
            "build_success": False,
            "build_status": "blocked",
            "disconnected_components_count": 0,
            "floating_parts_count": 0,
            "roles_completed": 0,
            "roles_total": 0,
            "iteration_count": 0,
            "risky_steps_executed": 0,
            "risky_steps_total": 0,
            "attachment_plausibility": 0.0,
        },
    ]
    metrics = compute_quality_metrics(rows)
    for key in (
        "build_success_rate",
        "blocked_honesty_rate",
        "disconnected_parts_metric",
        "floating_parts_metric",
        "primary_role_completion",
        "silhouette_plausibility_proxy",
        "average_iterations_to_stop",
        "risky_detail_execution_rate",
    ):
        assert key in metrics


if __name__ == "__main__":
    test_benchmark_harness_smoke()
    test_blocked_case_honesty()
    test_family_subtype_benchmark_routing_matrix()
    test_quality_metrics_aggregate_contract()
    print("All benchmark tests passed.")

