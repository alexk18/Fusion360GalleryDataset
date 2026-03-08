"""
Quality metrics for create-mode benchmark runs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _avg(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    if not rows:
        return 0.0
    return float(sum(rows)) / float(len(rows))


def _safe_ratio(num: float, den: float) -> float:
    d = float(den)
    if d <= 0.0:
        return 0.0
    return float(num) / d


def silhouette_plausibility_proxy(case_result: Dict[str, Any]) -> float:
    role_ratio = _safe_ratio(
        float(case_result.get("roles_completed", 0) or 0),
        float(case_result.get("roles_total", 0) or 0),
    )
    disconnected = float(case_result.get("disconnected_components_count", 0) or 0)
    floating = float(case_result.get("floating_parts_count", 0) or 0)
    attachment = float(case_result.get("attachment_plausibility", 0.0) or 0.0)
    score = (
        0.45 * role_ratio
        + 0.20 * (1.0 / (1.0 + disconnected))
        + 0.20 * (1.0 / (1.0 + floating))
        + 0.15 * max(0.0, min(1.0, attachment))
    )
    return max(0.0, min(1.0, score))


def compute_quality_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, float]:
    rows = [r for r in (case_results or []) if isinstance(r, dict)]
    if not rows:
        return {
            "build_success_rate": 0.0,
            "blocked_honesty_rate": 0.0,
            "disconnected_parts_metric": 0.0,
            "floating_parts_metric": 0.0,
            "primary_role_completion": 0.0,
            "silhouette_plausibility_proxy": 0.0,
            "average_iterations_to_stop": 0.0,
            "risky_detail_execution_rate": 0.0,
        }

    runnable = [r for r in rows if str(r.get("outcome_target") or "") != "blocked"]
    blocked = [r for r in rows if str(r.get("outcome_target") or "") == "blocked"]

    success_rate = _safe_ratio(
        sum(1 for r in runnable if bool(r.get("build_success"))),
        len(runnable),
    )
    blocked_honesty = _safe_ratio(
        sum(1 for r in blocked if str(r.get("build_status") or "") == "blocked"),
        len(blocked),
    )
    disconnected_metric = _avg(float(r.get("disconnected_components_count", 0) or 0) for r in runnable)
    floating_metric = _avg(float(r.get("floating_parts_count", 0) or 0) for r in runnable)
    primary_role_completion = _avg(
        _safe_ratio(
            float(r.get("roles_completed", 0) or 0),
            float(r.get("roles_total", 0) or 0),
        )
        for r in runnable
    )
    silhouette_proxy = _avg(silhouette_plausibility_proxy(r) for r in runnable)
    avg_iterations = _avg(float(r.get("iteration_count", 0) or 0) for r in runnable)

    risky_exec = sum(float(r.get("risky_steps_executed", 0) or 0) for r in runnable)
    risky_total = sum(float(r.get("risky_steps_total", 0) or 0) for r in runnable)
    risky_rate = _safe_ratio(risky_exec, risky_total)

    return {
        "build_success_rate": round(success_rate, 6),
        "blocked_honesty_rate": round(blocked_honesty, 6),
        "disconnected_parts_metric": round(disconnected_metric, 6),
        "floating_parts_metric": round(floating_metric, 6),
        "primary_role_completion": round(primary_role_completion, 6),
        "silhouette_plausibility_proxy": round(silhouette_proxy, 6),
        "average_iterations_to_stop": round(avg_iterations, 6),
        "risky_detail_execution_rate": round(risky_rate, 6),
    }

