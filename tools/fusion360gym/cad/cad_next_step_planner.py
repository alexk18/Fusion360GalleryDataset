"""
Next-step planner for iterative create-mode execution.

Consumes structural spec, relational assembly graph, runtime state snapshots,
and dataset-informed priors to choose the next small executable CAD fragment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _step_role(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    role = str(meta.get("role") or "").strip()
    if role:
        return role
    return str(step.get("id") or "").strip()


def _risk_level(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    risk = _norm(meta.get("risk_level"))
    if risk:
        return risk
    primitive = _norm(step.get("primitive"))
    operation = str(step.get("operation") or "").strip()
    if primitive == "cut_extrude" or operation in ("CutFeatureOperation", "IntersectFeatureOperation"):
        return "risky_detail"
    return "core"


def _detail_level(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    detail = _norm(meta.get("detail_level"))
    if detail:
        return detail
    if _risk_level(step) == "risky_detail":
        return "secondary"
    return "core"


def _dependencies_satisfied(role: str, graph: Dict[str, Any], built_roles: Set[str]) -> bool:
    deps = set((graph.get("dependencies") or {}).get(role, []) or [])
    return deps.issubset(built_roles)


def choose_next_fragment(
    *,
    plan_steps: List[Dict[str, Any]],
    graph: Dict[str, Any],
    state_snapshot: Dict[str, Any],
    priors: Dict[str, Any],
    max_steps_per_iteration: int = 2,
) -> Dict[str, Any]:
    executed_ids = set(state_snapshot.get("executed_step_ids") or [])
    role_to_step = graph.get("role_to_step") or {}
    built_roles = {role for role, sid in role_to_step.items() if sid and sid in executed_ids}
    required_roles = set(graph.get("required_roles") or [])
    core_complete = required_roles.issubset(built_roles) if required_roles else False
    assembly_eval = state_snapshot.get("assembly_eval") if isinstance(state_snapshot.get("assembly_eval"), dict) else {}
    disconnected_components = int(assembly_eval.get("disconnected_components", 0) or 0)
    floating_parts = int(assembly_eval.get("floating_parts_count", 0) or 0)
    attachment_plausibility = float(assembly_eval.get("attachment_plausibility", 1.0) or 0.0)
    enforce_stability = disconnected_components > 0 or floating_parts > 0 or attachment_plausibility < 0.6

    ordering = [r for r in (priors.get("ordering") or []) if isinstance(r, str) and r.strip()]
    if not ordering:
        ordering = [r for r in (graph.get("build_order") or []) if isinstance(r, str) and r.strip()]

    # Prioritize roles that can restore attachment/connectivity if runtime signals are weak.
    priority_roles: List[str] = []
    for pair in (assembly_eval.get("unattached_pairs") or []):
        if not isinstance(pair, dict):
            continue
        dep = str(pair.get("dependency") or "").strip()
        role = str(pair.get("role") or "").strip()
        if dep and dep not in built_roles and dep not in priority_roles:
            priority_roles.append(dep)
        if role and role not in built_roles and role not in priority_roles:
            priority_roles.append(role)
    if enforce_stability:
        for role in ordering:
            rr = _norm(role)
            if role in built_roles:
                continue
            if any(k in rr for k in ("support", "base", "leg", "hull", "frame", "module")):
                if role not in priority_roles:
                    priority_roles.append(role)
    if priority_roles:
        merged = []
        seen = set()
        for role in priority_roles + ordering:
            if role in seen:
                continue
            seen.add(role)
            merged.append(role)
        ordering = merged

    step_by_id = {
        str((s or {}).get("id") or ""): s
        for s in plan_steps
        if isinstance(s, dict) and str((s or {}).get("id") or "").strip()
    }
    selected_steps: List[Dict[str, Any]] = []
    selected_roles: List[str] = []
    reason = "no suitable role"

    for role in ordering:
        sid = str(role_to_step.get(role) or "")
        if not sid or sid in executed_ids:
            continue
        step = step_by_id.get(sid)
        if not isinstance(step, dict):
            continue
        if not _dependencies_satisfied(role, graph, built_roles):
            reason = f"waiting dependencies for role={role}"
            continue
        risk = _risk_level(step)
        detail = _detail_level(step)
        if not core_complete and (risk == "risky_detail" or detail == "secondary"):
            reason = f"defer non-core role={role} until primary masses complete"
            continue
        if core_complete and enforce_stability and (risk == "risky_detail" or detail == "secondary"):
            reason = f"defer detail role={role} while assembly stability is low"
            continue
        selected_steps.append(step)
        selected_roles.append(role)
        reason = f"selected role={role}"
        if enforce_stability:
            reason += " (stability-priority)"
        if len(selected_steps) >= max(1, int(max_steps_per_iteration)):
            break

    # Fallback: if no role matched but core is complete, allow one safe secondary step.
    if not selected_steps and core_complete and not enforce_stability:
        for role in ordering:
            sid = str(role_to_step.get(role) or "")
            if not sid or sid in executed_ids:
                continue
            step = step_by_id.get(sid)
            if not isinstance(step, dict):
                continue
            if _risk_level(step) == "risky_detail":
                continue
            selected_steps.append(step)
            selected_roles.append(role)
            reason = f"fallback secondary role={role}"
            break

    return {
        "selected_roles": selected_roles,
        "steps": selected_steps,
        "phase": "core" if not core_complete else "secondary",
        "reason": reason,
        "core_complete_before_selection": core_complete,
        "stability_guard_active": enforce_stability,
    }
