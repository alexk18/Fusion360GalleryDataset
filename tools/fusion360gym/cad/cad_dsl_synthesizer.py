"""
Structural Spec -> CAD DSL synthesis layer for create mode.

Primary objective: produce role-aware, family-aware, construction-order-aware
DSL plans before any optional LLM augmentation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .cad_capabilities import CapabilityModel, default_capability_model, capability_required_for_primitive
from .cad_dsl import default_budget
from .cad_family_grammar import grammar_id_for_family, synthesize_steps_from_family_grammar


def _expected_roles(spec: Dict[str, Any]) -> List[str]:
    roles: List[str] = []
    for key in ("main_masses", "supporting_parts", "secondary_parts"):
        for part in (spec.get(key) or []):
            if not isinstance(part, dict):
                continue
            role = str(part.get("role") or "").strip()
            if role:
                roles.append(role)
    return roles


def _build_role_coverage_summary(spec: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = _expected_roles(spec)
    step_ids = {str((s or {}).get("id") or "").strip().lower() for s in steps if isinstance(s, dict)}

    matched = []
    missing = []
    for role in expected:
        rid = role.strip().lower()
        if not rid:
            continue
        tokens = [t for t in rid.replace("-", "_").split("_") if t]
        hit = any(rid in sid for sid in step_ids) or any(
            any(len(t) >= 3 and t in sid for t in tokens)
            for sid in step_ids
        )
        if hit:
            matched.append(role)
        else:
            missing.append(role)

    ratio = float(len(matched)) / float(max(1, len(expected)))
    return {
        "expected_count": len(expected),
        "matched_count": len(matched),
        "coverage_ratio": round(ratio, 6),
        "matched_roles": matched,
        "missing_roles": missing,
    }


def _capability_ok_for_plan(plan: Dict[str, Any], cap_model: CapabilityModel) -> bool:
    for step in (plan.get("steps") or []):
        if not isinstance(step, dict):
            return False
        primitive = str(step.get("primitive") or "").strip().lower()
        operation = str(step.get("operation") or "").strip()
        required = capability_required_for_primitive(primitive, operation)
        if required == "unsupported":
            return False
        if not cap_model.supports(required):
            return False
    return True


def _parse_plane(plane: Any) -> tuple[str, float]:
    s = str(plane or "").strip().upper()
    if "@" in s:
        base, off = s.split("@", 1)
        try:
            return base.strip(), float(off.strip())
        except Exception:
            return base.strip(), 0.0
    if s in ("XY", "XZ", "YZ"):
        return s, 0.0
    return "XY", 0.0


def _step_bounds(step: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not isinstance(step, dict):
        return None
    profile = step.get("profile") if isinstance(step.get("profile"), dict) else {}
    ptype = str(profile.get("type") or "").strip().lower()
    if ptype == "rect":
        cx = float(profile.get("cx", 0.0) or 0.0)
        cy = float(profile.get("cy", 0.0) or 0.0)
        w = abs(float(profile.get("w", 0.0) or 0.0))
        h = abs(float(profile.get("h", 0.0) or 0.0))
        p0, p1, q0, q1 = (cx - w * 0.5, cx + w * 0.5, cy - h * 0.5, cy + h * 0.5)
    elif ptype == "circle":
        cx = float(profile.get("cx", 0.0) or 0.0)
        cy = float(profile.get("cy", 0.0) or 0.0)
        r = abs(float(profile.get("radius", 0.0) or 0.0))
        p0, p1, q0, q1 = (cx - r, cx + r, cy - r, cy + r)
    elif ptype == "poly":
        pts = [p for p in (profile.get("pts") or []) if isinstance(p, dict)]
        if not pts:
            return None
        xs = [float(p.get("x", 0.0) or 0.0) for p in pts]
        ys = [float(p.get("y", 0.0) or 0.0) for p in pts]
        p0, p1, q0, q1 = (min(xs), max(xs), min(ys), max(ys))
    else:
        return None

    base, off = _parse_plane(step.get("plane"))
    d = abs(float(step.get("distance", 0.0) or 0.0))
    if base == "XY":
        return {"x_min": p0, "x_max": p1, "y_min": q0, "y_max": q1, "z_min": off, "z_max": off + d}
    if base == "XZ":
        # In this Fusion setup, XZ plane sketch Y maps to -Z in world coords.
        return {"x_min": p0, "x_max": p1, "y_min": off, "y_max": off + d, "z_min": min(-q0, -q1), "z_max": max(-q0, -q1)}
    if base == "YZ":
        return {"x_min": off, "x_max": off + d, "y_min": p0, "y_max": p1, "z_min": q0, "z_max": q1}
    return None


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def _bounds_union(bounds_list: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not bounds_list:
        return None
    return {
        "x_min": min(b["x_min"] for b in bounds_list),
        "x_max": max(b["x_max"] for b in bounds_list),
        "y_min": min(b["y_min"] for b in bounds_list),
        "y_max": max(b["y_max"] for b in bounds_list),
        "z_min": min(b["z_min"] for b in bounds_list),
        "z_max": max(b["z_max"] for b in bounds_list),
    }


def _bounds_overlap(a: Dict[str, float], b: Dict[str, float]) -> bool:
    ox = _axis_overlap(a["x_min"], a["x_max"], b["x_min"], b["x_max"])
    oy = _axis_overlap(a["y_min"], a["y_max"], b["y_min"], b["y_max"])
    oz = _axis_overlap(a["z_min"], a["z_max"], b["z_min"], b["z_max"])
    return (ox > 0.0 and oy > 0.0) or (ox > 0.0 and oz > 0.0) or (oy > 0.0 and oz > 0.0)


def _step_meta(step: Dict[str, Any]) -> Dict[str, Any]:
    meta = step.get("meta")
    return meta if isinstance(meta, dict) else {}


def _step_risk_level(step: Dict[str, Any]) -> str:
    meta = _step_meta(step)
    risk = str(meta.get("risk_level") or "").strip().lower()
    if risk:
        return risk
    primitive = str(step.get("primitive") or "").strip().lower()
    operation = str(step.get("operation") or "").strip()
    if primitive == "cut_extrude" or operation in ("CutFeatureOperation", "IntersectFeatureOperation"):
        return "risky_detail"
    return "core"


def _is_risky_detail(step: Dict[str, Any]) -> bool:
    return _step_risk_level(step) == "risky_detail"


def _is_essential(step: Dict[str, Any]) -> bool:
    meta = _step_meta(step)
    if "essential" in meta:
        return bool(meta.get("essential"))
    return not _is_risky_detail(step)


def _core_or_secondary_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in steps if not _is_risky_detail(s)]


def _safe_boolean_step(step: Dict[str, Any], committed_steps: List[Dict[str, Any]]) -> bool:
    # Non-essential risky details must not break first-pass create robustness.
    if not _is_essential(step):
        return False
    if not committed_steps:
        return False
    step_b = _step_bounds(step)
    if step_b is None:
        return False
    committed_bounds = [b for b in (_step_bounds(s) for s in committed_steps) if b is not None]
    union = _bounds_union(committed_bounds)
    if union is None:
        return False
    return _bounds_overlap(step_b, union)


def _build_detail_safe_steps(steps: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    committed: List[Dict[str, Any]] = []
    pruned = 0
    for step in steps:
        if _is_risky_detail(step):
            if _safe_boolean_step(step, committed):
                committed.append(step)
            else:
                pruned += 1
            continue
        committed.append(step)
    return committed, pruned


def synthesize_dsl_candidates_from_structural_spec(
    structural_spec: Dict[str, Any],
    *,
    user_request: str,
    candidate_count: int = 3,
    mode: str = "create",
    capabilities: Optional[CapabilityModel] = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic family-grammar candidate synthesis.
    Returns capability-gated candidates only.

    Create execution safety policy:
    - primary candidate is core-first (risky detail pruned)
    - detail candidate is emitted only when risky steps pass safety checks
    """
    spec = dict(structural_spec or {})
    cap_model = capabilities or default_capability_model()

    if str(spec.get("build_outcome_target") or "") == "blocked":
        return []

    family = str(spec.get("object_family") or "").strip().lower()
    grammar_id = grammar_id_for_family(family)
    count = max(1, min(6, int(candidate_count or 1)))

    candidates: List[Dict[str, Any]] = []
    variant = 0
    while len(candidates) < count:
        session = f"syn_{family}_{int(time.time() * 1000)}_{variant+1}"
        steps_full = synthesize_steps_from_family_grammar(
            spec,
            session=session,
            user_request=user_request,
            variant=variant,
        )
        variant += 1
        if not steps_full:
            if variant > count + 2:
                break
            continue

        # Primary-first candidate: prune all risky details.
        steps_primary = _core_or_secondary_steps(steps_full)
        pruned_primary = max(0, len(steps_full) - len(steps_primary))
        role_coverage_primary = _build_role_coverage_summary(spec, steps_primary)
        plan_primary = {
            "units": "cm",
            "session": session,
            "mode": str(mode or "create").strip().lower() or "create",
            "budget": default_budget(),
            "global": {
                "source": "family_grammar_synthesizer",
                "shape_family": family,
                "build_outcome_target": spec.get("build_outcome_target", "exact"),
                "approximation_policy": dict(spec.get("approximation_policy") or {}),
                "synthesis_grammar": grammar_id,
                "construction_strategy": spec.get("construction_strategy", ""),
                "structural_role_coverage": role_coverage_primary,
                "create_safety": {
                    "mode": "primary_first",
                    "risky_pruned_count": pruned_primary,
                    "risky_retained_count": 0,
                },
                "request": str(user_request or "")[:240],
            },
            "steps": steps_primary,
        }
        if _capability_ok_for_plan(plan_primary, cap_model):
            candidates.append(plan_primary)
        if len(candidates) >= count:
            break

        # Optional detail-safe candidate: retain only risky steps that are plausibly safe.
        steps_safe_detail, pruned_detail = _build_detail_safe_steps(steps_full)
        retained_risky = sum(1 for s in steps_safe_detail if _is_risky_detail(s))
        if retained_risky <= 0:
            continue
        role_coverage_detail = _build_role_coverage_summary(spec, steps_safe_detail)
        plan_detail = {
            "units": "cm",
            "session": f"{session}_detail",
            "mode": str(mode or "create").strip().lower() or "create",
            "budget": default_budget(),
            "global": {
                "source": "family_grammar_synthesizer",
                "shape_family": family,
                "build_outcome_target": spec.get("build_outcome_target", "exact"),
                "approximation_policy": dict(spec.get("approximation_policy") or {}),
                "synthesis_grammar": grammar_id,
                "construction_strategy": spec.get("construction_strategy", ""),
                "structural_role_coverage": role_coverage_detail,
                "create_safety": {
                    "mode": "safe_detail_candidate",
                    "risky_pruned_count": pruned_detail,
                    "risky_retained_count": retained_risky,
                },
                "request": str(user_request or "")[:240],
            },
            "steps": steps_safe_detail,
        }
        if _capability_ok_for_plan(plan_detail, cap_model):
            candidates.append(plan_detail)

    return candidates[:count]
