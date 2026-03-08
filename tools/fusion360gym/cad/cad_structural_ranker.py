"""
Ranking utilities for Structural Build Spec candidates and DSL plans.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cad_capabilities import CapabilityModel, default_capability_model, capability_required_for_primitive


_FAMILY_HINTS = {
    "furniture_boxy_panel": (
        "chair",
        "table",
        "cabinet",
        "panel",
        "shelf",
        "desk",
        "box",
    ),
    "lowpoly_hard_surface_vehicle": (
        "tank",
        "vehicle",
        "truck",
        "car",
        "track",
        "turret",
        "robot",
        "drone",
    ),
    "profile_driven_symmetric": (
        "bracket",
        "frame",
        "profile",
        "symmetric",
        "chassis",
    ),
    "rotational_bodies": (
        "bottle",
        "vase",
        "cup",
        "mug",
        "cylinder",
        "spool",
    ),
    "unsupported_smooth_freeform": (
        "human",
        "face",
        "character",
        "organic",
        "creature",
        "freeform",
        "sculpt",
    ),
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _family_fit_score(spec: Dict[str, Any], user_request: str) -> float:
    family = _norm(spec.get("object_family"))
    hints = _FAMILY_HINTS.get(family, ())
    text = _norm(user_request)
    if not hints:
        return 0.5
    hits = sum(1 for h in hints if h in text)
    return min(1.0, 0.25 + (0.75 * hits / max(1, len(hints) / 2)))


def _capability_fit_score(spec: Dict[str, Any], capabilities: CapabilityModel) -> float:
    req = spec.get("required_capabilities") or []
    if not req:
        return 1.0
    ok = sum(1 for cap in req if capabilities.supports(str(cap)))
    return ok / max(1, len(req))


def _extract_roles(spec: Dict[str, Any]) -> List[str]:
    roles: List[str] = []
    for key in ("main_masses", "supporting_parts", "secondary_parts"):
        for part in (spec.get(key) or []):
            if not isinstance(part, dict):
                continue
            role = _norm(part.get("role"))
            if role:
                roles.append(role)
    return roles


def _structural_coverage_score(spec: Dict[str, Any]) -> float:
    roles = _extract_roles(spec)
    if not roles:
        return 0.0
    main = [r for r in roles if r in {_norm((p or {}).get("role")) for p in (spec.get("main_masses") or []) if isinstance(p, dict)}]
    has_order = isinstance(spec.get("build_order"), list) and len(spec.get("build_order") or []) > 0
    role_bonus = min(1.0, len(set(roles)) / 6.0)
    main_bonus = min(1.0, len(set(main)) / 3.0)
    order_bonus = 1.0 if has_order else 0.3
    return (0.45 * role_bonus) + (0.35 * main_bonus) + (0.20 * order_bonus)


def _build_order_coherence_score(spec: Dict[str, Any]) -> float:
    order = [
        _norm(r) for r in (spec.get("build_order") or []) if _norm(r)
    ]
    if not order:
        return 0.0
    roles = set(_extract_roles(spec))
    if not roles:
        return 0.0
    valid = sum(1 for role in order if role in roles)
    starts_with_main = False
    first = order[0]
    for part in (spec.get("main_masses") or []):
        if isinstance(part, dict) and _norm(part.get("role")) == first:
            starts_with_main = True
            break
    base = valid / max(1, len(order))
    if starts_with_main:
        base = min(1.0, base + 0.15)
    return base


def _silhouette_consistency_score(spec: Dict[str, Any]) -> float:
    silhouette_constraints = spec.get("silhouette_constraints") or []
    symmetry = _norm(spec.get("symmetry"))
    c = 0.0
    if isinstance(silhouette_constraints, list) and len(silhouette_constraints) >= 1:
        c += 0.6
    if symmetry and symmetry != "none":
        c += 0.4
    return min(1.0, c)


def _support_plausibility_score(spec: Dict[str, Any], capabilities: CapabilityModel) -> float:
    outcome = _norm(spec.get("build_outcome_target"))
    allowed = [_norm(p) for p in (spec.get("allowed_primitives") or []) if _norm(p)]
    if outcome == "blocked":
        return 1.0 if _norm(spec.get("blocked_reason")) else 0.4
    if not allowed:
        return 0.0
    supported = 0
    for prim in allowed:
        required = capability_required_for_primitive(prim, "")
        if required != "unsupported" and capabilities.supports(required):
            supported += 1
    return supported / max(1, len(allowed))


def score_structural_spec(
    spec: Dict[str, Any],
    user_request: str,
    *,
    capabilities: Optional[CapabilityModel] = None,
) -> Dict[str, Any]:
    cap_model = capabilities or default_capability_model()
    family_fit = _family_fit_score(spec, user_request)
    capability_fit = _capability_fit_score(spec, cap_model)
    structural_coverage = _structural_coverage_score(spec)
    build_order_coherence = _build_order_coherence_score(spec)
    silhouette_consistency = _silhouette_consistency_score(spec)
    support_plausibility = _support_plausibility_score(spec, cap_model)

    total = (
        0.20 * family_fit
        + 0.20 * capability_fit
        + 0.20 * structural_coverage
        + 0.15 * build_order_coherence
        + 0.10 * silhouette_consistency
        + 0.15 * support_plausibility
    )

    return {
        "total": round(total, 6),
        "family_fit": round(family_fit, 6),
        "capability_fit": round(capability_fit, 6),
        "structural_coverage": round(structural_coverage, 6),
        "build_order_coherence": round(build_order_coherence, 6),
        "silhouette_consistency": round(silhouette_consistency, 6),
        "support_plausibility": round(support_plausibility, 6),
    }


def rank_structural_specs(
    specs: Sequence[Dict[str, Any]],
    user_request: str,
    *,
    capabilities: Optional[CapabilityModel] = None,
) -> List[Dict[str, Any]]:
    ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        breakdown = score_structural_spec(spec, user_request, capabilities=capabilities)
        ranked.append((float(breakdown["total"]), spec, breakdown))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "spec": spec,
            "score": score,
            "breakdown": breakdown,
        }
        for score, spec, breakdown in ranked
    ]


def _role_match_fraction(plan: Dict[str, Any], spec: Dict[str, Any]) -> float:
    roles = _extract_roles(spec)
    if not roles:
        return 0.0

    step_ids = [_norm((s or {}).get("id")) for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not step_ids:
        return 0.0

    matched = 0
    for role in roles:
        role_tokens = [t for t in role.replace("-", "_").split("_") if t]
        hit = False
        for sid in step_ids:
            if not sid:
                continue
            if role in sid:
                hit = True
                break
            token_hits = sum(1 for t in role_tokens if len(t) >= 3 and t in sid)
            if token_hits >= 1:
                hit = True
                break
        if hit:
            matched += 1
    return matched / max(1, len(roles))


def _plan_build_order_coherence(plan: Dict[str, Any], spec: Dict[str, Any]) -> float:
    order = [_norm(r) for r in (spec.get("build_order") or []) if _norm(r)]
    if not order:
        return 0.0

    step_ids = [_norm((s or {}).get("id")) for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not step_ids:
        return 0.0

    last_idx = -1
    matched = 0
    for role in order:
        found_idx = -1
        for i, sid in enumerate(step_ids):
            if role in sid:
                found_idx = i
                break
            tokens = [t for t in role.split("_") if t]
            if any(len(t) >= 3 and t in sid for t in tokens):
                found_idx = i
                break
        if found_idx < 0:
            continue
        if found_idx >= last_idx:
            matched += 1
            last_idx = found_idx
    return matched / max(1, len(order))


def _plan_capability_fit(plan: Dict[str, Any], capabilities: CapabilityModel) -> float:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return 0.0
    ok = 0
    for step in steps:
        primitive = _norm(step.get("primitive"))
        op = str(step.get("operation") or "")
        required = capability_required_for_primitive(primitive, op)
        if required != "unsupported" and capabilities.supports(required):
            ok += 1
    return ok / max(1, len(steps))


def _parse_plane(plane: Any) -> Tuple[str, float]:
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


def _rect_2d_bounds(profile: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cx = float(profile.get("cx", 0.0) or 0.0)
    cy = float(profile.get("cy", 0.0) or 0.0)
    w = abs(float(profile.get("w", 0.0) or 0.0))
    h = abs(float(profile.get("h", 0.0) or 0.0))
    return (cx - w * 0.5, cx + w * 0.5, cy - h * 0.5, cy + h * 0.5)


def _circle_2d_bounds(profile: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cx = float(profile.get("cx", 0.0) or 0.0)
    cy = float(profile.get("cy", 0.0) or 0.0)
    r = abs(float(profile.get("radius", 0.0) or 0.0))
    return (cx - r, cx + r, cy - r, cy + r)


def _poly_2d_bounds(profile: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    pts = [p for p in (profile.get("pts") or []) if isinstance(p, dict)]
    if not pts:
        return None
    xs = [float(p.get("x", 0.0) or 0.0) for p in pts]
    ys = [float(p.get("y", 0.0) or 0.0) for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))


def _step_bounds(step: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not isinstance(step, dict):
        return None
    profile = step.get("profile") if isinstance(step.get("profile"), dict) else {}
    ptype = _norm(profile.get("type"))
    if ptype == "rect":
        b2 = _rect_2d_bounds(profile)
    elif ptype == "circle":
        b2 = _circle_2d_bounds(profile)
    elif ptype == "poly":
        b2 = _poly_2d_bounds(profile)
        if b2 is None:
            return None
    else:
        return None

    base, off = _parse_plane(step.get("plane"))
    d = abs(float(step.get("distance", 0.0) or 0.0))
    p0, p1, q0, q1 = b2
    if base == "XY":
        return {"x_min": p0, "x_max": p1, "y_min": q0, "y_max": q1, "z_min": off, "z_max": off + d}
    if base == "XZ":
        return {"x_min": p0, "x_max": p1, "y_min": off, "y_max": off + d, "z_min": q0, "z_max": q1}
    if base == "YZ":
        return {"x_min": off, "x_max": off + d, "y_min": p0, "y_max": p1, "z_min": q0, "z_max": q1}
    return None


def _center(bounds: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        (bounds["x_min"] + bounds["x_max"]) * 0.5,
        (bounds["y_min"] + bounds["y_max"]) * 0.5,
        (bounds["z_min"] + bounds["z_max"]) * 0.5,
    )


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def _bounds_overlap_xy(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return _axis_overlap(a["x_min"], a["x_max"], b["x_min"], b["x_max"]) > 0.0 and _axis_overlap(
        a["y_min"], a["y_max"], b["y_min"], b["y_max"]
    ) > 0.0


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


def _step_id_matches_role(step_id: str, role: str) -> bool:
    sid = _norm(step_id)
    rr = _norm(role)
    if not sid or not rr:
        return False
    if rr in sid:
        return True
    tokens = [t for t in rr.replace("-", "_").split("_") if t]
    return any(len(t) >= 3 and t in sid for t in tokens)


def _match_step_for_role(steps: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for step in steps:
        if _step_id_matches_role(step.get("id"), role):
            return step
    return None


def _role_to_step_coherence(plan: Dict[str, Any], spec: Dict[str, Any]) -> float:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    role_parts = []
    for key in ("main_masses", "supporting_parts", "secondary_parts"):
        role_parts.extend([p for p in (spec.get(key) or []) if isinstance(p, dict)])
    if not role_parts:
        return 0.0

    matched = 0.0
    for part in role_parts:
        role = str(part.get("role") or "").strip()
        hint = _norm(part.get("primitive_hint"))
        step = _match_step_for_role(steps, role)
        if step is None:
            continue
        matched += 0.6
        primitive = _norm(step.get("primitive"))
        if hint and hint in primitive:
            matched += 0.4
    return matched / max(1.0, float(len(role_parts)))


def _primitive_appropriateness(plan: Dict[str, Any], spec: Dict[str, Any]) -> float:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return 0.0
    counts: Dict[str, int] = {}
    for step in steps:
        p = _norm(step.get("primitive"))
        counts[p] = counts.get(p, 0) + 1

    family = _norm(spec.get("object_family"))
    total = float(len(steps))
    rect = counts.get("rect_extrude", 0) / total
    poly = counts.get("poly_extrude", 0) / total
    wedge = counts.get("wedge_extrude", 0) / total
    circle = counts.get("circle_extrude", 0) / total

    if family == "lowpoly_hard_surface_vehicle":
        return max(0.0, min(1.0, 0.55 * (poly + wedge) + 0.35 * circle + 0.10 * max(0.0, 1.0 - rect)))
    if family == "furniture_boxy_panel":
        return max(0.0, min(1.0, 0.75 * rect + 0.25 * max(0.0, 1.0 - wedge)))
    if family == "profile_driven_symmetric":
        return max(0.0, min(1.0, 0.45 * poly + 0.45 * rect + 0.10 * circle))
    if family == "rotational_bodies":
        return max(0.0, min(1.0, 0.70 * circle + 0.30 * poly))
    return 0.5


def _attachment_plausibility(plan: Dict[str, Any], spec: Dict[str, Any]) -> float:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return 0.0
    family = _norm(spec.get("object_family"))
    by_id = {str((s or {}).get("id") or ""): s for s in steps}
    bounds = {sid: _step_bounds(step) for sid, step in by_id.items()}

    if family == "lowpoly_hard_surface_vehicle":
        checks = []
        upper = bounds.get("upper_hull")
        turret = bounds.get("turret")
        gun = bounds.get("gun")
        left = bounds.get("left_track_module")
        right = bounds.get("right_track_module")
        if upper and turret:
            checks.append(1.0 if turret["z_min"] >= upper["z_max"] - 2.0 and _bounds_overlap_xy(upper, turret) else 0.0)
        if turret and gun:
            checks.append(1.0 if gun["y_max"] > turret["y_max"] and _axis_overlap(gun["z_min"], gun["z_max"], turret["z_min"], turret["z_max"]) > 0 else 0.0)
        if left and right:
            lc = _center(left)[0]
            rc = _center(right)[0]
            checks.append(1.0 if lc < 0.0 < rc and _axis_overlap(left["y_min"], left["y_max"], right["y_min"], right["y_max"]) > 0 else 0.0)
        return sum(checks) / max(1, len(checks))

    if family == "furniture_boxy_panel":
        checks = []
        supports = [b for sid, b in bounds.items() if b and "support" in _norm(sid)]
        base = bounds.get("base_panel")
        if supports:
            checks.append(sum(1.0 for b in supports if b["z_min"] <= 0.15) / float(len(supports)))
        if supports and base:
            top_touch = sum(1.0 for b in supports if abs(b["z_max"] - base["z_min"]) <= 5.0 and _bounds_overlap_xy(b, base))
            checks.append(top_touch / float(len(supports)))
        return sum(checks) / max(1, len(checks))

    if family == "profile_driven_symmetric":
        left = bounds.get("left_support")
        right = bounds.get("right_support")
        body = bounds.get("primary_profile_body")
        checks = []
        if left and right:
            lc = _center(left)[0]
            rc = _center(right)[0]
            checks.append(1.0 if abs(abs(lc) - abs(rc)) < 4.0 and lc < 0 < rc else 0.0)
        if body and left and right:
            checks.append(1.0 if _bounds_overlap_xy(body, left) and _bounds_overlap_xy(body, right) else 0.0)
        return sum(checks) / max(1, len(checks))

    if family == "rotational_bodies":
        core = bounds.get("radial_core")
        neck = bounds.get("neck_or_top") or bounds.get("neck_cap")
        if core and neck:
            ccore = _center(core)
            cneck = _center(neck)
            center_ok = abs(ccore[0] - cneck[0]) < 1.5 and abs(ccore[1] - cneck[1]) < 1.5
            stacked = neck["z_min"] >= core["z_min"] and neck["z_max"] >= core["z_max"] - 1.0
            return 1.0 if center_ok and stacked else 0.0
        return 0.5

    return 0.5


def _silhouette_plausibility(plan: Dict[str, Any], spec: Dict[str, Any]) -> float:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    bounds_list = [b for b in (_step_bounds(s) for s in steps) if b is not None]
    union = _bounds_union(bounds_list)
    if union is None:
        return 0.0
    family = _norm(spec.get("object_family"))
    span_x = max(0.1, union["x_max"] - union["x_min"])
    span_y = max(0.1, union["y_max"] - union["y_min"])
    span_z = max(0.1, union["z_max"] - union["z_min"])

    if family == "lowpoly_hard_surface_vehicle":
        return 1.0 if span_y > span_z * 2.0 and span_y > span_x * 1.2 else 0.35
    if family == "furniture_boxy_panel":
        footprint = span_x * span_y
        return 1.0 if footprint > span_z * 12.0 else 0.45
    if family == "profile_driven_symmetric":
        return 1.0 if span_y > span_z * 1.4 else 0.5
    if family == "rotational_bodies":
        xy_balance = abs(span_x - span_y) / max(span_x, span_y)
        return 1.0 if xy_balance < 0.30 else 0.55
    return 0.5


def _support_plausibility(plan: Dict[str, Any], spec: Dict[str, Any], capabilities: CapabilityModel) -> float:
    allowed = {_norm(p) for p in (spec.get("allowed_primitives") or []) if _norm(p)}
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return 0.0
    if not allowed:
        return 0.0

    cap_ok = 0
    for step in steps:
        primitive = _norm(step.get("primitive"))
        op = str(step.get("operation") or "")
        if primitive not in allowed:
            continue
        req = capability_required_for_primitive(primitive, op)
        if req != "unsupported" and capabilities.supports(req):
            cap_ok += 1

    bounds_list = [b for b in (_step_bounds(s) for s in steps) if b is not None]
    touches_floor = sum(1 for b in bounds_list if b["z_min"] <= 0.2)
    floor_score = (touches_floor / max(1, len(bounds_list))) if bounds_list else 0.0
    cap_score = cap_ok / max(1, len(steps))
    return 0.7 * cap_score + 0.3 * floor_score


def _risky_detail_ratio(plan: Dict[str, Any]) -> float:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return 1.0

    risky = 0
    for step in steps:
        meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
        risk = _norm(meta.get("risk_level"))
        primitive = _norm(step.get("primitive"))
        operation = str(step.get("operation") or "").strip()
        if risk == "risky_detail":
            risky += 1
            continue
        if primitive == "cut_extrude" or operation in ("CutFeatureOperation", "IntersectFeatureOperation"):
            risky += 1
    return risky / max(1, len(steps))


def _create_safety_score(plan: Dict[str, Any]) -> float:
    global_cfg = plan.get("global") if isinstance(plan.get("global"), dict) else {}
    safety = global_cfg.get("create_safety") if isinstance(global_cfg.get("create_safety"), dict) else {}
    mode = _norm(safety.get("mode"))
    if mode == "primary_first":
        return 1.0
    if mode == "safe_detail_candidate":
        pruned = float(safety.get("risky_pruned_count", 0) or 0)
        kept = float(safety.get("risky_retained_count", 0) or 0)
        if kept <= 0:
            return 0.9
        return max(0.45, min(0.90, (pruned + 1.0) / (pruned + kept + 1.0)))
    return 0.6


def score_dsl_candidate_against_spec(
    plan: Dict[str, Any],
    spec: Dict[str, Any],
    *,
    capabilities: Optional[CapabilityModel] = None,
) -> Dict[str, Any]:
    cap_model = capabilities or default_capability_model()
    family_fit = 1.0 if _norm(spec.get("object_family")) in _norm((plan.get("global") or {}).get("shape_family")) else 0.75
    capability_fit = _plan_capability_fit(plan, cap_model)
    structural_coverage = _role_match_fraction(plan, spec)
    role_to_step_coherence = _role_to_step_coherence(plan, spec)
    primitive_appropriateness = _primitive_appropriateness(plan, spec)
    attachment_plausibility = _attachment_plausibility(plan, spec)
    build_order_coherence = _plan_build_order_coherence(plan, spec)
    ordering_plausibility = build_order_coherence
    silhouette_consistency = _silhouette_plausibility(plan, spec)
    support_plausibility = _support_plausibility(plan, spec, cap_model)
    risky_detail_ratio = _risky_detail_ratio(plan)
    robust_primary_score = max(0.0, min(1.0, 1.0 - risky_detail_ratio))
    create_safety_score = _create_safety_score(plan)

    total = (
        0.10 * family_fit
        + 0.10 * capability_fit
        + 0.14 * structural_coverage
        + 0.12 * role_to_step_coherence
        + 0.10 * primitive_appropriateness
        + 0.12 * attachment_plausibility
        + 0.09 * silhouette_consistency
        + 0.08 * support_plausibility
        + 0.09 * ordering_plausibility
        + 0.10 * robust_primary_score
        + 0.08 * create_safety_score
        - 0.12 * risky_detail_ratio
    )

    return {
        "total": round(total, 6),
        "family_fit": round(family_fit, 6),
        "capability_fit": round(capability_fit, 6),
        "structural_coverage": round(structural_coverage, 6),
        "role_to_step_coherence": round(role_to_step_coherence, 6),
        "primitive_appropriateness": round(primitive_appropriateness, 6),
        "attachment_plausibility": round(attachment_plausibility, 6),
        "build_order_coherence": round(build_order_coherence, 6),
        "ordering_plausibility": round(ordering_plausibility, 6),
        "silhouette_consistency": round(silhouette_consistency, 6),
        "support_plausibility": round(support_plausibility, 6),
        "risky_detail_ratio": round(risky_detail_ratio, 6),
        "robust_primary_score": round(robust_primary_score, 6),
        "create_safety_score": round(create_safety_score, 6),
    }


def rank_dsl_candidates(
    candidates: Sequence[Dict[str, Any]],
    spec: Dict[str, Any],
    *,
    capabilities: Optional[CapabilityModel] = None,
) -> List[Dict[str, Any]]:
    ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for plan in candidates or []:
        if not isinstance(plan, dict):
            continue
        breakdown = score_dsl_candidate_against_spec(plan, spec, capabilities=capabilities)
        ranked.append((float(breakdown["total"]), plan, breakdown))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "plan": plan,
            "score": score,
            "breakdown": breakdown,
        }
        for score, plan, breakdown in ranked
    ]
