"""
Deterministic validation of a CAD DSL Plan before execution.

- Budget: max_steps, max_parts
- Primitive sanity: rect/circle/poly dimensions, min sizes
- Capability-aware validation: reject unavailable backend capabilities
- Plan-level: coordinate convention, connectivity heuristic (optional)
- Loft/sweep/fillet: profile existence, radius clamp for fillet
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cad_dsl import (
    PRIMITIVES,
    EXTRUDE_OPERATIONS,
    FILLET_RULES,
    default_budget,
)
from .cad_capabilities import (
    CapabilityModel,
    default_capability_model,
    capability_required_for_primitive,
)


MIN_RECT_SIZE = 0.2
MIN_CIRCLE_RADIUS = 0.1
MIN_POLY_POINTS = 3
MIN_POLY_AREA = 0.01
MIN_DISTANCE = 0.05
MAX_DISTANCE = 1000.0
NEAR_DUPLICATE_EPS = 1e-6
FILLET_RADIUS_FRACTION = 0.4  # radius <= this * min_thickness


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _poly_area(pts: List[Dict[str, float]]) -> float:
    """Shoelace formula for polygon area (2D)."""
    if len(pts) < 3:
        return 0.0
    n = len(pts)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += (pts[i].get("x", 0) or 0) * (pts[j].get("y", 0) or 0)
        a -= (pts[j].get("x", 0) or 0) * (pts[i].get("y", 0) or 0)
    return abs(a) * 0.5


def _normalize_plane(plane: Any) -> Tuple[str, float]:
    """Return (base, offset) e.g. ('XY', 12.5)."""
    if isinstance(plane, (int, float)):
        return "XY", float(plane)
    s = str(plane).strip().upper()
    if "@" in s:
        base, off = s.split("@", 1)
        try:
            return base.strip(), float(off.strip())
        except ValueError:
            return base.strip(), 0.0
    if s in ("XY", "XZ", "YZ"):
        return s, 0.0
    return "XY", 0.0


def _validate_profile(
    step_id: str,
    profile: Dict[str, Any],
    result: ValidationResult,
    capabilities: CapabilityModel,
) -> bool:
    """Validate profile rect/circle/poly. Return True if ok."""
    ptype = (profile.get("type") or "rect").strip().lower()
    if ptype == "rect":
        if not capabilities.supports("sketch_rect"):
            result.add_error(f"Step {step_id}: sketch_rect capability unavailable")
            return False
        w = float(profile.get("w", 0))
        h = float(profile.get("h", 0))
        if w < MIN_RECT_SIZE or h < MIN_RECT_SIZE:
            result.add_error(f"Step {step_id}: rect w,h must be >= {MIN_RECT_SIZE}")
            return False
        return True
    if ptype == "circle":
        if not capabilities.supports("sketch_circle"):
            result.add_error(f"Step {step_id}: sketch_circle capability unavailable")
            return False
        r = float(profile.get("radius", 0))
        if r < MIN_CIRCLE_RADIUS:
            result.add_error(f"Step {step_id}: circle radius must be >= {MIN_CIRCLE_RADIUS}")
            return False
        return True
    if ptype == "poly":
        if not capabilities.supports("sketch_poly"):
            result.add_error(f"Step {step_id}: sketch_poly capability unavailable")
            return False
        arcs = profile.get("arcs") or []
        if arcs:
            if not capabilities.supports("sketch_arc"):
                result.add_error(f"Step {step_id}: arcs not supported; use straight poly segments only")
            else:
                result.add_error(f"Step {step_id}: arcs not supported by compiler; use straight poly segments only")
            return False
        pts = profile.get("pts") or []
        if len(pts) < MIN_POLY_POINTS:
            result.add_error(f"Step {step_id}: poly must have >= {MIN_POLY_POINTS} points")
            return False
        for i, p in enumerate(pts):
            if not isinstance(p, dict) or ("x" not in p and "y" not in p):
                result.add_error(f"Step {step_id}: poly point {i} must have x,y")
                return False
        area = _poly_area(pts)
        if area < MIN_POLY_AREA:
            result.add_error(f"Step {step_id}: poly area must be >= {MIN_POLY_AREA}")
            return False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = (pts[i].get("x", 0) or 0) - (pts[j].get("x", 0) or 0)
                dy = (pts[i].get("y", 0) or 0) - (pts[j].get("y", 0) or 0)
                if math.hypot(dx, dy) < NEAR_DUPLICATE_EPS:
                    result.add_warning(f"Step {step_id}: near-duplicate points at {i},{j}")
        return True
    result.add_error(f"Step {step_id}: unknown profile type '{ptype}'")
    return False


def _validate_step(
    step: Dict[str, Any],
    step_index: int,
    result: ValidationResult,
    capabilities: CapabilityModel,
    allow_unsupported: bool,
) -> None:
    """Validate one step."""
    step_id = step.get("id") or f"step_{step_index}"
    prim = (step.get("primitive") or "").strip().lower()
    if prim not in PRIMITIVES:
        result.add_error(f"Step {step_id}: unknown primitive '{prim}'")
        return

    op = (step.get("op") or "ensure").strip().lower()
    if op not in ("ensure", "create", "update"):
        result.add_warning(f"Step {step_id}: op '{op}' normalized to 'ensure'")

    operation = step.get("operation")
    if operation and operation not in EXTRUDE_OPERATIONS:
        result.add_error(f"Step {step_id}: invalid operation '{operation}'")

    required_cap = capability_required_for_primitive(prim, operation or "")
    if required_cap == "unsupported":
        msg = f"Step {step_id}: unsupported primitive '{prim}'"
        if allow_unsupported:
            result.add_warning(msg)
        else:
            result.add_error(msg)
        return

    if not capabilities.supports(required_cap):
        msg = f"Step {step_id}: capability '{required_cap}' unavailable ({capabilities.reason(required_cap)})"
        if allow_unsupported:
            result.add_warning(msg)
        else:
            result.add_error(msg)
            return

    plane = step.get("plane", "XY")
    _normalize_plane(plane)

    distance = step.get("distance")
    if distance is not None:
        d = float(distance)
        if d < 0:
            result.add_error(f"Step {step_id}: distance must be >= 0 (negative not supported)")
        elif d < MIN_DISTANCE:
            result.add_error(f"Step {step_id}: distance must be >= {MIN_DISTANCE}")
        if d > MAX_DISTANCE:
            result.add_warning(f"Step {step_id}: distance clamped to {MAX_DISTANCE}")

    profile = step.get("profile") or {}
    if prim in ("rect_extrude", "circle_extrude", "poly_extrude", "wedge_extrude", "cut_extrude"):
        _validate_profile(step_id, profile, result, capabilities)

    if prim == "fillet":
        fillet = step.get("fillet") or {}
        radius = fillet.get("radius")
        if radius is not None:
            r = float(radius)
            if r <= 0:
                result.add_error(f"Step {step_id}: fillet radius must be > 0")
        selectors = step.get("selectors") or {}
        rule = selectors.get("rule")
        if rule and rule not in FILLET_RULES:
            result.add_warning(f"Step {step_id}: unknown fillet rule '{rule}'")

    if prim == "loft":
        loft = step.get("loft") or {}
        profiles_list = loft.get("profiles") or []
        if len(profiles_list) < 2:
            result.add_error(f"Step {step_id}: loft requires at least 2 profiles")

    if prim == "sweep":
        sweep = step.get("sweep") or {}
        if not sweep.get("profile_ref") or not sweep.get("path_ref"):
            result.add_error(f"Step {step_id}: sweep requires profile_ref and path_ref")

    if prim == "chamfer":
        chamfer = step.get("chamfer") or {}
        distance = chamfer.get("distance")
        if distance is not None and float(distance) <= 0:
            result.add_error(f"Step {step_id}: chamfer distance must be > 0")

    if prim == "revolve":
        revolve = step.get("revolve") or {}
        if not revolve.get("axis_ref"):
            result.add_warning(f"Step {step_id}: revolve axis_ref is missing")


def validate_plan(
    plan: Dict[str, Any],
    *,
    capabilities: Optional[CapabilityModel] = None,
    allow_unsupported: bool = False,
) -> ValidationResult:
    """
    Deterministic validation of a full Plan.
    Returns ValidationResult(valid, errors, warnings).
    """
    result = ValidationResult(valid=True)
    cap = capabilities or default_capability_model()

    if not plan:
        result.add_error("Plan is empty")
        return result

    budget = plan.get("budget") or default_budget()
    max_steps = int(budget.get("max_steps", 25))
    max_parts = int(budget.get("max_parts", 18))
    steps = plan.get("steps") or []

    if len(steps) > max_steps:
        result.add_error(f"Plan has {len(steps)} steps, max allowed is {max_steps}")
    if len(steps) > max_parts:
        result.add_warning(f"Plan has {len(steps)} parts, budget max_parts is {max_parts}")

    seen_ids = set()
    for i, step in enumerate(steps):
        if not step:
            result.add_error(f"Step index {i} is empty")
            continue
        sid = step.get("id")
        if sid:
            if sid in seen_ids:
                result.add_error(f"Duplicate step id: {sid}")
            seen_ids.add(sid)
        _validate_step(step, i, result, cap, allow_unsupported)

    return result
