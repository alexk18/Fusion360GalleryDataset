"""
Deterministic compiler: CAD DSL Plan -> sequence of server call descriptors.

Each step is compiled to: add_sketch, draw profile (add_point/add_line/add_circle/close_profile),
add_extrude (or stub for loft/sweep/fillet until server supports them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cad_dsl import Step, make_step_names, PLANES


@dataclass
class CompiledCall:
    """Single server call: command name + data payload."""
    command: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledStep:
    """One DSL step compiled to a sequence of server calls. profile_id_source: which prior call yields profile_id for extrude."""
    step_id: str
    primitive: str
    calls: List[CompiledCall] = field(default_factory=list)
    profile_id_source: int = -1  # index in calls that returns profile_id (e.g. close_profile or add_circle)


def _plane_to_server(plane: str) -> str:
    """Normalize plane to server form: XY, XZ, YZ or XY@z."""
    s = str(plane).strip().upper()
    if not s:
        return "XY"
    if "@" in s:
        base, off = s.split("@", 1)
        base = base.strip()
        try:
            off_f = float(off.strip())
            return f"{base}@{off_f}"
        except ValueError:
            return base
    return s if s in PLANES else "XY"


def _rect_profile_calls(cx: float, cy: float, w: float, h: float, sketch_name_key: str) -> List[CompiledCall]:
    """Four corners: left-bottom, right-bottom, right-top, left-top. Sequential add_point creates lines."""
    half_w, half_h = w / 2, h / 2
    pts = [
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    ]
    out = []
    for i, (px, py) in enumerate(pts):
        out.append(CompiledCall("add_point", {
            "sketch_name": f"__ref:{sketch_name_key}",
            "pt": {"x": px, "y": py, "z": 0},
        }))
    out.append(CompiledCall("close_profile", {"sketch_name": f"__ref:{sketch_name_key}"}))
    return out


def _circle_profile_calls(cx: float, cy: float, radius: float, sketch_name_key: str) -> List[CompiledCall]:
    return [
        CompiledCall("add_circle", {
            "sketch_name": f"__ref:{sketch_name_key}",
            "pt": {"x": cx, "y": cy, "z": 0},
            "radius": radius,
        }),
    ]


def _poly_profile_calls(pts: List[Dict], arcs: Optional[List[Dict]], sketch_name_key: str) -> List[CompiledCall]:
    """Polygon: add_point for each vertex; optional arcs approximated by segments or single arc if one segment."""
    out = []
    for p in pts:
        x = float(p.get("x", 0))
        y = float(p.get("y", 0))
        out.append(CompiledCall("add_point", {
            "sketch_name": f"__ref:{sketch_name_key}",
            "pt": {"x": x, "y": y, "z": 0},
        }))
    out.append(CompiledCall("close_profile", {"sketch_name": f"__ref:{sketch_name_key}"}))
    return out


def _compile_step(step: Dict[str, Any], session: str) -> Optional[CompiledStep]:
    """Compile one DSL step to server calls. Returns None if primitive not yet supported."""
    step_id = str(step.get("id") or "step")
    prim = (step.get("primitive") or "").strip().lower()
    names = step.get("names") or make_step_names(session, step_id)
    sketch_name = names.get("sketch") or f"{session}__{step_id}__sk"
    feature_name = names.get("feature") or f"{session}__{step_id}__feat"
    plane = _plane_to_server(step.get("plane", "XY"))
    profile = step.get("profile") or {}
    distance = float(step.get("distance", 1.0))
    operation = step.get("operation", "NewBodyFeatureOperation")

    if prim in ("rect_extrude", "cut_extrude"):
        ptype = (profile.get("type") or "rect").lower()
        cx = float(profile.get("cx", 0))
        cy = float(profile.get("cy", 0))
        w = float(profile.get("w", 1))
        h = float(profile.get("h", 1))
        calls = [
            CompiledCall("add_sketch", {"sketch_plane": plane, "sketch_name": sketch_name}),
            *_rect_profile_calls(cx, cy, w, h, "sketch_name"),
            CompiledCall("add_extrude", {
                "sketch_name": "__ref:sketch_name",
                "profile_id": "__ref:profile_id",
                "distance": distance,
                "operation": operation,
                "feature_name": feature_name,
            }),
        ]
        profile_id_source = len(calls) - 2
        return CompiledStep(step_id=step_id, primitive=prim, calls=calls, profile_id_source=profile_id_source)

    if prim == "circle_extrude":
        ptype = (profile.get("type") or "circle").lower()
        cx = float(profile.get("cx", 0))
        cy = float(profile.get("cy", 0))
        radius = float(profile.get("radius", 1))
        calls = [
            CompiledCall("add_sketch", {"sketch_plane": plane, "sketch_name": sketch_name}),
            *_circle_profile_calls(cx, cy, radius, "sketch_name"),
            CompiledCall("add_extrude", {
                "sketch_name": "__ref:sketch_name",
                "profile_id": "__ref:profile_id",
                "distance": distance,
                "operation": operation,
                "feature_name": feature_name,
            }),
        ]
        profile_id_source = 1
        return CompiledStep(step_id=step_id, primitive=prim, calls=calls, profile_id_source=profile_id_source)

    if prim in ("poly_extrude", "wedge_extrude"):
        pts = profile.get("pts") or []
        arcs = profile.get("arcs")
        if len(pts) < 3:
            return None
        calls = [
            CompiledCall("add_sketch", {"sketch_plane": plane, "sketch_name": sketch_name}),
            *_poly_profile_calls(pts, arcs, "sketch_name"),
            CompiledCall("add_extrude", {
                "sketch_name": "__ref:sketch_name",
                "profile_id": "__ref:profile_id",
                "distance": distance,
                "operation": operation,
                "feature_name": feature_name,
            }),
        ]
        profile_id_source = len(calls) - 2
        return CompiledStep(step_id=step_id, primitive=prim, calls=calls, profile_id_source=profile_id_source)

    if prim in ("loft", "sweep", "fillet"):
        return CompiledStep(step_id=step_id, primitive=prim, calls=[], profile_id_source=-1)

    return None


def compile_plan(plan: Dict[str, Any]) -> List[CompiledStep]:
    """
    Compile a full Plan to a list of CompiledStep.
    Steps with unsupported primitives (loft/sweep/fillet) get empty calls (executor can skip or stub).
    """
    session = str(plan.get("session") or "session")
    steps = plan.get("steps") or []
    result = []
    for s in steps:
        if not s:
            continue
        cs = _compile_step(s, session)
        if cs is not None:
            result.append(cs)
    return result
