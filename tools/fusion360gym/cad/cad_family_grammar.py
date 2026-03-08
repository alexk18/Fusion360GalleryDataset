"""
Family-specific deterministic CAD DSL construction grammars.

This module maps Structural Build Spec families/roles to CAD DSL step patterns,
with explicit construction order and silhouette/support constraints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .cad_dsl import make_step_names


def grammar_id_for_family(family: str) -> str:
    f = str(family or "").strip().lower()
    if f == "furniture_boxy_panel":
        return "furniture_panel_support_grammar_v1"
    if f == "lowpoly_hard_surface_vehicle":
        return "lowpoly_vehicle_mass_grammar_v1"
    if f == "profile_driven_symmetric":
        return "profile_symmetric_body_grammar_v1"
    if f == "rotational_bodies":
        return "rotational_stepped_approx_grammar_v1"
    return "blocked_family_grammar_v1"


def _role_buckets(spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    main = [p for p in (spec.get("main_masses") or []) if isinstance(p, dict)]
    supporting = [p for p in (spec.get("supporting_parts") or []) if isinstance(p, dict)]
    secondary = [p for p in (spec.get("secondary_parts") or []) if isinstance(p, dict)]
    return main, supporting, secondary


def _has_role(spec: Dict[str, Any], role_name: str) -> bool:
    rn = str(role_name or "").strip().lower()
    if not rn:
        return False
    for bucket in _role_buckets(spec):
        for part in bucket:
            if str(part.get("role") or "").strip().lower() == rn:
                return True
    return False


def _step(
    *,
    session: str,
    step_id: str,
    primitive: str,
    plane: str,
    profile: Dict[str, Any],
    distance: float,
    operation: str,
    detail_level: str = "",
    risk_level: str = "",
    essential: bool = True,
) -> Dict[str, Any]:
    sid = str(step_id or "").strip().lower()
    prim = str(primitive or "").strip().lower()
    op = str(operation or "").strip()

    if not detail_level:
        if prim == "cut_extrude" or op in ("CutFeatureOperation", "IntersectFeatureOperation"):
            detail_level = "secondary"
        elif any(k in sid for k in ("wheel", "ring", "cap", "top_panel", "back_panel")):
            detail_level = "secondary"
        else:
            detail_level = "core"

    if not risk_level:
        if prim == "cut_extrude" or op in ("CutFeatureOperation", "IntersectFeatureOperation"):
            risk_level = "risky_detail"
        elif detail_level == "core":
            risk_level = "core"
        else:
            risk_level = "secondary"

    if risk_level == "risky_detail" and essential is True:
        # Boolean/detail steps default to non-essential in first-pass create mode.
        essential = False

    return {
        "id": step_id,
        "op": "ensure",
        "primitive": primitive,
        "names": make_step_names(session, step_id),
        "plane": plane,
        "profile": profile,
        "distance": float(distance),
        "operation": operation,
        "meta": {
            "role": step_id,
            "detail_level": detail_level,
            "risk_level": risk_level,
            "essential": bool(essential),
        },
    }


def _synthesize_lowpoly_vehicle(spec: Dict[str, Any], session: str, variant: int = 0) -> List[Dict[str, Any]]:
    length = 70.0 + 4.0 * variant
    hull_w = 34.0 + 1.5 * variant
    lower_h = 10.0 + 0.8 * variant
    upper_h = 8.0
    track_w = 9.0
    track_h = 9.5
    turret_w = 20.0 + 1.0 * variant
    turret_l = 18.0 + 1.5 * variant
    turret_h = 6.0
    gun_radius = 1.2
    gun_len = 22.0 + 3.0 * variant

    steps: List[Dict[str, Any]] = []

    if _has_role(spec, "lower_hull"):
        steps.append(
            _step(
                session=session,
                step_id="lower_hull",
                primitive="wedge_extrude",
                plane="XY@0",
                profile={
                    "type": "poly",
                    "pts": [
                        {"x": -hull_w * 0.50, "y": 0.0},
                        {"x": hull_w * 0.50, "y": 0.0},
                        {"x": hull_w * 0.42, "y": length},
                        {"x": -hull_w * 0.42, "y": length},
                    ],
                },
                distance=lower_h,
                operation="NewBodyFeatureOperation",
            )
        )

    if _has_role(spec, "upper_hull"):
        steps.append(
            _step(
                session=session,
                step_id="upper_hull",
                primitive="poly_extrude",
                plane=f"XY@{lower_h}",
                profile={
                    "type": "poly",
                    "pts": [
                        {"x": -hull_w * 0.40, "y": length * 0.18},
                        {"x": hull_w * 0.40, "y": length * 0.18},
                        {"x": hull_w * 0.36, "y": length * 0.82},
                        {"x": -hull_w * 0.36, "y": length * 0.82},
                    ],
                },
                distance=upper_h,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "left_track_module"):
        steps.append(
            _step(
                session=session,
                step_id="left_track_module",
                primitive="poly_extrude",
                plane="XY@0",
                profile={
                    "type": "poly",
                    "pts": [
                        {"x": -(hull_w * 0.5 + track_w), "y": length * 0.05},
                        {"x": -(hull_w * 0.5), "y": length * 0.05},
                        {"x": -(hull_w * 0.5), "y": length * 0.95},
                        {"x": -(hull_w * 0.5 + track_w), "y": length * 0.95},
                    ],
                },
                distance=track_h,
                operation="NewBodyFeatureOperation",
            )
        )

    if _has_role(spec, "right_track_module"):
        steps.append(
            _step(
                session=session,
                step_id="right_track_module",
                primitive="poly_extrude",
                plane="XY@0",
                profile={
                    "type": "poly",
                    "pts": [
                        {"x": hull_w * 0.5, "y": length * 0.05},
                        {"x": hull_w * 0.5 + track_w, "y": length * 0.05},
                        {"x": hull_w * 0.5 + track_w, "y": length * 0.95},
                        {"x": hull_w * 0.5, "y": length * 0.95},
                    ],
                },
                distance=track_h,
                operation="NewBodyFeatureOperation",
            )
        )

    turret_z = lower_h + upper_h
    turret_cy = length * 0.60
    if _has_role(spec, "turret"):
        steps.append(
            _step(
                session=session,
                step_id="turret",
                primitive="poly_extrude",
                plane=f"XY@{turret_z}",
                profile={
                    "type": "poly",
                    "pts": [
                        {"x": -turret_w * 0.5, "y": turret_cy - turret_l * 0.5},
                        {"x": turret_w * 0.5, "y": turret_cy - turret_l * 0.5},
                        {"x": turret_w * 0.44, "y": turret_cy + turret_l * 0.5},
                        {"x": -turret_w * 0.44, "y": turret_cy + turret_l * 0.5},
                    ],
                },
                distance=turret_h,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "gun"):
        steps.append(
            _step(
                session=session,
                step_id="gun",
                primitive="circle_extrude",
                plane=f"XZ@{turret_cy + turret_l * 0.35}",
                profile={
                    "type": "circle",
                    "cx": 0.0,
                    "cy": turret_z + turret_h * 0.55,
                    "radius": gun_radius,
                },
                distance=gun_len,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "front_wheels"):
        steps.append(
            _step(
                session=session,
                step_id="front_wheels",
                primitive="circle_extrude",
                plane=f"YZ@{-hull_w * 0.5 - track_w * 0.85}",
                profile={
                    "type": "circle",
                    "cx": length * 0.15,
                    "cy": track_h * 0.45,
                    "radius": 2.4,
                },
                distance=track_w * 0.7,
                operation="NewBodyFeatureOperation",
            )
        )

    if _has_role(spec, "rear_wheels"):
        steps.append(
            _step(
                session=session,
                step_id="rear_wheels",
                primitive="circle_extrude",
                plane=f"YZ@{hull_w * 0.5 + track_w * 0.15}",
                profile={
                    "type": "circle",
                    "cx": length * 0.86,
                    "cy": track_h * 0.45,
                    "radius": 2.4,
                },
                distance=track_w * 0.7,
                operation="NewBodyFeatureOperation",
            )
        )

    if _has_role(spec, "service_cutouts"):
        steps.append(
            _step(
                session=session,
                step_id="service_cutouts",
                primitive="cut_extrude",
                plane=f"XY@{lower_h * 0.35}",
                profile={
                    "type": "rect",
                    "cx": 0.0,
                    "cy": length * 0.52,
                    "w": hull_w * 0.22,
                    "h": length * 0.10,
                },
                distance=2.5,
                operation="CutFeatureOperation",
            )
        )

    return steps


def _synthesize_furniture(spec: Dict[str, Any], session: str, user_request: str, variant: int = 0) -> List[Dict[str, Any]]:
    text = str(user_request or "").strip().lower()
    support_first = any(k in text for k in ("chair", "stool", "bench", "table", "desk"))
    symmetry_required = (str(spec.get("symmetry") or "").strip().lower() in ("approx_x", "x", "mirror_x")) or (
        "symmetric" in text or "symmetry" in text
    )

    span_w = 52.0 + 3.0 * variant
    span_d = 40.0 + 2.0 * variant
    support_h = 42.0 + 1.5 * variant
    support_w = 4.2
    panel_t = 2.8
    seat_z = support_h

    left_x = -span_w * 0.35
    right_x = span_w * 0.35 if symmetry_required else span_w * 0.28
    front_y = span_d * 0.28
    back_y = span_d * 0.58

    steps: List[Dict[str, Any]] = []

    def _add_support(step_id: str, cx: float, cy: float) -> None:
        steps.append(
            _step(
                session=session,
                step_id=step_id,
                primitive="rect_extrude",
                plane="XY@0",
                profile={"type": "rect", "cx": cx, "cy": cy, "w": support_w, "h": support_w},
                distance=support_h,
                operation="NewBodyFeatureOperation",
            )
        )

    if support_first:
        _add_support("left_support", left_x, front_y)
        _add_support("right_support", right_x, front_y)

    if _has_role(spec, "base_panel"):
        steps.append(
            _step(
                session=session,
                step_id="base_panel",
                primitive="rect_extrude",
                plane=f"XY@{seat_z}",
                profile={"type": "rect", "cx": 0.0, "cy": front_y, "w": span_w * 0.85, "h": span_d * 0.50},
                distance=panel_t,
                operation="JoinFeatureOperation" if support_first else "NewBodyFeatureOperation",
            )
        )

    if not support_first:
        _add_support("left_support", left_x, front_y)
        _add_support("right_support", right_x, front_y)

    if _has_role(spec, "top_panel"):
        steps.append(
            _step(
                session=session,
                step_id="top_panel",
                primitive="rect_extrude",
                plane=f"XY@{seat_z + panel_t}",
                profile={"type": "rect", "cx": 0.0, "cy": front_y, "w": span_w * 0.80, "h": span_d * 0.46},
                distance=panel_t,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "back_panel"):
        steps.append(
            _step(
                session=session,
                step_id="back_panel",
                primitive="rect_extrude",
                plane=f"XZ@{back_y}",
                profile={
                    "type": "rect",
                    "cx": 0.0,
                    "cy": seat_z + panel_t + support_h * 0.22,
                    "w": span_w * 0.78,
                    "h": support_h * 0.55,
                },
                distance=panel_t,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "utility_cutout"):
        steps.append(
            _step(
                session=session,
                step_id="utility_cutout",
                primitive="cut_extrude",
                plane=f"XY@{seat_z + panel_t * 0.3}",
                profile={"type": "rect", "cx": 0.0, "cy": front_y, "w": span_w * 0.22, "h": span_d * 0.18},
                distance=panel_t * 1.3,
                operation="CutFeatureOperation",
            )
        )

    return steps


def _synthesize_profile_symmetric(spec: Dict[str, Any], session: str, variant: int = 0) -> List[Dict[str, Any]]:
    body_len = 56.0 + 3.0 * variant
    body_w = 34.0 + 1.5 * variant
    body_h = 10.0
    support_w = 4.5
    support_h = 12.0

    steps: List[Dict[str, Any]] = []

    if _has_role(spec, "primary_profile_body"):
        steps.append(
            _step(
                session=session,
                step_id="primary_profile_body",
                primitive="poly_extrude",
                plane="XY@0",
                profile={
                    "type": "poly",
                    "pts": [
                        {"x": -body_w * 0.50, "y": 0.0},
                        {"x": body_w * 0.50, "y": 0.0},
                        {"x": body_w * 0.45, "y": body_len},
                        {"x": -body_w * 0.45, "y": body_len},
                    ],
                },
                distance=body_h,
                operation="NewBodyFeatureOperation",
            )
        )

    if _has_role(spec, "left_support"):
        steps.append(
            _step(
                session=session,
                step_id="left_support",
                primitive="rect_extrude",
                plane="XY@0",
                profile={"type": "rect", "cx": -body_w * 0.42, "cy": body_len * 0.60, "w": support_w, "h": support_w},
                distance=support_h,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "right_support"):
        steps.append(
            _step(
                session=session,
                step_id="right_support",
                primitive="rect_extrude",
                plane="XY@0",
                profile={"type": "rect", "cx": body_w * 0.42, "cy": body_len * 0.60, "w": support_w, "h": support_w},
                distance=support_h,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "functional_holes"):
        steps.append(
            _step(
                session=session,
                step_id="functional_holes",
                primitive="cut_extrude",
                plane=f"XY@{body_h * 0.35}",
                profile={"type": "rect", "cx": 0.0, "cy": body_len * 0.48, "w": body_w * 0.16, "h": body_len * 0.16},
                distance=body_h * 0.7,
                operation="CutFeatureOperation",
            )
        )

    return steps


def _synthesize_rotational(spec: Dict[str, Any], session: str, variant: int = 0) -> List[Dict[str, Any]]:
    base_r = 8.5 + 0.5 * variant
    core_h = 11.0 + 1.0 * variant
    mid_r = base_r * 0.75
    mid_h = 8.0
    neck_r = base_r * 0.43
    neck_h = 5.0

    steps: List[Dict[str, Any]] = []

    if _has_role(spec, "radial_core"):
        steps.append(
            _step(
                session=session,
                step_id="radial_core",
                primitive="circle_extrude",
                plane="XY@0",
                profile={"type": "circle", "cx": 0.0, "cy": 0.0, "radius": base_r},
                distance=core_h,
                operation="NewBodyFeatureOperation",
            )
        )

    if _has_role(spec, "neck_or_top"):
        steps.append(
            _step(
                session=session,
                step_id="neck_or_top",
                primitive="circle_extrude",
                plane=f"XY@{core_h}",
                profile={"type": "circle", "cx": 0.0, "cy": 0.0, "radius": mid_r},
                distance=mid_h,
                operation="JoinFeatureOperation",
            )
        )
        steps.append(
            _step(
                session=session,
                step_id="neck_cap",
                primitive="circle_extrude",
                plane=f"XY@{core_h + mid_h}",
                profile={"type": "circle", "cx": 0.0, "cy": 0.0, "radius": neck_r},
                distance=neck_h,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "base_ring"):
        steps.append(
            _step(
                session=session,
                step_id="base_ring",
                primitive="circle_extrude",
                plane="XY@0",
                profile={"type": "circle", "cx": 0.0, "cy": 0.0, "radius": base_r * 1.08},
                distance=1.6,
                operation="JoinFeatureOperation",
            )
        )

    if _has_role(spec, "profile_relief"):
        steps.append(
            _step(
                session=session,
                step_id="profile_relief",
                primitive="cut_extrude",
                plane=f"XY@{core_h * 0.45}",
                profile={"type": "rect", "cx": 0.0, "cy": 0.0, "w": base_r * 0.35, "h": base_r * 0.25},
                distance=1.0,
                operation="CutFeatureOperation",
            )
        )

    return steps


def synthesize_steps_from_family_grammar(
    spec: Dict[str, Any],
    *,
    session: str,
    user_request: str,
    variant: int = 0,
) -> List[Dict[str, Any]]:
    family = str(spec.get("object_family") or "").strip().lower()
    if family == "lowpoly_hard_surface_vehicle":
        return _synthesize_lowpoly_vehicle(spec, session, variant=variant)
    if family == "furniture_boxy_panel":
        return _synthesize_furniture(spec, session, user_request=user_request, variant=variant)
    if family == "profile_driven_symmetric":
        return _synthesize_profile_symmetric(spec, session, variant=variant)
    if family == "rotational_bodies":
        return _synthesize_rotational(spec, session, variant=variant)
    return []
