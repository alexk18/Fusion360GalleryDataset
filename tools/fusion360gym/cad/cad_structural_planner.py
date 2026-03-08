"""
Deterministic Structural Build Spec planner for create-mode.

Create pipeline contract:
text/image/folder -> shape family -> Structural Build Spec -> CAD DSL synthesis
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .cad_capabilities import CapabilityModel, default_capability_model
from .cad_structural_spec import (
    StructuralBuildSpec,
    StructuralPart,
    required_capabilities_for_primitives,
)


def _norm(text: str) -> str:
    return str(text or "").strip().lower()


def classify_shape_family(user_request: str, images: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    text = _norm(user_request)

    smooth_freeform = (
        "organic",
        "creature",
        "character",
        "human",
        "face",
        "head",
        "animal",
        "cloth",
        "fabric",
        "draped",
        "flowing",
        "sculpt",
        "statue",
        "freeform",
        "smooth surface",
        "biomorphic",
        "figurine",
    )
    rotational = (
        "vase",
        "bottle",
        "cup",
        "mug",
        "can",
        "cylinder",
        "cone",
        "spool",
        "pulley",
        "knob",
        "roller",
        "nozzle",
        "lathe",
        "axisymmetric",
        "axis-symmetric",
    )
    vehicle = (
        "tank",
        "vehicle",
        "truck",
        "car",
        "aircraft",
        "plane",
        "jet",
        "boat",
        "ship",
        "apc",
        "jeep",
        "drone",
        "mech",
        "robot",
        "crawler",
        "track",
        "armored",
        "turret",
    )
    profile_symmetric = (
        "bracket",
        "frame",
        "profile",
        "symmetric",
        "symmetry",
        "chassis",
        "airfoil",
        "wing section",
        "cross-section",
        "plate with cutouts",
    )
    furniture_boxy = (
        "table",
        "chair",
        "stool",
        "cabinet",
        "shelf",
        "desk",
        "bed",
        "bench",
        "drawer",
        "box",
        "case",
        "panel",
        "wardrobe",
    )

    # Check specific buildable families BEFORE the smooth_freeform catch-all.
    # This prevents generic freeform keywords (e.g. "head") from shadowing
    # compound words that belong to specific families (e.g. "headboard" -> bed).
    if any(k in text for k in furniture_boxy):
        return "furniture_boxy_panel"
    if any(k in text for k in vehicle):
        return "lowpoly_hard_surface_vehicle"
    if any(k in text for k in rotational):
        return "rotational_bodies"
    if any(k in text for k in profile_symmetric):
        return "profile_driven_symmetric"
    if any(k in text for k in smooth_freeform):
        return "unsupported_smooth_freeform"
    if images and len(images) >= 3:
        return "profile_driven_symmetric"
    return "furniture_boxy_panel"


def _part(role: str, primitive_hint: str, notes: str = "") -> StructuralPart:
    return StructuralPart(role=role, primitive_hint=primitive_hint, notes=notes)


def _furniture_spec() -> StructuralBuildSpec:
    allowed = ["rect_extrude", "cut_extrude", "circle_extrude"]
    return StructuralBuildSpec(
        object_family="furniture_boxy_panel",
        build_outcome_target="exact",
        approximation_policy={"mode": "panel_first", "detail_level": "conservative"},
        symmetry="optional",
        main_masses=[
            _part("base_panel", "rect_extrude", "primary support or seat area"),
            _part("top_panel", "rect_extrude", "main horizontal closure"),
        ],
        supporting_parts=[
            _part("left_support", "rect_extrude", "vertical support/leg"),
            _part("right_support", "rect_extrude", "vertical support/leg"),
        ],
        secondary_parts=[
            _part("back_panel", "rect_extrude", "optional vertical plane"),
            _part("utility_cutout", "cut_extrude", "optional openings"),
        ],
        silhouette_constraints=[
            "dominant flat panels",
            "stable base contact",
            "avoid micro details",
        ],
        construction_strategy="panel-first extrusions with optional cutouts",
        build_order=["left_support", "right_support", "base_panel", "top_panel", "back_panel", "utility_cutout"],
        allowed_primitives=allowed,
        required_capabilities=required_capabilities_for_primitives(allowed),
        confidence_notes=["conservative boxy decomposition"],
        blocked_reason="",
    )


def _vehicle_spec() -> StructuralBuildSpec:
    allowed = ["rect_extrude", "poly_extrude", "wedge_extrude", "circle_extrude", "cut_extrude"]
    return StructuralBuildSpec(
        object_family="lowpoly_hard_surface_vehicle",
        build_outcome_target="exact",
        approximation_policy={"mode": "hard_surface_lowpoly", "detail_level": "medium"},
        symmetry="approx_x",
        main_masses=[
            _part("lower_hull", "wedge_extrude", "dominant lower armored mass"),
            _part("upper_hull", "poly_extrude", "upper structural shell"),
            _part("turret", "poly_extrude", "turret mass"),
        ],
        supporting_parts=[
            _part("left_track_module", "poly_extrude", "left running gear envelope"),
            _part("right_track_module", "poly_extrude", "right running gear envelope"),
            _part("gun", "circle_extrude", "barrel aligned with turret"),
        ],
        secondary_parts=[
            _part("front_wheels", "circle_extrude", "drive wheel pair"),
            _part("rear_wheels", "circle_extrude", "idler wheel pair"),
            _part("service_cutouts", "cut_extrude", "functional cuts"),
        ],
        silhouette_constraints=[
            "longitudinal hull silhouette",
            "turret above upper hull",
            "left/right running gear symmetry",
        ],
        construction_strategy="hull core then running gear modules then secondary details",
        build_order=[
            "lower_hull",
            "upper_hull",
            "turret",
            "gun",
            "left_track_module",
            "right_track_module",
            "front_wheels",
            "rear_wheels",
            "service_cutouts",
        ],
        allowed_primitives=allowed,
        required_capabilities=required_capabilities_for_primitives(allowed),
        confidence_notes=["vehicle heuristics enforce role-oriented masses"],
        blocked_reason="",
    )


def _profile_symmetric_spec() -> StructuralBuildSpec:
    allowed = ["poly_extrude", "rect_extrude", "cut_extrude", "circle_extrude"]
    return StructuralBuildSpec(
        object_family="profile_driven_symmetric",
        build_outcome_target="exact",
        approximation_policy={"mode": "profile_first", "detail_level": "conservative"},
        symmetry="approx_x",
        main_masses=[
            _part("primary_profile_body", "poly_extrude", "dominant cross-section mass"),
        ],
        supporting_parts=[
            _part("left_support", "rect_extrude", "left symmetric support"),
            _part("right_support", "rect_extrude", "right symmetric support"),
        ],
        secondary_parts=[
            _part("functional_holes", "cut_extrude", "slots/ports/holes"),
        ],
        silhouette_constraints=[
            "consistent bilateral symmetry",
            "profile dominates silhouette",
        ],
        construction_strategy="profile-first body, then mirrored supports and cutouts",
        build_order=["primary_profile_body", "left_support", "right_support", "functional_holes"],
        allowed_primitives=allowed,
        required_capabilities=required_capabilities_for_primitives(allowed),
        confidence_notes=["symmetric profile decomposition"],
        blocked_reason="",
    )


def _rotational_spec(capabilities: CapabilityModel) -> StructuralBuildSpec:
    revolve_supported = capabilities.supports("revolve")
    if revolve_supported:
        allowed = ["revolve", "cut_extrude", "circle_extrude"]
        outcome = "exact"
        approximation = {"mode": "revolve_exact"}
        strategy = "axis profile with revolve"
        blocked_reason = ""
    else:
        allowed = ["circle_extrude", "poly_extrude", "cut_extrude"]
        outcome = "lowpoly_approx"
        approximation = {
            "mode": "stepped_radial_approx",
            "segments": 6,
            "reason": "revolve capability unavailable",
        }
        strategy = "stacked radial segments for stepped rotational approximation"
        blocked_reason = ""

    return StructuralBuildSpec(
        object_family="rotational_bodies",
        build_outcome_target=outcome,
        approximation_policy=approximation,
        symmetry="axis_y",
        main_masses=[
            _part("radial_core", "circle_extrude", "main axisymmetric body"),
            _part("neck_or_top", "circle_extrude", "upper narrowing section"),
        ],
        supporting_parts=[
            _part("base_ring", "circle_extrude", "stability base"),
        ],
        secondary_parts=[
            _part("profile_relief", "cut_extrude", "minor profile details"),
        ],
        silhouette_constraints=[
            "monotonic radial transitions",
            "axisymmetric silhouette intent",
        ],
        construction_strategy=strategy,
        build_order=["radial_core", "neck_or_top", "base_ring", "profile_relief"],
        allowed_primitives=allowed,
        required_capabilities=required_capabilities_for_primitives(allowed),
        confidence_notes=["rotational family normalized to explicit exact/approx policy"],
        blocked_reason=blocked_reason,
    )


def _unsupported_smooth_spec() -> StructuralBuildSpec:
    return StructuralBuildSpec(
        object_family="unsupported_smooth_freeform",
        build_outcome_target="blocked",
        approximation_policy={
            "mode": "blocked",
            "reason": "smooth/freeform geometry cannot be represented faithfully with current runtime primitives",
        },
        symmetry="none",
        main_masses=[_part("freeform_mass", "unsupported", "organic continuous surface")],
        supporting_parts=[],
        secondary_parts=[],
        silhouette_constraints=["continuous smooth curvature", "freeform local detail"],
        construction_strategy="blocked",
        build_order=["freeform_mass"],
        allowed_primitives=[],
        required_capabilities=["loft", "sweep", "fillet", "chamfer", "revolve"],
        confidence_notes=["family explicitly blocked to avoid silent junk geometry"],
        blocked_reason=(
            "unsupported_smooth_freeform: backend lacks faithful smooth/freeform primitives "
            "(loft/sweep/revolve/fillet/chamfer)"
        ),
    )


def build_structural_spec(
    family: str,
    *,
    capabilities: Optional[CapabilityModel] = None,
) -> StructuralBuildSpec:
    cap_model = capabilities or default_capability_model()
    fam = _norm(family)
    if fam == "furniture_boxy_panel":
        return _furniture_spec()
    if fam == "lowpoly_hard_surface_vehicle":
        return _vehicle_spec()
    if fam == "profile_driven_symmetric":
        return _profile_symmetric_spec()
    if fam == "rotational_bodies":
        return _rotational_spec(cap_model)
    return _unsupported_smooth_spec()


def plan_structural_specs(
    user_request: str,
    *,
    images: Optional[Sequence[Dict[str, Any]]] = None,
    capabilities: Optional[CapabilityModel] = None,
) -> List[Dict[str, Any]]:
    """
    Create-mode structural planner entrypoint.
    Returns ordered candidate specs as plain dicts.
    """
    cap_model = capabilities or default_capability_model()
    family = classify_shape_family(user_request, images=images)

    candidates: List[StructuralBuildSpec] = [build_structural_spec(family, capabilities=cap_model)]

    # Add one deterministic fallback candidate for ambiguous non-blocked families.
    if family not in ("unsupported_smooth_freeform", "lowpoly_hard_surface_vehicle"):
        fallback_family = "profile_driven_symmetric" if (images and len(images) >= 2) else "furniture_boxy_panel"
        if fallback_family != family:
            candidates.append(build_structural_spec(fallback_family, capabilities=cap_model))

    return [spec.to_dict() for spec in candidates]
