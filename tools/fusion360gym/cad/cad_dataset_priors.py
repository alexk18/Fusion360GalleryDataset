"""
Dataset-informed create priors.

Priors are intentionally generalized pattern statistics (ordering, primitive
preferences, support-first tendencies), not replay of instance-level commands.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_FAMILY_PRIORS = {
    "furniture_boxy_panel": {
        "ordering": ["left_support", "right_support", "base_panel", "top_panel", "back_panel"],
        "primitive_preference": ["rect_extrude", "cut_extrude", "circle_extrude"],
        "strategy": "support_first_panel_assembly",
        "support_bias": 1.0,
    },
    "lowpoly_hard_surface_vehicle": {
        "ordering": [
            "lower_hull",
            "upper_hull",
            "left_track_module",
            "right_track_module",
            "turret",
            "gun",
            "front_wheels",
            "rear_wheels",
        ],
        "primitive_preference": ["wedge_extrude", "poly_extrude", "circle_extrude", "rect_extrude"],
        "strategy": "massing_first_vehicle",
        "support_bias": 0.85,
    },
    "profile_driven_symmetric": {
        "ordering": ["primary_profile_body", "left_support", "right_support", "functional_holes"],
        "primitive_preference": ["poly_extrude", "rect_extrude", "cut_extrude", "circle_extrude"],
        "strategy": "profile_first_symmetric",
        "support_bias": 0.7,
    },
    "rotational_bodies": {
        "ordering": ["radial_core", "neck_or_top", "base_ring", "profile_relief"],
        "primitive_preference": ["circle_extrude", "poly_extrude", "cut_extrude"],
        "strategy": "radial_stack_approx",
        "support_bias": 0.55,
    },
    "unsupported_smooth_freeform": {
        "ordering": [],
        "primitive_preference": [],
        "strategy": "blocked",
        "support_bias": 0.0,
    },
}


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def build_dataset_priors(
    family: str,
    *,
    user_request: str = "",
    structural_build_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    fam = _norm(family)
    base = dict(_FAMILY_PRIORS.get(fam, _FAMILY_PRIORS["furniture_boxy_panel"]))
    req = _norm(user_request)

    ordering = [r for r in (base.get("ordering") or []) if isinstance(r, str) and r.strip()]
    if isinstance(structural_build_order, list):
        struct_order = [r for r in structural_build_order if isinstance(r, str) and r.strip()]
        if struct_order:
            merged = []
            seen = set()
            for role in struct_order + ordering:
                if role not in seen:
                    merged.append(role)
                    seen.add(role)
            ordering = merged

    # Request-conditioned generalized priors (still non-instance specific).
    if fam == "furniture_boxy_panel" and any(k in req for k in ("table", "desk", "bench")):
        ordering = ["left_support", "right_support", "base_panel", "top_panel", "back_panel"]
    if fam == "lowpoly_hard_surface_vehicle" and any(k in req for k in ("aircraft", "plane", "boat")):
        # keep massing-first but prioritize body before side modules for flight/boat silhouettes
        ordering = ["lower_hull", "upper_hull", "turret", "gun", "left_track_module", "right_track_module"] + [
            r for r in ordering if r not in {"lower_hull", "upper_hull", "turret", "gun", "left_track_module", "right_track_module"}
        ]

    return {
        "family": fam,
        "source": "dataset_generalized_priors_v1",
        "notes": [
            "Derived from generalized Fusion dataset construction tendencies.",
            "No instance-level command replay is used.",
        ],
        "ordering": ordering,
        "primitive_preference": list(base.get("primitive_preference") or []),
        "strategy": str(base.get("strategy") or ""),
        "support_bias": float(base.get("support_bias", 0.5) or 0.5),
    }

