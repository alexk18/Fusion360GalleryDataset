"""
Structural Build Spec schema for create-mode planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .cad_capabilities import capability_required_for_primitive


CREATE_FAMILIES = (
    "furniture_boxy_panel",
    "lowpoly_hard_surface_vehicle",
    "profile_driven_symmetric",
    "rotational_bodies",
    "unsupported_smooth_freeform",
    "unknown_best_effort",
)

CREATE_OUTCOMES = ("exact", "lowpoly_approx", "best_effort", "blocked")

SCENARIO_TYPES = ("single_object", "assembly", "multi_object", "scene_or_environment", "unknown")


@dataclass
class StructuralPart:
    role: str
    primitive_hint: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "primitive_hint": self.primitive_hint,
            "notes": self.notes,
        }


@dataclass
class StructuralBuildSpec:
    object_family: str
    build_outcome_target: str
    approximation_policy: Dict[str, Any]
    symmetry: str
    main_masses: List[StructuralPart] = field(default_factory=list)
    supporting_parts: List[StructuralPart] = field(default_factory=list)
    secondary_parts: List[StructuralPart] = field(default_factory=list)
    silhouette_constraints: List[str] = field(default_factory=list)
    construction_strategy: str = ""
    build_order: List[str] = field(default_factory=list)
    allowed_primitives: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    confidence_notes: List[str] = field(default_factory=list)
    blocked_reason: str = ""
    subtype: str = ""
    scenario_type: str = "single_object"
    routing_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_family": self.object_family,
            "build_outcome_target": self.build_outcome_target,
            "approximation_policy": dict(self.approximation_policy or {}),
            "symmetry": self.symmetry,
            "main_masses": [p.to_dict() for p in self.main_masses],
            "supporting_parts": [p.to_dict() for p in self.supporting_parts],
            "secondary_parts": [p.to_dict() for p in self.secondary_parts],
            "silhouette_constraints": list(self.silhouette_constraints or []),
            "construction_strategy": self.construction_strategy,
            "build_order": list(self.build_order or []),
            "allowed_primitives": list(self.allowed_primitives or []),
            "required_capabilities": list(self.required_capabilities or []),
            "confidence_notes": list(self.confidence_notes or []),
            "blocked_reason": self.blocked_reason,
            "subtype": self.subtype,
            "scenario_type": self.scenario_type,
            "routing_confidence": self.routing_confidence,
        }


def required_capabilities_for_primitives(allowed_primitives: List[str]) -> List[str]:
    caps: List[str] = []
    for primitive in allowed_primitives or []:
        req = capability_required_for_primitive(str(primitive or "").strip().lower(), "")
        if req == "unsupported":
            continue
        if req not in caps:
            caps.append(req)
    return caps


def validate_structural_spec(spec: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ["spec must be an object"]

    required_fields = (
        "object_family",
        "build_outcome_target",
        "approximation_policy",
        "symmetry",
        "main_masses",
        "supporting_parts",
        "secondary_parts",
        "silhouette_constraints",
        "construction_strategy",
        "build_order",
        "allowed_primitives",
        "required_capabilities",
        "confidence_notes",
        "blocked_reason",
    )
    for k in required_fields:
        if k not in spec:
            errors.append(f"missing required field '{k}'")

    family = str(spec.get("object_family") or "")
    if family not in CREATE_FAMILIES:
        errors.append(f"invalid object_family '{family}'")

    outcome = str(spec.get("build_outcome_target") or "")
    if outcome not in CREATE_OUTCOMES:
        errors.append(f"invalid build_outcome_target '{outcome}'")

    for bucket_name in ("main_masses", "supporting_parts", "secondary_parts"):
        bucket = spec.get(bucket_name)
        if not isinstance(bucket, list):
            errors.append(f"{bucket_name} must be a list")
            continue
        for i, part in enumerate(bucket):
            if not isinstance(part, dict):
                errors.append(f"{bucket_name}[{i}] must be an object")
                continue
            if not str(part.get("role") or "").strip():
                errors.append(f"{bucket_name}[{i}].role is required")
            if not str(part.get("primitive_hint") or "").strip():
                errors.append(f"{bucket_name}[{i}].primitive_hint is required")

    build_order = spec.get("build_order")
    if not isinstance(build_order, list):
        errors.append("build_order must be a list")
    else:
        roles = set()
        for bucket_name in ("main_masses", "supporting_parts", "secondary_parts"):
            for part in (spec.get(bucket_name) or []):
                if isinstance(part, dict):
                    role = str(part.get("role") or "").strip()
                    if role:
                        roles.add(role)
        for role in build_order:
            if str(role or "").strip() and str(role).strip() not in roles:
                errors.append(f"build_order role '{role}' is not present in part buckets")

    return errors
