"""
Capability model for Fusion CAD backend.

This module is the single source of truth for what the backend can do now:
- create capabilities
- edit capabilities
- query capabilities
- partial/unsupported features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


STATUS_SUPPORTED = "supported"
STATUS_PARTIAL = "partial"
STATUS_UNSUPPORTED = "unsupported"


@dataclass
class Capability:
    status: str
    reason: str = ""


@dataclass
class CapabilityModel:
    capabilities: Dict[str, Capability] = field(default_factory=dict)

    def supports(self, name: str) -> bool:
        cap = self.capabilities.get(name)
        return cap is not None and cap.status == STATUS_SUPPORTED

    def is_partial(self, name: str) -> bool:
        cap = self.capabilities.get(name)
        return cap is not None and cap.status == STATUS_PARTIAL

    def status(self, name: str) -> str:
        cap = self.capabilities.get(name)
        if cap is None:
            return STATUS_UNSUPPORTED
        return cap.status

    def reason(self, name: str) -> str:
        cap = self.capabilities.get(name)
        if cap is None:
            return "capability not declared"
        return cap.reason or ""


def default_capability_model() -> CapabilityModel:
    """
    Backend capabilities aligned with current implementation.
    """
    caps = {
        # Create sketch/profile capabilities
        "sketch_rect": Capability(STATUS_SUPPORTED),
        "sketch_circle": Capability(STATUS_SUPPORTED),
        "sketch_poly": Capability(STATUS_SUPPORTED),
        "sketch_arc": Capability(STATUS_UNSUPPORTED, "arcs are rejected by validator/compiler in current backend"),
        "profile_close": Capability(STATUS_SUPPORTED),
        # Extrude capabilities
        "extrude_new_body": Capability(STATUS_SUPPORTED),
        "extrude_cut": Capability(STATUS_SUPPORTED),
        "update_extrude_distance": Capability(STATUS_SUPPORTED),
        # Query capabilities
        "find_entity_by_name": Capability(STATUS_SUPPORTED),
        "list_features": Capability(STATUS_SUPPORTED),
        "screenshot": Capability(STATUS_SUPPORTED),
        "bbox_query": Capability(STATUS_SUPPORTED),
        # Advanced operations (declared but unsupported now)
        "fillet": Capability(STATUS_UNSUPPORTED, "dsl-level only; execution not implemented"),
        "chamfer": Capability(STATUS_UNSUPPORTED, "dsl-level placeholder"),
        "revolve": Capability(STATUS_UNSUPPORTED, "dsl-level placeholder"),
        "loft": Capability(STATUS_UNSUPPORTED, "dsl-level only; execution not implemented"),
        "sweep": Capability(STATUS_UNSUPPORTED, "dsl-level only; execution not implemented"),
        "sketch_edit_existing": Capability(
            STATUS_PARTIAL,
            "existing sketch is authoritative by name; geometry edit-in-place is not implemented",
        ),
        "suppress_feature": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "delete_feature": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "edge_query": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "face_query": Capability(STATUS_UNSUPPORTED, "not implemented"),
    }
    return CapabilityModel(capabilities=caps)


def capability_required_for_primitive(primitive: str, operation: str = "") -> str:
    p = (primitive or "").strip().lower()
    op = (operation or "").strip()
    if p in ("rect_extrude", "cut_extrude"):
        if op == "CutFeatureOperation" or p == "cut_extrude":
            return "extrude_cut"
        return "extrude_new_body"
    if p in ("circle_extrude", "poly_extrude", "wedge_extrude"):
        if op == "CutFeatureOperation":
            return "extrude_cut"
        return "extrude_new_body"
    if p in ("loft", "sweep", "fillet", "chamfer"):
        return p
    if p == "revolve":
        return "revolve"
    return "unsupported"
