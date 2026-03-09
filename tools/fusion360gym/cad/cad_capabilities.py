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
        "list_tools": Capability(STATUS_SUPPORTED),
        "list_capabilities": Capability(STATUS_SUPPORTED),
        "find_entity_by_name": Capability(STATUS_SUPPORTED),
        "list_entities": Capability(STATUS_SUPPORTED),
        "list_features": Capability(STATUS_SUPPORTED),
        "get_features": Capability(STATUS_SUPPORTED, "exact list when get_features command is available"),
        "get_model_state": Capability(STATUS_SUPPORTED, "exact summary when server command is available"),
        "model_state_summary": Capability(STATUS_SUPPORTED),
        "get_sketches": Capability(STATUS_SUPPORTED, "exact list when get_sketches command is available"),
        "get_bodies": Capability(STATUS_PARTIAL, "exact body list when command exists, otherwise feature-proxy"),
        "get_parts": Capability(STATUS_PARTIAL, "alias of bodies"),
        "get_faces": Capability(STATUS_PARTIAL, "exact summary when command exists, otherwise heuristic estimate"),
        "get_edges": Capability(STATUS_PARTIAL, "exact summary when command exists, otherwise heuristic estimate"),
        "get_body_bbox": Capability(STATUS_PARTIAL, "exact per-body bbox when command exists, otherwise model bbox fallback"),
        "get_feature_bbox": Capability(STATUS_PARTIAL, "exact per-feature bbox when command exists"),
        "get_body_relations": Capability(STATUS_PARTIAL, "exact intersections + contact estimates"),
        "get_feature_body_relations": Capability(STATUS_PARTIAL, "exact feature-to-body mapping when command exists"),
        "get_connected_components": Capability(STATUS_PARTIAL, "exact/derived when geometry graph is available"),
        "get_overlaps": Capability(STATUS_PARTIAL, "exact intersections when geometry graph is available"),
        "active_construction_context": Capability(STATUS_SUPPORTED, "derived from feature/sketch recency"),
        "screenshot": Capability(STATUS_SUPPORTED),
        "bbox_query": Capability(STATUS_SUPPORTED),
        # Advanced operations
        "fillet": Capability(STATUS_SUPPORTED, "fillet edges of a body with constant radius"),
        "chamfer": Capability(STATUS_SUPPORTED, "chamfer edges of a body with equal distance"),
        "shell": Capability(STATUS_SUPPORTED, "shell (hollow out) a body with wall thickness"),
        "edge_query_by_body": Capability(STATUS_SUPPORTED, "get edge indices and positions for a body"),
        "revolve": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "loft": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "sweep": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "sketch_edit_existing": Capability(
            STATUS_PARTIAL,
            "existing sketch is authoritative by name; geometry edit-in-place is not implemented",
        ),
        "suppress_feature": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "delete_feature": Capability(STATUS_UNSUPPORTED, "not implemented"),
        "edge_query": Capability(STATUS_SUPPORTED, "edge count per body; detailed query via get_edges_by_body"),
        "face_query": Capability(STATUS_PARTIAL, "face count per body; no individual face selection yet"),
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
