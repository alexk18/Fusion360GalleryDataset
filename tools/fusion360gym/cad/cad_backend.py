"""
Fusion CAD backend tool-surface (MCP-style contract).

LLM never calls Fusion commands directly. Executor/operator use this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .cad_capabilities import CapabilityModel, default_capability_model


@dataclass
class BackendResult:
    ok: bool
    response: Any = None
    reason: str = ""


class FusionCadBackend:
    """
    Capability-aware backend wrapper around Fusion360Gym client/send_command.

    Tool semantics:
    - create_sketch
    - add_point (legacy helper for polygonal profile compilation)
    - add_line
    - add_arc
    - add_circle
    - close_profile
    - extrude
    - cut
    - find_entity_by_name
    - list_features
    - screenshot
    - query_bounding_box
    - update_extrude
    """

    def __init__(self, client: Any, capabilities: Optional[CapabilityModel] = None):
        self.client = client
        self.capabilities = capabilities or default_capability_model()

    def _unsupported(self, cap: str) -> BackendResult:
        return BackendResult(ok=False, reason=f"capability '{cap}' unavailable: {self.capabilities.reason(cap)}")

    def _ok(self, response: Any) -> BackendResult:
        return BackendResult(ok=True, response=response)

    def _call(self, command: str, data: Optional[Dict[str, Any]] = None) -> Any:
        if data is None:
            data = {}
        if hasattr(self.client, "send_command"):
            return self.client.send_command(command, data)
        return None

    def create_sketch(self, sketch_plane: str, sketch_name: Optional[str] = None) -> BackendResult:
        if hasattr(self.client, "add_sketch"):
            return self._ok(self.client.add_sketch(sketch_plane, sketch_name=sketch_name))
        payload: Dict[str, Any] = {"sketch_plane": sketch_plane}
        if sketch_name:
            payload["sketch_name"] = sketch_name
        return self._ok(self._call("add_sketch", payload))

    def add_line(self, sketch_name: str, pt1: Dict[str, Any], pt2: Dict[str, Any]) -> BackendResult:
        if hasattr(self.client, "add_line"):
            return self._ok(self.client.add_line(sketch_name, pt1, pt2))
        return self._ok(self._call("add_line", {"sketch_name": sketch_name, "pt1": pt1, "pt2": pt2}))

    def add_point(self, sketch_name: str, pt: Dict[str, Any]) -> BackendResult:
        if hasattr(self.client, "add_point"):
            return self._ok(self.client.add_point(sketch_name, pt))
        return self._ok(self._call("add_point", {"sketch_name": sketch_name, "pt": pt}))

    def add_arc(self, sketch_name: str, pt1: Dict[str, Any], pt2: Dict[str, Any], angle: float) -> BackendResult:
        if not self.capabilities.supports("sketch_arc"):
            return self._unsupported("sketch_arc")
        if hasattr(self.client, "add_arc"):
            return self._ok(self.client.add_arc(sketch_name, pt1, pt2, angle))
        return self._ok(self._call("add_arc", {"sketch_name": sketch_name, "pt1": pt1, "pt2": pt2, "angle": angle}))

    def add_circle(self, sketch_name: str, pt: Dict[str, Any], radius: float) -> BackendResult:
        if hasattr(self.client, "add_circle"):
            return self._ok(self.client.add_circle(sketch_name, pt, radius))
        return self._ok(self._call("add_circle", {"sketch_name": sketch_name, "pt": pt, "radius": radius}))

    def close_profile(self, sketch_name: str) -> BackendResult:
        if hasattr(self.client, "close_profile"):
            return self._ok(self.client.close_profile(sketch_name))
        return self._ok(self._call("close_profile", {"sketch_name": sketch_name}))

    def extrude(
        self,
        sketch_name: str,
        profile_id: str,
        distance: float,
        operation: str,
        feature_name: Optional[str] = None,
    ) -> BackendResult:
        if operation == "CutFeatureOperation":
            if not self.capabilities.supports("extrude_cut"):
                return self._unsupported("extrude_cut")
        else:
            if not self.capabilities.supports("extrude_new_body"):
                return self._unsupported("extrude_new_body")
        if hasattr(self.client, "add_extrude"):
            return self._ok(self.client.add_extrude(sketch_name, profile_id, distance, operation, feature_name=feature_name))
        payload: Dict[str, Any] = {
            "sketch_name": sketch_name,
            "profile_id": profile_id,
            "distance": distance,
            "operation": operation,
        }
        if feature_name:
            payload["feature_name"] = feature_name
        return self._ok(self._call("add_extrude", payload))

    def cut(self, sketch_name: str, profile_id: str, distance: float, feature_name: Optional[str] = None) -> BackendResult:
        return self.extrude(
            sketch_name=sketch_name,
            profile_id=profile_id,
            distance=distance,
            operation="CutFeatureOperation",
            feature_name=feature_name,
        )

    def find_entity_by_name(self, entity_type: str, name: str) -> BackendResult:
        if not self.capabilities.supports("find_entity_by_name"):
            return self._unsupported("find_entity_by_name")
        if hasattr(self.client, "find_entity_by_name"):
            return self._ok(self.client.find_entity_by_name(entity_type, name))
        return self._ok(self._call("find_entity_by_name", {"type": entity_type, "name": name}))

    def list_features(self) -> BackendResult:
        if not self.capabilities.supports("list_features"):
            return self._unsupported("list_features")
        if hasattr(self.client, "list_features"):
            return self._ok(self.client.list_features())
        return self._ok(self._call("list_features", {}))

    def query_features(self) -> BackendResult:
        return self.list_features()

    def screenshot(self, file: str, width: int = 512, height: int = 512) -> BackendResult:
        if not self.capabilities.supports("screenshot"):
            return self._unsupported("screenshot")
        if hasattr(self.client, "screenshot"):
            return self._ok(self.client.screenshot(file, width=width, height=height))
        return self._ok(self._call("screenshot", {"file": file, "width": width, "height": height}))

    def query_bounding_box(self) -> BackendResult:
        if not self.capabilities.supports("bbox_query"):
            return self._unsupported("bbox_query")
        if hasattr(self.client, "query_bounding_box"):
            return self._ok(self.client.query_bounding_box())
        return self._ok(self._call("query_bounding_box", {}))

    def update_extrude(self, feature_name: str, distance: float) -> BackendResult:
        if not self.capabilities.supports("update_extrude_distance"):
            return self._unsupported("update_extrude_distance")
        if hasattr(self.client, "update_extrude"):
            return self._ok(self.client.update_extrude(feature_name, distance))
        return self._ok(self._call("update_extrude", {"feature_name": feature_name, "distance": float(distance)}))

    # Reserved extension points (declared but unsupported in current backend)
    def fillet(self, *_args, **_kwargs) -> BackendResult:
        return self._unsupported("fillet")

    def chamfer(self, *_args, **_kwargs) -> BackendResult:
        return self._unsupported("chamfer")

    def revolve(self, *_args, **_kwargs) -> BackendResult:
        return self._unsupported("revolve")

    def loft(self, *_args, **_kwargs) -> BackendResult:
        return self._unsupported("loft")

    def sweep(self, *_args, **_kwargs) -> BackendResult:
        return self._unsupported("sweep")
