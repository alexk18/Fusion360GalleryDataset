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
    - list_tools
    - list_capabilities
    - create_sketch
    - add_point (legacy helper for polygonal profile compilation)
    - add_line
    - add_arc
    - add_circle
    - close_profile
    - extrude
    - cut
    - find_entity_by_name
    - list_entities
    - list_features
    - get_model_state
    - screenshot
    - query_bounding_box
    - update_extrude
    - get_features / get_sketches / get_bodies / get_parts
    - get_faces / get_edges (heuristic summaries)
    - get_body_bbox / get_feature_bbox
    - get_body_relations / get_feature_body_relations
    - get_connected_components / get_overlaps / get_active_construction_context
    """

    def __init__(self, client: Any, capabilities: Optional[CapabilityModel] = None):
        self.client = client
        self.capabilities = capabilities or default_capability_model()

    def _unsupported(self, cap: str) -> BackendResult:
        return BackendResult(ok=False, reason=f"capability '{cap}' unavailable: {self.capabilities.reason(cap)}")

    def _ok(self, response: Any) -> BackendResult:
        return BackendResult(ok=True, response=response)

    def _enabled(self, cap: str) -> bool:
        return self.capabilities.supports(cap) or self.capabilities.is_partial(cap)

    @staticmethod
    def _unwrap_response_payload(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return dict(response)
        if hasattr(response, "json"):
            try:
                body = response.json()
                if isinstance(body, dict):
                    return body
            except Exception:
                pass
        return {"raw": str(response)}

    def _call(self, command: str, data: Optional[Dict[str, Any]] = None) -> Any:
        if data is None:
            data = {}
        if hasattr(self.client, "send_command"):
            return self.client.send_command(command, data)
        return None

    def _response_ok(self, response: Any) -> bool:
        if response is None:
            return False
        if hasattr(response, "status_code"):
            try:
                if int(getattr(response, "status_code")) >= 400:
                    return False
            except Exception:
                return False
        payload = self._unwrap_response_payload(response)
        if isinstance(payload, dict):
            if payload.get("ok") is False:
                return False
            status = payload.get("status")
            if isinstance(status, int) and status >= 400:
                return False
        return True

    def _response_reason(self, response: Any, fallback: str = "command failed") -> str:
        payload = self._unwrap_response_payload(response)
        if isinstance(payload, dict):
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        return fallback

    def _call_checked(self, command: str, data: Optional[Dict[str, Any]] = None) -> BackendResult:
        response = self._call(command, data or {})
        if self._response_ok(response):
            return self._ok(response)
        return BackendResult(ok=False, response=response, reason=self._response_reason(response, f"{command} failed"))

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
            response = self.client.list_features()
            if self._response_ok(response):
                return self._ok(response)
            return BackendResult(ok=False, response=response, reason=self._response_reason(response, "list_features failed"))
        return self._call_checked("list_features", {})

    def query_features(self) -> BackendResult:
        return self.list_features()

    def list_capabilities(self) -> BackendResult:
        return self._ok(
            {
                "capabilities": {
                    name: {"status": cap.status, "reason": cap.reason}
                    for name, cap in (self.capabilities.capabilities or {}).items()
                }
            }
        )

    def list_tools(self) -> BackendResult:
        if not self.capabilities.supports("list_tools"):
            return self._unsupported("list_tools")
        if hasattr(self.client, "list_tools"):
            response = self.client.list_tools()
            if self._response_ok(response):
                return self._ok(response)
            return BackendResult(ok=False, response=response, reason=self._response_reason(response, "list_tools failed"))
        return self._call_checked("list_tools", {})

    def list_entities(self) -> BackendResult:
        if not self.capabilities.supports("list_entities"):
            return self._unsupported("list_entities")
        # Current backend entity registry is feature-driven.
        return self.list_features()

    def get_features(self) -> BackendResult:
        if self._enabled("get_features"):
            out = self._call_checked("get_features", {})
            if out.ok:
                return out
        # Compatibility fallback.
        return self.list_features()

    def get_sketches(self) -> BackendResult:
        if self._enabled("get_sketches"):
            out = self._call_checked("get_sketches", {})
            if out.ok:
                return out

        features_result = self.get_features()
        if not features_result.ok:
            return features_result
        payload = self._unwrap_response_payload(features_result.response)
        data = payload.get("data", payload)
        sketches = data.get("sketches", []) if isinstance(data, dict) else []
        return self._ok({"sketches": sketches, "source_kind": "derived_exact"})

    def get_bodies(self) -> BackendResult:
        if self._enabled("get_bodies"):
            out = self._call_checked("get_bodies", {})
            if out.ok:
                return out

        # Legacy fallback: feature proxies.
        features_result = self.get_features()
        if not features_result.ok:
            return features_result
        payload = self._unwrap_response_payload(features_result.response)
        data = payload.get("data", payload)
        extrudes = data.get("extrude_features", []) if isinstance(data, dict) else []
        bodies = [
            {"name": str((item or {}).get("name") or f"body_{i+1}"), "source": "feature_proxy"}
            for i, item in enumerate(extrudes if isinstance(extrudes, list) else [])
        ]
        return self._ok({"bodies": bodies, "source_kind": "heuristic_estimate"})

    def get_parts(self) -> BackendResult:
        if self._enabled("get_parts"):
            out = self._call_checked("get_parts", {})
            if out.ok:
                return out
        bodies = self.get_bodies()
        if not bodies.ok:
            return bodies
        payload = self._unwrap_response_payload(bodies.response)
        data = payload.get("data", payload)
        return self._ok({"parts": data.get("bodies", []) if isinstance(data, dict) else [], "source_kind": "derived_exact"})

    def get_faces(self) -> BackendResult:
        if self._enabled("get_faces"):
            out = self._call_checked("get_faces", {})
            if out.ok:
                return out
        bodies_result = self.get_bodies()
        if not bodies_result.ok:
            return bodies_result
        payload = self._unwrap_response_payload(bodies_result.response)
        data = payload.get("data", payload)
        count = len(data.get("bodies", []) if isinstance(data, dict) else [])
        return self._ok({"faces": {"count_estimate": 6 * count, "source_kind": "heuristic_estimate"}})

    def get_edges(self) -> BackendResult:
        if self._enabled("get_edges"):
            out = self._call_checked("get_edges", {})
            if out.ok:
                return out
        bodies_result = self.get_bodies()
        if not bodies_result.ok:
            return bodies_result
        payload = self._unwrap_response_payload(bodies_result.response)
        data = payload.get("data", payload)
        count = len(data.get("bodies", []) if isinstance(data, dict) else [])
        return self._ok({"edges": {"count_estimate": 12 * count, "source_kind": "heuristic_estimate"}})

    def get_body_bbox(self, body_name: str = "") -> BackendResult:
        if self._enabled("get_body_bbox"):
            payload = {"body_name": body_name} if str(body_name or "").strip() else {}
            out = self._call_checked("get_body_bbox", payload)
            if out.ok:
                return out
        bbox = self.query_bounding_box()
        if not bbox.ok:
            return bbox
        payload = self._unwrap_response_payload(bbox.response)
        data = payload.get("data", payload)
        return self._ok({"body_bbox": {"model": data.get("bounding_box", data)}, "source_kind": "heuristic_estimate"})

    def get_feature_bbox(self) -> BackendResult:
        if self._enabled("get_feature_bbox"):
            out = self._call_checked("get_feature_bbox", {})
            if out.ok:
                return out
        # Runtime may not expose per-feature bbox directly.
        return self._ok({"feature_bbox": {}, "source_kind": "heuristic_estimate"})

    def get_connected_components(self) -> BackendResult:
        if self._enabled("get_connected_components"):
            out = self._call_checked("get_connected_components", {})
            if out.ok:
                return out
        return self._ok({"components": [], "source_kind": "heuristic_estimate"})

    def get_overlaps(self) -> BackendResult:
        if self._enabled("get_overlaps"):
            out = self._call_checked("get_overlaps", {})
            if out.ok:
                return out
        return self._ok({"overlaps": [], "source_kind": "heuristic_estimate"})

    def get_body_relations(self) -> BackendResult:
        if self._enabled("get_body_relations"):
            out = self._call_checked("get_body_relations", {})
            if out.ok:
                return out
        return self._ok({"relations": [], "source_kind": "heuristic_estimate"})

    def get_feature_body_relations(self) -> BackendResult:
        if self._enabled("get_feature_body_relations"):
            out = self._call_checked("get_feature_body_relations", {})
            if out.ok:
                return out
        return self._ok({"relations": [], "source_kind": "heuristic_estimate"})

    def get_active_construction_context(self) -> BackendResult:
        if self._enabled("active_construction_context"):
            out = self._call_checked("get_active_construction_context", {})
            if out.ok:
                return out

        features_result = self.list_features()
        if not features_result.ok:
            return features_result
        payload = self._unwrap_response_payload(features_result.response)
        data = payload.get("data", payload)
        sketches = data.get("sketches", []) if isinstance(data, dict) else []
        extrudes = data.get("extrude_features", []) if isinstance(data, dict) else []
        last_sketch = str((sketches[-1] or {}).get("name") or "") if isinstance(sketches, list) and sketches else ""
        last_feature = str((extrudes[-1] or {}).get("name") or "") if isinstance(extrudes, list) and extrudes else ""
        return self._ok({"active_context": {"last_sketch": last_sketch, "last_feature": last_feature}, "source_kind": "derived_exact"})

    def get_model_state(self) -> BackendResult:
        if not self.capabilities.supports("model_state_summary"):
            return self._unsupported("model_state_summary")
        if self._enabled("get_model_state"):
            out = self._call_checked("get_model_state", {})
            if out.ok:
                return out
        features_result = self.list_features()
        if not features_result.ok:
            return BackendResult(ok=False, reason=f"get_model_state failed: {features_result.reason}")
        bbox_result = self.query_bounding_box()
        if not bbox_result.ok:
            return BackendResult(ok=False, reason=f"get_model_state failed: {bbox_result.reason}")
        features_payload = self._unwrap_response_payload(features_result.response)
        bbox_payload = self._unwrap_response_payload(bbox_result.response)
        features_data = features_payload.get("data", features_payload)
        bbox_data = bbox_payload.get("data", bbox_payload)
        return self._ok(
            {
                "features": features_data,
                "bounding_box": bbox_data,
                "source_kind": "derived_exact",
            }
        )

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
            response = self.client.query_bounding_box()
            if self._response_ok(response):
                return self._ok(response)
            return BackendResult(ok=False, response=response, reason=self._response_reason(response, "query_bounding_box failed"))
        return self._call_checked("query_bounding_box", {})

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
