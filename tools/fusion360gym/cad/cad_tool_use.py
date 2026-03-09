"""
Tool-use definitions and dispatcher for Claude tool_use integration.

Converts Fusion360Gym backend operations into Claude-compatible tool
definitions, and dispatches Claude tool_use calls to the backend.

This is the bridge that lets Claude directly call CAD operations
instead of going through template/grammar synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cad_backend import BackendResult, FusionCadBackend


# ---------------------------------------------------------------------------
# Tool definitions in Claude tool_use format
# ---------------------------------------------------------------------------

CAD_TOOLS: List[Dict[str, Any]] = [
    # === SKETCH CREATION ===
    {
        "name": "create_sketch",
        "description": (
            "Create a new sketch on a construction plane. "
            "Plane can be 'XY', 'XZ', 'YZ', or with offset like 'XY@10.0'. "
            "XY extrudes along +Z, XZ along +Y, YZ along +X. "
            "Returns the sketch name for subsequent operations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_plane": {
                    "type": "string",
                    "description": "Construction plane: 'XY', 'XZ', 'YZ', or with offset e.g. 'XY@5.0'",
                },
                "sketch_name": {
                    "type": "string",
                    "description": "Optional name for the sketch. Auto-generated if omitted.",
                },
            },
            "required": ["sketch_plane"],
        },
    },
    {
        "name": "add_line",
        "description": (
            "Add a line segment to an existing sketch. "
            "Points are 2D coordinates {x, y} in the sketch plane (cm)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_name": {"type": "string", "description": "Name of the target sketch"},
                "pt1": {
                    "type": "object",
                    "description": "Start point {x, y} in cm",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                },
                "pt2": {
                    "type": "object",
                    "description": "End point {x, y} in cm",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                },
            },
            "required": ["sketch_name", "pt1", "pt2"],
        },
    },
    {
        "name": "add_circle",
        "description": (
            "Add a circle to an existing sketch. "
            "Center is a 2D point {x, y} in cm, radius in cm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_name": {"type": "string", "description": "Name of the target sketch"},
                "center": {
                    "type": "object",
                    "description": "Center point {x, y} in cm",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                },
                "radius": {"type": "number", "description": "Circle radius in cm"},
            },
            "required": ["sketch_name", "center", "radius"],
        },
    },
    {
        "name": "add_rectangle",
        "description": (
            "Add a rectangle to a sketch by drawing 4 lines. "
            "Specify center (cx, cy) and dimensions (width, height) in cm. "
            "This is a convenience tool — it adds 4 points and closes the profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_name": {"type": "string", "description": "Name of the target sketch"},
                "cx": {"type": "number", "description": "Center X coordinate in cm"},
                "cy": {"type": "number", "description": "Center Y coordinate in cm"},
                "width": {"type": "number", "description": "Rectangle width in cm"},
                "height": {"type": "number", "description": "Rectangle height in cm"},
            },
            "required": ["sketch_name", "cx", "cy", "width", "height"],
        },
    },
    {
        "name": "add_polygon",
        "description": (
            "Add a polygon to a sketch by adding points sequentially. "
            "Points are 2D coordinates [{x, y}, ...] in cm. "
            "The profile is automatically closed after the last point."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_name": {"type": "string", "description": "Name of the target sketch"},
                "points": {
                    "type": "array",
                    "description": "Array of 2D points [{x, y}, ...] in cm",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                        "required": ["x", "y"],
                    },
                },
            },
            "required": ["sketch_name", "points"],
        },
    },
    {
        "name": "close_profile",
        "description": "Close the current sketch profile so it can be extruded.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_name": {"type": "string", "description": "Name of the sketch to close"},
            },
            "required": ["sketch_name"],
        },
    },
    # === FEATURE CREATION ===
    {
        "name": "extrude",
        "description": (
            "Extrude a sketch profile into a 3D solid body. "
            "Operations: 'NewBodyFeatureOperation' (new separate body), "
            "'JoinFeatureOperation' (merge with existing body — requires overlap), "
            "'CutFeatureOperation' (cut material from existing body). "
            "Distance is the extrude height in cm. "
            "profile_id is typically 'profile_0' for the first profile in the sketch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sketch_name": {"type": "string", "description": "Name of the sketch containing the profile"},
                "profile_id": {
                    "type": "string",
                    "description": "Profile identifier, usually 'profile_0'",
                    "default": "profile_0",
                },
                "distance": {"type": "number", "description": "Extrude distance/height in cm"},
                "operation": {
                    "type": "string",
                    "description": "Feature operation type",
                    "enum": ["NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation"],
                },
                "feature_name": {
                    "type": "string",
                    "description": "Optional name for the extrude feature",
                },
            },
            "required": ["sketch_name", "distance", "operation"],
        },
    },
    # === QUERY / INTROSPECTION ===
    {
        "name": "get_model_state",
        "description": (
            "Get current model state including all features, bodies, and bounding box. "
            "Use this to understand what exists in the model before adding more geometry."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_bodies",
        "description": "Get list of all solid bodies currently in the model.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_features",
        "description": "Get list of all features (sketches and extrudes) in the model.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_body_bbox",
        "description": (
            "Get bounding box of a specific body or the entire model. "
            "Returns min/max coordinates in cm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "body_name": {
                    "type": "string",
                    "description": "Name of body to query. Omit for entire model.",
                },
            },
        },
    },
    {
        "name": "get_connected_components",
        "description": (
            "Get connected components — groups of bodies that are touching or merged. "
            "Use to check if the assembly is properly connected."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_body_relations",
        "description": (
            "Get spatial relations between bodies: touching, overlapping, or separate."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_bounding_box",
        "description": "Get the overall bounding box of the entire model.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "screenshot",
        "description": "Capture a viewport screenshot of the current model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Output file path for the screenshot"},
                "width": {"type": "integer", "description": "Image width in pixels", "default": 512},
                "height": {"type": "integer", "description": "Image height in pixels", "default": 512},
            },
            "required": ["file"],
        },
    },
    # === MODEL MANAGEMENT ===
    {
        "name": "clear",
        "description": "Clear the current design — remove all features and bodies. Use before starting a new build.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return tool definitions in Claude tool_use format."""
    return list(CAD_TOOLS)


# ---------------------------------------------------------------------------
# Tool dispatcher — routes Claude tool_use calls to backend methods
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Routes Claude tool_use calls to FusionCadBackend methods."""

    def __init__(self, backend: FusionCadBackend):
        self.backend = backend

    def dispatch(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call and return a result dict.

        Returns:
            {"ok": True/False, "result": ..., "error": ...}
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            result = handler(self, tool_input)
            if isinstance(result, BackendResult):
                if result.ok:
                    return {"ok": True, "result": self._extract_data(result)}
                return {"ok": False, "error": result.reason or "command failed"}
            return {"ok": True, "result": result}
        except Exception as ex:
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}

    @staticmethod
    def _extract_data(result: BackendResult) -> Any:
        """Extract meaningful data from BackendResult."""
        resp = result.response
        if resp is None:
            return "ok"
        if isinstance(resp, dict):
            return resp.get("data", resp)
        if hasattr(resp, "json"):
            try:
                body = resp.json()
                if isinstance(body, dict):
                    return body.get("data", body)
            except Exception:
                pass
        return str(resp)

    # --- Handler implementations ---

    def _handle_create_sketch(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.create_sketch(
            sketch_plane=inp["sketch_plane"],
            sketch_name=inp.get("sketch_name"),
        )

    def _handle_add_line(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.add_line(
            sketch_name=inp["sketch_name"],
            pt1=inp["pt1"],
            pt2=inp["pt2"],
        )

    def _handle_add_circle(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.add_circle(
            sketch_name=inp["sketch_name"],
            pt=inp["center"],
            radius=inp["radius"],
        )

    def _handle_add_rectangle(self, inp: Dict[str, Any]) -> BackendResult:
        """Convenience: draw rectangle as 4 points + close."""
        sk = inp["sketch_name"]
        cx, cy = inp["cx"], inp["cy"]
        w, h = inp["width"], inp["height"]
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        pts = [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ]
        last_result = None
        for pt in pts:
            last_result = self.backend.add_point(sk, pt)
            if not last_result.ok:
                return last_result
        close_result = self.backend.close_profile(sk)
        return close_result if close_result.ok else last_result

    def _handle_add_polygon(self, inp: Dict[str, Any]) -> BackendResult:
        """Add polygon points + close."""
        sk = inp["sketch_name"]
        points = inp["points"]
        last_result = None
        for pt in points:
            last_result = self.backend.add_point(sk, pt)
            if not last_result.ok:
                return last_result
        close_result = self.backend.close_profile(sk)
        return close_result if close_result.ok else last_result

    def _handle_close_profile(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.close_profile(inp["sketch_name"])

    def _handle_extrude(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.extrude(
            sketch_name=inp["sketch_name"],
            profile_id=inp.get("profile_id", "profile_0"),
            distance=inp["distance"],
            operation=inp["operation"],
            feature_name=inp.get("feature_name"),
        )

    def _handle_get_model_state(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.get_model_state()

    def _handle_get_bodies(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.get_bodies()

    def _handle_get_features(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.get_features()

    def _handle_get_body_bbox(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.get_body_bbox(body_name=inp.get("body_name", ""))

    def _handle_get_connected_components(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.get_connected_components()

    def _handle_get_body_relations(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.get_body_relations()

    def _handle_query_bounding_box(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.query_bounding_box()

    def _handle_screenshot(self, inp: Dict[str, Any]) -> BackendResult:
        return self.backend.screenshot(
            file=inp["file"],
            width=inp.get("width", 512),
            height=inp.get("height", 512),
        )

    def _handle_clear(self, inp: Dict[str, Any]) -> BackendResult:
        result = self.backend._call("clear", {})
        return BackendResult(ok=True, response=result)

    # Handler dispatch table
    _handlers: Dict[str, Any] = {
        "create_sketch": _handle_create_sketch,
        "add_line": _handle_add_line,
        "add_circle": _handle_add_circle,
        "add_rectangle": _handle_add_rectangle,
        "add_polygon": _handle_add_polygon,
        "close_profile": _handle_close_profile,
        "extrude": _handle_extrude,
        "get_model_state": _handle_get_model_state,
        "get_bodies": _handle_get_bodies,
        "get_features": _handle_get_features,
        "get_body_bbox": _handle_get_body_bbox,
        "get_connected_components": _handle_get_connected_components,
        "get_body_relations": _handle_get_body_relations,
        "query_bounding_box": _handle_query_bounding_box,
        "screenshot": _handle_screenshot,
        "clear": _handle_clear,
    }
