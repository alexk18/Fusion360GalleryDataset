"""
Universal CAD DSL: Plan (create/edit) and Patch (edit-only) JSON structures.

The LLM outputs ONLY:
- A high-level CAD DSL Plan (create mode), or
- A CAD DSL Patch (edit mode).

All server execution is deterministic: DSL → validated → compiled → executed.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

PLAN_SCHEMA_VERSION = "1.0"

# Operations for extrude/loft/sweep
EXTRUDE_OPERATIONS = [
    "NewBodyFeatureOperation",
    "JoinFeatureOperation",
    "CutFeatureOperation",
    "IntersectFeatureOperation",
]

# Plan primitives
PRIMITIVES = [
    "rect_extrude",
    "circle_extrude",
    "poly_extrude",
    "wedge_extrude",
    "cut_extrude",
    "loft",
    "sweep",
    "fillet",
    "chamfer",
    "revolve",
]

# Plan planes
PLANES = ("XY", "XZ", "YZ")
PLANE_OFFSET_PATTERN = "XY@z|XZ@y|YZ@x"  # e.g. XY@12.5

# Fillet rule-based selectors
FILLET_RULES = (
    "outer_edges",
    "top_perimeter",
    "tip_edges",
    "front_edges",
    "sharp_edges",
)


def default_budget() -> Dict[str, int]:
    return {"max_steps": 25, "max_parts": 18}


def make_step_names(session: str, step_id: str) -> Dict[str, str]:
    """Deterministic naming for persistence across restart."""
    prefix = f"{session}__{step_id}"
    return {
        "sketch": f"{prefix}__sk",
        "feature": f"{prefix}__feat",
        "body": f"{prefix}__body",
        "group": f"{prefix}__grp",
    }


# ---------------------------------------------------------------------------
# Step (one build step in a Plan)
# ---------------------------------------------------------------------------


class Step:
    """One step in a CAD plan. Immutable-ish dict view with helpers."""

    __slots__ = ("_d",)

    def __init__(self, data: Dict[str, Any]):
        self._d = dict(data) if data else {}

    @property
    def id(self) -> str:
        return self._d.get("id", "")

    @property
    def op(self) -> str:
        return self._d.get("op", "ensure")

    @property
    def primitive(self) -> str:
        return self._d.get("primitive", "")

    @property
    def names(self) -> Dict[str, str]:
        return self._d.get("names") or {}

    @property
    def plane(self) -> str:
        return self._d.get("plane", "XY")

    @property
    def profile(self) -> Dict[str, Any]:
        return self._d.get("profile") or {}

    @property
    def distance(self) -> float:
        return float(self._d.get("distance", 0.0))

    @property
    def operation(self) -> str:
        return self._d.get("operation", "NewBodyFeatureOperation")

    @property
    def params(self) -> Dict[str, str]:
        return self._d.get("params") or {}

    @property
    def selectors(self) -> Dict[str, Any]:
        return self._d.get("selectors") or {}

    @property
    def loft(self) -> Dict[str, Any]:
        return self._d.get("loft") or {}

    @property
    def sweep(self) -> Dict[str, Any]:
        return self._d.get("sweep") or {}

    @property
    def fillet(self) -> Dict[str, Any]:
        return self._d.get("fillet") or {}

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)


# ---------------------------------------------------------------------------
# Plan (full create/edit specification)
# ---------------------------------------------------------------------------


class Plan:
    """Full CAD plan: units, session, mode, budget, global, steps."""

    __slots__ = ("_d",)

    def __init__(self, data: Dict[str, Any] | None = None):
        self._d = dict(data) if data else {}

    @property
    def units(self) -> str:
        return self._d.get("units", "cm")

    @property
    def session(self) -> str:
        return self._d.get("session", "")

    @property
    def mode(self) -> str:
        return self._d.get("mode", "create")

    @property
    def budget(self) -> Dict[str, int]:
        return self._d.get("budget") or default_budget()

    @property
    def global_config(self) -> Dict[str, Any]:
        return self._d.get("global") or {}

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return list(self._d.get("steps") or [])

    def step_by_id(self, step_id: str) -> Optional[Dict[str, Any]]:
        for s in self.steps:
            if (s or {}).get("id") == step_id:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._d, indent=indent)


# ---------------------------------------------------------------------------
# Patch (edit-only: replace/add/remove)
# ---------------------------------------------------------------------------


class Patch:
    """Patch format: list of replace/add/remove operations + intent."""

    __slots__ = ("_d",)

    def __init__(self, data: Dict[str, Any] | None = None):
        self._d = dict(data) if data else {}

    @property
    def patches(self) -> List[Dict[str, Any]]:
        return list(self._d.get("patches") or [])

    @property
    def intent(self) -> str:
        return self._d.get("intent", "minimal_change")

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._d)


# ---------------------------------------------------------------------------
# Example plans (for docs and tests)
# ---------------------------------------------------------------------------

EXAMPLE_PLAN_LOWPOLY_TANK = {
    "units": "cm",
    "session": "demo_tank_01",
    "mode": "create",
    "budget": {"max_steps": 25, "max_parts": 18},
    "global": {"origin": "X-centered,Y-back=0,Z-floor=0", "symmetry": "approx_x"},
    "steps": [
        {
            "id": "hull",
            "op": "ensure",
            "primitive": "wedge_extrude",
            "names": {"sketch": "demo_tank_01__hull__sk", "feature": "demo_tank_01__hull__feat", "body": "demo_tank_01__hull__body"},
            "plane": "XY@0",
            "profile": {"type": "poly", "pts": [{"x": -15, "y": 0}, {"x": 15, "y": 0}, {"x": 12, "y": 8}, {"x": -12, "y": 8}]},
            "distance": 8,
            "operation": "NewBodyFeatureOperation",
            "params": {"distance_param": "hull_height"},
        },
        {
            "id": "turret",
            "op": "ensure",
            "primitive": "poly_extrude",
            "names": {"sketch": "demo_tank_01__turret__sk", "feature": "demo_tank_01__turret__feat", "body": "demo_tank_01__turret__body"},
            "plane": "XY@8",
            "profile": {"type": "poly", "pts": [{"x": -4, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 6}, {"x": -4, "y": 6}]},
            "distance": 4,
            "operation": "JoinFeatureOperation",
            "params": {"distance_param": "turret_height"},
        },
        {
            "id": "gun",
            "op": "ensure",
            "primitive": "circle_extrude",
            "names": {"sketch": "demo_tank_01__gun__sk", "feature": "demo_tank_01__gun__feat", "body": "demo_tank_01__gun__body"},
            "plane": "YZ@6",
            "profile": {"type": "circle", "cx": 0, "cy": 5, "radius": 0.8},
            "distance": 20,
            "operation": "JoinFeatureOperation",
            "params": {"distance_param": "gun_length"},
        },
        {
            "id": "wheel_fl",
            "op": "ensure",
            "primitive": "circle_extrude",
            "names": {"sketch": "demo_tank_01__wheel_fl__sk", "feature": "demo_tank_01__wheel_fl__feat", "body": "demo_tank_01__wheel_fl__body"},
            "plane": "YZ@-6",
            "profile": {"type": "circle", "cx": -8, "cy": 4, "radius": 2.5},
            "distance": 2,
            "operation": "NewBodyFeatureOperation",
        },
    ],
}

EXAMPLE_PLAN_LOWPOLY_PLANE = {
    "units": "cm",
    "session": "demo_plane_01",
    "mode": "create",
    "budget": {"max_steps": 25, "max_parts": 18},
    "global": {"origin": "X-centered,Y-back=0,Z-floor=0", "symmetry": "approx_x"},
    "steps": [
        {
            "id": "fuselage",
            "op": "ensure",
            "primitive": "poly_extrude",
            "names": {"sketch": "demo_plane_01__fuselage__sk", "feature": "demo_plane_01__fuselage__feat", "body": "demo_plane_01__fuselage__body"},
            "plane": "XY@0",
            "profile": {"type": "poly", "pts": [{"x": -2, "y": 0}, {"x": 2, "y": 0}, {"x": 2.2, "y": 25}, {"x": -2.2, "y": 25}]},
            "distance": 4,
            "operation": "NewBodyFeatureOperation",
            "params": {"distance_param": "fuselage_w"},
        },
        {
            "id": "wing",
            "op": "ensure",
            "primitive": "poly_extrude",
            "names": {"sketch": "demo_plane_01__wing__sk", "feature": "demo_plane_01__wing__feat", "body": "demo_plane_01__wing__body"},
            "plane": "XZ@12",
            "profile": {
                "type": "poly",
                "pts": [
                    {"x": -15, "y": 0},
                    {"x": 15, "y": 0},
                    {"x": 14, "y": 3},
                    {"x": -14, "y": 3},
                ],
            },
            "distance": 1.5,
            "operation": "JoinFeatureOperation",
            "params": {"distance_param": "wing_thick"},
        },
        {
            "id": "tail",
            "op": "ensure",
            "primitive": "rect_extrude",
            "names": {"sketch": "demo_plane_01__tail__sk", "feature": "demo_plane_01__tail__feat", "body": "demo_plane_01__tail__body"},
            "plane": "XZ@24",
            "profile": {"type": "rect", "cx": 0, "cy": 2, "w": 8, "h": 4},
            "distance": 1,
            "operation": "JoinFeatureOperation",
        },
    ],
}
