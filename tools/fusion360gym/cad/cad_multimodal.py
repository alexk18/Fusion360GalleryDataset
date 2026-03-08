"""
Text-to-CAD and image-to-CAD convergence contract.

Both input paths must converge into one CAD DSL execution path:
text/images -> structural representation -> CAD plan -> validate/compile/execute

This module intentionally keeps visual interpretation stubbed while exposing
deterministic integration points for future image-to-CAD models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .cad_dsl import make_step_names


@dataclass
class StructuralPart:
    name: str
    bbox: Dict[str, float]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralObjectRepresentation:
    source: str
    parts: List[StructuralPart]
    notes: List[str] = field(default_factory=list)


class VisualInterpreter(Protocol):
    def interpret(self, images: List[Dict[str, Any]]) -> StructuralObjectRepresentation:
        ...


class TextInterpreter(Protocol):
    def interpret(self, text_request: str) -> StructuralObjectRepresentation:
        ...


class CadPlanner(Protocol):
    def plan(self, structural: StructuralObjectRepresentation, session: str, mode: str = "create") -> Dict[str, Any]:
        ...


def structural_from_legacy_bboxes(parts: List[Dict[str, Any]]) -> StructuralObjectRepresentation:
    """
    Legacy adapter only. Converts bbox parts into structural representation.
    This is a fallback bridge, not the primary truth model.
    """
    structural_parts: List[StructuralPart] = []
    for index, part in enumerate(parts or []):
        name = str(part.get("name") or f"part_{index}")
        bbox = {
            "x_min": float(part.get("x_min", 0.0)),
            "x_max": float(part.get("x_max", 0.0)),
            "y_min": float(part.get("y_min", 0.0)),
            "y_max": float(part.get("y_max", 0.0)),
            "z_min": float(part.get("z_min", 0.0)),
            "z_max": float(part.get("z_max", 0.0)),
        }
        structural_parts.append(StructuralPart(name=name, bbox=bbox))
    return StructuralObjectRepresentation(
        source="legacy_bbox_fallback",
        parts=structural_parts,
        notes=["Converted from legacy bbox format; downstream path is unified CAD DSL"],
    )


class LegacyBboxCadPlanner:
    """
    Deterministic planner from structural bboxes to CAD DSL.
    Used only as transition fallback and for integration tests.
    """

    def plan(self, structural: StructuralObjectRepresentation, session: str, mode: str = "create") -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        for index, part in enumerate(structural.parts):
            bbox = part.bbox
            width = max(0.1, bbox["x_max"] - bbox["x_min"])
            depth = max(0.1, bbox["y_max"] - bbox["y_min"])
            height = max(0.1, bbox["z_max"] - bbox["z_min"])
            cx = (bbox["x_min"] + bbox["x_max"]) * 0.5
            cy = (bbox["y_min"] + bbox["y_max"]) * 0.5
            base_z = bbox["z_min"]
            step_id = f"part_{index}"
            names = make_step_names(session, step_id)
            steps.append(
                {
                    "id": step_id,
                    "op": "ensure",
                    "primitive": "rect_extrude",
                    "names": names,
                    "plane": f"XY@{base_z}",
                    "profile": {"type": "rect", "cx": cx, "cy": cy, "w": width, "h": depth},
                    "distance": height,
                    "operation": "NewBodyFeatureOperation",
                }
            )
        return {
            "units": "cm",
            "session": session,
            "mode": mode,
            "global": {"source": structural.source},
            "steps": steps,
        }


def text_to_cad_plan(
    text_request: str,
    *,
    session: str,
    text_interpreter: TextInterpreter,
    planner: CadPlanner,
    mode: str = "create",
) -> Dict[str, Any]:
    structural = text_interpreter.interpret(text_request)
    return planner.plan(structural, session=session, mode=mode)


def image_to_cad_plan(
    images: List[Dict[str, Any]],
    *,
    session: str,
    visual_interpreter: VisualInterpreter,
    planner: CadPlanner,
    mode: str = "create",
) -> Dict[str, Any]:
    structural = visual_interpreter.interpret(images)
    return planner.plan(structural, session=session, mode=mode)
