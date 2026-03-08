"""
Declarative tool registry + lightweight payload validation for Fusion360Gym server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


TypeSpec = Any
ValidatorFn = Callable[[Dict[str, Any]], Optional[str]]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_vector3d(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "x" not in value or "y" not in value or "z" not in value:
        return False
    return _is_number(value["x"]) and _is_number(value["y"]) and _is_number(value["z"])


def _is_point2_or_3d(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "x" not in value or "y" not in value:
        return False
    if not _is_number(value["x"]) or not _is_number(value["y"]):
        return False
    if "z" in value and not _is_number(value["z"]):
        return False
    return True


def _is_plane(value: Any) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        base = value.split("@", 1)[0].strip().upper()
        return base in ("XY", "XZ", "YZ")
    if isinstance(value, dict):
        return _is_vector3d(value)
    return False


def _check_type(value: Any, expected: TypeSpec) -> bool:
    if expected in (None, "any"):
        return True
    if isinstance(expected, (list, tuple, set)):
        return any(_check_type(value, e) for e in expected)
    if expected == "dict":
        return isinstance(value, dict)
    if expected == "list":
        return isinstance(value, list)
    if expected == "str":
        return isinstance(value, str)
    if expected == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "float":
        return isinstance(value, float)
    if expected == "number":
        return _is_number(value)
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "vector3d":
        return _is_vector3d(value)
    if expected == "point3d":
        return _is_point2_or_3d(value)
    if expected == "plane":
        return _is_plane(value)
    return False


@dataclass
class CommandSchema:
    required: Dict[str, TypeSpec]
    optional: Dict[str, TypeSpec] | None = None
    allow_extra: bool = True
    root_type: TypeSpec = "dict"


@dataclass
class CommandSpec:
    name: str
    description: str
    handler: Callable[[Any], Tuple[int, str, Any]]
    schema: CommandSchema | None = None
    validator: ValidatorFn | None = None
    category: str = "general"


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, command_name: str) -> Optional[CommandSpec]:
        return self._specs.get(command_name)

    def validate(self, spec: CommandSpec, payload: Any) -> Tuple[bool, Optional[str]]:
        if spec.schema is None:
            if spec.validator is None:
                return True, None
            if not isinstance(payload, dict):
                return False, "payload must be an object"
            custom_error = spec.validator(payload)
            return (custom_error is None), custom_error

        schema = spec.schema
        if payload is None:
            payload_obj = {}
        else:
            payload_obj = payload

        if not _check_type(payload_obj, schema.root_type):
            return False, f"payload must match root type '{schema.root_type}'"

        if not isinstance(payload_obj, dict):
            return False, "payload must be an object"

        optional = schema.optional or {}
        for key, expected in schema.required.items():
            if key not in payload_obj:
                return False, f"missing required field '{key}'"
            if not _check_type(payload_obj[key], expected):
                return False, f"field '{key}' has invalid type (expected {expected})"

        for key, expected in optional.items():
            if key in payload_obj and not _check_type(payload_obj[key], expected):
                return False, f"field '{key}' has invalid type (expected {expected})"

        if not schema.allow_extra:
            allowed = set(schema.required.keys()) | set(optional.keys())
            for key in payload_obj.keys():
                if key not in allowed:
                    return False, f"unexpected field '{key}'"

        if spec.validator is not None:
            custom_error = spec.validator(payload_obj)
            if custom_error is not None:
                return False, custom_error

        return True, None

    def describe(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for spec in sorted(self._specs.values(), key=lambda s: s.name):
            required = []
            optional = []
            if spec.schema is not None:
                required = sorted(list(spec.schema.required.keys()))
                optional = sorted(list((spec.schema.optional or {}).keys()))
            rows.append(
                {
                    "name": spec.name,
                    "category": spec.category,
                    "description": spec.description,
                    "required_fields": required,
                    "optional_fields": optional,
                }
            )
        return rows

