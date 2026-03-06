"""
Stateful executor: run compiled plan against Fusion client with ensure/update semantics.

- In create mode: run all compiled steps; never call clear().
- In edit mode: for each step, if feature exists by name -> skip create (or later: update_extrude); else create.
- Edit policy (Risk D): Only update existing features via update_extrude (distance etc.). Do not "add to"
  existing sketches (add_point/add_line on an existing sketch would use server state that may be
  inconsistent). To change geometry, use a new sketch/feature with a new name; old feature suppress/delete
  is a future option.
- Maintains registry: step_id -> {sketch_name, feature_name, body_name, last_profile_id, ...}.
- Dry-run: print compiled calls only, no client calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cad_compiler import compile_plan, CompiledStep, CompiledCall


@dataclass
class ExecutionResult:
    success: bool
    steps_ok: int = 0
    steps_total: int = 0
    message: str = ""
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _resolve_data(data: Dict[str, Any], refs: Dict[str, Any]) -> Dict[str, Any]:
    """Replace __ref:key with refs[key] in data (shallow)."""
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and v.startswith("__ref:"):
            key = v[6:]
            out[k] = refs.get(key)
        elif isinstance(v, dict):
            out[k] = _resolve_data(v, refs)
        else:
            out[k] = v
    return out


def _get_profile_id_from_response(response: Any) -> Optional[str]:
    """Extract first profile id from client response (data.profiles dict)."""
    if response is None:
        return None
    try:
        if hasattr(response, "json"):
            body = response.json()
        else:
            body = response
        data = (body or {}).get("data") or {}
        profiles = data.get("profiles") or {}
        if profiles:
            return next(iter(profiles))
    except Exception:
        pass
    return None


def _get_sketch_name_from_response(response: Any) -> Optional[str]:
    try:
        if hasattr(response, "json"):
            body = response.json()
        else:
            body = response
        data = (body or {}).get("data") or {}
        return data.get("sketch_name")
    except Exception:
        pass
    return None


class CadExecutor:
    """
    Executes compiled plan via a Fusion client (send_command or method calls).
    client_interface: object with send_command(command, data) returning response with .status_code and .json().
    """

    def __init__(
        self,
        client: Any,
        dry_run: bool = False,
        step_delay: float = 0.4,
        edit_mode: bool = False,
        session_id: Optional[str] = None,
    ):
        self.client = client
        self.dry_run = dry_run
        self.step_delay = step_delay
        self.edit_mode = edit_mode
        self.session_id = session_id or ""
        self.registry: Dict[str, Dict[str, Any]] = {}

    def _send(self, command: str, data: Optional[Dict] = None):
        if self.dry_run:
            return type("Resp", (), {"status_code": 200, "json": lambda: {"data": {}}})()

        if data is None:
            data = {}
        if hasattr(self.client, "send_command"):
            return self.client.send_command(command, data)
        return self.client.send_command(command, data)

    def _add_sketch(self, data: Dict) -> Any:
        sketch_plane = data.get("sketch_plane", "XY")
        sketch_name = data.get("sketch_name")
        if hasattr(self.client, "add_sketch"):
            return self.client.add_sketch(sketch_plane, sketch_name=sketch_name)
        payload = {"sketch_plane": sketch_plane}
        if sketch_name:
            payload["sketch_name"] = sketch_name
        return self._send("add_sketch", payload)

    def _add_point(self, sketch_name: str, pt: Dict) -> Any:
        if hasattr(self.client, "add_point"):
            return self.client.add_point(sketch_name, pt)
        return self._send("add_point", {"sketch_name": sketch_name, "pt": pt})

    def _add_line(self, sketch_name: str, pt1: Dict, pt2: Dict) -> Any:
        if hasattr(self.client, "add_line"):
            return self.client.add_line(sketch_name, pt1, pt2)
        return self._send("add_line", {"sketch_name": sketch_name, "pt1": pt1, "pt2": pt2})

    def _add_circle(self, sketch_name: str, pt: Dict, radius: float) -> Any:
        if hasattr(self.client, "add_circle"):
            return self.client.add_circle(sketch_name, pt, radius)
        return self._send("add_circle", {"sketch_name": sketch_name, "pt": pt, "radius": radius})

    def _close_profile(self, sketch_name: str) -> Any:
        if hasattr(self.client, "close_profile"):
            return self.client.close_profile(sketch_name)
        return self._send("close_profile", {"sketch_name": sketch_name})

    def _add_extrude(
        self,
        sketch_name: str,
        profile_id: str,
        distance: float,
        operation: str,
        feature_name: Optional[str] = None,
    ) -> Any:
        if hasattr(self.client, "add_extrude"):
            return self.client.add_extrude(
                sketch_name, profile_id, distance, operation, feature_name=feature_name
            )
        payload = {
            "sketch_name": sketch_name,
            "profile_id": profile_id,
            "distance": distance,
            "operation": operation,
        }
        if feature_name:
            payload["feature_name"] = feature_name
        return self._send("add_extrude", payload)

    def _find_feature_by_name(self, name: str) -> tuple[bool, Optional[str]]:
        """Check if feature exists by name. Returns (found, error_message).
        If server returns failure (e.g. duplicate name), returns (False, message) so caller can fail deterministically."""
        if self.dry_run:
            return (name in (r.get("feature_name") for r in self.registry.values()), None)
        if hasattr(self.client, "send_command"):
            r = self._send("find_entity_by_name", {"type": "ExtrudeFeature", "name": name})
            if r is None:
                return (False, None)
            code = getattr(r, "status_code", 0)
            try:
                body = r.json() if hasattr(r, "json") else {}
                msg = (body.get("message") or "") if isinstance(body, dict) else ""
            except Exception:
                msg = ""
            if code != 200:
                return (False, msg or f"find_entity_by_name failed ({code})")
            data = (body.get("data") or {}) if isinstance(body, dict) else {}
            found = data.get("found", False)
            count = data.get("count", 0)
            if count > 1:
                return (False, data.get("message") or f"Duplicate feature name: {name} (count={count})")
            return (found, None)
        return (False, None)

    def _update_extrude(self, feature_name: str, distance: float) -> Any:
        if self.dry_run:
            return type("Resp", (), {"status_code": 200})()
        if hasattr(self.client, "send_command"):
            return self._send("update_extrude", {"feature_name": feature_name, "distance": distance})
        return None

    def execute_compiled_step(self, compiled: CompiledStep, plan_session: str) -> bool:
        """Execute one CompiledStep. Returns True if step succeeded."""
        refs: Dict[str, Any] = {}
        last_profile_id: Optional[str] = None
        sketch_name: Optional[str] = None
        feature_name = (compiled.calls[-1].data.get("feature_name") if compiled.calls else None) or ""

        if compiled.primitive in ("loft", "sweep", "fillet"):
            if self.dry_run:
                print(f"  [dry-run] skip unsupported primitive: {compiled.primitive}")
            return True

        for i, call in enumerate(compiled.calls):
            if self.dry_run:
                print(f"  [dry-run] {call.command} {call.data}")
                continue

            data = _resolve_data(dict(call.data), refs)
            if call.command == "add_sketch":
                r = self._add_sketch(data)
                if r and getattr(r, "status_code", 0) != 200:
                    return False
                sketch_name = _get_sketch_name_from_response(r) or data.get("sketch_name")
                if sketch_name:
                    refs["sketch_name"] = sketch_name

            elif call.command == "add_point":
                sn = data.get("sketch_name")
                pt = data.get("pt", {})
                if sn:
                    r = self._add_point(sn, pt)
                    if r and getattr(r, "status_code", 0) != 200:
                        return False
                    pid = _get_profile_id_from_response(r)
                    if pid:
                        last_profile_id = pid

            elif call.command == "add_line":
                sn = data.get("sketch_name")
                if sn:
                    r = self._add_line(sn, data.get("pt1", {}), data.get("pt2", {}))
                    if r and getattr(r, "status_code", 0) != 200:
                        return False
                    last_profile_id = _get_profile_id_from_response(r) or last_profile_id

            elif call.command == "add_circle":
                sn = data.get("sketch_name")
                if sn:
                    r = self._add_circle(sn, data.get("pt", {}), float(data.get("radius", 1)))
                    if r and getattr(r, "status_code", 0) != 200:
                        return False
                    last_profile_id = _get_profile_id_from_response(r) or last_profile_id

            elif call.command == "close_profile":
                sn = data.get("sketch_name")
                if sn:
                    r = self._close_profile(sn)
                    if r and getattr(r, "status_code", 0) != 200:
                        return False
                    last_profile_id = _get_profile_id_from_response(r) or last_profile_id

            elif call.command == "add_extrude":
                refs["profile_id"] = last_profile_id
                data = _resolve_data(dict(call.data), refs)
                sn = data.get("sketch_name")
                pid = data.get("profile_id")
                if not sn or not pid:
                    return False
                r = self._add_extrude(
                    sn,
                    pid,
                    float(data.get("distance", 1)),
                    data.get("operation", "NewBodyFeatureOperation"),
                    feature_name=data.get("feature_name"),
                )
                if r and getattr(r, "status_code", 0) != 200:
                    return False
                feature_name = data.get("feature_name") or ""
                if feature_name and compiled.step_id:
                    self.registry[compiled.step_id] = {
                        "sketch_name": sketch_name,
                        "feature_name": feature_name,
                        "body_name": (data.get("body_name") or ""),
                        "last_profile_id": last_profile_id,
                    }

            if self.step_delay > 0:
                time.sleep(self.step_delay)

        return True

    def execute_plan(self, plan: Dict[str, Any]) -> ExecutionResult:
        """
        Compile and execute plan. In edit mode, skip create for steps whose feature already exists (by name).
        Never calls clear().
        """
        compiled_list = compile_plan(plan)
        session = str(plan.get("session") or self.session_id or "session")
        mode = (plan.get("mode") or "create").strip().lower()
        edit_mode = mode == "edit" or self.edit_mode

        steps_ok = 0
        for cs in compiled_list:
            if edit_mode and cs.primitive in ("rect_extrude", "circle_extrude", "poly_extrude", "wedge_extrude", "cut_extrude"):
                feature_name = None
                for c in cs.calls:
                    if c.command == "add_extrude":
                        feature_name = c.data.get("feature_name")
                        break
                if feature_name:
                    found, err = self._find_feature_by_name(feature_name)
                    if err:
                        return ExecutionResult(
                            success=False,
                            steps_ok=steps_ok,
                            steps_total=len(compiled_list),
                            message=f"find_entity_by_name failed: {err}",
                            registry=dict(self.registry),
                        )
                    if found:
                        if self.dry_run:
                            print(f"  [dry-run] skip (exists): {cs.step_id}")
                        steps_ok += 1
                        continue
            if self.execute_compiled_step(cs, session):
                steps_ok += 1

        return ExecutionResult(
            success=steps_ok == len(compiled_list),
            steps_ok=steps_ok,
            steps_total=len(compiled_list),
            message=f"Executed {steps_ok}/{len(compiled_list)} steps",
            registry=dict(self.registry),
        )
