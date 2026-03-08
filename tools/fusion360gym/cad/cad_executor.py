"""
Stateful executor: run compiled CAD DSL plan via capability-aware backend contract.

Semantics:
- No direct raw Fusion commands from LLM output.
- In edit mode existing feature is updated only via update_extrude when edit is in-place editable.
- Unsupported in-place edits are deterministically classified as recreate-required or unsupported.
- No fake success for unsupported primitives in normal execution.
- Dry-run may emit unsupported primitives as stubbed (not built geometry).
- clear() is never called.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cad_backend import FusionCadBackend
from .cad_capabilities import (
    CapabilityModel,
    capability_required_for_primitive,
    default_capability_model,
)
from .cad_compiler import CompiledStep, compile_plan


EDIT_IN_PLACE_EDITABLE = "in_place_editable"
EDIT_RECREATE_REQUIRED = "recreate_required"
EDIT_UNSUPPORTED = "unsupported_change"

EXTRUDE_PRIMITIVES = (
    "rect_extrude",
    "circle_extrude",
    "poly_extrude",
    "wedge_extrude",
    "cut_extrude",
)

UNSUPPORTED_EXECUTION_PRIMITIVES = ("loft", "sweep", "fillet", "chamfer", "revolve")


@dataclass
class EditChangeDecision:
    classification: str
    reason: str = ""


def _normalize_plane(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_name(value: Any) -> str:
    return str(value or "").strip()


def _profile_signature(profile: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Optional[str]]:
    profile = profile or {}
    ptype = str(profile.get("type") or "").strip().lower()
    if ptype in ("", "rect"):
        return (
            (
                "rect",
                float(profile.get("w", 0.0)),
                float(profile.get("h", 0.0)),
                float(profile.get("cx", 0.0)),
                float(profile.get("cy", 0.0)),
            ),
            None,
        )
    if ptype == "circle":
        return (
            (
                "circle",
                float(profile.get("radius", 0.0)),
                float(profile.get("cx", 0.0)),
                float(profile.get("cy", 0.0)),
            ),
            None,
        )
    if ptype == "poly":
        pts = tuple((float(p.get("x", 0.0)), float(p.get("y", 0.0))) for p in (profile.get("pts") or []))
        arcs = tuple(
            (
                int(a.get("i_start", -1)),
                int(a.get("i_end", -1)),
                float(a.get("angle_deg", 0.0)),
            )
            for a in (profile.get("arcs") or [])
        )
        if arcs:
            return (("poly", pts, arcs), "poly profile arcs are unsupported in current backend")
        return (("poly", pts), None)
    return ((ptype,), f"unsupported profile type: {ptype!r}")


def classify_edit_change(step_before: Dict[str, Any], step_after: Dict[str, Any]) -> EditChangeDecision:
    """
    Classify change between previous and new step:
    - in_place_editable: only distance (or no geometry-affecting change)
    - recreate_required: plane/operation/profile/name changed
    - unsupported_change: primitive/profile unsupported for comparison
    """
    before = step_before or {}
    after = step_after or {}

    primitive_before = str(before.get("primitive") or "").strip().lower()
    primitive_after = str(after.get("primitive") or "").strip().lower()
    if not primitive_before and primitive_after:
        primitive_before = primitive_after
    if not primitive_after and primitive_before:
        primitive_after = primitive_before
    if not primitive_before and not primitive_after:
        primitive_before = "rect_extrude"
        primitive_after = "rect_extrude"
    if primitive_before != primitive_after:
        return EditChangeDecision(
            EDIT_UNSUPPORTED,
            "changing primitive type in-place is unsupported",
        )
    if primitive_after not in EXTRUDE_PRIMITIVES:
        return EditChangeDecision(
            EDIT_UNSUPPORTED,
            f"in-place edit is unsupported for primitive {primitive_after!r}",
        )
    if _normalize_plane(before.get("plane")) != _normalize_plane(after.get("plane")):
        return EditChangeDecision(
            EDIT_RECREATE_REQUIRED,
            "plane change is not in-place editable",
        )
    if _normalize_name(before.get("operation")) != _normalize_name(after.get("operation")):
        return EditChangeDecision(
            EDIT_RECREATE_REQUIRED,
            "operation change is not in-place editable",
        )
    before_profile, before_profile_reason = _profile_signature(before.get("profile") or {})
    after_profile, after_profile_reason = _profile_signature(after.get("profile") or {})
    if before_profile_reason or after_profile_reason:
        reason = before_profile_reason or after_profile_reason or "unsupported profile change"
        return EditChangeDecision(EDIT_UNSUPPORTED, reason)
    if before_profile != after_profile:
        return EditChangeDecision(
            EDIT_RECREATE_REQUIRED,
            "profile geometry change is not in-place editable",
        )
    names_before = before.get("names") or {}
    names_after = after.get("names") or {}
    if _normalize_name(names_before.get("feature")) != _normalize_name(names_after.get("feature")):
        return EditChangeDecision(
            EDIT_RECREATE_REQUIRED,
            "feature name change requires recreate path",
        )
    return EditChangeDecision(EDIT_IN_PLACE_EDITABLE, "")


def detect_unsupported_edit(step_before: Dict[str, Any], step_after: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Backward-compatible helper for tests/callers.
    Returns (supported, reason), where supported=True means in-place editable.
    """
    decision = classify_edit_change(step_before, step_after)
    return (decision.classification == EDIT_IN_PLACE_EDITABLE, decision.reason)


def _get_distance_from_compiled_step(compiled: CompiledStep) -> float:
    for call in compiled.calls:
        if call.command == "add_extrude":
            return float(call.data.get("distance", 1.0))
    return 1.0


def _get_feature_name_from_compiled_step(compiled: CompiledStep) -> Optional[str]:
    for call in compiled.calls:
        if call.command == "add_extrude":
            return _normalize_name(call.data.get("feature_name")) or None
    return None


@dataclass
class ExecutionTraceEvent:
    step_id: str
    primitive: str
    action: str
    backend_path: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ExecutionResult:
    success: bool
    steps_ok: int = 0
    steps_total: int = 0
    message: str = ""
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trace: List[ExecutionTraceEvent] = field(default_factory=list)


def _resolve_data(data: Dict[str, Any], refs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and value.startswith("__ref:"):
            out[key] = refs.get(value[6:])
        elif isinstance(value, dict):
            out[key] = _resolve_data(value, refs)
        else:
            out[key] = value
    return out


def _response_status_code(response: Any) -> int:
    if response is None:
        return 0
    if hasattr(response, "status_code"):
        try:
            return int(getattr(response, "status_code"))
        except Exception:
            return 0
    if isinstance(response, dict):
        status = response.get("status")
        if isinstance(status, int):
            return status
    return 200


def _response_json(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    if isinstance(response, dict):
        return response
    if hasattr(response, "json"):
        try:
            body = response.json()
            if isinstance(body, dict):
                return body
        except Exception:
            return {}
    return {}


def _response_error_message(response: Any) -> str:
    body = _response_json(response)
    message = body.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if response is None:
        return "no response"
    return "command failed"


def _get_profile_id_from_response(response: Any) -> Optional[str]:
    body = _response_json(response)
    data = body.get("data") or {}
    profiles = data.get("profiles") or {}
    if profiles:
        return next(iter(profiles))
    return None


def _get_sketch_name_from_response(response: Any) -> Optional[str]:
    body = _response_json(response)
    data = body.get("data") or {}
    name = data.get("sketch_name")
    return name if isinstance(name, str) and name.strip() else None


class CadExecutor:
    def __init__(
        self,
        client: Any,
        dry_run: bool = False,
        step_delay: float = 0.4,
        edit_mode: bool = False,
        session_id: Optional[str] = None,
        capabilities: Optional[CapabilityModel] = None,
    ):
        self.client = client
        self.dry_run = dry_run
        self.step_delay = step_delay
        self.edit_mode = edit_mode
        self.session_id = session_id or ""
        self.capabilities = capabilities or default_capability_model()
        self.backend = FusionCadBackend(client, capabilities=self.capabilities)
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.trace: List[ExecutionTraceEvent] = []
        self._last_step_error = ""
        self._last_step_backend_path: List[str] = []

    def _execute_controlled_recreate_suffix(
        self,
        compiled_list: List[CompiledStep],
        start_index: int,
        steps_ok_so_far: int,
        steps_total: int,
        session: str,
        decision_reason: str,
    ) -> ExecutionResult:
        suffix = compiled_list[start_index:]
        start_step = compiled_list[start_index]
        start_step_id = start_step.step_id
        self._record_trace(
            step_id=start_step_id,
            primitive=start_step.primitive,
            action="recreate-required",
            backend_path=["controlled_recreate_suffix"],
            reason=f"{decision_reason}; suffix_steps={len(suffix)}",
        )

        blocked_features: List[str] = []
        for cs in suffix:
            feature_name = _get_feature_name_from_compiled_step(cs)
            if not feature_name:
                continue
            found, find_err, _find_path = self._feature_lookup(feature_name)
            if find_err:
                self._record_trace(
                    step_id=cs.step_id,
                    primitive=cs.primitive,
                    action="failed",
                    backend_path=["controlled_recreate_suffix", "find_entity_by_name"],
                    reason=find_err,
                )
                return ExecutionResult(
                    success=False,
                    steps_ok=steps_ok_so_far,
                    steps_total=steps_total,
                    message=f"controlled recreate failed during lookup: {find_err}",
                    registry=dict(self.registry),
                    trace=list(self.trace),
                )
            if found:
                blocked_features.append(feature_name)

        if blocked_features:
            reason = (
                "controlled recreate blocked: existing suffix features require delete/suppress path, "
                "which is unavailable in current backend; existing="
                + ", ".join(blocked_features[:8])
            )
            self._record_trace(
                step_id=start_step_id,
                primitive=start_step.primitive,
                action="failed",
                backend_path=["controlled_recreate_suffix"],
                reason=reason,
            )
            return ExecutionResult(
                success=False,
                steps_ok=steps_ok_so_far,
                steps_total=steps_total,
                message=reason,
                registry=dict(self.registry),
                trace=list(self.trace),
            )

        suffix_ok = 0
        for cs in suffix:
            ok = self.execute_compiled_step(cs, session)
            if not ok:
                reason = self._last_step_error or "controlled recreate suffix step failed"
                self._record_trace(
                    step_id=cs.step_id,
                    primitive=cs.primitive,
                    action="failed",
                    backend_path=list(self._last_step_backend_path),
                    reason=reason,
                )
                return ExecutionResult(
                    success=False,
                    steps_ok=steps_ok_so_far + suffix_ok,
                    steps_total=steps_total,
                    message=reason,
                    registry=dict(self.registry),
                    trace=list(self.trace),
                )
            self._record_trace(
                step_id=cs.step_id,
                primitive=cs.primitive,
                action="created",
                backend_path=list(self._last_step_backend_path),
                reason="controlled_recreate_suffix",
            )
            suffix_ok += 1

        total_ok = steps_ok_so_far + suffix_ok
        return ExecutionResult(
            success=total_ok == steps_total,
            steps_ok=total_ok,
            steps_total=steps_total,
            message=(
                f"Executed {total_ok}/{steps_total} steps "
                f"(edit with controlled recreate from '{start_step_id}')"
            ),
            registry=dict(self.registry),
            trace=list(self.trace),
        )

    def _record_trace(
        self,
        *,
        step_id: str,
        primitive: str,
        action: str,
        backend_path: Optional[List[str]] = None,
        reason: str = "",
    ) -> None:
        self.trace.append(
            ExecutionTraceEvent(
                step_id=step_id,
                primitive=primitive,
                action=action,
                backend_path=list(backend_path or []),
                reason=reason,
            )
        )

    def _feature_lookup(self, feature_name: str) -> Tuple[bool, Optional[str], List[str]]:
        path = ["find_entity_by_name"]
        if self.dry_run:
            found = feature_name in (entry.get("feature_name") for entry in self.registry.values())
            return (found, None, path)
        result = self.backend.find_entity_by_name("ExtrudeFeature", feature_name)
        if not result.ok:
            return (False, result.reason or "find_entity_by_name failed", path)
        response = result.response
        code = _response_status_code(response)
        body = _response_json(response)
        if code != 200:
            return (False, _response_error_message(response), path)
        data = body.get("data") or {}
        count = int(data.get("count", 0) or 0)
        if count > 1:
            return (
                False,
                data.get("message")
                or f"Duplicate entity name: '{feature_name}' found {count} times. Rename to ensure uniqueness.",
                path,
            )
        return (bool(data.get("found", False)), None, path)

    def _update_feature_distance(self, feature_name: str, distance: float) -> Tuple[bool, str, List[str]]:
        path = ["update_extrude"]
        if self.dry_run:
            return (True, "", path)
        result = self.backend.update_extrude(feature_name, distance)
        if not result.ok:
            return (False, result.reason or "update_extrude capability rejection", path)
        response = result.response
        if _response_status_code(response) != 200:
            return (False, _response_error_message(response), path)
        return (True, "", path)

    def _check_capability_for_step(self, step_id: str, primitive: str, operation: str) -> Tuple[bool, str]:
        required = capability_required_for_primitive(primitive, operation)
        if required == "unsupported":
            return (False, f"unsupported primitive '{primitive}'")
        if not self.capabilities.supports(required):
            return (
                False,
                f"capability '{required}' unavailable: {self.capabilities.reason(required)}",
            )
        return (True, "")

    def execute_compiled_step(self, compiled: CompiledStep, _plan_session: str) -> bool:
        self._last_step_error = ""
        self._last_step_backend_path = []
        refs: Dict[str, Any] = {}
        sketch_name: Optional[str] = None
        last_profile_id: Optional[str] = None

        for call in compiled.calls:
            self._last_step_backend_path.append(call.command)
            data = _resolve_data(dict(call.data), refs)
            if self.dry_run:
                print(f"  [dry-run] {call.command} {data}")
                if call.command == "add_sketch":
                    refs["sketch_name"] = data.get("sketch_name")
                if call.command in ("add_point", "add_line", "add_circle", "close_profile"):
                    refs["profile_id"] = refs.get("profile_id", "__dry_run_profile__")
                continue

            if call.command == "add_sketch":
                result = self.backend.create_sketch(
                    data.get("sketch_plane", "XY"),
                    sketch_name=data.get("sketch_name"),
                )
            elif call.command == "add_point":
                result = self.backend.add_point(data.get("sketch_name", ""), data.get("pt", {}))
            elif call.command == "add_line":
                result = self.backend.add_line(
                    data.get("sketch_name", ""),
                    data.get("pt1", {}),
                    data.get("pt2", {}),
                )
            elif call.command == "add_circle":
                result = self.backend.add_circle(
                    data.get("sketch_name", ""),
                    data.get("pt", {}),
                    float(data.get("radius", 1.0)),
                )
            elif call.command == "close_profile":
                result = self.backend.close_profile(data.get("sketch_name", ""))
            elif call.command == "add_extrude":
                refs["profile_id"] = last_profile_id
                extrude_data = _resolve_data(dict(call.data), refs)
                sketch = extrude_data.get("sketch_name")
                profile_id = extrude_data.get("profile_id")
                if not sketch or not profile_id:
                    self._last_step_error = "add_extrude requires resolved sketch_name and profile_id"
                    return False
                result = self.backend.extrude(
                    sketch_name=sketch,
                    profile_id=profile_id,
                    distance=float(extrude_data.get("distance", 1.0)),
                    operation=extrude_data.get("operation", "NewBodyFeatureOperation"),
                    feature_name=extrude_data.get("feature_name"),
                )
            else:
                self._last_step_error = f"unknown compiled command: {call.command}"
                return False

            if not result.ok:
                self._last_step_error = result.reason or f"backend rejected command {call.command}"
                return False
            response = result.response
            if _response_status_code(response) != 200:
                self._last_step_error = _response_error_message(response)
                return False

            if call.command == "add_sketch":
                sketch_name = _get_sketch_name_from_response(response) or data.get("sketch_name")
                if sketch_name:
                    refs["sketch_name"] = sketch_name
            elif call.command in ("add_point", "add_line", "add_circle", "close_profile"):
                last_profile_id = _get_profile_id_from_response(response) or last_profile_id
                if last_profile_id:
                    refs["profile_id"] = last_profile_id
            elif call.command == "add_extrude":
                feature_name = _normalize_name(data.get("feature_name"))
                if feature_name and compiled.step_id:
                    self.registry[compiled.step_id] = {
                        "sketch_name": sketch_name or "",
                        "feature_name": feature_name,
                        "body_name": _normalize_name(data.get("body_name")),
                        "last_profile_id": last_profile_id,
                        "updated": False,
                    }

            if self.step_delay > 0:
                time.sleep(self.step_delay)

        return True

    def execute_plan(
        self,
        plan: Dict[str, Any],
        previous_plan: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        compiled_list = compile_plan(plan)
        raw_steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict) and s]
        compiled_ids = {cs.step_id for cs in compiled_list}
        missing_compiled_steps: List[str] = []
        for i, step in enumerate(raw_steps):
            sid = str(step.get("id") or f"step_{i}")
            if sid not in compiled_ids:
                missing_compiled_steps.append(sid)

        if missing_compiled_steps:
            reason = (
                "Plan contains non-compilable steps (unsupported or malformed): "
                + ", ".join(missing_compiled_steps[:8])
            )
            self._record_trace(
                step_id="compile",
                primitive="plan",
                action="failed",
                backend_path=["compile_plan"],
                reason=reason,
            )
            return ExecutionResult(
                success=False,
                steps_ok=0,
                steps_total=len(raw_steps),
                message=reason,
                registry=dict(self.registry),
                trace=list(self.trace),
            )

        steps_total = len(compiled_list)
        mode = _normalize_name(plan.get("mode") or "create").lower()
        edit_mode = mode == "edit" or self.edit_mode
        session = _normalize_name(plan.get("session") or self.session_id or "session")
        steps = plan.get("steps") or []
        steps_by_id = {s.get("id"): s for s in steps if isinstance(s, dict) and s.get("id")}
        previous_steps = {}
        if previous_plan:
            previous_steps = {
                s.get("id"): s for s in (previous_plan.get("steps") or []) if isinstance(s, dict) and s.get("id")
            }

        steps_ok = 0
        for compiled_index, compiled in enumerate(compiled_list):
            step_id = compiled.step_id
            primitive = compiled.primitive
            current_step = steps_by_id.get(step_id) or {}
            operation = _normalize_name(current_step.get("operation"))

            cap_ok, cap_reason = self._check_capability_for_step(step_id, primitive, operation)
            if not cap_ok:
                if self.dry_run:
                    self._record_trace(
                        step_id=step_id,
                        primitive=primitive,
                        action="unsupported",
                        backend_path=[],
                        reason=f"{cap_reason} (dry-run stub; no geometry built)",
                    )
                    steps_ok += 1
                    continue
                self._record_trace(
                    step_id=step_id,
                    primitive=primitive,
                    action="unsupported",
                    backend_path=[],
                    reason=cap_reason,
                )
                return ExecutionResult(
                    success=False,
                    steps_ok=steps_ok,
                    steps_total=steps_total,
                    message=cap_reason,
                    registry=dict(self.registry),
                    trace=list(self.trace),
                )

            if primitive in UNSUPPORTED_EXECUTION_PRIMITIVES:
                reason = f"unsupported primitive in execution: {primitive}"
                if self.dry_run:
                    self._record_trace(
                        step_id=step_id,
                        primitive=primitive,
                        action="unsupported",
                        backend_path=[],
                        reason=f"{reason} (dry-run stub; no geometry built)",
                    )
                    steps_ok += 1
                    continue
                self._record_trace(
                    step_id=step_id,
                    primitive=primitive,
                    action="unsupported",
                    backend_path=[],
                    reason=reason,
                )
                return ExecutionResult(
                    success=False,
                    steps_ok=steps_ok,
                    steps_total=steps_total,
                    message=reason,
                    registry=dict(self.registry),
                    trace=list(self.trace),
                )

            feature_name = _get_feature_name_from_compiled_step(compiled)
            if edit_mode and feature_name and primitive in EXTRUDE_PRIMITIVES:
                decision = None
                if step_id in previous_steps and step_id in steps_by_id:
                    decision = classify_edit_change(previous_steps[step_id], steps_by_id[step_id])
                    if decision.classification == EDIT_UNSUPPORTED:
                        reason = f"unsupported in-place edit for step '{step_id}': {decision.reason}"
                        self._record_trace(
                            step_id=step_id,
                            primitive=primitive,
                            action="failed",
                            backend_path=["classify_edit_change"],
                            reason=reason,
                        )
                        return ExecutionResult(
                            success=False,
                            steps_ok=steps_ok,
                            steps_total=steps_total,
                            message=reason,
                            registry=dict(self.registry),
                            trace=list(self.trace),
                        )
                    if decision.classification == EDIT_RECREATE_REQUIRED:
                        return self._execute_controlled_recreate_suffix(
                            compiled_list=compiled_list,
                            start_index=compiled_index,
                            steps_ok_so_far=steps_ok,
                            steps_total=steps_total,
                            session=session,
                            decision_reason=f"recreate-required for step '{step_id}': {decision.reason}",
                        )

                found, find_err, find_path = self._feature_lookup(feature_name)
                if find_err:
                    self._record_trace(
                        step_id=step_id,
                        primitive=primitive,
                        action="failed",
                        backend_path=find_path,
                        reason=find_err,
                    )
                    return ExecutionResult(
                        success=False,
                        steps_ok=steps_ok,
                        steps_total=steps_total,
                        message=f"duplicate entity name or find failed: {find_err}",
                        registry=dict(self.registry),
                        trace=list(self.trace),
                    )
                if found:
                    new_distance = _get_distance_from_compiled_step(compiled)
                    updated, update_err, update_path = self._update_feature_distance(feature_name, new_distance)
                    backend_path = find_path + update_path
                    if not updated:
                        self._record_trace(
                            step_id=step_id,
                            primitive=primitive,
                            action="failed",
                            backend_path=backend_path,
                            reason=update_err,
                        )
                        return ExecutionResult(
                            success=False,
                            steps_ok=steps_ok,
                            steps_total=steps_total,
                            message=f"existing feature found, update_extrude failed: {update_err}",
                            registry=dict(self.registry),
                            trace=list(self.trace),
                        )
                    if self.dry_run:
                        print(f"  [dry-run] update_extrude feature={feature_name!r} distance={new_distance}")
                    self.registry[step_id] = {
                        "sketch_name": "",
                        "feature_name": feature_name,
                        "body_name": "",
                        "last_profile_id": None,
                        "updated": True,
                    }
                    self._record_trace(
                        step_id=step_id,
                        primitive=primitive,
                        action="updated",
                        backend_path=backend_path,
                        reason="",
                    )
                    steps_ok += 1
                    if self.step_delay > 0:
                        time.sleep(self.step_delay)
                    continue

            ok = self.execute_compiled_step(compiled, session)
            if not ok:
                reason = self._last_step_error or "step execution failed"
                self._record_trace(
                    step_id=step_id,
                    primitive=primitive,
                    action="failed",
                    backend_path=list(self._last_step_backend_path),
                    reason=reason,
                )
                return ExecutionResult(
                    success=False,
                    steps_ok=steps_ok,
                    steps_total=steps_total,
                    message=reason,
                    registry=dict(self.registry),
                    trace=list(self.trace),
                )
            self._record_trace(
                step_id=step_id,
                primitive=primitive,
                action="created",
                backend_path=list(self._last_step_backend_path),
                reason="dry-run" if self.dry_run else "",
            )
            steps_ok += 1

        mode_suffix = "create + in-place update" if edit_mode else "create"
        return ExecutionResult(
            success=steps_ok == steps_total,
            steps_ok=steps_ok,
            steps_total=steps_total,
            message=f"Executed {steps_ok}/{steps_total} steps ({mode_suffix})",
            registry=dict(self.registry),
            trace=list(self.trace),
        )
