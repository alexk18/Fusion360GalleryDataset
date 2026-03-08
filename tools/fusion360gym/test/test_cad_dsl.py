"""
Unit tests for CAD DSL: patch apply, validation, compiler (dry-run), selectors.
"""

import os
import sys
import json

# Add fusion360gym (parent of test) so "cad" is a package
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from cad.cad_dsl import (
    Plan,
    Patch,
    Step,
    default_budget,
    make_step_names,
    EXAMPLE_PLAN_LOWPOLY_TANK,
)
from cad.cad_patch import apply_patch, _parse_path
from cad.cad_validate import validate_plan, _poly_area
from cad.cad_compiler import compile_plan, CompiledStep
from cad.cad_capabilities import Capability, CapabilityModel, STATUS_UNSUPPORTED, default_capability_model
from cad.cad_backend import FusionCadBackend
from cad.cad_operator import run_cad_operator
import cad.cad_operator as cad_operator_module
from cad.cad_executor import (
    CadExecutor,
    ExecutionResult,
    _resolve_data,
    _get_profile_id_from_response,
    detect_unsupported_edit,
    classify_edit_change,
    EDIT_RECREATE_REQUIRED,
    EDIT_UNSUPPORTED,
    EDIT_IN_PLACE_EDITABLE,
    _get_distance_from_compiled_step,
    _get_feature_name_from_compiled_step,
)


def test_make_step_names():
    names = make_step_names("sess1", "hull")
    assert names["sketch"] == "sess1__hull__sk"
    assert names["feature"] == "sess1__hull__feat"
    assert names["body"] == "sess1__hull__body"


def test_parse_path():
    assert _parse_path("steps[hull].distance") == ["steps", "hull", "distance"]
    assert _parse_path("steps[a]") == ["steps", "a"]


def test_patch_replace_distance():
    plan = {
        "units": "cm",
        "session": "s1",
        "mode": "create",
        "budget": {"max_steps": 25, "max_parts": 18},
        "steps": [
            {"id": "hull", "primitive": "rect_extrude", "distance": 5, "profile": {"type": "rect", "w": 10, "h": 8}},
        ],
    }
    patch = {
        "patches": [
            {"op": "replace", "path": "steps[hull].distance", "value": 18.0},
        ],
        "intent": "minimal_change",
    }
    out = apply_patch(plan, patch)
    assert isinstance(out["steps"][0], dict), "step should remain a dict"
    assert out["steps"][0]["distance"] == 18.0
    assert out["steps"][0]["id"] == "hull"


def test_patch_add_step():
    plan = {"session": "s1", "steps": [{"id": "a", "distance": 1}]}
    patch = {
        "patches": [
            {"op": "add", "path": "steps", "value": {"id": "b", "distance": 2}},
        ],
        "intent": "minimal_change",
    }
    out = apply_patch(plan, patch)
    assert len(out["steps"]) == 2
    assert out["steps"][1]["id"] == "b"
    assert out["steps"][1]["distance"] == 2


def test_patch_remove_step():
    plan = {"session": "s1", "steps": [{"id": "a", "d": 1}, {"id": "b", "d": 2}]}
    patch = {"patches": [{"op": "remove", "path": "steps[b]"}]}
    out = apply_patch(plan, patch)
    assert len(out["steps"]) == 1
    assert out["steps"][0]["id"] == "a"


def test_poly_area():
    pts = [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}, {"x": 0, "y": 2}]
    assert _poly_area(pts) == 4.0


def test_validate_plan_ok():
    plan = {
        "units": "cm",
        "session": "s1",
        "budget": {"max_steps": 25, "max_parts": 18},
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    vr = validate_plan(plan)
    assert vr.valid, vr.errors


def test_validate_plan_budget():
    plan = {
        "session": "s1",
        "budget": {"max_steps": 3, "max_parts": 3},
        "steps": [
            {"id": f"s{i}", "primitive": "rect_extrude", "profile": {"type": "rect", "w": 1, "h": 1}, "distance": 1}
            for i in range(5)
        ],
    }
    vr = validate_plan(plan)
    assert not vr.valid
    assert any("steps" in e for e in vr.errors)


def test_validate_plan_rect_min_size():
    plan = {
        "session": "s1",
        "steps": [
            {
                "id": "tiny",
                "primitive": "rect_extrude",
                "profile": {"type": "rect", "w": 0.01, "h": 0.01},
                "distance": 1,
            },
        ],
    }
    vr = validate_plan(plan)
    assert not vr.valid
    assert any("rect" in e.lower() or "min" in e.lower() for e in vr.errors)


def test_validate_profile_type_mismatch_is_rejected():
    plan = {
        "session": "s1",
        "steps": [
            {
                "id": "bad_rect",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "circle", "radius": 2},
                "distance": 1,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    vr = validate_plan(plan)
    assert not vr.valid
    assert any("requires profile.type='rect'" in e.lower() for e in vr.errors)


def test_compile_plan_tank():
    compiled = compile_plan(EXAMPLE_PLAN_LOWPOLY_TANK)
    assert len(compiled) >= 1
    for cs in compiled:
        assert isinstance(cs, CompiledStep)
        assert cs.step_id
        assert cs.primitive in ("rect_extrude", "circle_extrude", "poly_extrude", "wedge_extrude", "cut_extrude")
    hull = next((c for c in compiled if c.step_id == "hull"), None)
    assert hull is not None
    assert hull.primitive == "wedge_extrude"
    assert any(c.command == "add_sketch" for c in hull.calls)
    assert any(c.command == "add_extrude" for c in hull.calls)


def test_resolve_data():
    refs = {"sketch_name": "Sk1", "profile_id": "prof_abc"}
    data = {"sketch_name": "__ref:sketch_name", "profile_id": "__ref:profile_id", "distance": 5}
    out = _resolve_data(data, refs)
    assert out["sketch_name"] == "Sk1"
    assert out["profile_id"] == "prof_abc"
    assert out["distance"] == 5


def test_get_profile_id_from_response():
    class R:
        def json(self):
            return {"data": {"profiles": {"pid1": {}}}}
    assert _get_profile_id_from_response(R()) == "pid1"
    assert _get_profile_id_from_response(None) is None


def test_executor_dry_run():
    class FakeClient:
        def send_command(self, cmd, data=None):
            return type("Resp", (), {"status_code": 200, "json": lambda: {"data": {}}})()

    from cad.cad_compiler import compile_plan
    plan = {
        "session": "t",
        "steps": [
            {
                "id": "box",
                "primitive": "rect_extrude",
                "names": {"sketch": "t__box__sk", "feature": "t__box__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 5, "h": 5},
                "distance": 2,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    exec = CadExecutor(FakeClient(), dry_run=True)
    result = exec.execute_plan(plan)
    assert result.steps_total >= 1
    assert result.success


def test_get_distance_and_feature_name_from_compiled():
    from cad.cad_compiler import compile_plan
    plan = {"session": "s", "steps": [{"id": "a", "primitive": "rect_extrude", "plane": "XY", "profile": {"type": "rect", "w": 2, "h": 2}, "distance": 7.5, "operation": "NewBodyFeatureOperation"}]}
    compiled = compile_plan(plan)
    assert len(compiled) == 1
    assert _get_distance_from_compiled_step(compiled[0]) == 7.5
    assert _get_feature_name_from_compiled_step(compiled[0]) == "s__a__feat"


def test_detect_unsupported_edit_distance_only():
    before = {"id": "hull", "plane": "XY", "operation": "NewBodyFeatureOperation", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 5}
    after = {"id": "hull", "plane": "XY", "operation": "NewBodyFeatureOperation", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 12}
    ok, reason = detect_unsupported_edit(before, after)
    assert ok is True
    assert reason == ""


def test_detect_unsupported_edit_profile_change():
    before = {"id": "hull", "plane": "XY", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 5}
    after = {"id": "hull", "plane": "XY", "profile": {"type": "rect", "w": 15, "h": 8}, "distance": 5}
    ok, reason = detect_unsupported_edit(before, after)
    assert ok is False
    assert "profile" in reason.lower() or "unsupported" in reason.lower()


def test_detect_unsupported_edit_plane_change():
    before = {"id": "hull", "plane": "XY", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 5}
    after = {"id": "hull", "plane": "XZ", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 5}
    ok, reason = detect_unsupported_edit(before, after)
    assert ok is False
    assert "plane" in reason.lower() or "unsupported" in reason.lower()


class _MockResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return {"data": self._data, "message": ""}


def test_edit_mode_calls_update_extrude():
    """In edit mode when feature 'exists', executor must call update_extrude, not skip silently."""
    calls = []
    class MockClient:
        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def update_extrude(self, feature_name, distance):
            calls.append(("update_extrude", feature_name, distance))
            return _MockResponse(200)

        def send_command(self, cmd, data=None):
            return _MockResponse(200, {})

        def add_sketch(self, plane, sketch_name=None):
            return _MockResponse(200, {"sketch_name": "s1__hull__sk"})

        def add_point(self, sn, pt):
            return _MockResponse(200, {"profiles": {"p1": {}}})

        def close_profile(self, sn):
            return _MockResponse(200, {"profiles": {"p1": {}}})

        def add_extrude(self, sn, pid, distance, operation, feature_name=None):
            return _MockResponse(200)

    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {"id": "hull", "primitive": "rect_extrude", "names": {"sketch": "s1__hull__sk", "feature": "s1__hull__feat"}, "plane": "XY", "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 10, "h": 8}, "distance": 6.0, "operation": "NewBodyFeatureOperation"},
        ],
    }
    client = MockClient()
    executor = CadExecutor(client, dry_run=False, edit_mode=True)
    result = executor.execute_plan(plan)
    assert result.success, result.message
    assert len(calls) == 1
    assert calls[0][0] == "update_extrude"
    assert calls[0][1] == "s1__hull__feat"
    assert calls[0][2] == 6.0


def test_edit_mode_registry_after_update():
    """Registry must be updated after successful update_extrude."""
    class MockClient:
        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def update_extrude(self, feature_name, distance):
            return _MockResponse(200)

        def send_command(self, cmd, data=None):
            return _MockResponse(200, {})

    plan = {"session": "s1", "mode": "edit", "steps": [{"id": "hull", "primitive": "rect_extrude", "names": {"feature": "s1__hull__feat"}, "plane": "XY", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 3, "operation": "NewBodyFeatureOperation"}]}
    executor = CadExecutor(MockClient(), dry_run=False, edit_mode=True)
    result = executor.execute_plan(plan)
    assert result.success
    assert "hull" in result.registry
    assert result.registry["hull"].get("feature_name") == "s1__hull__feat"
    assert result.registry["hull"].get("updated") is True


def test_create_path_supported_primitive_and_registry():
    """Create path works and registry updates after add_extrude."""
    class MockClient:
        def add_sketch(self, plane, sketch_name=None):
            return _MockResponse(200, {"sketch_name": sketch_name or "s1__box__sk"})

        def add_point(self, sketch_name, pt):
            return _MockResponse(200, {"profiles": {"p1": {}}})

        def close_profile(self, sketch_name):
            return _MockResponse(200, {"profiles": {"p1": {}}})

        def add_extrude(self, sketch_name, profile_id, distance, operation, feature_name=None):
            return _MockResponse(200, {})

    plan = {
        "session": "s1",
        "mode": "create",
        "steps": [
            {
                "id": "box",
                "primitive": "rect_extrude",
                "names": {"sketch": "s1__box__sk", "feature": "s1__box__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 10, "h": 8},
                "distance": 2.0,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = CadExecutor(MockClient(), dry_run=False).execute_plan(plan)
    assert result.success, result.message
    assert "box" in result.registry
    assert result.registry["box"]["feature_name"] == "s1__box__feat"
    assert result.registry["box"]["updated"] is False


def test_no_silent_skip_on_existing_feature_when_edit_expected():
    """Existing feature must be updated, not silently skipped or recreated."""
    update_calls = []

    class MockClient:
        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def update_extrude(self, feature_name, distance):
            update_calls.append((feature_name, distance))
            return _MockResponse(200, {})

        def add_sketch(self, plane, sketch_name=None):
            raise AssertionError("add_sketch should not run for existing feature update")

        def add_point(self, sketch_name, pt):
            raise AssertionError("add_point should not run for existing feature update")

        def close_profile(self, sketch_name):
            raise AssertionError("close_profile should not run for existing feature update")

        def add_extrude(self, sketch_name, profile_id, distance, operation, feature_name=None):
            raise AssertionError("add_extrude should not run for existing feature update")

    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 11.0,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = CadExecutor(MockClient(), dry_run=False, edit_mode=True).execute_plan(plan)
    assert result.success, result.message
    assert update_calls == [("s1__hull__feat", 11.0)]
    assert any(ev.action == "updated" for ev in result.trace)


def test_unsupported_edit_fails_when_previous_plan_given():
    """When previous_plan is provided and step changed unsupported field, execute_plan fails."""
    class MockClient:
        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def send_command(self, cmd, data=None):
            return _MockResponse(200, {})

    previous = {"session": "s1", "mode": "edit", "steps": [{"id": "hull", "primitive": "rect_extrude", "names": {"feature": "s1__hull__feat"}, "plane": "XY", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 5, "operation": "NewBodyFeatureOperation"}]}
    plan = {"session": "s1", "mode": "edit", "steps": [{"id": "hull", "primitive": "rect_extrude", "names": {"feature": "s1__hull__feat"}, "plane": "XY", "profile": {"type": "rect", "w": 20, "h": 8}, "distance": 5, "operation": "NewBodyFeatureOperation"}]}
    executor = CadExecutor(MockClient(), dry_run=False, edit_mode=True)
    result = executor.execute_plan(plan, previous_plan=previous)
    assert not result.success
    assert (
        "recreate-required" in result.message.lower()
        or "profile" in result.message.lower()
        or "controlled recreate blocked" in result.message.lower()
    )


def test_run_cad_operator_edit_distance_update_with_previous_plan():
    calls = []

    class MockClient:
        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def update_extrude(self, feature_name, distance):
            calls.append((feature_name, distance))
            return _MockResponse(200, {})

    previous = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 9,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = run_cad_operator(
        MockClient(),
        plan,
        edit_mode=True,
        previous_plan=previous,
        dry_run=False,
        step_delay=0.0,
    )
    assert result.success, result.message
    assert calls == [("s1__hull__feat", 9.0)]
    assert any(ev.action == "updated" for ev in result.trace)


def test_run_cad_operator_edit_recreate_required_with_previous_plan():
    class MockClient:
        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def update_extrude(self, feature_name, distance):
            return _MockResponse(200, {})

    previous = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XZ",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = run_cad_operator(
        MockClient(),
        plan,
        edit_mode=True,
        previous_plan=previous,
        dry_run=False,
        step_delay=0.0,
    )
    assert not result.success
    assert (
        "recreate-required" in result.message.lower()
        or "plane" in result.message.lower()
        or "controlled recreate blocked" in result.message.lower()
    )
    assert any(ev.action == "recreate-required" for ev in result.trace)


def test_controlled_recreate_executes_suffix_when_feature_absent():
    class MockClient:
        def clear(self):
            raise AssertionError("clear() must never be called in controlled recreate")

        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": False, "count": 0})

        def add_sketch(self, plane, sketch_name=None):
            return _MockResponse(200, {"sketch_name": sketch_name or "s1__hull__sk"})

        def add_point(self, sketch_name, pt):
            return _MockResponse(200, {"profiles": {"p1": {}}})

        def close_profile(self, sketch_name):
            return _MockResponse(200, {"profiles": {"p1": {}}})

        def add_extrude(self, sketch_name, profile_id, distance, operation, feature_name=None):
            return _MockResponse(200, {})

    previous = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"sketch": "s1__hull__sk", "feature": "s1__hull__feat"},
                "plane": "XZ",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = CadExecutor(MockClient(), dry_run=False, edit_mode=True).execute_plan(plan, previous_plan=previous)
    assert result.success, result.message
    assert any(ev.action == "recreate-required" for ev in result.trace)
    assert any(ev.action == "created" for ev in result.trace)
    assert "hull" in result.registry
    assert result.registry["hull"]["updated"] is False


def test_controlled_recreate_blocked_when_existing_suffix_features_present():
    class MockClient:
        def clear(self):
            raise AssertionError("clear() must never be called in controlled recreate")

        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

    previous = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XZ",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = CadExecutor(MockClient(), dry_run=False, edit_mode=True).execute_plan(plan, previous_plan=previous)
    assert not result.success
    assert "controlled recreate blocked" in result.message.lower()
    assert any(ev.action == "recreate-required" for ev in result.trace)
    assert any(ev.action == "failed" for ev in result.trace)


def test_run_cad_operator_post_fix_preserves_previous_plan():
    previous = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 5,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 8},
                "distance": 6,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }

    calls = []

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def execute_plan(self, _plan, previous_plan=None):
            calls.append(previous_plan)
            return ExecutionResult(success=True, steps_ok=1, steps_total=1, message="ok", trace=[])

    old_exec = cad_operator_module.CadExecutor
    old_inspector = cad_operator_module.run_inspector
    old_fixer = cad_operator_module.run_fixer
    try:
        cad_operator_module.CadExecutor = FakeExecutor
        cad_operator_module.run_inspector = lambda *_args, **_kwargs: {"score": 0.0, "issues": ["adjust"]}
        cad_operator_module.run_fixer = lambda *_args, **_kwargs: {
            "patches": [{"op": "replace", "path": "steps[hull].distance", "value": 7.0}],
            "intent": "minimal_change",
        }
        run_cad_operator(
            object(),
            plan,
            edit_mode=True,
            previous_plan=previous,
            run_post_render_fix=True,
            max_post_fix_iterations=1,
            take_screenshot=lambda: "img",
            call_llm_with_image=lambda *_args, **_kwargs: "{}",
            call_llm=lambda *_args, **_kwargs: "{}",
            step_delay=0.0,
        )
    finally:
        cad_operator_module.CadExecutor = old_exec
        cad_operator_module.run_inspector = old_inspector
        cad_operator_module.run_fixer = old_fixer

    assert len(calls) == 2
    assert calls[0] is previous
    assert calls[1] is previous


def test_classify_edit_change_semantics():
    before = {
        "id": "h",
        "primitive": "rect_extrude",
        "plane": "XY",
        "operation": "NewBodyFeatureOperation",
        "profile": {"type": "rect", "w": 10, "h": 8},
        "distance": 3,
    }
    after_distance = dict(before)
    after_distance["distance"] = 9
    decision = classify_edit_change(before, after_distance)
    assert decision.classification == EDIT_IN_PLACE_EDITABLE

    after_plane = dict(before)
    after_plane["plane"] = "XZ"
    decision = classify_edit_change(before, after_plane)
    assert decision.classification == EDIT_RECREATE_REQUIRED

    after_unsupported = dict(before)
    after_unsupported["profile"] = {"type": "spline"}
    decision = classify_edit_change(before, after_unsupported)
    assert decision.classification == EDIT_UNSUPPORTED


def test_executor_unknown_primitive_does_not_succeed():
    class MockClient:
        pass

    plan = {
        "session": "s1",
        "mode": "create",
        "steps": [
            {"id": "mystery", "primitive": "unknown_primitive_xyz", "distance": 1},
        ],
    }
    result = CadExecutor(MockClient(), dry_run=False).execute_plan(plan)
    assert not result.success
    assert "non-compilable" in result.message.lower() or "unsupported" in result.message.lower()
    assert any(ev.action == "failed" for ev in result.trace)


def test_unsupported_primitive_loft_fails_in_normal_run():
    """Loft/sweep/fillet must not report success in normal execution."""
    from cad.cad_compiler import compile_plan
    plan = {"session": "s1", "steps": [{"id": "wing", "primitive": "loft", "loft": {"profiles": [{"sketch": "a", "profile_id": "p1"}, {"sketch": "b", "profile_id": "p2"}]}}]}
    compiled = compile_plan(plan)
    assert len(compiled) == 1
    assert compiled[0].primitive == "loft"
    client = type("C", (), {"send_command": lambda s, d: None})()
    executor = CadExecutor(client, dry_run=False)
    result = executor.execute_plan(plan)
    assert not result.success
    assert "loft" in result.message.lower() or "unsupported" in result.message.lower()


def test_dry_run_unsupported_primitive_is_stubbed_not_built():
    plan = {"session": "s1", "steps": [{"id": "wing", "primitive": "loft", "loft": {"profiles": [{"sketch": "a", "profile_id": "p1"}, {"sketch": "b", "profile_id": "p2"}]}}]}
    client = type("C", (), {"send_command": lambda self, cmd, data=None: _MockResponse(200, {})})()
    result = CadExecutor(client, dry_run=True).execute_plan(plan)
    assert result.success
    assert result.steps_ok == 1
    assert not result.registry
    assert any(ev.action == "unsupported" for ev in result.trace)


def test_validate_rejects_arcs():
    """Validator must reject poly profile with arcs."""
    plan = {
        "session": "s1",
        "steps": [
            {"id": "curved", "primitive": "poly_extrude", "plane": "XY", "profile": {"type": "poly", "pts": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}, {"x": 0, "y": 2}], "arcs": [{"i_start": 0, "i_end": 1, "angle_deg": 90}]}, "distance": 1, "operation": "NewBodyFeatureOperation"},
        ],
    }
    vr = validate_plan(plan)
    assert not vr.valid
    assert any("arc" in e.lower() for e in vr.errors)


def test_capability_aware_validation_rejects_unavailable_feature():
    caps = default_capability_model()
    caps.capabilities["extrude_new_body"] = Capability(STATUS_UNSUPPORTED, "disabled for test")
    plan = {
        "session": "s1",
        "steps": [
            {
                "id": "box",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 2, "h": 2},
                "distance": 1.0,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    vr = validate_plan(plan, capabilities=caps)
    assert not vr.valid
    assert any("capability" in e.lower() for e in vr.errors)


def test_add_sketch_existing_name_contract_authoritative():
    """When sketch name already exists, backend accepts existing sketch response (plane ignored by server)."""
    class MockClient:
        def add_sketch(self, sketch_plane, sketch_name=None):
            return _MockResponse(200, {"sketch_name": "existing_sketch"})

    backend = FusionCadBackend(MockClient())
    result = backend.create_sketch("XZ", sketch_name="existing_sketch")
    assert result.ok
    assert result.response.json()["data"]["sketch_name"] == "existing_sketch"


def test_backend_list_tools_and_model_state_summary():
    class MockClient:
        def list_tools(self):
            return _MockResponse(200, {"tools": [{"name": "add_sketch"}]})

        def list_features(self):
            return _MockResponse(
                200,
                {
                    "sketches": [{"name": "s1", "type": "Sketch"}],
                    "extrude_features": [{"name": "f1", "type": "ExtrudeFeature"}],
                    "counts": {"sketches": 1, "extrude_features": 1},
                },
            )

        def query_bounding_box(self):
            return _MockResponse(200, {"bounding_box": {"min": {"x": 0}, "max": {"x": 1}}})

    backend = FusionCadBackend(MockClient())
    tools_result = backend.list_tools()
    assert tools_result.ok
    assert isinstance(tools_result.response.json().get("data", {}).get("tools", []), list)

    state_result = backend.get_model_state()
    assert state_result.ok
    state = state_result.response
    assert "features" in state and "bounding_box" in state
    assert state["features"]["counts"]["sketches"] == 1
    assert "bounding_box" in state["bounding_box"]


def test_backend_model_state_capability_rejection():
    class MockClient:
        def list_features(self):
            return _MockResponse(200, {"counts": {"sketches": 0, "extrude_features": 0}})

        def query_bounding_box(self):
            return _MockResponse(200, {"bounding_box": {}})

    caps = default_capability_model()
    caps.capabilities["model_state_summary"] = Capability(STATUS_UNSUPPORTED, "disabled in test")
    backend = FusionCadBackend(MockClient(), capabilities=caps)
    state_result = backend.get_model_state()
    assert not state_result.ok
    assert "model_state_summary" in state_result.reason


def test_duplicate_name_propagates_to_executor():
    """When find_entity_by_name returns error (e.g. duplicate), executor fails with that message."""
    class DupResponse:
        status_code = 500
        def json(self):
            return {"message": "Duplicate entity name: 'x' found 2 times."}

    class MockClient:
        def find_entity_by_name(self, etype, name):
            return DupResponse()

    plan = {"session": "s1", "mode": "edit", "steps": [{"id": "hull", "primitive": "rect_extrude", "names": {"feature": "x"}, "plane": "XY", "profile": {"type": "rect", "w": 10, "h": 8}, "distance": 5, "operation": "NewBodyFeatureOperation"}]}
    executor = CadExecutor(MockClient(), dry_run=False, edit_mode=True)
    result = executor.execute_plan(plan)
    assert not result.success
    assert "duplicate" in result.message.lower() or "find" in result.message.lower()


def test_clear_not_used_in_edit_mode():
    class MockClient:
        def clear(self):
            raise AssertionError("clear() must never be called by CAD executor in edit mode")

        def find_entity_by_name(self, etype, name):
            return _MockResponse(200, {"found": True, "count": 1})

        def update_extrude(self, feature_name, distance):
            return _MockResponse(200, {})

    plan = {
        "session": "s1",
        "mode": "edit",
        "steps": [
            {
                "id": "hull",
                "primitive": "rect_extrude",
                "names": {"feature": "s1__hull__feat"},
                "plane": "XY",
                "profile": {"type": "rect", "w": 5, "h": 5},
                "distance": 4,
                "operation": "NewBodyFeatureOperation",
            },
        ],
    }
    result = CadExecutor(MockClient(), dry_run=False, edit_mode=True).execute_plan(plan)
    assert result.success


def test_docs_contract_mentions_actual_semantics():
    readme_path = os.path.join(FUSION_DIR, "cad", "README_CAD_DSL.md")
    with open(readme_path, "r", encoding="utf-8") as fh:
        text = fh.read().lower()
    assert "no fake success" in text
    assert "add_sketch contract" in text
    assert "arcs" in text and "not supported" in text
    assert "legacy" in text and "bbox" in text


if __name__ == "__main__":
    test_make_step_names()
    test_parse_path()
    test_patch_replace_distance()
    test_patch_add_step()
    test_patch_remove_step()
    test_poly_area()
    test_validate_plan_ok()
    test_validate_plan_budget()
    test_validate_plan_rect_min_size()
    test_validate_profile_type_mismatch_is_rejected()
    test_validate_rejects_arcs()
    test_compile_plan_tank()
    test_resolve_data()
    test_get_profile_id_from_response()
    test_executor_dry_run()
    test_get_distance_and_feature_name_from_compiled()
    test_detect_unsupported_edit_distance_only()
    test_detect_unsupported_edit_profile_change()
    test_detect_unsupported_edit_plane_change()
    test_edit_mode_calls_update_extrude()
    test_edit_mode_registry_after_update()
    test_create_path_supported_primitive_and_registry()
    test_no_silent_skip_on_existing_feature_when_edit_expected()
    test_unsupported_edit_fails_when_previous_plan_given()
    test_run_cad_operator_edit_distance_update_with_previous_plan()
    test_run_cad_operator_edit_recreate_required_with_previous_plan()
    test_controlled_recreate_executes_suffix_when_feature_absent()
    test_controlled_recreate_blocked_when_existing_suffix_features_present()
    test_run_cad_operator_post_fix_preserves_previous_plan()
    test_classify_edit_change_semantics()
    test_executor_unknown_primitive_does_not_succeed()
    test_unsupported_primitive_loft_fails_in_normal_run()
    test_dry_run_unsupported_primitive_is_stubbed_not_built()
    test_capability_aware_validation_rejects_unavailable_feature()
    test_add_sketch_existing_name_contract_authoritative()
    test_backend_list_tools_and_model_state_summary()
    test_backend_model_state_capability_rejection()
    test_duplicate_name_propagates_to_executor()
    test_clear_not_used_in_edit_mode()
    test_docs_contract_mentions_actual_semantics()
    print("All tests passed.")
