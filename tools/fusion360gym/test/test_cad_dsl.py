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
from cad.cad_executor import CadExecutor, ExecutionResult, _resolve_data, _get_profile_id_from_response


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
    test_compile_plan_tank()
    test_resolve_data()
    test_get_profile_id_from_response()
    test_executor_dry_run()
    print("All tests passed.")
