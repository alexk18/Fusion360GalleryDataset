"""
Orchestration-level tests for ai_assistant DSL-first routing.
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
EXAMPLES_DIR = os.path.join(FUSION_DIR, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

import ai_assistant as ai_module
from cad.cad_structural_planner import plan_structural_specs
from cad.cad_structural_ranker import rank_dsl_candidates
from cad.cad_executor import ExecutionResult


class _OrchestrationHarness(ai_module.FusionAIAssistant):
    def __init__(self):
        self.fusion = object()
        self.provider = "test"
        self.llm = object()
        self.model = "test-model"
        self.review_enabled = False
        self.last_parts = []
        self.last_request = ""
        self.last_images = []
        self.last_cad_plan = None
        self.last_create_policy = None
        self.last_structural_spec = None
        self._text_modes = []
        self._image_modes = []
        self._exec_modes = []
        self._exec_prev = []
        self._legacy_called = False

    def _make_plan(self, mode):
        return {
            "units": "cm",
            "session": "test_session",
            "mode": mode,
            "steps": [
                {
                    "id": "box",
                    "primitive": "rect_extrude",
                    "plane": "XY",
                    "profile": {"type": "rect", "w": 2, "h": 2},
                    "distance": 1.0,
                    "operation": "NewBodyFeatureOperation",
                }
            ],
        }

    def check_connection(self):
        return True

    def ensure_connection(self):
        return True

    def _plan_text_to_cad_candidates(self, user_request, mode="create"):
        self._text_modes.append(mode)
        return [self._make_plan(mode)]

    def _plan_images_to_cad_candidates(self, user_request, images, mode="create"):
        self._image_modes.append(mode)
        return [self._make_plan(mode)]

    def _execute_cad_dsl_best(self, candidates, mode="create", previous_plan=None, user_request=""):
        self._exec_modes.append(mode)
        self._exec_prev.append(previous_plan)
        return True, self._make_plan(mode), None

    def _load_images_from_paths(self, paths):
        return [
            {
                "data": "ZmFrZQ==",
                "media_type": "image/png",
                "name": os.path.basename(str(paths[0])),
                "label": "front",
            }
        ]

    def _gather_image_paths(self, folder, max_images=8):
        return [Path(folder) / "view.png"]

    def decompose_from_images(self, images):
        self._legacy_called = True
        return {"parts": []}


class _NoImagePlanHarness(_OrchestrationHarness):
    def _plan_images_to_cad_candidates(self, user_request, images, mode="create"):
        self._image_modes.append(mode)
        return []


class _CreatePlannerHarness(_OrchestrationHarness):
    def __init__(self, payload):
        super().__init__()
        self._payload = payload

    def _call_llm(self, _system_prompt, _user_content):
        return json.dumps(self._payload)

    def _call_llm_with_images(self, _system_prompt, _text, _images):
        return json.dumps(self._payload)

    def _plan_text_to_cad_candidates(self, user_request, mode="create"):
        return ai_module.FusionAIAssistant._plan_text_to_cad_candidates(self, user_request, mode=mode)

    def _plan_images_to_cad_candidates(self, user_request, images, mode="create"):
        return ai_module.FusionAIAssistant._plan_images_to_cad_candidates(self, user_request, images, mode=mode)


class _StructuralCreateHarness(_CreatePlannerHarness):
    def __init__(self, payload):
        super().__init__(payload)
        self._structural_calls = []

    def _resolve_create_structural_context(self, user_request, images=None):
        self._structural_calls.append(
            {
                "request": str(user_request or ""),
                "images_count": len(images or []),
            }
        )
        return ai_module.FusionAIAssistant._resolve_create_structural_context(
            self,
            user_request,
            images=images,
        )


class _IterativeExecuteHarness(_StructuralCreateHarness):
    def __init__(self, payload):
        super().__init__(payload)
        self.fusion = object()

    def _execute_cad_dsl_best(self, candidates, mode="create", previous_plan=None, user_request=""):
        return ai_module.FusionAIAssistant._execute_cad_dsl_best(
            self,
            candidates,
            mode=mode,
            previous_plan=previous_plan,
            user_request=user_request,
        )


def _run_with_inputs(assistant, commands):
    stream = iter(commands)
    with patch("builtins.input", side_effect=lambda _prompt="": next(stream)):
        assistant.run()


def test_text_path_create_then_edit_uses_previous_plan():
    assistant = _OrchestrationHarness()
    with patch.object(ai_module, "USE_CAD_DSL_PLANNER", True), patch.object(
        ai_module, "USE_LEGACY_BBOX_FALLBACK", False
    ):
        _run_with_inputs(assistant, ["Build a chair", "add armrests", "exit"])

    assert assistant._text_modes == ["create", "edit"]
    assert assistant._exec_modes == ["create", "edit"]
    assert assistant._exec_prev[0] is None
    assert isinstance(assistant._exec_prev[1], dict)


def test_image_command_uses_dsl_path_only():
    assistant = _OrchestrationHarness()
    with tempfile.TemporaryDirectory() as td:
        image_path = os.path.join(td, "photo.png")
        with open(image_path, "wb") as fh:
            fh.write(b"fake")
        with patch.object(ai_module, "USE_CAD_DSL_PLANNER", True), patch.object(
            ai_module, "USE_LEGACY_BBOX_FALLBACK", False
        ):
            _run_with_inputs(assistant, [f"build from image {image_path}", "exit"])

    assert assistant._image_modes == ["create"]
    assert assistant._exec_modes == ["create"]
    assert assistant._legacy_called is False


def test_folder_command_uses_dsl_path_only():
    assistant = _OrchestrationHarness()
    with tempfile.TemporaryDirectory() as td:
        with patch.object(ai_module, "USE_CAD_DSL_PLANNER", True), patch.object(
            ai_module, "USE_LEGACY_BBOX_FALLBACK", False
        ):
            _run_with_inputs(assistant, [f"build from folder {td}", "exit"])

    assert assistant._image_modes == ["create"]
    assert assistant._exec_modes == ["create"]
    assert assistant._legacy_called is False


def test_image_no_candidate_does_not_fallback_when_disabled():
    assistant = _NoImagePlanHarness()
    with tempfile.TemporaryDirectory() as td:
        image_path = os.path.join(td, "photo.png")
        with open(image_path, "wb") as fh:
            fh.write(b"fake")
        with patch.object(ai_module, "USE_CAD_DSL_PLANNER", True), patch.object(
            ai_module, "USE_LEGACY_BBOX_FALLBACK", False
        ):
            _run_with_inputs(assistant, [f"build from image {image_path}", "exit"])

    assert assistant._image_modes == ["create"]
    assert assistant._exec_modes == []
    assert assistant._legacy_called is False


def test_sanitize_plan_does_not_silently_downgrade_advanced_primitive():
    assistant = _OrchestrationHarness()
    raw_plan = {
        "units": "cm",
        "session": "s_adv",
        "mode": "create",
        "steps": [
            {
                "id": "edge_round",
                "primitive": "fillet",
                "selectors": {"rule": "outer_edges"},
                "fillet": {"radius": 1.2},
            }
        ],
    }
    plan = assistant._sanitize_cad_plan(raw_plan, "add fillet", mode="create")
    assert isinstance(plan, dict)
    assert plan["steps"][0]["primitive"] == "fillet"
    assert plan["steps"][0].get("fillet", {}).get("radius") == 1.2


def test_create_policy_blocks_smooth_freeform_family():
    payload = {
        "units": "cm",
        "session": "s1",
        "mode": "create",
        "steps": [
            {
                "id": "x",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 2, "h": 2},
                "distance": 1,
                "operation": "NewBodyFeatureOperation",
            }
        ],
    }
    assistant = _CreatePlannerHarness(payload)
    with patch.object(ai_module, "DSL_CANDIDATE_COUNT", 1):
        candidates = assistant._plan_text_to_cad_candidates("Build an organic human face sculpture", mode="create")
    assert candidates == []
    assert assistant.last_create_policy["family"] == "unsupported_smooth_freeform"
    assert assistant.last_create_policy["outcome"] == "blocked"


def test_create_policy_marks_rotational_as_lowpoly_approx():
    payload = {
        "units": "cm",
        "session": "s_rot",
        "mode": "create",
        "steps": [
            {
                "id": "body",
                "primitive": "circle_extrude",
                "plane": "XY",
                "profile": {"type": "circle", "cx": 0, "cy": 0, "radius": 3},
                "distance": 2.0,
                "operation": "NewBodyFeatureOperation",
            }
        ],
    }
    assistant = _CreatePlannerHarness(payload)
    with patch.object(ai_module, "DSL_CANDIDATE_COUNT", 1):
        candidates = assistant._plan_text_to_cad_candidates("Build a bottle", mode="create")
    assert len(candidates) >= 1
    global_cfg = candidates[0].get("global", {})
    assert global_cfg.get("shape_family") == "rotational_bodies"
    assert global_cfg.get("build_outcome_target") == "lowpoly_approx"
    assert global_cfg.get("approximation_policy", {}).get("strategy") == "stepped_radial_approx"


def test_create_policy_filters_unsupported_primitives_before_execute():
    assistant = _OrchestrationHarness()
    policy = {
        "family": "lowpoly_hard_surface_vehicle",
        "outcome": "exact",
        "allowed_primitives": ["rect_extrude", "circle_extrude", "poly_extrude", "wedge_extrude", "cut_extrude"],
        "approximation_strategy": "hard_surface_lowpoly",
        "reason": "",
    }
    bad_candidate = {
        "units": "cm",
        "session": "s_bad",
        "mode": "create",
        "steps": [
            {"id": "edge_round", "primitive": "fillet", "fillet": {"radius": 0.5}, "selectors": {"rule": "outer_edges"}}
        ],
    }
    prepared = assistant._prepare_create_candidates([bad_candidate], policy, structural_spec=None)
    assert prepared == []


def test_structural_spec_tank_family_and_roles():
    specs = plan_structural_specs("Build me a lowpoly tank", images=[], capabilities=ai_module.default_capability_model())
    assert specs
    tank = specs[0]
    assert tank.get("object_family") == "lowpoly_hard_surface_vehicle"
    assert tank.get("build_outcome_target") == "exact"
    roles = [
        p.get("role")
        for bucket in ("main_masses", "supporting_parts", "secondary_parts")
        for p in (tank.get(bucket) or [])
        if isinstance(p, dict)
    ]
    assert "lower_hull" in roles
    assert "upper_hull" in roles
    assert "turret" in roles
    assert "gun" in roles


def test_structural_spec_bottle_lowpoly_when_revolve_unavailable():
    specs = plan_structural_specs("Build a bottle", images=[], capabilities=ai_module.default_capability_model())
    assert specs
    bottle = specs[0]
    assert bottle.get("object_family") == "rotational_bodies"
    assert bottle.get("build_outcome_target") == "lowpoly_approx"
    assert bottle.get("approximation_policy", {}).get("mode") == "stepped_radial_approx"


def test_structural_spec_human_face_is_blocked():
    specs = plan_structural_specs("Build a human face sculpture", images=[], capabilities=ai_module.default_capability_model())
    assert specs
    blocked = specs[0]
    assert blocked.get("object_family") == "unsupported_smooth_freeform"
    assert blocked.get("build_outcome_target") == "blocked"
    assert blocked.get("blocked_reason")


def test_structural_ranking_prefers_complete_candidate_over_box_junk():
    spec = plan_structural_specs("Build me a lowpoly tank", images=[], capabilities=ai_module.default_capability_model())[0]
    complete = {
        "session": "s1",
        "mode": "create",
        "global": {"shape_family": "lowpoly_hard_surface_vehicle"},
        "steps": [
            {"id": "lower_hull", "primitive": "wedge_extrude", "operation": "NewBodyFeatureOperation"},
            {"id": "upper_hull", "primitive": "poly_extrude", "operation": "JoinFeatureOperation"},
            {"id": "turret", "primitive": "poly_extrude", "operation": "JoinFeatureOperation"},
            {"id": "gun", "primitive": "circle_extrude", "operation": "JoinFeatureOperation"},
            {"id": "left_track_module", "primitive": "poly_extrude", "operation": "NewBodyFeatureOperation"},
            {"id": "right_track_module", "primitive": "poly_extrude", "operation": "NewBodyFeatureOperation"},
        ],
    }
    junk = {
        "session": "s2",
        "mode": "create",
        "global": {"shape_family": "lowpoly_hard_surface_vehicle"},
        "steps": [
            {"id": "box_1", "primitive": "rect_extrude", "operation": "NewBodyFeatureOperation"},
            {"id": "box_2", "primitive": "rect_extrude", "operation": "NewBodyFeatureOperation"},
        ],
    }
    ranked = rank_dsl_candidates([junk, complete], spec, capabilities=ai_module.default_capability_model())
    assert len(ranked) == 2
    assert ranked[0]["plan"]["session"] == "s1"


def test_text_image_folder_create_planning_uses_structural_stage():
    payload = {
        "units": "cm",
        "session": "s_struct",
        "mode": "create",
        "steps": [
            {
                "id": "lower_hull",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 10, "h": 6, "cx": 0, "cy": 0},
                "distance": 2,
                "operation": "NewBodyFeatureOperation",
            }
        ],
    }
    assistant = _StructuralCreateHarness(payload)
    with patch.object(ai_module, "DSL_CANDIDATE_COUNT", 1):
        text_candidates = assistant._plan_text_to_cad_candidates("Build a chair", mode="create")
        image_candidates = assistant._plan_images_to_cad_candidates(
            "Build from image",
            images=[{"data": "ZmFrZQ==", "media_type": "image/png", "label": "front"}],
            mode="create",
        )
        folder_candidates = assistant._plan_images_to_cad_candidates(
            "Build from folder",
            images=[
                {"data": "ZmFrZQ==", "media_type": "image/png", "label": "front"},
                {"data": "ZmFrZQ==", "media_type": "image/png", "label": "left"},
                {"data": "ZmFrZQ==", "media_type": "image/png", "label": "back"},
            ],
            mode="create",
        )
    assert text_candidates
    assert image_candidates
    assert folder_candidates
    valid_sources = {"structural_spec_family_grammar", "family_grammar_synthesizer"}
    assert any((c.get("global") or {}).get("source") in valid_sources for c in text_candidates)
    assert any((c.get("global") or {}).get("source") in valid_sources for c in image_candidates)
    assert any((c.get("global") or {}).get("source") in valid_sources for c in folder_candidates)
    assert any((c.get("global") or {}).get("synthesis_grammar") for c in text_candidates)
    assert ((text_candidates[0].get("global") or {}).get("create_safety") or {}).get("mode") == "primary_first"
    assert ((image_candidates[0].get("global") or {}).get("create_safety") or {}).get("mode") == "primary_first"
    assert ((folder_candidates[0].get("global") or {}).get("create_safety") or {}).get("mode") == "primary_first"
    assert all(str((s or {}).get("primitive") or "") != "cut_extrude" for s in (text_candidates[0].get("steps") or []))
    assert len(assistant._structural_calls) == 3
    assert assistant._structural_calls[0]["images_count"] == 0
    assert assistant._structural_calls[1]["images_count"] == 1
    assert assistant._structural_calls[2]["images_count"] == 3


def test_no_silent_downgrade_smooth_freeform_in_create():
    payload = {
        "units": "cm",
        "session": "blocked_payload",
        "mode": "create",
        "steps": [
            {
                "id": "base",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 5, "h": 5},
                "distance": 1,
                "operation": "NewBodyFeatureOperation",
            }
        ],
    }
    assistant = _StructuralCreateHarness(payload)
    with patch.object(ai_module, "DSL_CANDIDATE_COUNT", 1):
        candidates = assistant._plan_text_to_cad_candidates("Build a smooth human face sculpture", mode="create")
    assert candidates == []
    assert assistant.last_create_policy["outcome"] == "blocked"
    assert "unsupported_smooth_freeform" in assistant.last_create_policy["reason"]


def test_create_execution_uses_iterative_agent_loop():
    payload = {
        "units": "cm",
        "session": "iterative_src",
        "mode": "create",
        "steps": [
            {
                "id": "lower_hull",
                "primitive": "rect_extrude",
                "plane": "XY",
                "profile": {"type": "rect", "w": 12, "h": 8, "cx": 0, "cy": 0},
                "distance": 3,
                "operation": "NewBodyFeatureOperation",
            }
        ],
    }
    assistant = _IterativeExecuteHarness(payload)
    with patch.object(ai_module, "DSL_CANDIDATE_COUNT", 1):
        candidates = assistant._plan_text_to_cad_candidates("Build a lowpoly tank", mode="create")
    assert candidates

    calls = {"run": 0}

    class _FakeIterativeAgent:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **kwargs):
            calls["run"] += 1
            assert kwargs.get("user_request") == "Build a lowpoly tank"
            assert kwargs.get("structural_spec", {}).get("object_family") == "lowpoly_hard_surface_vehicle"
            assert isinstance(kwargs.get("candidates"), list) and kwargs["candidates"]
            out = type("Out", (), {})()
            out.success = True
            out.best_index = 0
            out.best_plan = kwargs["candidates"][0]
            out.result = ExecutionResult(success=True, steps_ok=1, steps_total=1, message="ok", trace=[])
            out.loop_trace = [
                {"iteration": 1, "action": "created", "selected_roles": ["lower_hull"], "executed_step_ids": ["lower_hull"]},
                {"iteration": 2, "action": "stop_required_complete"},
            ]
            out.state_snapshots = [{"executed_step_ids": []}, {"executed_step_ids": ["lower_hull"]}]
            return out

    with patch.object(ai_module, "IterativeCreateAgent", _FakeIterativeAgent):
        ok, best_plan, result = assistant._execute_cad_dsl_best(
            candidates,
            mode="create",
            user_request="Build a lowpoly tank",
        )
    assert ok is True
    assert isinstance(best_plan, dict)
    assert result.success is True
    assert calls["run"] == 1
    assert len(assistant.last_iterative_loop_trace) == 2
    assert len(assistant.last_iterative_state_snapshots) == 2


if __name__ == "__main__":
    test_text_path_create_then_edit_uses_previous_plan()
    test_image_command_uses_dsl_path_only()
    test_folder_command_uses_dsl_path_only()
    test_image_no_candidate_does_not_fallback_when_disabled()
    test_sanitize_plan_does_not_silently_downgrade_advanced_primitive()
    test_create_policy_blocks_smooth_freeform_family()
    test_create_policy_marks_rotational_as_lowpoly_approx()
    test_create_policy_filters_unsupported_primitives_before_execute()
    test_structural_spec_tank_family_and_roles()
    test_structural_spec_bottle_lowpoly_when_revolve_unavailable()
    test_structural_spec_human_face_is_blocked()
    test_structural_ranking_prefers_complete_candidate_over_box_junk()
    test_text_image_folder_create_planning_uses_structural_stage()
    test_no_silent_downgrade_smooth_freeform_in_create()
    test_create_execution_uses_iterative_agent_loop()
    print("All ai_assistant orchestration tests passed.")
