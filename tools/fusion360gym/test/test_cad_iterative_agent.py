"""
Tests for iterative create-mode CAD agent loop and its reasoning layers.
"""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from cad.cad_agent_loop import IterativeCreateAgent
from cad.cad_backend import FusionCadBackend
from cad.cad_dataset_priors import build_dataset_priors
from cad.cad_executor import ExecutionResult
from cad.cad_relational_assembly import build_relational_assembly_graph, evaluate_relational_graph
from cad.cad_state_introspection import inspect_runtime_state
from cad.cad_structural_planner import plan_structural_specs
from cad.cad_dsl_synthesizer import synthesize_dsl_candidates_from_structural_spec
from cad.cad_capabilities import default_capability_model


class _MockResponse:
    def __init__(self, data):
        self.status_code = 200
        self._data = data

    def json(self):
        return {"data": self._data}


class _MockClient:
    def __init__(self):
        self.clear_calls = 0
        self.refresh_calls = 0

    def clear(self):
        self.clear_calls += 1
        return _MockResponse({})

    def refresh(self):
        self.refresh_calls += 1
        return _MockResponse({})

    def list_tools(self):
        return _MockResponse({"tools": [{"name": "add_sketch"}, {"name": "add_extrude"}]})

    def list_features(self):
        return _MockResponse(
            {
                "sketches": [{"name": "sk_1", "type": "Sketch"}],
                "extrude_features": [{"name": "feat_1", "type": "ExtrudeFeature"}],
                "counts": {"sketches": 1, "extrude_features": 1},
            }
        )

    def query_bounding_box(self):
        return _MockResponse({"bounding_box": {"min": {"x": -1, "y": -1, "z": 0}, "max": {"x": 2, "y": 3, "z": 4}}})


def _make_success_execute_fn():
    def _execute(fragment_plan):
        steps = [s for s in (fragment_plan.get("steps") or []) if isinstance(s, dict)]
        reg = {}
        for step in steps:
            sid = str(step.get("id") or "")
            if sid:
                reg[sid] = {"feature_name": f"{sid}__feat", "updated": False}
        return ExecutionResult(
            success=True,
            steps_ok=len(steps),
            steps_total=len(steps),
            message="ok",
            registry=reg,
            trace=[],
        )

    return _execute


def _inspect_from_executed(executed_steps=None, planned_steps=None):
    executed = [s for s in (executed_steps or []) if isinstance(s, dict)]
    ids = [str(s.get("id") or "") for s in executed if str(s.get("id") or "")]
    # Approximate one connected component when 2+ elements exist.
    comps = [ids] if ids else []
    rels = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            rels.append({"a": a, "b": b, "relation": "touches_estimate"})
    return {
        "executed_step_ids": ids,
        "get_connected_components": comps,
        "get_step_relations": rels,
        "get_active_construction_context": {"grounded_step_ids": ids[:1], "floating_step_ids": []},
        "get_feature_bbox": {},
        "introspection_sources": {
            "get_connected_components": {"source_kind": "heuristic_estimate", "confidence": 0.6},
            "get_step_relations": {"source_kind": "heuristic_estimate", "confidence": 0.6},
            "get_active_construction_context": {"source_kind": "heuristic_estimate", "confidence": 0.6},
        },
    }


def test_iterative_agent_runs_multi_cycle_and_prioritizes_primary_massing():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    candidate = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]
    candidate["steps"].append(
        {
            "id": "service_cutouts",
            "primitive": "cut_extrude",
            "plane": "XY@2",
            "profile": {"type": "rect", "cx": 0, "cy": 10, "w": 2, "h": 2},
            "distance": 1,
            "operation": "CutFeatureOperation",
            "meta": {"risk_level": "risky_detail", "detail_level": "secondary", "essential": False},
        }
    )

    agent = IterativeCreateAgent(
        _MockClient(),
        capabilities=default_capability_model(),
        step_delay=0.0,
        inspect_fn=_inspect_from_executed,
        execute_fn=_make_success_execute_fn(),
    )
    out = agent.run(
        user_request="Build a lowpoly tank",
        structural_spec=spec,
        candidates=[candidate],
        max_iterations=32,
    )

    assert out.success
    assert out.result.success
    assert len(out.loop_trace) >= 2
    executed_ids = [str((s or {}).get("id") or "") for s in (out.best_plan.get("steps") or [])]
    assert "lower_hull" in executed_ids
    assert "upper_hull" in executed_ids
    assert "turret" in executed_ids
    assert "gun" in executed_ids
    assert "service_cutouts" not in executed_ids


def test_state_introspection_provides_runtime_query_surface():
    backend = FusionCadBackend(_MockClient(), capabilities=default_capability_model())
    step = {
        "id": "base",
        "primitive": "rect_extrude",
        "plane": "XY@0",
        "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 10, "h": 6},
        "distance": 2,
    }
    state = inspect_runtime_state(backend, executed_steps=[step], planned_steps=[step])
    assert "list_tools" in state
    assert "list_capabilities" in state
    assert "get_model_state" in state
    assert "get_features" in state
    assert "get_bodies" in state
    assert "get_parts" in state
    assert "get_sketches" in state
    assert "get_faces" in state
    assert "get_edges" in state
    assert "get_body_bbox" in state
    assert "get_feature_bbox" in state
    assert "get_connected_components" in state
    assert "get_overlaps" in state
    assert "get_active_construction_context" in state
    assert "get_body_to_body_relations" in state
    assert "get_feature_to_body_relations" in state
    assert "get_step_relations" in state
    assert "introspection_sources" in state
    assert state["executed_step_ids"] == ["base"]
    src = state["introspection_sources"]
    assert src["get_model_state"]["source_kind"] in ("exact_fusion_api", "derived_exact", "heuristic_estimate")
    assert src["get_feature_bbox"]["source_kind"] in ("exact_fusion_api", "derived_exact", "heuristic_estimate")


def test_relational_assembly_graph_has_parent_support_and_connectivity_penalty():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    plan = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]
    graph = build_relational_assembly_graph(spec, plan.get("steps") or [])
    rel_types = {str((r or {}).get("type") or "") for r in (graph.get("relations") or [])}
    assert "parent" in rel_types
    assert "support_relation" in rel_types
    assert "mirrored_with" in rel_types
    assert "must_be_above" in rel_types

    # Two disconnected components should reduce assembly score.
    state = {
        "executed_step_ids": ["lower_hull", "turret"],
        "get_connected_components": [["lower_hull"], ["turret"]],
        "get_step_relations": [],
        "get_active_construction_context": {"grounded_step_ids": ["lower_hull"], "floating_step_ids": ["turret"]},
    }
    eval_out = evaluate_relational_graph(graph, state)
    assert eval_out["disconnected_components"] == 1
    assert eval_out["floating_parts_count"] >= 1
    assert eval_out["assembly_score"] < 1.0


def test_dataset_priors_are_generalized_and_influence_ordering():
    priors = build_dataset_priors("furniture_boxy_panel", user_request="Build a desk")
    assert priors["source"] == "dataset_generalized_priors_v1"
    assert "No instance-level command replay is used." in priors["notes"]
    assert priors["ordering"][:2] == ["left_support", "right_support"]
    assert priors["strategy"] == "support_first_panel_assembly"


def test_supported_family_routing_for_create_classes():
    cap = default_capability_model()
    tank = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=cap)[0]
    chair = plan_structural_specs("Build a chair", images=[], capabilities=cap)[0]
    aircraft = plan_structural_specs("Build a lowpoly aircraft", images=[], capabilities=cap)[0]
    boat = plan_structural_specs("Build a lowpoly boat", images=[], capabilities=cap)[0]
    bottle = plan_structural_specs("Build a bottle", images=[], capabilities=cap)[0]
    blocked = plan_structural_specs("Build a smooth human face sculpture", images=[], capabilities=cap)[0]

    assert tank["object_family"] == "lowpoly_hard_surface_vehicle"
    assert chair["object_family"] == "furniture_boxy_panel"
    assert aircraft["object_family"] == "lowpoly_hard_surface_vehicle"
    assert boat["object_family"] == "lowpoly_hard_surface_vehicle"
    assert bottle["object_family"] == "rotational_bodies"
    assert bottle["build_outcome_target"] == "lowpoly_approx"
    assert blocked["object_family"] == "unsupported_smooth_freeform"
    assert blocked["build_outcome_target"] == "blocked"


def test_overlap_and_contact_summary_behavior_from_state_introspection():
    backend = FusionCadBackend(_MockClient(), capabilities=default_capability_model())
    step_a = {
        "id": "a",
        "primitive": "rect_extrude",
        "plane": "XY@0",
        "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 4, "h": 4},
        "distance": 2,
    }
    step_b = {
        "id": "b",
        "primitive": "rect_extrude",
        "plane": "XY@2",
        "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 4, "h": 4},
        "distance": 2,
    }
    state = inspect_runtime_state(backend, executed_steps=[step_a, step_b], planned_steps=[step_a, step_b])
    comps = state.get("get_connected_components") or []
    rels = state.get("get_step_relations") or []
    assert len(comps) == 1
    assert any("touch" in str((r or {}).get("relation") or "") or "intersect" in str((r or {}).get("relation") or "") for r in rels)


def test_iterative_loop_uses_runtime_state_signals_for_success_gate():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    candidate = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]

    calls = {"n": 0}

    def inspect_fn(executed_steps=None, planned_steps=None):
        calls["n"] += 1
        ids = [str(s.get("id") or "") for s in (executed_steps or []) if str(s.get("id") or "")]
        comps = [ids] if ids else []
        rels = []
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                rels.append({"a": a, "b": b, "relation": "touches_estimate"})
        return {
            "executed_step_ids": ids,
            "get_connected_components": comps,
            "get_step_relations": rels,
            "get_active_construction_context": {"grounded_step_ids": ids[:1], "floating_step_ids": []},
        }

    agent = IterativeCreateAgent(
        _MockClient(),
        capabilities=default_capability_model(),
        step_delay=0.0,
        inspect_fn=inspect_fn,
        execute_fn=_make_success_execute_fn(),
    )
    out = agent.run(
        user_request="Build a lowpoly tank",
        structural_spec=spec,
        candidates=[candidate],
        max_iterations=16,
    )
    assert calls["n"] >= 2
    assert out.result.success


if __name__ == "__main__":
    test_iterative_agent_runs_multi_cycle_and_prioritizes_primary_massing()
    test_state_introspection_provides_runtime_query_surface()
    test_relational_assembly_graph_has_parent_support_and_connectivity_penalty()
    test_dataset_priors_are_generalized_and_influence_ordering()
    test_supported_family_routing_for_create_classes()
    test_overlap_and_contact_summary_behavior_from_state_introspection()
    test_iterative_loop_uses_runtime_state_signals_for_success_gate()
    print("All iterative CAD agent tests passed.")
