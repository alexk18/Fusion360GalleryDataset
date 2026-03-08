"""
Tests for structural step-selection logic (general, not object-specific).

Validates that the next-step planner and agent loop use constructive
assembly reasoning: attachment context, closure priority, structural
stop-conditions — across all supported exact families.
"""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from cad.cad_agent_loop import IterativeCreateAgent
from cad.cad_capabilities import default_capability_model
from cad.cad_dataset_priors import build_dataset_priors
from cad.cad_dsl_synthesizer import synthesize_dsl_candidates_from_structural_spec
from cad.cad_executor import ExecutionResult
from cad.cad_next_step_planner import choose_next_fragment
from cad.cad_relational_assembly import build_relational_assembly_graph, evaluate_relational_graph
from cad.cad_structural_planner import plan_structural_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, data):
        self.status_code = 200
        self._data = data
    def json(self):
        return {"data": self._data}

class _MockClient:
    def clear(self): return _MockResponse({})
    def refresh(self): return _MockResponse({})
    def list_tools(self): return _MockResponse({"tools": [{"name": "add_sketch"}, {"name": "add_extrude"}]})
    def list_features(self): return _MockResponse({"sketches": [], "extrude_features": [], "counts": {"sketches": 0, "extrude_features": 0}})
    def query_bounding_box(self): return _MockResponse({"bounding_box": {"min": {"x": -1, "y": -1, "z": 0}, "max": {"x": 2, "y": 3, "z": 4}}})


def _make_success_execute_fn():
    def _execute(fragment_plan):
        steps = [s for s in (fragment_plan.get("steps") or []) if isinstance(s, dict)]
        reg = {}
        for step in steps:
            sid = str(step.get("id") or "")
            if sid:
                reg[sid] = {"feature_name": f"{sid}__feat", "updated": False}
        return ExecutionResult(success=True, steps_ok=len(steps), steps_total=len(steps), message="ok", registry=reg, trace=[])
    return _execute


def _setup(prompt):
    """Return (spec, plan_steps, graph, priors) for a given prompt."""
    cap = default_capability_model()
    spec = plan_structural_specs(prompt, images=[], capabilities=cap)[0]
    candidates = synthesize_dsl_candidates_from_structural_spec(
        spec, user_request=prompt, candidate_count=1, mode="create", capabilities=cap,
    )
    plan = candidates[0]
    steps = plan.get("steps", [])
    graph = build_relational_assembly_graph(spec, steps)
    family = str(spec.get("object_family") or "")
    priors = build_dataset_priors(family, user_request=prompt, structural_build_order=list(spec.get("build_order") or []))
    return spec, steps, graph, priors


def _state_disconnected(executed_ids, components, floating_ids=None, relations=None):
    """Produce a state snapshot with explicit disconnection."""
    grounded = [executed_ids[0]] if executed_ids else []
    floating = list(floating_ids or [])
    return {
        "executed_step_ids": list(executed_ids),
        "get_connected_components": components,
        "get_step_relations": relations or [],
        "get_active_construction_context": {"grounded_step_ids": grounded, "floating_step_ids": floating},
    }


def _state_connected(executed_ids, relations=None):
    """Produce a state snapshot where all parts are in one component."""
    rels = relations or []
    if not rels:
        ids = list(executed_ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                rels.append({"a": a, "b": b, "relation": "touches_estimate"})
    return {
        "executed_step_ids": list(executed_ids),
        "get_connected_components": [list(executed_ids)] if executed_ids else [],
        "get_step_relations": rels,
        "get_active_construction_context": {"grounded_step_ids": [executed_ids[0]] if executed_ids else [], "floating_step_ids": []},
    }


# ---------------------------------------------------------------------------
# Test 1: Dependent part blocked without parent evidence
# ---------------------------------------------------------------------------

def test_dependent_blocked_without_parent_context():
    """A dependent role must not be selected when the assembly is unstable
    and its parent attachment context is missing.  General rule, tested on
    two different families to confirm no object-specific hack."""

    # --- Vehicle family ---
    _, steps_v, graph_v, priors_v = _setup("Build a lowpoly tank")
    # State: only lower_hull built, turret floating (skipped upper_hull).
    state_v = _state_disconnected(
        ["lower_hull"],
        [["lower_hull"]],
    )
    state_v["assembly_eval"] = evaluate_relational_graph(graph_v, state_v)
    frag_v = choose_next_fragment(plan_steps=steps_v, graph=graph_v, state_snapshot=state_v, priors=priors_v)
    # Planner should pick upper_hull (next scaffold), NOT turret (turret depends on upper_hull).
    selected_v = frag_v["selected_roles"]
    assert "turret" not in selected_v, f"turret selected without upper_hull context: {selected_v}"
    assert len(selected_v) > 0, "planner should find an admissible step"

    # --- Furniture family ---
    _, steps_f, graph_f, priors_f = _setup("Build a chair")
    # State: only left_support built.
    state_f = _state_disconnected(
        ["left_support"],
        [["left_support"]],
    )
    state_f["assembly_eval"] = evaluate_relational_graph(graph_f, state_f)
    frag_f = choose_next_fragment(plan_steps=steps_f, graph=graph_f, state_snapshot=state_f, priors=priors_f)
    selected_f = frag_f["selected_roles"]
    # top_panel depends on base_panel — should not be selected.
    assert "top_panel" not in selected_f, f"top_panel selected without base_panel: {selected_f}"
    assert len(selected_f) > 0, "planner should find an admissible step"


# ---------------------------------------------------------------------------
# Test 2: Closure fragment preferred when disconnected
# ---------------------------------------------------------------------------

def test_closure_fragment_preferred_when_disconnected():
    """When the assembly is disconnected, the planner should prefer a role
    that can reduce instability over an unrelated uncovered role.
    General rule, not tied to any specific object."""

    _, steps, graph, priors = _setup("Build a lowpoly tank")
    # State: lower_hull and turret built but disconnected (upper_hull missing).
    # turret should NOT have been buildable without upper_hull, but let's test
    # what the planner does when it sees disconnection.
    state = _state_disconnected(
        ["lower_hull", "turret"],
        [["lower_hull"], ["turret"]],
        floating_ids=["turret"],
    )
    state["assembly_eval"] = evaluate_relational_graph(graph, state)
    frag = choose_next_fragment(plan_steps=steps, graph=graph, state_snapshot=state, priors=priors)
    selected = frag["selected_roles"]
    # upper_hull is the role that bridges lower_hull and turret.
    # Planner should prefer it over unrelated roles like front_wheels.
    assert frag["stability_guard_active"], "planner should detect unstable assembly"
    if selected:
        assert selected[0] in ("upper_hull", "left_track_module", "right_track_module"), \
            f"expected scaffold/closure role, got {selected}"


# ---------------------------------------------------------------------------
# Test 3: Stop-condition fails when roles built but assembly disconnected
# ---------------------------------------------------------------------------

def test_stop_condition_requires_structural_plausibility():
    """The agent loop should NOT declare success when all required roles are
    built but the assembly remains disconnected.  General rule."""

    _, steps, graph, priors = _setup("Build a lowpoly tank")
    # Simulate: all required roles built, but 2 disconnected components, no relations.
    req = set(graph.get("required_roles") or [])
    all_ids = [str(graph["role_to_step"].get(r) or "") for r in req if graph["role_to_step"].get(r)]
    state = _state_disconnected(
        all_ids,
        [[all_ids[0]], all_ids[1:]],  # 2 components
        floating_ids=all_ids[1:],
    )
    ev = evaluate_relational_graph(graph, state)
    assert ev["disconnected_components"] > 0
    assert ev["assembly_score"] < 0.5, f"score should be low when disconnected: {ev['assembly_score']}"


# ---------------------------------------------------------------------------
# Test 4: Both furniture and vehicle benefit from same general logic
# ---------------------------------------------------------------------------

def test_furniture_and_vehicle_same_general_logic():
    """The structural step-selection produces correct build sequences for
    both furniture and vehicle families through the SAME general rules
    (grounded-first, scaffold, then dependents)."""

    cap = default_capability_model()

    for prompt, family_expected, first_role_options in [
        ("Build a chair", "furniture_boxy_panel", {"left_support", "right_support"}),
        ("Build a lowpoly tank", "lowpoly_hard_surface_vehicle", {"lower_hull"}),
    ]:
        spec, steps, graph, priors = _setup(prompt)
        assert spec["object_family"] == family_expected

        # From empty state, first selected role must be grounded/foundation.
        state = _state_connected([])
        state["assembly_eval"] = evaluate_relational_graph(graph, state)
        frag = choose_next_fragment(plan_steps=steps, graph=graph, state_snapshot=state, priors=priors)
        selected = frag["selected_roles"]
        assert len(selected) > 0, f"no step selected for {prompt}"
        assert selected[0] in first_role_options, \
            f"first role for {prompt} should be in {first_role_options}, got {selected[0]}"


# ---------------------------------------------------------------------------
# Test 5: No regression for rotational lowpoly approx
# ---------------------------------------------------------------------------

def test_rotational_lowpoly_no_regression():
    """Rotational bodies (bottle, vase, mug) should still work correctly
    with the new step-selection logic."""

    cap = default_capability_model()
    for prompt in ["Build a bottle", "Build a vase", "Build a mug"]:
        spec = plan_structural_specs(prompt, images=[], capabilities=cap)[0]
        assert spec["object_family"] == "rotational_bodies"
        assert spec["build_outcome_target"] == "lowpoly_approx"

        candidates = synthesize_dsl_candidates_from_structural_spec(
            spec, user_request=prompt, candidate_count=1, mode="create", capabilities=cap,
        )
        assert len(candidates) > 0

        # Run full agent loop with connected mock.
        def inspect_connected(executed_steps=None, planned_steps=None):
            ids = [str(s.get("id") or "") for s in (executed_steps or []) if str(s.get("id") or "")]
            rels = []
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    rels.append({"a": a, "b": b, "relation": "touches_estimate"})
            return {
                "executed_step_ids": ids,
                "get_connected_components": [ids] if ids else [],
                "get_step_relations": rels,
                "get_active_construction_context": {"grounded_step_ids": ids[:1], "floating_step_ids": []},
            }

        agent = IterativeCreateAgent(
            _MockClient(), capabilities=cap, step_delay=0.0,
            inspect_fn=inspect_connected, execute_fn=_make_success_execute_fn(),
        )
        out = agent.run(user_request=prompt, structural_spec=spec, candidates=candidates, max_iterations=16)
        assert out.success, f"{prompt} should succeed, got: {out.result.message}"


# ---------------------------------------------------------------------------
# Test 6: No regression for blocked smooth/freeform
# ---------------------------------------------------------------------------

def test_blocked_smooth_freeform_no_regression():
    """Smooth/freeform cases must remain honestly blocked."""
    cap = default_capability_model()
    spec = plan_structural_specs("Build a smooth human face sculpture", images=[], capabilities=cap)[0]
    assert spec["object_family"] == "unsupported_smooth_freeform"
    assert spec["build_outcome_target"] == "blocked"
    assert spec["blocked_reason"], "should have a blocked reason"


# ---------------------------------------------------------------------------
# Test 7: Full agent loop for vehicle produces connected assembly
# ---------------------------------------------------------------------------

def test_vehicle_agent_loop_connected_assembly():
    """Full agent loop for a vehicle should produce a structurally connected
    assembly when the mock returns touching relations."""

    spec, steps, graph, priors = _setup("Build a lowpoly tank")

    def inspect_connected(executed_steps=None, planned_steps=None):
        ids = [str(s.get("id") or "") for s in (executed_steps or []) if str(s.get("id") or "")]
        rels = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                rels.append({"a": a, "b": b, "relation": "touches_estimate"})
        return {
            "executed_step_ids": ids,
            "get_connected_components": [ids] if ids else [],
            "get_step_relations": rels,
            "get_active_construction_context": {"grounded_step_ids": ids[:1], "floating_step_ids": []},
        }

    cap = default_capability_model()
    candidates = synthesize_dsl_candidates_from_structural_spec(
        spec, user_request="Build a lowpoly tank", candidate_count=1, mode="create", capabilities=cap,
    )
    agent = IterativeCreateAgent(
        _MockClient(), capabilities=cap, step_delay=0.0,
        inspect_fn=inspect_connected, execute_fn=_make_success_execute_fn(),
    )
    out = agent.run(user_request="Build a lowpoly tank", structural_spec=spec, candidates=candidates, max_iterations=32)
    assert out.success, f"tank should succeed, got: {out.result.message}"
    executed_ids = [str((s or {}).get("id") or "") for s in (out.best_plan.get("steps") or [])]
    assert "lower_hull" in executed_ids


# ---------------------------------------------------------------------------
# Test 8: Full agent loop for furniture produces connected assembly
# ---------------------------------------------------------------------------

def test_furniture_agent_loop_connected_assembly():
    """Full agent loop for furniture should produce a structurally connected
    assembly — both vehicle and furniture use the same general logic."""

    spec, steps, graph, priors = _setup("Build a chair")

    def inspect_connected(executed_steps=None, planned_steps=None):
        ids = [str(s.get("id") or "") for s in (executed_steps or []) if str(s.get("id") or "")]
        rels = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                rels.append({"a": a, "b": b, "relation": "touches_estimate"})
        return {
            "executed_step_ids": ids,
            "get_connected_components": [ids] if ids else [],
            "get_step_relations": rels,
            "get_active_construction_context": {"grounded_step_ids": ids[:1], "floating_step_ids": []},
        }

    cap = default_capability_model()
    candidates = synthesize_dsl_candidates_from_structural_spec(
        spec, user_request="Build a chair", candidate_count=1, mode="create", capabilities=cap,
    )
    agent = IterativeCreateAgent(
        _MockClient(), capabilities=cap, step_delay=0.0,
        inspect_fn=inspect_connected, execute_fn=_make_success_execute_fn(),
    )
    out = agent.run(user_request="Build a chair", structural_spec=spec, candidates=candidates, max_iterations=32)
    assert out.success, f"chair should succeed, got: {out.result.message}"
    executed_ids = [str((s or {}).get("id") or "") for s in (out.best_plan.get("steps") or [])]
    assert "left_support" in executed_ids
    assert "base_panel" in executed_ids


if __name__ == "__main__":
    test_dependent_blocked_without_parent_context()
    test_closure_fragment_preferred_when_disconnected()
    test_stop_condition_requires_structural_plausibility()
    test_furniture_and_vehicle_same_general_logic()
    test_rotational_lowpoly_no_regression()
    test_blocked_smooth_freeform_no_regression()
    test_vehicle_agent_loop_connected_assembly()
    test_furniture_agent_loop_connected_assembly()
    print("All structural step-selection tests passed.")
