"""
Build-quality tests for family-aware Structural Spec -> CAD DSL synthesis.
"""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from cad.cad_capabilities import default_capability_model
from cad.cad_structural_planner import plan_structural_specs
from cad.cad_dsl_synthesizer import synthesize_dsl_candidates_from_structural_spec
from cad.cad_structural_ranker import rank_dsl_candidates


def _parse_plane_offset(plane):
    s = str(plane or "").strip().upper()
    if "@" not in s:
        return s, 0.0
    base, off = s.split("@", 1)
    try:
        return base, float(off)
    except Exception:
        return base, 0.0


def _rect_cx(step):
    profile = step.get("profile") or {}
    return float(profile.get("cx", 0.0) or 0.0)


def test_vehicle_family_grammar_order_and_attachment_signals():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    plans = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=2,
        mode="create",
        capabilities=default_capability_model(),
    )
    assert plans
    plan = plans[0]
    step_ids = [s.get("id") for s in (plan.get("steps") or []) if isinstance(s, dict)]
    assert ((plan.get("global") or {}).get("create_safety") or {}).get("mode") == "primary_first"

    assert "lower_hull" in step_ids
    assert "upper_hull" in step_ids
    assert "turret" in step_ids
    assert "gun" in step_ids
    assert "left_track_module" in step_ids
    assert "right_track_module" in step_ids
    assert "service_cutouts" not in step_ids

    assert step_ids.index("lower_hull") < step_ids.index("turret")
    assert step_ids.index("turret") < step_ids.index("gun")

    left = next(s for s in plan["steps"] if s.get("id") == "left_track_module")
    right = next(s for s in plan["steps"] if s.get("id") == "right_track_module")
    left_cx = sum(p["x"] for p in left["profile"]["pts"]) / float(len(left["profile"]["pts"]))
    right_cx = sum(p["x"] for p in right["profile"]["pts"]) / float(len(right["profile"]["pts"]))
    assert left_cx < 0.0 < right_cx


def test_furniture_family_grammar_support_order_no_floating_and_symmetry():
    spec = plan_structural_specs("Build a chair", images=[], capabilities=default_capability_model())[0]
    assert spec.get("object_family") == "furniture_boxy_panel"
    plans = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a symmetric chair",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )
    assert plans
    plan = plans[0]
    step_ids = [s.get("id") for s in (plan.get("steps") or []) if isinstance(s, dict)]

    assert "left_support" in step_ids
    assert "right_support" in step_ids
    assert "base_panel" in step_ids
    assert step_ids.index("left_support") < step_ids.index("base_panel")
    assert step_ids.index("right_support") < step_ids.index("base_panel")

    left = next(s for s in plan["steps"] if s.get("id") == "left_support")
    right = next(s for s in plan["steps"] if s.get("id") == "right_support")
    lbase, loff = _parse_plane_offset(left.get("plane"))
    rbase, roff = _parse_plane_offset(right.get("plane"))
    assert lbase == "XY" and loff == 0.0
    assert rbase == "XY" and roff == 0.0
    assert left.get("distance", 0.0) > 0.0
    assert right.get("distance", 0.0) > 0.0

    # symmetry signal
    assert _rect_cx(left) < 0.0
    assert _rect_cx(right) > 0.0
    assert abs(abs(_rect_cx(left)) - abs(_rect_cx(right))) < 4.0


def test_rotational_family_grammar_lowpoly_approx_without_revolve_claim():
    spec = plan_structural_specs("Build a bottle", images=[], capabilities=default_capability_model())[0]
    assert spec.get("build_outcome_target") == "lowpoly_approx"

    plans = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a bottle",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )
    assert plans
    plan = plans[0]

    primitives = [str((s or {}).get("primitive") or "") for s in (plan.get("steps") or [])]
    assert "revolve" not in primitives
    assert "circle_extrude" in primitives
    assert (plan.get("global") or {}).get("build_outcome_target") == "lowpoly_approx"


def test_ranking_prefers_family_coherent_tank_over_box_pile():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    coherent = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]

    junk = {
        "session": "junk",
        "mode": "create",
        "global": {"shape_family": "lowpoly_hard_surface_vehicle"},
        "steps": [
            {"id": "box_1", "primitive": "rect_extrude", "plane": "XY", "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 12, "h": 12}, "distance": 6, "operation": "NewBodyFeatureOperation"},
            {"id": "box_2", "primitive": "rect_extrude", "plane": "XY@6", "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 10, "h": 10}, "distance": 5, "operation": "JoinFeatureOperation"},
        ],
    }

    ranked = rank_dsl_candidates([junk, coherent], spec, capabilities=default_capability_model())
    assert ranked[0]["plan"]["session"] == coherent["session"]


def test_ranking_prefers_family_coherent_furniture_over_disconnected_slab():
    spec = plan_structural_specs("Build a chair", images=[], capabilities=default_capability_model())[0]
    coherent = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a chair",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]

    slab = {
        "session": "slab",
        "mode": "create",
        "global": {"shape_family": "furniture_boxy_panel"},
        "steps": [
            {"id": "top", "primitive": "rect_extrude", "plane": "XY@30", "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 50, "h": 30}, "distance": 2, "operation": "NewBodyFeatureOperation"},
        ],
    }

    ranked = rank_dsl_candidates([slab, coherent], spec, capabilities=default_capability_model())
    assert ranked[0]["plan"]["session"] == coherent["session"]


def test_create_safety_prunes_nonessential_cut_steps_in_first_pass():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    plans = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=2,
        mode="create",
        capabilities=default_capability_model(),
    )
    assert len(plans) >= 1
    first = plans[0]
    safety = (first.get("global") or {}).get("create_safety") or {}
    assert safety.get("mode") == "primary_first"
    assert int(float(safety.get("risky_pruned_count", 0) or 0)) >= 1
    assert int(float(safety.get("risky_retained_count", 0) or 0)) == 0
    assert all(str((s or {}).get("primitive") or "") != "cut_extrude" for s in (first.get("steps") or []))


def test_ranking_prefers_robust_primary_over_fragile_cut_heavy_tank():
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=default_capability_model())[0]
    robust = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a lowpoly tank",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]
    fragile = {
        "session": "fragile_tank",
        "mode": "create",
        "global": {
            "shape_family": "lowpoly_hard_surface_vehicle",
            "create_safety": {"mode": "safe_detail_candidate", "risky_pruned_count": 0, "risky_retained_count": 6},
        },
        "steps": [
            {"id": "lower_hull", "primitive": "rect_extrude", "plane": "XY", "profile": {"type": "rect", "cx": 0, "cy": 25, "w": 30, "h": 60}, "distance": 8, "operation": "NewBodyFeatureOperation"},
            {"id": "upper_hull", "primitive": "rect_extrude", "plane": "XY@8", "profile": {"type": "rect", "cx": 0, "cy": 30, "w": 24, "h": 36}, "distance": 6, "operation": "JoinFeatureOperation"},
            {"id": "turret", "primitive": "rect_extrude", "plane": "XY@14", "profile": {"type": "rect", "cx": 0, "cy": 35, "w": 14, "h": 12}, "distance": 4, "operation": "JoinFeatureOperation"},
            {"id": "gun", "primitive": "circle_extrude", "plane": "XZ@40", "profile": {"type": "circle", "cx": 0, "cy": 16, "radius": 1}, "distance": 20, "operation": "JoinFeatureOperation"},
            {"id": "service_cut_1", "primitive": "cut_extrude", "plane": "XY@3", "profile": {"type": "rect", "cx": 0, "cy": 30, "w": 5, "h": 5}, "distance": 2, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
            {"id": "service_cut_2", "primitive": "cut_extrude", "plane": "XY@3", "profile": {"type": "rect", "cx": 5, "cy": 33, "w": 4, "h": 4}, "distance": 2, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
            {"id": "service_cut_3", "primitive": "cut_extrude", "plane": "XY@3", "profile": {"type": "rect", "cx": -5, "cy": 33, "w": 4, "h": 4}, "distance": 2, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
            {"id": "service_cut_4", "primitive": "cut_extrude", "plane": "XY@3", "profile": {"type": "rect", "cx": 0, "cy": 20, "w": 6, "h": 4}, "distance": 2, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
        ],
    }
    ranked = rank_dsl_candidates([fragile, robust], spec, capabilities=default_capability_model())
    assert ranked[0]["plan"]["session"] == robust["session"]


def test_ranking_prefers_stable_furniture_over_cut_heavy_variant():
    spec = plan_structural_specs("Build a chair", images=[], capabilities=default_capability_model())[0]
    stable = synthesize_dsl_candidates_from_structural_spec(
        spec,
        user_request="Build a chair",
        candidate_count=1,
        mode="create",
        capabilities=default_capability_model(),
    )[0]
    cut_heavy = {
        "session": "fragile_furniture",
        "mode": "create",
        "global": {
            "shape_family": "furniture_boxy_panel",
            "create_safety": {"mode": "safe_detail_candidate", "risky_pruned_count": 0, "risky_retained_count": 5},
        },
        "steps": [
            {"id": "base_panel", "primitive": "rect_extrude", "plane": "XY@40", "profile": {"type": "rect", "cx": 0, "cy": 10, "w": 40, "h": 20}, "distance": 2, "operation": "NewBodyFeatureOperation"},
            {"id": "left_support", "primitive": "rect_extrude", "plane": "XY@0", "profile": {"type": "rect", "cx": -10, "cy": 10, "w": 3, "h": 3}, "distance": 40, "operation": "NewBodyFeatureOperation"},
            {"id": "right_support", "primitive": "rect_extrude", "plane": "XY@0", "profile": {"type": "rect", "cx": 10, "cy": 10, "w": 3, "h": 3}, "distance": 40, "operation": "NewBodyFeatureOperation"},
            {"id": "decor_cut_1", "primitive": "cut_extrude", "plane": "XY@40", "profile": {"type": "rect", "cx": 0, "cy": 10, "w": 10, "h": 2}, "distance": 1, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
            {"id": "decor_cut_2", "primitive": "cut_extrude", "plane": "XY@40", "profile": {"type": "rect", "cx": 0, "cy": 8, "w": 8, "h": 2}, "distance": 1, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
            {"id": "decor_cut_3", "primitive": "cut_extrude", "plane": "XY@40", "profile": {"type": "rect", "cx": 0, "cy": 12, "w": 8, "h": 2}, "distance": 1, "operation": "CutFeatureOperation", "meta": {"risk_level": "risky_detail", "essential": False}},
        ],
    }
    ranked = rank_dsl_candidates([cut_heavy, stable], spec, capabilities=default_capability_model())
    assert ranked[0]["plan"]["session"] == stable["session"]


if __name__ == "__main__":
    test_vehicle_family_grammar_order_and_attachment_signals()
    test_furniture_family_grammar_support_order_no_floating_and_symmetry()
    test_rotational_family_grammar_lowpoly_approx_without_revolve_claim()
    test_ranking_prefers_family_coherent_tank_over_box_pile()
    test_ranking_prefers_family_coherent_furniture_over_disconnected_slab()
    test_create_safety_prunes_nonessential_cut_steps_in_first_pass()
    test_ranking_prefers_robust_primary_over_fragile_cut_heavy_tank()
    test_ranking_prefers_stable_furniture_over_cut_heavy_variant()
    print("All family grammar synthesis tests passed.")
