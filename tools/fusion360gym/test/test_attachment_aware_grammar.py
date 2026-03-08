"""
Tests for attachment-aware geometry grammar.

Validates that:
- dependent attached parts use JoinFeatureOperation (not NewBody)
- attached geometry has overlap-compatible placement with parent
- standalone masses still use NewBody
- furniture panels/supports use correct attachment intent
- vehicle appendages use attachment-aware operation selection
- no regression for rotational lowpoly approx
- no regression for blocked smooth/freeform
"""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.dirname(TEST_DIR)
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from cad.cad_capabilities import default_capability_model
from cad.cad_dsl_synthesizer import synthesize_dsl_candidates_from_structural_spec
from cad.cad_family_grammar import (
    _ATTACH_OVERLAP,
    _resolve_operation,
    synthesize_steps_from_family_grammar,
)
from cad.cad_geometry_queries import step_bounds
from cad.cad_structural_planner import plan_structural_specs


def _steps_by_id(steps):
    return {str(s.get("id") or ""): s for s in steps if isinstance(s, dict)}


def _meta(step):
    return step.get("meta") or {}


def _bounds_overlap_3d(a, b):
    """Check if two 3D bounding boxes have volumetric overlap."""
    if a is None or b is None:
        return False
    return (
        a["x_min"] < b["x_max"]
        and a["x_max"] > b["x_min"]
        and a["y_min"] < b["y_max"]
        and a["y_max"] > b["y_min"]
        and a["z_min"] < b["z_max"]
        and a["z_max"] > b["z_min"]
    )


# ---------------------------------------------------------------------------
# Test 1: Vehicle tracks use Join and overlap hull
# ---------------------------------------------------------------------------

def test_vehicle_tracks_join_and_overlap():
    cap = default_capability_model()
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=cap)[0]
    steps = synthesize_steps_from_family_grammar(spec, session="test", user_request="Build a lowpoly tank")
    by_id = _steps_by_id(steps)

    hull_bounds = step_bounds(by_id["lower_hull"])

    for track_id in ("left_track_module", "right_track_module"):
        if track_id not in by_id:
            continue
        step = by_id[track_id]
        assert step["operation"] == "JoinFeatureOperation", \
            "{} should use Join, got {}".format(track_id, step["operation"])
        assert _meta(step)["attachment_intent"] == "attached"
        assert _meta(step)["parent_role"] == "lower_hull"
        assert _meta(step)["merge_required"] is True

        track_bounds = step_bounds(step)
        assert _bounds_overlap_3d(hull_bounds, track_bounds), \
            "{} must volumetrically overlap lower_hull".format(track_id)


# ---------------------------------------------------------------------------
# Test 2: Vehicle gun uses Join and has correct attachment metadata
# ---------------------------------------------------------------------------

def test_vehicle_gun_join_and_metadata():
    cap = default_capability_model()
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=cap)[0]
    steps = synthesize_steps_from_family_grammar(spec, session="test", user_request="Build a lowpoly tank")
    by_id = _steps_by_id(steps)

    gun = by_id["gun"]
    assert gun["operation"] == "JoinFeatureOperation"
    assert _meta(gun)["attachment_intent"] == "attached"
    assert _meta(gun)["parent_role"] == "turret"
    assert _meta(gun)["merge_required"] is True

    # Gun bounds should place it at positive z (inside turret z-range).
    gun_b = step_bounds(gun)
    turret_b = step_bounds(by_id["turret"])
    assert gun_b is not None and turret_b is not None
    assert gun_b["z_min"] >= turret_b["z_min"], \
        "gun z_min={} should be >= turret z_min={}".format(gun_b["z_min"], turret_b["z_min"])
    assert gun_b["z_max"] <= turret_b["z_max"] + 1.0, \
        "gun z_max={} should be near turret z_max={}".format(gun_b["z_max"], turret_b["z_max"])


# ---------------------------------------------------------------------------
# Test 3: Standalone base uses NewBody
# ---------------------------------------------------------------------------

def test_standalone_base_uses_newbody():
    cap = default_capability_model()
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=cap)[0]
    steps = synthesize_steps_from_family_grammar(spec, session="test", user_request="Build a lowpoly tank")
    by_id = _steps_by_id(steps)

    hull = by_id["lower_hull"]
    assert hull["operation"] == "NewBodyFeatureOperation"
    assert _meta(hull)["attachment_intent"] == "standalone"
    assert _meta(hull)["merge_required"] is False


# ---------------------------------------------------------------------------
# Test 4: Furniture panels use attachment-aware operations with overlap
# ---------------------------------------------------------------------------

def test_furniture_panels_attachment_and_overlap():
    cap = default_capability_model()
    spec = plan_structural_specs("Build a chair", images=[], capabilities=cap)[0]
    steps = synthesize_steps_from_family_grammar(spec, session="test", user_request="Build a chair")
    by_id = _steps_by_id(steps)

    # Supports are standalone (grounded).
    for support_id in ("left_support", "right_support"):
        if support_id in by_id:
            assert by_id[support_id]["operation"] == "NewBodyFeatureOperation"
            assert _meta(by_id[support_id])["attachment_intent"] == "standalone"

    # Base panel bridges supports.
    if "base_panel" in by_id:
        bp = by_id["base_panel"]
        assert bp["operation"] == "JoinFeatureOperation"
        assert _meta(bp)["attachment_intent"] == "bridging"
        assert _meta(bp)["parent_role"] == "left_support"

        # Base panel must volumetrically overlap with at least one support.
        panel_b = step_bounds(bp)
        for support_id in ("left_support", "right_support"):
            if support_id in by_id:
                support_b = step_bounds(by_id[support_id])
                assert _bounds_overlap_3d(support_b, panel_b), \
                    "base_panel must overlap {}".format(support_id)


# ---------------------------------------------------------------------------
# Test 5: General operation resolution logic
# ---------------------------------------------------------------------------

def test_operation_resolution():
    # Attached parts should get Join.
    assert _resolve_operation("NewBodyFeatureOperation", "attached") == "JoinFeatureOperation"
    assert _resolve_operation("NewBodyFeatureOperation", "bridging") == "JoinFeatureOperation"
    assert _resolve_operation("NewBodyFeatureOperation", "support_merged") == "JoinFeatureOperation"

    # Standalone keeps NewBody.
    assert _resolve_operation("NewBodyFeatureOperation", "standalone") == "NewBodyFeatureOperation"

    # Cut/Intersect never overridden.
    assert _resolve_operation("CutFeatureOperation", "attached") == "CutFeatureOperation"
    assert _resolve_operation("IntersectFeatureOperation", "bridging") == "IntersectFeatureOperation"


# ---------------------------------------------------------------------------
# Test 6: No regression for rotational lowpoly
# ---------------------------------------------------------------------------

def test_rotational_no_regression():
    cap = default_capability_model()
    for prompt in ("Build a bottle", "Build a vase", "Build a mug"):
        spec = plan_structural_specs(prompt, images=[], capabilities=cap)[0]
        assert spec["object_family"] == "rotational_bodies"
        assert spec["build_outcome_target"] == "lowpoly_approx"
        candidates = synthesize_dsl_candidates_from_structural_spec(
            spec, user_request=prompt, candidate_count=1, mode="create", capabilities=cap,
        )
        assert len(candidates) > 0
        steps = candidates[0].get("steps") or []
        assert len(steps) >= 2
        # Radial core must be standalone NewBody.
        core = [s for s in steps if str(s.get("id") or "") == "radial_core"]
        if core:
            assert core[0]["operation"] == "NewBodyFeatureOperation"
            assert _meta(core[0])["attachment_intent"] == "standalone"


# ---------------------------------------------------------------------------
# Test 7: No regression for blocked smooth/freeform
# ---------------------------------------------------------------------------

def test_blocked_smooth_freeform_no_regression():
    cap = default_capability_model()
    spec = plan_structural_specs("Build a smooth human face sculpture", images=[], capabilities=cap)[0]
    assert spec["object_family"] == "unsupported_smooth_freeform"
    assert spec["build_outcome_target"] == "blocked"


# ---------------------------------------------------------------------------
# Test 8: All attached parts carry merge metadata
# ---------------------------------------------------------------------------

def test_attached_parts_have_merge_metadata():
    cap = default_capability_model()
    for prompt, family in [
        ("Build a lowpoly tank", "lowpoly_hard_surface_vehicle"),
        ("Build a chair", "furniture_boxy_panel"),
    ]:
        spec = plan_structural_specs(prompt, images=[], capabilities=cap)[0]
        steps = synthesize_steps_from_family_grammar(spec, session="test", user_request=prompt)
        for step in steps:
            meta = _meta(step)
            intent = meta.get("attachment_intent", "")
            if intent in ("attached", "bridging", "support_merged"):
                assert meta.get("merge_required") is True, \
                    "step {} has intent={} but merge_required is not True".format(step["id"], intent)
                assert meta.get("parent_role"), \
                    "step {} has intent={} but no parent_role".format(step["id"], intent)
                assert step["operation"] == "JoinFeatureOperation", \
                    "step {} has intent={} but operation is {}".format(step["id"], intent, step["operation"])


# ---------------------------------------------------------------------------
# Test 9: XZ plane bounds produce correct positive-z for negated cy
# ---------------------------------------------------------------------------

def test_xz_plane_bounds_positive_z():
    """Steps on XZ plane with negated cy should produce positive z bounds."""
    cap = default_capability_model()
    spec = plan_structural_specs("Build a lowpoly tank", images=[], capabilities=cap)[0]
    steps = synthesize_steps_from_family_grammar(spec, session="test", user_request="Build a lowpoly tank")
    by_id = _steps_by_id(steps)

    gun = by_id.get("gun")
    if gun:
        b = step_bounds(gun)
        assert b is not None
        # Gun should be at positive z (inside turret/hull z-range).
        assert b["z_min"] > 0.0, "gun z_min={} should be positive".format(b["z_min"])
        assert b["z_max"] > 0.0, "gun z_max={} should be positive".format(b["z_max"])


# ---------------------------------------------------------------------------
# Test 10: Full candidate synthesis still produces valid plans
# ---------------------------------------------------------------------------

def test_full_candidate_synthesis_valid():
    cap = default_capability_model()
    for prompt in [
        "Build a lowpoly tank",
        "Build a chair",
        "Build a bottle",
    ]:
        candidates = synthesize_dsl_candidates_from_structural_spec(
            plan_structural_specs(prompt, images=[], capabilities=cap)[0],
            user_request=prompt,
            candidate_count=1,
            mode="create",
            capabilities=cap,
        )
        assert len(candidates) > 0
        plan = candidates[0]
        steps = plan.get("steps") or []
        assert len(steps) >= 2
        # All steps must have valid operations.
        for step in steps:
            assert step["operation"] in (
                "NewBodyFeatureOperation",
                "JoinFeatureOperation",
                "CutFeatureOperation",
                "IntersectFeatureOperation",
            ), "step {} has invalid operation: {}".format(step["id"], step["operation"])


if __name__ == "__main__":
    test_vehicle_tracks_join_and_overlap()
    test_vehicle_gun_join_and_metadata()
    test_standalone_base_uses_newbody()
    test_furniture_panels_attachment_and_overlap()
    test_operation_resolution()
    test_rotational_no_regression()
    test_blocked_smooth_freeform_no_regression()
    test_attached_parts_have_merge_metadata()
    test_xz_plane_bounds_positive_z()
    test_full_candidate_synthesis_valid()
    print("All attachment-aware grammar tests passed.")
