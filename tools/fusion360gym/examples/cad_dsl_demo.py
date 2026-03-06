"""
Demo: Universal CAD DSL pipeline — tank, plane, edit without clear.

Usage:
  1. Set DRY_RUN=1 to print compiled steps only (no Fusion).
  2. With Fusion 360 + add-in running: DRY_RUN=0 python cad_dsl_demo.py [tank|plane|edit]

  tank  - build lowpoly tank (wedge hull, poly turret, circle gun, wheel)
  plane - build lowpoly plane (fuselage, wing, tail)
  edit  - apply a patch to change one distance (edit mode; no clear)
"""

import os
import sys

FUSION_DIR = os.path.join(os.path.dirname(__file__), "..")
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from cad.cad_dsl import EXAMPLE_PLAN_LOWPOLY_TANK, EXAMPLE_PLAN_LOWPOLY_PLANE
from cad.cad_validate import validate_plan
from cad.cad_compiler import compile_plan
from cad.cad_executor import CadExecutor
from cad.cad_patch import apply_patch

DRY_RUN = os.environ.get("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
HOST = os.environ.get("FUSION_HOST", "127.0.0.1")
PORT = int(os.environ.get("FUSION_PORT", "8080"))


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "tank").strip().lower()

    if which == "tank":
        plan = dict(EXAMPLE_PLAN_LOWPOLY_TANK)
        print("Plan: lowpoly tank (hull, turret, gun, wheel)")
    elif which == "plane":
        plan = dict(EXAMPLE_PLAN_LOWPOLY_PLANE)
        print("Plan: lowpoly plane (fuselage, wing, tail)")
    elif which == "edit":
        plan = dict(EXAMPLE_PLAN_LOWPOLY_TANK)
        plan["mode"] = "edit"
        patch = {
            "patches": [
                {"op": "replace", "path": "steps[hull].distance", "value": 10.0},
            ],
            "intent": "minimal_change",
        }
        plan = apply_patch(plan, patch)
        print("Plan: edit tank — set hull distance to 10 (no clear)")
    else:
        print("Usage: python cad_dsl_demo.py [tank|plane|edit]")
        return 0

    vr = validate_plan(plan)
    if not vr.valid:
        print("Validation failed:", vr.errors)
        return 1
    print("Validation OK.")

    compiled = compile_plan(plan)
    print(f"Compiled {len(compiled)} steps.")

    if DRY_RUN:
        print("\n--- DRY RUN (no Fusion) ---")
        for cs in compiled:
            print(f"  Step {cs.step_id} ({cs.primitive}):")
            for c in cs.calls:
                print(f"    {c.command}: {c.data}")
        print("Done (dry run).")
        return 0

    try:
        from fusion360gym_client import Fusion360GymClient
    except ImportError:
        print("Fusion360GymClient not found. Use DRY_RUN=1 to test without Fusion.")
        return 1

    client = Fusion360GymClient(f"http://{HOST}:{PORT}")
    executor = CadExecutor(
        client,
        dry_run=False,
        step_delay=0.4,
        edit_mode=(plan.get("mode") == "edit"),
    )
    result = executor.execute_plan(plan)
    if result.success:
        print(f"Done: {result.message}")
    else:
        print(f"Execution failed: {result.message}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
