"""
Demo scenarios — detailed build instructions for complex mechanical parts.

Each scenario provides an explicit prompt with exact dimensions and build order
that the tool-driven agent can follow to produce a complex CAD model.

Usage:
    from cad.demo_scenarios import SCENARIOS
    prompt = SCENARIOS["bearing_housing"]["prompt"]
    agent.run(prompt)
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Scenario: Pillow Block Bearing Housing
# ---------------------------------------------------------------------------
# This is a simplified but recognizable bearing housing with:
# - Rectangular base plate with bolt holes
# - Cylindrical central boss for the bearing
# - Through-hole bore for the shaft
# - Support ribs connecting boss to base
# - Fillets on transitions for structural/visual quality
#
# Total estimated iterations: ~35 out of 50 budget
# ---------------------------------------------------------------------------

BEARING_HOUSING_PROMPT = r"""Build a pillow block bearing housing with the following EXACT specifications.
Follow the build order precisely. Do NOT skip steps or combine operations.

## Target object
A pillow block bearing housing: a rectangular base plate with a tall cylindrical
boss at the center (holds the bearing), a through-hole bore for the shaft, 4 bolt
holes in the base corners, 2 support ribs, and fillets for a finished look.

## Dimensions (all in cm)

| Part               | Shape     | Dimensions                              |
|--------------------|-----------|----------------------------------------|
| Base plate         | Rectangle | 24 × 12 cm, 2.5 cm thick              |
| Central boss       | Cylinder  | radius 5 cm, 8 cm tall above base     |
| Shaft bore (cut)   | Cylinder  | radius 2.5 cm, through entire height  |
| Bolt holes (cut)   | Cylinder  | radius 0.8 cm, through base           |
| Front support rib  | Rectangle | 2 × 4 cm, 5 cm tall                   |
| Back support rib   | Rectangle | 2 × 4 cm, 5 cm tall                   |

## Coordinate layout (top view, XY plane)

```
     Y
     ^
     |
  (-9,+4)  bolt    (+9,+4)  bolt
     |                 |
     |    rib(0,+4)    |
     |   ___________   |
     |  /  BOSS r=5 \  |
  ---+--| bore r=2.5|--+---> X
     |  \___________|  |
     |    rib(0,-4)    |
     |                 |
  (-9,-4)  bolt    (+9,-4)  bolt
```

## Build order (follow EXACTLY)

### Step 1: Base plate
- create_sketch("XY@0")
- add_rectangle(cx=0, cy=0, width=24, height=12)
- extrude(distance=2.5, operation="NewBodyFeatureOperation")
- screenshot + get_model_state → verify base plate at Z=0..2.5

### Step 2: Central boss
The boss sits on top of the base. Sketch at XY@2.0 for 0.5cm overlap with base top (Z=2.5).
- create_sketch("XY@2.0")
- add_circle(center={x:0, y:0}, radius=5)
- extrude(distance=8.5, operation="JoinFeatureOperation")
  → Boss from Z=2.0 to Z=10.5 (8cm above base top). Overlap = 0.5cm with base. ✓

### Step 3: Front support rib
Rib connects front side of boss to base. Sketch at XY@2.0 (same overlap).
- create_sketch("XY@2.0")
- add_rectangle(cx=0, cy=4.5, width=2, height=3)
  → Rib centered at Y=4.5, spanning Y=3..6. Boss edge at Y=5 overlaps. ✓
- extrude(distance=4.5, operation="JoinFeatureOperation")
  → Rib from Z=2.0 to Z=6.5

### Step 4: Back support rib (mirror of front)
- create_sketch("XY@2.0")
- add_rectangle(cx=0, cy=-4.5, width=2, height=3)
- extrude(distance=4.5, operation="JoinFeatureOperation")
- screenshot → verify boss + ribs look correct

### Step 5: Shaft bore (through-hole cut)
Cut a cylindrical hole through the entire boss and base for the shaft.
- create_sketch("XY@0")
- add_circle(center={x:0, y:0}, radius=2.5)
- extrude(distance=11, operation="CutFeatureOperation")
  → Cuts from Z=0 to Z=11, through base (0..2.5) and boss (2.0..10.5). Clean through-hole. ✓

### Step 6: Bolt hole — front-left
- create_sketch("XY@0")
- add_circle(center={x:-9, y:4}, radius=0.8)
- extrude(distance=3, operation="CutFeatureOperation")

### Step 7: Bolt hole — front-right
- create_sketch("XY@0")
- add_circle(center={x:9, y:4}, radius=0.8)
- extrude(distance=3, operation="CutFeatureOperation")

### Step 8: Bolt hole — back-left
- create_sketch("XY@0")
- add_circle(center={x:-9, y:-4}, radius=0.8)
- extrude(distance=3, operation="CutFeatureOperation")

### Step 9: Bolt hole — back-right
- create_sketch("XY@0")
- add_circle(center={x:9, y:-4}, radius=0.8)
- extrude(distance=3, operation="CutFeatureOperation")
- screenshot → verify all 4 bolt holes and shaft bore visible

### Step 10: Finishing — fillet edges
Apply fillet to round the sharp transitions, especially where boss meets base.
- fillet(body_name="Body1", radius=0.3)
  If this fails, try radius=0.2. If that also fails, skip — the model is still valid.

### Step 11: Final verification
- screenshot → final visual check
- get_model_state → verify body count = 1 (all parts joined)

## Important notes
- Each sketch has ONE profile only — create a new sketch for each extrusion.
- CutFeatureOperation subtracts material from the existing body.
- JoinFeatureOperation merges new material with the existing body (needs ≥0.5cm overlap).
- The bolt holes must be in the corners of the base plate, not near the boss.
- The shaft bore must go completely through the boss from bottom to top.
- Do NOT restart or call clear after the first extrude.
"""

# ---------------------------------------------------------------------------
# Scenario: Flanged Pipe Connector
# ---------------------------------------------------------------------------

FLANGED_PIPE_PROMPT = r"""Build a flanged pipe connector with the following EXACT specifications.
Follow the build order precisely.

## Target object
A short pipe section with circular flanges at both ends, bolt holes in each flange,
and a through-bore. Used for connecting pipe segments.

## Dimensions (all in cm)

| Part              | Shape     | Dimensions                              |
|-------------------|-----------|----------------------------------------|
| Bottom flange     | Cylinder  | radius 6 cm, 1.5 cm thick             |
| Pipe body         | Cylinder  | radius 3 cm, 10 cm tall               |
| Top flange        | Cylinder  | radius 6 cm, 1.5 cm thick             |
| Through bore (cut)| Cylinder  | radius 2.2 cm, full height             |
| Bolt holes (cut)  | Cylinder  | radius 0.6 cm, through each flange    |

## Coordinate layout (top view)
```
        Y
        ^
   BH3  |  BH4      (bolt holes at radius 4.5 from center)
        \|/
  BH2 --+-- BH1 --> X     (pipe center at origin)
        /|\
   BH5  |  BH6
```

6 bolt holes evenly distributed is complex with circles only.
Use 4 bolt holes at ±4.5 on X and Y axes instead:
- (4.5, 0), (-4.5, 0), (0, 4.5), (0, -4.5)

## Build order

### Step 1: Bottom flange
- create_sketch("XY@0")
- add_circle(center={x:0, y:0}, radius=6)
- extrude(distance=1.5, operation="NewBodyFeatureOperation")
- screenshot

### Step 2: Pipe body
Sketch at XY@1.0 (0.5cm overlap into flange top at Z=1.5).
- create_sketch("XY@1.0")
- add_circle(center={x:0, y:0}, radius=3)
- extrude(distance=10.5, operation="JoinFeatureOperation")
  → Pipe from Z=1.0 to Z=11.5

### Step 3: Top flange
Sketch at XY@11.0 (0.5cm overlap into pipe top at Z=11.5).
- create_sketch("XY@11.0")
- add_circle(center={x:0, y:0}, radius=6)
- extrude(distance=2.0, operation="JoinFeatureOperation")
  → Top flange from Z=11.0 to Z=13.0. 0.5cm overlap with pipe. ✓
- screenshot

### Step 4: Through bore
- create_sketch("XY@0")
- add_circle(center={x:0, y:0}, radius=2.2)
- extrude(distance=14, operation="CutFeatureOperation")
  → Cuts through entire height (0..13). Clean through-hole. ✓

### Step 5-8: Bolt holes in bottom flange (4 holes)
For each hole position: (4.5,0), (-4.5,0), (0,4.5), (0,-4.5):
- create_sketch("XY@0")
- add_circle(center={x:POS_X, y:POS_Y}, radius=0.6)
- extrude(distance=2, operation="CutFeatureOperation")

### Step 9-12: Bolt holes in top flange (4 holes)
For each hole position: (4.5,0), (-4.5,0), (0,4.5), (0,-4.5):
- create_sketch("XY@11.0")
- add_circle(center={x:POS_X, y:POS_Y}, radius=0.6)
- extrude(distance=2.5, operation="CutFeatureOperation")
- screenshot after last hole

### Step 13: Finishing
- chamfer(body_name="Body1", distance=0.2)
  If fails, try fillet with radius=0.15 instead.

### Step 14: Final
- screenshot → verify flanged pipe with bore and bolt holes
- get_model_state
"""

# ---------------------------------------------------------------------------
# Scenario: Motor Mount Bracket
# ---------------------------------------------------------------------------

MOTOR_MOUNT_PROMPT = r"""Build a motor mount bracket with the following EXACT specifications.
Follow the build order precisely.

## Target object
An L-shaped bracket with a vertical back plate, a horizontal base plate,
mounting holes in both plates, and a triangular gusset for rigidity.

## Dimensions (all in cm)

| Part              | Shape      | Dimensions                             |
|-------------------|------------|---------------------------------------|
| Base plate        | Rectangle  | 16 × 10 cm, 1.5 cm thick             |
| Back plate        | Rectangle  | 16 × 12 cm, 1.5 cm thick (vertical)  |
| Gusset (triangle) | Polygon   | right triangle, 6 × 8 cm, 1.5 thick  |
| Motor hole (cut)  | Circle     | radius 2.5 cm, in back plate center  |
| Base holes (cut)  | Circle     | radius 0.7 cm, 4 corners of base     |

## Side view (XZ plane at Y=0):
```
    Z
    ^
    |  +-----------+
    | |back plate  |  12cm tall
    | |   (hole)   |
    | |            |
    | +--+         |
    |  / |gusset   |
    | /  |         |
    +/---+---------+---> X (or Y)
    base plate (10cm deep)
```

## Build order

### Step 1: Base plate
- create_sketch("XY@0")
- add_rectangle(cx=0, cy=-5, width=16, height=10)
  → Base from Y=-10 to Y=0, X=-8 to X=8
- extrude(distance=1.5, operation="NewBodyFeatureOperation")
- screenshot

### Step 2: Back plate (vertical wall)
The back plate rises from the back edge of the base plate (Y=0).
Use XZ plane at Y=-0.5 so it overlaps 0.5cm into the base (base goes to Y=0).
Wait — base is at cy=-5, height=10, so Y=-10 to Y=0. Back edge is at Y=0.
Actually let me rethink. The back plate should be at the back edge.

Use XZ plane: sketch X maps to world X, sketch Y maps to world Z.
- create_sketch("XZ@-0.5")
  → XZ plane at Y=-0.5 (overlaps with base which extends to Y=0, overlap=0.5cm)
- add_rectangle(cx=0, cy=7.25, width=16, height=12.5)
  → sketch cy=7.25, half-height=6.25 → Z from 1.0 to 13.5.
  Z=1.0 overlaps with base top (Z=1.5) by 0.5cm. ✓
  Actually, base top is at Z=1.5. Wall should start overlapping with base.
  Let me set: cy = (1.0 + 13.5)/2 = 7.25, height = 12.5 → Z from 1.0 to 13.5.
  Overlap with base (Z=0..1.5): from Z=1.0 to Z=1.5 = 0.5cm. ✓
- extrude(distance=1.5, operation="JoinFeatureOperation")
  → Wall extrudes in +Y direction from Y=-0.5, 1.5cm → Y=-0.5 to Y=1.0
  → But base is at Y=-10 to Y=0. Overlap in Y: -0.5 to 0 = 0.5cm. ✓
- screenshot

### Step 3: Gusset (triangular support)
A right triangle connecting the base plate to the back plate.
Use polygon on XZ plane.
- create_sketch("XZ@-4.5")
  → At Y=-4.5 (well within base Y range of -10..0). Extrude will go +Y.
- add_polygon with points:
  [{x:-0.5, y:1.0}, {x:-0.5, y:9.0}, {x:-6.5, y:1.0}]
  → Triangle: bottom-left at (-0.5, 1.0), top at (-0.5, 9.0), bottom-right at (-6.5, 1.0).
  sketch Y = world Z, so this spans Z=1.0..9.0. Overlaps with base (Z≤1.5). ✓
  sketch X = world X, centered around X=0 on left side. Let me center it:
  [{x:-0.75, y:1.0}, {x:-0.75, y:9.0}, {x:-7.0, y:1.0}]
- extrude(distance=3, operation="JoinFeatureOperation")
  → Extrudes from Y=-4.5 to Y=-1.5. Within base Y range. ✓

Repeat mirrored gusset on right side:
- create_sketch("XZ@-4.5")
- add_polygon: [{x:0.75, y:1.0}, {x:0.75, y:9.0}, {x:7.0, y:1.0}]
- extrude(distance=3, operation="JoinFeatureOperation")
- screenshot

### Step 4: Motor mounting hole (in back plate)
The hole is in the center of the back plate. Back plate center is at approximately X=0, Z=7.5.
Use XZ plane at Y=-1 (within the back plate Y range of -0.5 to 1.0).
- create_sketch("XZ@-0.5")
  Wait, we need the cut to go through the plate. Plate is at Y=-0.5 to Y=1.0.
  Sketch at XZ@-1, extrude 3 (+Y direction) → Y=-1 to Y=2. Cuts through plate. ✓
- create_sketch("XZ@-1")
- add_circle(center={x:0, y:7.5}, radius=2.5)
  → sketch (0, 7.5) → world (0, -1, 7.5). Circle in back plate.
- extrude(distance=3, operation="CutFeatureOperation")

### Step 5-8: Base mounting holes
Hole positions in base plate corners: (-5.5,-8), (5.5,-8), (-5.5,-2), (5.5,-2)
For each:
- create_sketch("XY@0")
- add_circle(center={x:POS_X, y:POS_Y}, radius=0.7)
- extrude(distance=2, operation="CutFeatureOperation")
- screenshot after last hole

### Step 9: Finishing
- fillet(body_name="Body1", radius=0.3)
- screenshot → final verification
"""

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "bearing_housing": {
        "name": "Pillow Block Bearing Housing",
        "description": (
            "Rectangular base plate with cylindrical boss, through-bore for shaft, "
            "4 bolt holes, support ribs, and fillets. ~35 iterations."
        ),
        "prompt": BEARING_HOUSING_PROMPT,
        "expected_iterations": 35,
        "expected_body_count": 1,  # All parts joined
        "key_features": [
            "rectangular base plate",
            "cylindrical boss (Join)",
            "through-bore (Cut)",
            "4 bolt holes (Cut)",
            "2 support ribs (Join)",
            "fillet finishing",
        ],
    },
    "flanged_pipe": {
        "name": "Flanged Pipe Connector",
        "description": (
            "Pipe section with circular flanges at both ends, through-bore, "
            "and bolt holes in each flange. ~40 iterations."
        ),
        "prompt": FLANGED_PIPE_PROMPT,
        "expected_iterations": 40,
        "expected_body_count": 1,
        "key_features": [
            "bottom flange (NewBody)",
            "pipe body (Join)",
            "top flange (Join)",
            "through-bore (Cut)",
            "8 bolt holes (Cut)",
            "chamfer finishing",
        ],
    },
    "motor_mount": {
        "name": "Motor Mount Bracket",
        "description": (
            "L-shaped bracket with vertical back plate, horizontal base plate, "
            "triangular gussets, motor hole, and base bolt holes. ~38 iterations."
        ),
        "prompt": MOTOR_MOUNT_PROMPT,
        "expected_iterations": 38,
        "expected_body_count": 1,
        "key_features": [
            "base plate (NewBody)",
            "vertical back plate (Join, XZ plane)",
            "2 triangular gussets (Join, polygon)",
            "motor hole (Cut, XZ plane)",
            "4 base holes (Cut)",
            "fillet finishing",
        ],
    },
}


def get_scenario(name: str) -> Dict[str, Any]:
    """Get a demo scenario by name. Raises KeyError if not found."""
    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    """Return available scenario names."""
    return list(SCENARIOS.keys())
