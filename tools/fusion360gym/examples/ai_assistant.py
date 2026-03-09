"""
Fusion 360 AI Assistant.

Primary architecture:
  1. LLM planner (text/vision) -> CAD Plan DSL
  2. Deterministic validator/compiler/executor -> Fusion 360 Gym
  3. Optional screenshot review loop

Legacy fallback (explicit opt-in only):
  - bbox decomposition -> deterministic encoder

Usage:
  1. Launch Fusion 360 and run the Fusion 360 Gym add-in
  2. Set API keys in the .env file next to this script
  3. python ai_assistant.py
"""

import sys
import os
import json
import math
import time
import base64
import tempfile
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")
if CLIENT_DIR not in sys.path:
    sys.path.append(CLIENT_DIR)
FUSION_DIR = os.path.join(os.path.dirname(__file__), "..")
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)

from fusion360gym_client import Fusion360GymClient
from cad.cad_operator import run_best_of_n_create, pre_render_fix, run_cad_operator
from cad.cad_capabilities import default_capability_model, capability_required_for_primitive
from cad.cad_validate import validate_plan
from cad.cad_multimodal import structural_from_legacy_bboxes, LegacyBboxCadPlanner
from cad.cad_agent_loop import IterativeCreateAgent
from cad.cad_dsl import PRIMITIVES as CAD_DSL_PRIMITIVES
from cad.cad_structural_spec import validate_structural_spec
from cad.cad_structural_planner import plan_structural_specs
from cad.cad_structural_ranker import rank_structural_specs, rank_dsl_candidates
from cad.cad_dsl_synthesizer import synthesize_dsl_candidates_from_structural_spec
from cad.cad_tool_agent import ToolDrivenAgent
from cad.cad_backend import FusionCadBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST_NAME = "127.0.0.1"
PORT_NUMBER = 8080
STEP_DELAY = 0.4
SCREENSHOT_WIDTH = 512
SCREENSHOT_HEIGHT = 512
MAX_FEEDBACK_ITERATIONS = int(os.environ.get("MAX_FEEDBACK_ITERATIONS", "1"))
ENABLE_VISUAL_REVIEW = os.environ.get("ENABLE_VISUAL_REVIEW", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
ENABLE_DETAILING_PASS = os.environ.get("ENABLE_DETAILING_PASS", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
DETAILING_MAX_PARTS = int(os.environ.get("DETAILING_MAX_PARTS", "22"))
PROGRAM_DETAIL_MAX_STEPS = int(os.environ.get("PROGRAM_DETAIL_MAX_STEPS", "12"))
ENABLE_AUTO_DETAIL_PROGRAM = os.environ.get("ENABLE_AUTO_DETAIL_PROGRAM", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
AUTO_DETAIL_PROGRAM_ROUNDS = int(os.environ.get("AUTO_DETAIL_PROGRAM_ROUNDS", "2"))
AUTO_DETAIL_MIN_CONFIDENCE = float(os.environ.get("AUTO_DETAIL_MIN_CONFIDENCE", "0.55"))

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

DESIGN_INDEX_FILE = os.environ.get("DESIGN_INDEX_FILE", "design_index.json")
DESIGN_INDEX_PATH = os.path.join(os.path.dirname(__file__), DESIGN_INDEX_FILE)
TRACK_DESIGN_INDEX_FILE = os.environ.get("TRACK_DESIGN_INDEX_FILE", "").strip()
TRACK_DESIGN_INDEX_PATH = (
    os.path.join(os.path.dirname(__file__), TRACK_DESIGN_INDEX_FILE)
    if TRACK_DESIGN_INDEX_FILE
    else ""
)
# Cap number of designs loaded into memory (0 = no cap). Use when index is huge.
DESIGN_INDEX_MAX_ENTRIES = int(os.environ.get("DESIGN_INDEX_MAX_ENTRIES", "0"))
RECON_DATASET_ROOT = os.environ.get("RECON_DATASET_ROOT", r"e:\Work\Fusion360\datasets\r1.0.1")
ASSEMBLY_DATASET_ROOT = os.environ.get(
    "ASSEMBLY_DATASET_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "assembly_examples")
)
SERVER_LAUNCH_PATH = os.path.join(os.path.dirname(__file__), "..", "server", "launch.py")
STEP_REPLAY_DELAY = float(os.environ.get("STEP_REPLAY_DELAY", "0.8"))

BEST_OF_N_CANDIDATES = int(os.environ.get("BEST_OF_N_CANDIDATES", "3"))
MULTIVIEW_BEST_OF_N_CANDIDATES = int(
    os.environ.get("MULTIVIEW_BEST_OF_N_CANDIDATES", "6")
)
MULTIVIEW_MIN_CONFIRM_VIEWS = int(os.environ.get("MULTIVIEW_MIN_CONFIRM_VIEWS", "2"))
USE_CAD_DSL_PLANNER = os.environ.get("USE_CAD_DSL_PLANNER", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
DSL_CANDIDATE_COUNT = int(os.environ.get("DSL_CANDIDATE_COUNT", str(max(3, BEST_OF_N_CANDIDATES))))
USE_LEGACY_BBOX_FALLBACK = os.environ.get("USE_LEGACY_BBOX_FALLBACK", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
USE_TOOL_DRIVEN_AGENT = os.environ.get("USE_TOOL_DRIVEN_AGENT", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
TOOL_AGENT_MAX_ITERATIONS = int(os.environ.get("TOOL_AGENT_MAX_ITERATIONS", "30"))
TOOL_AGENT_MODEL = os.environ.get("TOOL_AGENT_MODEL", "")

# ---------------------------------------------------------------------------
# Architect prompt вЂ" LLM decomposes objects into 3D bounding boxes
# ---------------------------------------------------------------------------

CAD_DSL_PLANNER_PROMPT = r"""You are a strict CAD planner.
Output ONLY a JSON CAD Plan for deterministic execution.

Allowed primitives:
- rect_extrude
- circle_extrude
- poly_extrude
- wedge_extrude
- cut_extrude

Allowed operations:
- NewBodyFeatureOperation
- JoinFeatureOperation
- CutFeatureOperation

Rules:
1) Keep the model physically plausible and connected.
2) Prefer 6-18 steps and structural simplicity over decorative detail.
3) Use planes intentionally: XY@z, XZ@y, YZ@x.
4) Put objects on the floor (z >= 0).
5) For repeated wheels/holes, create explicit separate steps.
6) Use stable ids and avoid duplicate step ids.
7) Do not output text, markdown, or comments.

Return ONLY JSON in this shape:
{
  "units": "cm",
  "session": "session_name",
  "mode": "create",
  "budget": {"max_steps": 25, "max_parts": 18},
  "global": {"symmetry": "approx_x"},
  "steps": [
    {
      "id": "base",
      "op": "ensure",
      "primitive": "rect_extrude",
      "plane": "XY@0",
      "profile": {"type": "rect", "cx": 0, "cy": 0, "w": 40, "h": 20},
      "distance": 8,
      "operation": "NewBodyFeatureOperation"
    }
  ]
}
"""

CAD_DSL_VISUAL_PLANNER_PROMPT = r"""You are a strict CAD planner for image-guided reconstruction.
Output ONLY a JSON CAD Plan for deterministic execution.

Use the provided images as geometric evidence and keep shape conservative.
Return only JSON in the same schema as the text CAD planner.
Do not output markdown or explanations.
"""

ARCHITECT_PROMPT = r"""You are a CAD decomposition engine.
Output axis-aligned 3D bounding boxes in centimeters.

You do NOT generate build commands. Only geometry.

Coordinate system (absolute):
- X: left/right (object center near X=0)
- Y: back->front (back-most usually near Y=0)
- Z: floor->up

Hard requirements:
1) Keep all parts physically plausible and connected (seat supported by legs/frame,
   backrest attached to seat/frame, etc.).
2) Prefer fewer larger structural volumes over many thin details.
3) If repeated thin elements exist (slats/rails), represent them as one group block
   (for example "slats_group") unless explicitly asked to model each element.
4) Keep left/right symmetry around X when appropriate.
5) Use floor at Z=0 for the final model (avoid negative Z).
6) Stay within realistic object proportions from the user request / brief.
7) For side/vertical details you may set optional "plane": "XY"|"XZ"|"YZ".
   Use "XZ"/"YZ" when XY-only extrusion would misorient the part.

Adjacency rule:
- Parts that touch in reality should share faces or have near-zero gap.

Return ONLY a JSON object, no markdown:
{format_hint}

{few_shot_section}"""

ARCHITECT_MULTIVIEW_PROMPT = r"""You reconstruct ONE real object from multiple photos.
Output axis-aligned 3D bounding boxes in centimeters.

The photos are the source of truth.

Rules:
- Do not invent details that are not visible.
- Every part must be explainable from at least two views; otherwise remove it or merge into a nearby structural part.
- Prioritize structural geometry and silhouette consistency over style details.
- Prefer 8-14 stable volumes over many tiny parts.
- Keep assembly physically connected and plausible.
- Use floor at Z=0 and avoid negative Z.
- Preserve left/right symmetry only when images support it.
- If repeated thin elements exist, group them into one volume.
- If uncertain, choose the conservative/simple interpretation.

Return ONLY a JSON object, no markdown:
{format_hint}
"""

CORRECTION_PROMPT = r"""You are editing an existing CAD part list (axis-aligned bboxes).

Rules:
- Keep the SAME part names and part count unless the user explicitly asks to add/remove parts.
- Only adjust dimensions/positions needed to satisfy the correction request.
- Preserve symmetry for left/right pairs when possible.
- Keep floor at Z=0 and avoid negative Z.
- Keep the assembly connected and physically plausible.

Return ONLY JSON:
{format_hint}
"""

DETAILING_PROMPT = r"""You refine an EXISTING CAD part list by adding functional detail.

Input:
1) Current model as axis-aligned bboxes.
2) User/object context.
3) Optional photos of the same object.

Goal:
- Keep the existing global silhouette and proportions.
- Add missing functional details that are clearly supported by context/photos.
- Do not replace the whole model with unrelated geometry.

Rules:
- Preserve structural core parts; add details conservatively.
- Prefer grouped detail volumes (e.g., track_segment_group, vent_group, wheel_group) over dozens of tiny parts.
- Keep physical connectivity and floor consistency (Z >= 0).
- Do not invent decorative parts without evidence.
- Return the COMPLETE updated parts list as JSON.

Return ONLY JSON:
{format_hint}
"""

PROGRAM_DETAIL_PLANNER_PROMPT = r"""You are a Fusion CAD action planner for detail refinement.

Task:
- You receive current model parts, a detail request, and optional image evidence.
- Generate a short sequence of additional build actions that add functional details.
- Steps must be conservative and physically plausible.

Allowed actions:
- refresh
- build (rect, circle, polygon, ring, gear extrude)

Build step schema:
{
  "action": "build",
  "description": "name",
  "plane": "XY|XZ|YZ" or "<PLANE>@<offset_cm>",
  "shape": "rect" or "circle",
  "cx": number,
  "cy": number,
  "w": number,      // required for rect
  "h": number,      // required for rect
  "radius": number, // required for circle
  "sides": number,  // required for polygon (>=3)
  "outer_radius": number, // required for ring/gear
  "inner_radius": number, // required for ring/gear
  "tooth_count": number,  // required for gear (>=6 recommended)
  "rotation_deg": number, // optional for polygon/gear
  "distance": number,
  "operation": "NewBodyFeatureOperation" or "JoinFeatureOperation" or "CutFeatureOperation",
  "repeat": number,  // optional >=1
  "dx": number,      // optional per-copy shift in X (cm)
  "dy": number,      // optional per-copy shift in Y (cm)
  "dz": number       // optional per-copy shift in Z/plane offset (cm)
}

Rules:
- Do NOT clear or rebuild the full model.
- Keep steps count small and impactful.
- Prefer grouped functional details (e.g. track block groups, wheel groups, vents).
- Use circle/ring/gear/polygon when mechanical detail is needed (holes, hubs, sprockets, perforation patterns).
- Keep dimensions realistic relative to existing model bounds.
- If uncertain, output fewer safer steps.
- Use plane orientation intentionally:
  XY extrudes along +Z, XZ extrudes along +Y, YZ extrudes along +X.
- Prefer Join/NewBody for additive detail.
- Use Cut only for explicit cutouts (holes, slots, ports, vents) and only when clearly intersecting an existing body.

Return ONLY JSON:
{"steps":[...]}
"""

DETAIL_EVALUATOR_PROMPT = r"""You are a CAD detail evaluator.

Input:
- current model parts (coarse + existing details)
- original user request/context
- optional photos

Task:
- Decide if more functional/mechanical details are needed.
- Propose a concise detail request for a downstream action planner.
- Stay object-agnostic: tank, aircraft, furniture, tools, gadgets, etc.

Rules:
- Do not request details that are not supported by context/photos.
- Prioritize functional details over decorative noise.
- If current detail level is already sufficient, say no.

Return ONLY JSON:
{
  "should_add_details": true,
  "confidence": 0.0,
  "detail_request": "short actionable detail brief",
  "missing_detail_groups": ["group_a", "group_b"],
  "notes": ["short reason"]
}
"""

ASSEMBLY_MECHANICAL_PRIORS = r"""Assembly reasoning priors for mechanical details:
- Preserve global frame; do not shift the whole model when adding details.
- Keep coaxial elements aligned (shaft-hole, hub-ring, sprocket-track).
- Keep bilateral symmetry for left/right repeated mechanisms unless context says otherwise.
- Place repeated elements with regular spacing and consistent size.
- Prefer functional groups: wheel sets, sprockets, idlers, hinge pairs, bolt rows, vent arrays.
- Keep added features near supporting parent parts, not floating in free space.
- Use cut operations for holes/ports and join/new-body for protrusions.
"""

EXTEND_REVISION_PROMPT = r"""The user has an EXISTING 3D model in the CAD program and wants to EXTEND or MODIFY it (add parts, change dimensions, add armrests, make back higher, etc.).

You receive:
1) The current model as a list of axis-aligned bounding boxes (name, x_min, x_max, y_min, y_max, z_min, z_max in cm).
2) The user's request describing what to add or change.

Your task: Output the COMPLETE updated parts list as a single JSON object with key "parts".
- You MAY add new parts (e.g. armrests, new back slats).
- You MAY remove or merge parts if the user asks.
- You MAY change dimensions/positions of existing parts.
- Keep the same coordinate system: floor Z=0, center Xв‰€0, back Yв‰€0. Stay connected and physically plausible.
- Preserve symmetry when appropriate.

Return ONLY JSON, no markdown:
{format_hint}
"""


REVIEW_PROMPT = r"""You are reviewing a 3D model built in Autodesk Fusion 360.
The user asked to build: "{user_request}"

You see a screenshot of the current result.
The model was built from these bounding boxes (centimeters):
{original_parts}

Evaluate:
1. Does the shape match what the user asked for?
2. Are proportions realistic for a real-world object?
3. Are all expected parts present and correctly positioned?

If the model looks correct, respond:
{{"satisfied": true, "comment": "Brief assessment"}}

If it needs fixes, describe WHAT is wrong in plain text. Do NOT generate coordinates
or build commands вЂ" just explain the problems clearly.
{{"satisfied": false, "comment": "Detailed description of what is wrong and how to fix it"}}

Return ONLY JSON. No markdown."""

IMAGE_BRIEF_PROMPT = r"""You analyze a single real-world object from one photo
for CAD reconstruction.

Goal: produce a robust engineering brief for downstream CAD generation.
Prioritize stable global proportions and structural parts over tiny details.

Rules:
- Focus on the primary object in the image.
- Infer object category, dominant symmetry, and structural parts.
- Estimate realistic overall dimensions in centimeters.
- Use robust assumptions (support contact, aligned faces), not style-specific details.
- If repeated thin elements exist, describe them as a group (do not enumerate).
- If an area is occluded, infer a plausible continuation.
- Ignore colors/materials/logos/background.

Return ONLY JSON, no markdown:
{
  "object_type": "single noun phrase",
  "design_brief": "short technical description",
  "overall_dims_cm": {"width": number, "depth": number, "height": number},
  "symmetry": "none|approx_x",
  "structural_parts_hint": [
    {"name":"...", "role":"seat|leg|backrest|panel|top|base|other", "count": number}
  ],
  "risk_notes": ["uncertain or occluded regions"],
  "features": [
    {"type": "tilt", "primary_axis": "x", "direction": "backward", "magnitude": {"type": "angle_deg", "value": 6}, "apply_to": "slender_posts"}
  ]
}
Optional "features": only add if the object has leaning or splayed elements (e.g. chair backrest leaning back, splayed legs).
- tilt: use primary_axis "x" for backward/forward lean (YZ plane), "y" for left/right; direction "backward"|"forward"|"outward"; magnitude angle_deg 0вЂ"12; apply_to "slender_posts"|"slender_rails"|"legs"|"handles".
- Omit "features" or use [] if no such elements.
"""

MULTIVIEW_BRIEF_PROMPT = r"""You analyze multiple photos of the SAME object
taken from different viewpoints for CAD reconstruction.

Goal: infer ONE consistent 3D interpretation and produce a robust engineering brief.

Rules:
- Treat all photos as the same object and resolve conflicts conservatively.
- Use provided view labels as orientation hints.
- Keep global geometry and proportions consistent across views.
- Estimate realistic overall dimensions in centimeters.
- Prefer structural volumes over tiny detail.
- If repeated thin elements exist, describe them as groups.
- Ignore color/material/background.

Return ONLY JSON, no markdown:
{
  "object_type": "single noun phrase",
  "design_brief": "short technical description",
  "overall_dims_cm": {"width": number, "depth": number, "height": number},
  "symmetry": "none|approx_x",
  "view_consistency_notes": ["conflicts and how they were resolved"],
  "structural_parts_hint": [
    {"name":"...", "role":"seat|leg|backrest|panel|top|base|other", "count": number}
  ],
  "features": [
    {"type": "tilt", "primary_axis": "x", "direction": "backward", "magnitude": {"type": "angle_deg", "value": 6}, "apply_to": "slender_posts"}
  ]
}
Optional "features": only add if the object has leaning or splayed elements (e.g. chair backrest leaning back).
- tilt: primary_axis "x" for backward/forward (YZ), "y" for left/right; direction "backward"|"forward"|"outward"; magnitude angle_deg 0вЂ"12; apply_to "slender_posts"|"slender_rails"|"legs"|"handles".
- Omit "features" or use [] if none.
"""

GEOMETRY_AUDIT_PROMPT = r"""You are a CAD geometry auditor.

You receive:
- a photo-derived engineering brief
- candidate axis-aligned 3D bounding boxes
- photos (possibly multi-view)

Task:
Return an improved parts list that is more stable and physically plausible.

Hard rules:
1) Keep global dimensions and silhouette consistent with the brief.
2) Enforce floor at Z=0 (no negative Z).
3) Remove floating parts and disconnected clusters.
4) Reduce part count if stability improves (merge thin repeated elements into groups).
5) Enforce left/right symmetry when declared.
6) Remove or merge any part that is not supported by at least two views.
7) Do not add common "typical" furniture details unless they are clearly visible.

Output ONLY JSON:
{"parts":[{"name":"...", "x_min":..., "x_max":..., "y_min":..., "y_max":..., "z_min":..., "z_max":...}]}
"""

MULTIVIEW_RANKER_PROMPT = r"""You score how well a candidate CAD part list matches multi-view photos.

Inputs:
- photos of the same object from different views
- optional brief/context
- candidate axis-aligned 3D boxes

Scoring rubric (0-100):
- silhouette/proportion consistency across views: 45
- no hallucinated extra parts: 25
- physical plausibility/connectivity: 20
- compactness (fewer stable parts): 10

Return ONLY JSON:
{"score": number, "notes": ["short reasons"]}
"""

VIEW_SUPPORT_FILTER_PROMPT = r"""You verify if each candidate part is grounded in multi-view evidence.

For each part index, estimate in how many views (0..N) this part is visually supported.
If uncertain, be conservative and prefer lower support.

Return ONLY JSON:
{
  "support_by_index": [{"index": 0, "views_supported": 3}],
  "drop_indices": [1, 4]
}
"""


# ---------------------------------------------------------------------------
# Design Index вЂ" few-shot retrieval from curated / dataset examples
# ---------------------------------------------------------------------------

class DesignIndex:

    def __init__(self, index_path):
        self.designs = []
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raw = [raw] if isinstance(raw, dict) else []
            self.designs = raw
            if DESIGN_INDEX_MAX_ENTRIES > 0 and len(self.designs) > DESIGN_INDEX_MAX_ENTRIES:
                self.designs = self.designs[:DESIGN_INDEX_MAX_ENTRIES]
                print(f"  Design index: loaded first {DESIGN_INDEX_MAX_ENTRIES} of {len(raw)} reference designs (cap)")
            else:
                print(f"  Design index: loaded {len(self.designs)} reference designs")
        else:
            print(f"  Design index: {index_path} not found, running without references")

    def find_similar(self, query, top_k=2):
        query_lower = query.lower()
        scored = []
        for design in self.designs:
            score = sum(2 for kw in design["keywords"] if kw in query_lower)
            score += sum(1 for kw in design["keywords"]
                         if any(w in query_lower for w in kw.split()))
            if score > 0:
                scored.append((score, design))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:top_k]]

    def format_few_shot(self, similar_designs):
        if not similar_designs:
            return ""
        lines = ["# Reference designs (use as proportion/structure guides)\n"]
        for i, design in enumerate(similar_designs, 1):
            lines.append(f"## Reference {i}: {design['description']}")
            parts_json = json.dumps({"parts": design["parts"]}, indent=2)
            lines.append(parts_json)
            lines.append("")
        return "\n".join(lines)

    def find_exact_model(self, token):
        """Find a design by exact token.

        Supports:
        - keyword exact match (e.g. "56045")
        - source filename stem exact match (e.g. "56045_d9d572d5_0000")
        - leading numeric id from filename (e.g. "56045")
        """
        if not token:
            return None
        t = token.strip().lower()
        for design in self.designs:
            # keywords
            for kw in design.get("keywords", []):
                if str(kw).strip().lower() == t:
                    return design
            # source file checks
            src = str(design.get("source_file", "")).strip().lower()
            if src:
                stem = src[:-5] if src.endswith(".json") else src
                if stem == t:
                    return design
                src_prefix = stem.split("_")[0]
                if src_prefix == t:
                    return design
        return None


# ---------------------------------------------------------------------------
# Deterministic encoder вЂ" bounding boxes в†’ Fusion 360 Gym JSON plan
# ---------------------------------------------------------------------------

def encode_bboxes_to_plan(parts, include_clear=True, include_refresh=False):
    """Convert a list of 3D bounding boxes into Fusion 360 Gym build steps.

    Each part is a dict with: name, x_min, x_max, y_min, y_max, z_min, z_max.
    Optional part["plane"] (XY/XZ/YZ) is supported. If missing, plane is inferred
    from bbox dimensions (smallest axis is used as extrusion direction).
    When include_clear is True, the first step is refresh (fit camera) so the view is usable during build.
    """
    parts = _expand_grouped_wheel_parts(parts)
    steps = []
    if include_clear:
        steps.append({"action": "refresh"})
        steps.append({"action": "clear"})

    for part in parts:
        step = _encode_single_bbox(part)
        if step:
            steps.append(step)

    if include_refresh:
        steps.append({"action": "refresh"})
    return steps


def _expand_grouped_wheel_parts(parts):
    """Expand grouped wheel-like bboxes into repeated wheel elements.

    This is a deterministic stability patch for prompts like "build a tank":
    LLMs often output one long grouped bbox (e.g. road_wheels_left), which
    becomes a "bar/leg" after extrusion. We split such grouped parts into
    repeated near-circular wheel bboxes before step encoding.
    """
    if not isinstance(parts, list):
        return []
    out = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        name = str(part.get("name", "")).strip().lower()
        group_like = any(
            k in name for k in (
                "road_wheels", "road wheel", "roadwheel",
                "rollers", "roller_group", "wheel_group",
                "колес", "катк", "ролик",
            )
        )
        if group_like:
            expanded = _split_bbox_into_repeated_wheels(part)
            if expanded and len(expanded) > 1:
                out.extend(expanded)
                continue
        out.append(part)
    return out


def _split_bbox_into_repeated_wheels(part):
    """Split one elongated wheel-group bbox into N wheel bboxes."""
    try:
        x0, x1 = float(part["x_min"]), float(part["x_max"])
        y0, y1 = float(part["y_min"]), float(part["y_max"])
        z0, z1 = float(part["z_min"]), float(part["z_max"])
    except Exception:
        return [part]

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if z0 > z1:
        z0, z1 = z1, z0

    dims = {
        "x": x1 - x0,
        "y": y1 - y0,
        "z": z1 - z0,
    }
    long_axis = max(dims, key=lambda a: dims[a])
    long_len = dims[long_axis]
    other_axes = [a for a in ("x", "y", "z") if a != long_axis]
    d1 = dims[other_axes[0]]
    d2 = dims[other_axes[1]]

    if min(d1, d2) <= 0.2 or long_len <= 0.2:
        return [part]
    # Keep only near-cylindrical cross-sections.
    if max(d1, d2) / max(0.01, min(d1, d2)) > 1.6:
        return [part]
    # Grouped wheels should be elongated.
    if long_len < 2.2 * max(d1, d2):
        return [part]

    avg_d = 0.5 * (d1 + d2)
    raw_count = int(round(long_len / max(1.0, avg_d * 1.45)))
    count = max(3, min(8, raw_count))

    # target wheel length along long axis (slightly thinner than diameter)
    wheel_len = min(avg_d, (long_len * 0.82) / count)
    if wheel_len <= 0.2:
        return [part]
    gap = (long_len - count * wheel_len) / (count + 1)
    if gap < 0.15:
        # if spacing is too tight, reduce wheel count until stable.
        while count > 2:
            count -= 1
            wheel_len = min(avg_d, (long_len * 0.82) / count)
            gap = (long_len - count * wheel_len) / (count + 1)
            if gap >= 0.15:
                break
    if count < 2:
        return [part]

    base_name = str(part.get("name", "wheel_group")).strip() or "wheel_group"
    plane = str(part.get("plane", "")).strip()
    result = []

    def part_with_axis(lo, hi):
        p = {
            "name": "",
            "x_min": x0, "x_max": x1,
            "y_min": y0, "y_max": y1,
            "z_min": z0, "z_max": z1,
        }
        if long_axis == "x":
            p["x_min"], p["x_max"] = lo, hi
        elif long_axis == "y":
            p["y_min"], p["y_max"] = lo, hi
        else:
            p["z_min"], p["z_max"] = lo, hi
        if plane:
            p["plane"] = plane
        return p

    if long_axis == "x":
        start = x0 + gap
    elif long_axis == "y":
        start = y0 + gap
    else:
        start = z0 + gap

    for i in range(count):
        lo = start + i * (wheel_len + gap)
        hi = lo + wheel_len
        p = part_with_axis(lo, hi)
        p["name"] = f"{base_name}_{i+1}"
        result.append(p)

    return result if result else [part]


def _encode_single_bbox(part):
    """Convert one absolute 3D bbox into a stable plane-aware build step."""
    name = part.get("name", "unnamed")
    x0, x1 = part["x_min"], part["x_max"]
    y0, y1 = part["y_min"], part["y_max"]
    z0, z1 = part["z_min"], part["z_max"]

    wx = abs(x1 - x0)
    wy = abs(y1 - y0)
    wz = abs(z1 - z0)

    if wx < 0.01 or wy < 0.01 or wz < 0.01:
        return None

    plane_raw = str(part.get("plane", "")).strip().upper()
    if "@" in plane_raw:
        plane_raw = plane_raw.split("@", 1)[0].strip()
    plane = plane_raw if plane_raw in ("XY", "XZ", "YZ") else None

    # Fallback: choose plane so extrusion follows the smallest bbox axis.
    if plane is None:
        axis_to_plane = {"x": "YZ", "y": "XZ", "z": "XY"}
        smallest_axis = min((("x", wx), ("y", wy), ("z", wz)), key=lambda kv: kv[1])[0]
        plane = axis_to_plane[smallest_axis]

    def choose_circle_shape(part_name, a, b, depth):
        n = str(part_name or "").lower()
        a = abs(float(a))
        b = abs(float(b))
        depth = abs(float(depth))
        if min(a, b, depth) <= 0.01:
            return False
        roundness = max(a, b) / max(0.01, min(a, b))
        cyl_keywords = (
            "wheel", "idler", "roller", "sprocket", "pulley", "hub", "axle",
            "shaft", "pipe", "tube", "barrel", "cylinder", "bearing", "ring",
            "колес", "катк", "ролик", "ось", "вал", "труб", "ствол", "цилинд",
        )
        has_kw = any(k in n for k in cyl_keywords)
        rod_like = depth >= 1.6 * ((a + b) * 0.5)
        return roundness <= 1.35 and (has_kw or rod_like)

    if plane == "XZ":
        shape = "circle" if choose_circle_shape(name, wx, wz, wy) else "rect"
        return _make_step(
            name=name,
            plane=f"XZ@{round(y0, 3)}",
            cx=(x0 + x1) / 2,
            cy=(z0 + z1) / 2,
            w=wx,
            h=wz,
            distance=wy,
            shape=shape,
        )
    if plane == "YZ":
        shape = "circle" if choose_circle_shape(name, wy, wz, wx) else "rect"
        return _make_step(
            name=name,
            plane=f"YZ@{round(x0, 3)}",
            cx=(y0 + y1) / 2,
            cy=(z0 + z1) / 2,
            w=wy,
            h=wz,
            distance=wx,
            shape=shape,
        )

    # Default XY.
    shape = "circle" if choose_circle_shape(name, wx, wy, wz) else "rect"
    return _make_step(
        name=name,
        plane=f"XY@{round(z0, 3)}",
        cx=(x0 + x1) / 2,
        cy=(y0 + y1) / 2,
        w=wx,
        h=wy,
        distance=wz,
        shape=shape,
    )


def _make_step(name, plane, cx, cy, w, h, distance, shape="rect"):
    step = {
        "action": "build",
        "description": name,
        "plane": plane,
        "shape": shape,
        "cx": round(cx, 1),
        "cy": round(cy, 1),
        "distance": round(distance, 1),
        "operation": "NewBodyFeatureOperation",
    }
    if shape == "circle":
        step["radius"] = round(max(0.1, min(float(w), float(h)) * 0.5), 1)
    else:
        step["w"] = round(w, 1)
        step["h"] = round(h, 1)
    return step


# ---------------------------------------------------------------------------
# LLM Provider abstraction
# ---------------------------------------------------------------------------

def create_llm_client():
    if LLM_PROVIDER == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("Package 'anthropic' is not installed. Run: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ANTHROPIC_API_KEY not found in .env")
            sys.exit(1)
        return "anthropic", anthropic.Anthropic(api_key=api_key)

    try:
        from openai import OpenAI
    except ImportError:
        print("Package 'openai' is not installed. Run: pip install openai")
        sys.exit(1)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        sys.exit(1)
    return "openai", OpenAI(api_key=api_key)


def get_model_name():
    if LLM_PROVIDER == "anthropic":
        return ANTHROPIC_MODEL
    return OPENAI_MODEL


# ---------------------------------------------------------------------------
# FusionAIAssistant
# ---------------------------------------------------------------------------

class FusionAIAssistant:

    def __init__(self, provider, llm_client, fusion_host, fusion_port):
        self.fusion = Fusion360GymClient(f"http://{fusion_host}:{fusion_port}")
        self.provider = provider
        self.llm = llm_client
        self.model = get_model_name()
        self.index = DesignIndex(DESIGN_INDEX_PATH)
        self.track_index = DesignIndex(TRACK_DESIGN_INDEX_PATH) if TRACK_DESIGN_INDEX_PATH else None
        self.review_enabled = ENABLE_VISUAL_REVIEW
        self.last_parts = []
        self.last_request = ""
        self.last_images = []
        self.last_cad_plan = None
        self.last_create_policy = None
        self.last_structural_spec = None
        self.last_iterative_loop_trace = []
        self.last_iterative_state_snapshots = []
        self.recon_root = Path(RECON_DATASET_ROOT)
        self.assembly_root = Path(ASSEMBLY_DATASET_ROOT)
        self._recon_file_cache = {}
        self._assembly_summary_cache = {}

    @staticmethod
    def _is_track_query(text):
        hay = str(text or "").lower()
        return any(
            k in hay for k in (
                "track", "tank", "caterpillar", "crawler", "sprocket", "idler",
                "гусениц", "гусеница", "трак", "танк",
            )
        )

    def _find_reference_designs(self, query, top_k=2):
        wanted = max(1, int(top_k))
        picked = []
        seen = set()

        def add_candidates(cands):
            for d in cands:
                key = (
                    str(d.get("source_file", "")).strip().lower(),
                    str(d.get("description", "")).strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                picked.append(d)
                if len(picked) >= wanted:
                    break

        if self.track_index is not None and self._is_track_query(query):
            add_candidates(self.track_index.find_similar(query, top_k=max(2, wanted)))
        if len(picked) < wanted:
            add_candidates(self.index.find_similar(query, top_k=max(2, wanted)))
        return picked[:wanted]

    def _find_exact_model_any(self, token):
        d = self.index.find_exact_model(token)
        if d is not None:
            return d
        if self.track_index is not None:
            d = self.track_index.find_exact_model(token)
        return d

    @staticmethod
    def _track_architect_addendum():
        return (
            "Track-specific decomposition constraints:\n"
            "- Include left/right drive wheel and idler wheel as explicit separate parts.\n"
            "- Include at least 4 road wheel placeholders per side (can be grouped as road_wheels_left_1..N and right_1..N).\n"
            "- Side mechanisms (track side frames, wheels) should prefer plane \"YZ\".\n"
            "- Keep strong left/right symmetry and keep all wheel centers aligned along Z where appropriate.\n"
            "- Do not collapse all wheels into one giant box."
        )

    def check_connection(self):
        try:
            r = self.fusion.ping()
            return r.status_code == 200
        except Exception:
            return False

    def ensure_connection(self):
        if self.check_connection():
            return True
        print("\n  Fusion 360 Gym is not responding.")
        print("  Re-open Fusion 360 (if needed) and run add-in: Add-ins -> Run.")
        print("  Then continue in this terminal.\n")
        return False

    def find_reconstruction_json(self, source_file):
        """Resolve a reconstruction JSON file path from source_file name."""
        if not source_file:
            return None
        if source_file in self._recon_file_cache:
            return self._recon_file_cache[source_file]
        if not self.recon_root.exists():
            return None
        # Dataset layout includes nested folders; search lazily and cache result.
        for p in self.recon_root.rglob(source_file):
            if p.is_file():
                self._recon_file_cache[source_file] = p
                return p
        self._recon_file_cache[source_file] = None
        return None

    def reconstruct_stepwise(self, json_path, only_extrudes=True):
        """Replay a reconstruction JSON incrementally for visual demos.

        Preferred mode uses server-side incremental reconstruction so geometry
        grows in-place (no full reset between steps). If server does not yet
        support that command, falls back to prefix replay.
        """
        self.fusion.clear()
        if only_extrudes:
            print(f"  Stepwise replay (in-place) with delay={STEP_REPLAY_DELAY:.2f}s")
        else:
            print(
                "  Stepwise replay requested for all timeline entities, "
                "but current server mode steps extrude boundaries."
            )

        r = self.fusion.reconstruct_stepwise(str(json_path), delay=STEP_REPLAY_DELAY)
        if r is not None and r.status_code == 200:
            self.fusion.refresh()
            print("  Stepwise replay completed.\n")
            return True

        # Show why in-place failed so user can fix (e.g. update add-in).
        err_msg = ""
        if r is not None:
            try:
                body = r.json()
                err_msg = body.get("message", "") or str(r.status_code)
            except Exception:
                err_msg = f"HTTP {r.status_code}"
        else:
            err_msg = "no response (check Fusion and add-in)"
        print(f"  In-place stepwise failed: {err_msg}")
        print("  Tip: Restart Fusion 360, then run Add-ins -> Run so the server loads the latest code.")
        print("  Using legacy prefix replay (model rebuilds each step).")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        timeline = data.get("timeline", [])
        entities = data.get("entities", {})
        if not timeline or not entities:
            print("  Invalid reconstruction JSON: missing timeline/entities.\n")
            return False
        step_indices = [
            i for i, t in enumerate(timeline)
            if entities.get(t.get("entity", ""), {}).get("type") == "ExtrudeFeature"
        ] if only_extrudes else list(range(len(timeline)))
        if not step_indices:
            print("  No replayable steps found in timeline.\n")
            return False
        for stage, t_idx in enumerate(step_indices, 1):
            prefix_timeline = timeline[:t_idx + 1]
            prefix_entities = {
                eid: entities[eid]
                for eid in (t.get("entity", "") for t in prefix_timeline)
                if eid in entities
            }
            partial = {
                "metadata": data.get("metadata", {}),
                "timeline": prefix_timeline,
                "entities": prefix_entities,
                "sequence": data.get("sequence", []),
                "properties": data.get("properties", {}),
            }
            self.fusion.clear()
            rr = self.fusion.send_command("reconstruct", partial)
            if rr is None or rr.status_code != 200:
                print(f"  Replay failed at stage {stage}/{len(step_indices)}.\n")
                return False
            self.fusion.refresh()
            time.sleep(STEP_REPLAY_DELAY)
        print("  Stepwise replay completed (legacy mode).\n")
        return True

    def relaunch_gym(self):
        """Launch a fresh Fusion instance via launch.py and wait for ping."""
        launch_py = Path(SERVER_LAUNCH_PATH).resolve()
        if not launch_py.exists():
            print(f"  launch.py not found: {launch_py}")
            return False
        try:
            print("  Launching Fusion 360 via launch.py ...")
            subprocess.Popen(
                [
                    sys.executable,
                    str(launch_py),
                    "--instances", "1",
                    "--host", HOST_NAME,
                    "--start_port", str(PORT_NUMBER),
                ],
                cwd=str(launch_py.parent),
            )
        except Exception as e:
            print(f"  Failed to launch Fusion: {e}")
            return False

        for i in range(1, 25):
            time.sleep(3)
            if self.check_connection():
                print(f"  Fusion 360 Gym connected after relaunch ({i * 3}s).\n")
                return True
        print("  Relaunch started, but ping still failing.")
        print("  If this is first-time setup, enable 'Run on startup' once in Fusion Add-ins.\n")
        return False

    # -- LLM calls (provider-agnostic) ----------------------------------------

    def _call_llm(self, system_prompt, user_content):
        if self.provider == "anthropic":
            if isinstance(user_content, str):
                messages = [{"role": "user", "content": user_content}]
            else:
                messages = [{"role": "user", "content": user_content}]
            response = self.llm.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )
            return response.content[0].text
        else:
            if isinstance(user_content, str):
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=0.2,
                max_tokens=8192,
            )
            return response.choices[0].message.content

    def _call_llm_with_image(self, system_prompt, text, image_b64, media_type="image/png"):
        return self._call_llm_with_images(
            system_prompt,
            text,
            [{"data": image_b64, "media_type": media_type}],
        )

    def _call_llm_with_images(self, system_prompt, text, images):
        """images: list[{'data': base64str, 'media_type': str, 'label'?: str, 'name'?: str}]"""
        if not images:
            raise ValueError("No images provided")
        if self.provider == "anthropic":
            content = []
            for idx, img in enumerate(images, 1):
                label = str(img.get("label", f"view_{idx}")).strip()
                name = str(img.get("name", "")).strip()
                content.append({
                    "type": "text",
                    "text": f"[View {idx}] label={label}; file={name}" if name else f"[View {idx}] label={label}",
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/png"),
                        "data": img["data"],
                    },
                })
            content.append({"type": "text", "text": text})
            response = self.llm.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=8192,
            )
            return response.content[0].text
        else:
            content = [{"type": "text", "text": text}]
            for idx, img in enumerate(images, 1):
                label = str(img.get("label", f"view_{idx}")).strip()
                name = str(img.get("name", "")).strip()
                content.append({
                    "type": "text",
                    "text": f"[View {idx}] label={label}; file={name}" if name else f"[View {idx}] label={label}",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.get('media_type', 'image/png')};base64,{img['data']}",
                        "detail": "high",
                    },
                })
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            return response.choices[0].message.content

    # -- Tool-driven agent (free-form LLM tool_use) --------------------------

    def _make_tool_agent_llm_call(self):
        """Create an LLM callable matching ToolDrivenAgent signature."""
        if self.provider != "anthropic":
            raise RuntimeError("Tool-driven agent requires Anthropic provider (LLM_PROVIDER=anthropic)")

        def llm_call(*, model, system, messages, tools, max_tokens=4096, temperature=0.1):
            return self.llm.messages.create(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return llm_call

    def _run_tool_driven_agent(self, user_request, images=None):
        """Run the tool-driven agent for a user request. Returns (ok, result)."""
        backend = FusionCadBackend(self.fusion)
        agent_model = TOOL_AGENT_MODEL or self.model
        agent = ToolDrivenAgent(
            backend=backend,
            llm_call=self._make_tool_agent_llm_call(),
            model=agent_model,
            max_iterations=TOOL_AGENT_MAX_ITERATIONS,
            step_delay=STEP_DELAY,
            verbose=True,
        )
        print(f"  [Tool Agent] model={agent_model}, max_iter={TOOL_AGENT_MAX_ITERATIONS}")
        result = agent.run(user_request, images=images)
        if result.ok:
            print(f"\n  Tool agent OK: {result.steps_executed} tool calls in {result.iterations} iterations")
            if result.final_state:
                bodies = result.final_state.get("bodies", [])
                if isinstance(bodies, list):
                    print(f"  Bodies in model: {len(bodies)}")
                    for b in bodies[:10]:
                        name = b if isinstance(b, str) else b.get("name", str(b))
                        print(f"    - {name}")
        else:
            print(f"\n  Tool agent failed: {result.error}")
        return result.ok, result

    # -- Step A: Architect вЂ" decompose into bounding boxes --------------------

    def decompose_object(self, user_request):
        similar = self._find_reference_designs(user_request, top_k=2)
        few_shot = self.index.format_few_shot(similar)

        if similar:
            names = [d["description"] for d in similar]
            print(f"  Found {len(similar)} reference design(s): {', '.join(names)}")

        format_hint = (
            '{"parts": [\n'
            '  {"name": "Part A", "x_min": -20, "x_max": 20, '
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
            '  ...\n'
            ']}'
        )

        prompt = ARCHITECT_PROMPT.replace("{format_hint}", format_hint)
        prompt = prompt.replace("{few_shot_section}", few_shot if few_shot else "")

        request = f'Decompose this object into 3D blocks: "{user_request}"'
        if self._is_track_query(user_request):
            request += "\n\n" + self._track_architect_addendum()
        best_payload = None
        best_score = float("-inf")
        attempts = max(1, BEST_OF_N_CANDIDATES)
        if self._is_track_query(user_request):
            attempts = max(attempts, 6)
        else:
            attempts = max(attempts, 4)
        for _ in range(attempts):
            raw = self._call_llm(prompt, request)
            payload = self._parse_json(raw)
            parts = self._sanitize_parts(payload.get("parts", []))
            score = self._score_parts(parts)
            if score > best_score:
                best_score = score
                best_payload = {"parts": parts}
        parts = (best_payload or {"parts": []}).get("parts", [])
        if not parts:
            return {"parts": []}
        repaired = self._repair_parts_with_audit(user_request, parts)
        return {"parts": repaired}

    def decompose_from_image(self, image_b64, media_type="image/png"):
        return self.decompose_from_images(
            [{"data": image_b64, "media_type": media_type}]
        )

    def decompose_from_images(self, images):
        """Template-first multi-view pipeline (same object, different angles)."""
        if not images:
            return {"parts": []}
        labeled_views = ", ".join(
            f"{i+1}:{img.get('label', 'unknown')}" for i, img in enumerate(images)
        )
        brief_text = (
            "Analyze these photos as different views of one object and produce "
            "a robust CAD-oriented design brief for reconstruction with "
            f"rectangular volumes. View labels/order: {labeled_views}"
        )
        try:
            brief_prompt = MULTIVIEW_BRIEF_PROMPT if len(images) > 1 else IMAGE_BRIEF_PROMPT
            raw_brief = self._call_llm_with_images(
                brief_prompt, brief_text, images
            )
            brief = self._parse_json(raw_brief)
        except Exception:
            # Fallback: if brief extraction fails, ask architect directly from image.
            format_hint = (
                '{"parts": [\n'
                '  {"name": "Part A", "x_min": -20, "x_max": 20, '
                '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
                '  ...\n'
                ']}'
            )
            prompt = ARCHITECT_MULTIVIEW_PROMPT.replace("{format_hint}", format_hint)
            text = (
                "Look at these images (same object from different views) and decompose "
                "the primary object into stable rectangular volumes while preserving "
                "visible proportions and cross-view consistency. "
                "Return only JSON with key parts."
            )
            raw = self._call_llm_with_images(prompt, text, images)
            direct = self._parse_json(raw)
            direct_parts = self._sanitize_parts(direct.get("parts", []))
            return {"parts": self._limit_parts_for_stability(direct_parts, max_parts=12)}

        design_brief = (brief.get("design_brief", "") or "").strip()
        object_type = (brief.get("object_type", "") or "").strip()
        props = brief.get("overall_dims_cm", {}) or brief.get("proportions", {}) or {}
        w = props.get("width")
        d = props.get("depth")
        h = props.get("height")
        symmetry = (brief.get("symmetry", "") or "").strip()
        hints = brief.get("structural_parts_hint", []) or brief.get("key_volumes", []) or []
        key_volumes_text = ", ".join(str(v) for v in hints[:8])
        target_dims = {"width": w, "depth": d, "height": h}

        # Feed a normalized engineering description into architect generation.
        enriched_request = (
            f"{design_brief}\n"
            f"Object type: {object_type}.\n"
            f"Target overall proportions (cm): width={w}, depth={d}, height={h}.\n"
            f"Symmetry: {symmetry}.\n"
            f"Key volumes: {key_volumes_text}.\n"
            "Important: preserve the global silhouette and relative part thicknesses "
            "from the photo. Use simple, stable rectangular solids."
        ).strip()

        # Candidate pool: template + vision-grounded architect + text fallback.
        candidates = []
        template_design = self._find_template_design(object_type, design_brief)
        if template_design is not None:
            template_parts = self._fit_template_parts(
                template_design.get("parts", []),
                target_dims=target_dims,
                symmetry=symmetry,
            )
            if template_parts:
                print(f"  Template-first: using '{template_design.get('description', 'template')}'")
                candidates.append(template_parts)

        vision_candidates = self._generate_multiview_candidates(
            images=images,
            enriched_request=enriched_request,
            target_dims=target_dims,
            symmetry=symmetry,
        )
        candidates.extend(vision_candidates)

        # Keep text-only branch as fallback only.
        if not candidates:
            first_pass = self.decompose_object(enriched_request)
            if first_pass.get("parts", []):
                candidates.append(first_pass.get("parts", []))

        first_parts = self._select_best_multiview_candidate(
            candidates=candidates,
            images=images,
            brief={
                "design_brief": design_brief,
                "object_type": object_type,
                "proportions": {"width": w, "depth": d, "height": h},
                "symmetry": symmetry,
                "key_volumes": hints,
            },
        )
        if not first_parts:
            return {"parts": []}

        # Universal geometry audit pass: improves proportion consistency and
        # removes unstable/spurious blocks without class-specific rules.
        audited_parts = first_parts
        try:
            audit_input = {
                "brief": {
                    "design_brief": design_brief,
                    "object_type": object_type,
                    "proportions": {"width": w, "depth": d, "height": h},
                    "symmetry": symmetry,
                    "key_volumes": hints,
                },
                "candidate_parts": first_parts,
            }
            raw_audit = self._call_llm_with_images(
                GEOMETRY_AUDIT_PROMPT,
                json.dumps(audit_input, ensure_ascii=True),
                images,
            )
            audited = self._parse_json(raw_audit)
            maybe_parts = audited.get("parts", [])
            if maybe_parts:
                audited_parts = maybe_parts
        except Exception:
            pass

        cleaned = self._sanitize_parts(
            audited_parts, target_dims=target_dims, symmetry=symmetry
        )
        features = brief.get("features")
        if not isinstance(features, list):
            features = []
        base = self._limit_parts_for_stability(cleaned, max_parts=12)
        deformed = self.apply_deformation_layer(base, features)
        final_cap = 18 if features else 12
        if len(deformed) > final_cap:
            deformed = self._limit_parts_for_stability(deformed, max_parts=final_cap)
        return {"parts": deformed}

    def _generate_multiview_candidates(self, images, enriched_request, target_dims, symmetry):
        format_hint = (
            '{"parts": [\n'
            '  {"name": "Part A", "x_min": -20, "x_max": 20, '
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
            '  ...\n'
            ']}'
        )
        prompt = ARCHITECT_MULTIVIEW_PROMPT.replace("{format_hint}", format_hint)
        text = (
            "Use all photos as views of the same object and produce a conservative "
            "structural bbox decomposition.\n\n"
            f"{enriched_request}\n\n"
            f"Require each part to be supported by at least {max(1, MULTIVIEW_MIN_CONFIRM_VIEWS)} views."
        )
        out = []
        attempts = max(1, MULTIVIEW_BEST_OF_N_CANDIDATES)
        for _ in range(attempts):
            try:
                raw = self._call_llm_with_images(prompt, text, images)
                payload = self._parse_json(raw)
            except Exception:
                continue
            parts = self._sanitize_parts(
                payload.get("parts", []), target_dims=target_dims, symmetry=symmetry
            )
            if not parts:
                continue
            parts = self._filter_parts_by_view_support(
                parts, images, min_views=MULTIVIEW_MIN_CONFIRM_VIEWS
            )
            parts = self._limit_parts_for_stability(parts, max_parts=14)
            if parts:
                out.append(parts)
        return out

    def _select_best_multiview_candidate(self, candidates, images, brief):
        if not candidates:
            return []
        best = []
        best_score = float("-inf")
        for parts in candidates:
            score = self._score_parts_with_images(parts, images, brief)
            if score > best_score:
                best_score = score
                best = parts
        return best

    def _score_parts_with_images(self, parts, images, brief):
        if not parts:
            return -10_000.0
        base = self._score_parts(parts)
        if not images:
            return base
        payload = {"brief": brief or {}, "candidate_parts": parts}
        try:
            raw = self._call_llm_with_images(
                MULTIVIEW_RANKER_PROMPT,
                json.dumps(payload, ensure_ascii=True),
                images,
            )
            data = self._parse_json(raw)
            vision_score = self._to_float(data.get("score"), base)
            return 0.45 * base + 0.55 * vision_score
        except Exception:
            return base

    def _filter_parts_by_view_support(self, parts, images, min_views=2):
        if not parts or not images or len(images) < 2:
            return parts
        payload = {
            "min_views": max(1, int(min_views)),
            "parts": parts,
        }
        try:
            raw = self._call_llm_with_images(
                VIEW_SUPPORT_FILTER_PROMPT,
                json.dumps(payload, ensure_ascii=True),
                images,
            )
            out = self._parse_json(raw)
        except Exception:
            return parts

        support_by_index = out.get("support_by_index", [])
        idx_to_views = {}
        if isinstance(support_by_index, list):
            for rec in support_by_index:
                if not isinstance(rec, dict):
                    continue
                i = rec.get("index")
                v = rec.get("views_supported")
                if isinstance(i, int):
                    idx_to_views[i] = self._to_float(v, 0.0)

        keep = []
        threshold = float(max(1, int(min_views)))
        for i, p in enumerate(parts):
            views_supported = idx_to_views.get(i)
            if views_supported is None:
                keep.append(p)
                continue
            if views_supported >= threshold:
                keep.append(p)

        # Fail-safe: do not drop into an empty model.
        return keep if keep else parts

    def _find_template_design(self, object_type, design_brief):
        query = f"{object_type} {design_brief}".strip()
        candidates = self._find_reference_designs(query, top_k=1)
        if candidates:
            return candidates[0]
        t = (object_type or "").strip().lower()
        if not t:
            return None
        for d in self.index.designs:
            for kw in d.get("keywords", []):
                if t in str(kw).lower() or str(kw).lower() in t:
                    return d
        return None

    def _fit_template_parts(self, parts, target_dims, symmetry):
        """Scale a known template to target dimensions before LLM audit."""
        if not parts:
            return []
        normalized = self._sanitize_parts(parts, target_dims=None, symmetry=symmetry)
        if not normalized:
            return []
        tw = self._to_float(target_dims.get("width"), 0.0)
        td = self._to_float(target_dims.get("depth"), 0.0)
        th = self._to_float(target_dims.get("height"), 0.0)
        cur_w = max(p["x_max"] for p in normalized) - min(p["x_min"] for p in normalized)
        cur_d = max(p["y_max"] for p in normalized) - min(p["y_min"] for p in normalized)
        cur_h = max(p["z_max"] for p in normalized) - min(p["z_min"] for p in normalized)
        sx = max(0.3, min(3.5, tw / cur_w)) if tw > 1.0 and cur_w > 0.1 else 1.0
        sy = max(0.3, min(3.5, td / cur_d)) if td > 1.0 and cur_d > 0.1 else 1.0
        sz = max(0.3, min(3.5, th / cur_h)) if th > 1.0 and cur_h > 0.1 else 1.0
        out = []
        for p in normalized:
            out.append({
                "name": p.get("name", "Part"),
                "x_min": p["x_min"] * sx,
                "x_max": p["x_max"] * sx,
                "y_min": p["y_min"] * sy,
                "y_max": p["y_max"] * sy,
                "z_min": p["z_min"] * sz,
                "z_max": p["z_max"] * sz,
            })
        final_parts = self._sanitize_parts(out, target_dims=target_dims, symmetry=symmetry)
        return self._limit_parts_for_stability(final_parts, max_parts=12)

    # -- Step B: Deterministic encoder вЂ" bboxes в†’ Gym commands ----------------

    @staticmethod
    def encode(parts, include_clear=True):
        return encode_bboxes_to_plan(parts, include_clear)

    @staticmethod
    def _format_build_step_line(step):
        def fnum(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return float(default)
        shape = str(step.get("shape", "rect")).lower()
        base = (
            f"{step.get('description', 'part'):30s}  {step.get('plane', 'XY')}  "
            f"cx={fnum(step.get('cx', 0.0)):6.1f}  cy={fnum(step.get('cy', 0.0)):6.1f}  "
        )
        if shape == "circle":
            return base + f"r={fnum(step.get('radius', 0.0)):6.1f}  d={fnum(step.get('distance', 0.0)):7.1f}"
        if shape == "polygon":
            return (
                base +
                f"r={fnum(step.get('radius', 0.0)):6.1f}  n={int(fnum(step.get('sides', 0), 0))}  "
                f"d={fnum(step.get('distance', 0.0)):7.1f}"
            )
        if shape in ("ring", "gear"):
            return (
                base +
                f"ro={fnum(step.get('outer_radius', 0.0)):5.1f}  ri={fnum(step.get('inner_radius', 0.0)):5.1f}  "
                f"d={fnum(step.get('distance', 0.0)):7.1f}"
            )
        return (
            base +
            f"w={fnum(step.get('w', 0.0)):6.1f}  h={fnum(step.get('h', 0.0)):6.1f}  "
            f"d={fnum(step.get('distance', 0.0)):7.1f}"
        )

    # -- intent: extend existing vs build new ---------------------------------

    @staticmethod
    def _intent_extend_or_new(user_input, has_existing_parts):
        """Classify request: 'extend' = modify/add to current model; 'new' = build from scratch."""
        if not has_existing_parts:
            return "new"
        text = (user_input or "").strip().lower()
        extend_cues = (
            "add ", "РґРѕР±Р°РІСЊ", "extend", "to existing", "to current", "to the current",
            "also add", "make the ", "change the ", "modify", "higher", "longer", "wider",
            "РґРѕСЃС‚СЂРѕР№", "РґРѕРїРѕР»РЅРё", "РµС‰С‘ ", "РїРѕРґР»РѕРєРѕС‚РЅРёРє", "СЂСѓС‡РєРё", "РЅР° С‚РµРєСѓС‰", "Рє С‚РµРєСѓС‰",
            "РІ С‚РµРєСѓС‰", "С‚РµРєСѓС‰СѓСЋ ", "С‚РµРєСѓС‰РёР№ ", "СЌС‚Сѓ РјРѕРґРµР»СЊ", "СЌС‚РѕС‚ ", "СЃСѓС‰РµСЃС‚РІСѓСЋС‰",
            "РЅР° СЌС‚РѕРј", "Рє СЌС‚РѕРјСѓ", "armrest", "backrest", "СЃРїРёРЅРєСѓ", "СЃРёРґРµРЅСЊРµ РІС‹С€Рµ",
        )
        new_cues = (
            "from scratch", "СЃ РЅСѓР»СЏ", "Р·Р°РЅРѕРІРѕ", "new ", "another ", "different ",
            "РЅРѕРІС‹Р№ ", "РґСЂСѓРіРѕР№ ", "build a new", "create a new", "start over",
        )
        for c in new_cues:
            if c in text:
                return "new"
        for c in extend_cues:
            if c in text:
                return "extend"
        return "extend"

    @staticmethod
    def _should_auto_detail(user_input):
        """Enable auto-detail only for explicit detail intent."""
        hay = str(user_input or "").lower()
        cues = (
            "detail", "details", "detailing", "refine", "refinement",
            "add details", "more detail", "mechanical detail",
            "детал", "проработ", "уточни детали", "добавь детали",
        )
        return any(c in hay for c in cues)

    @staticmethod
    def _classify_create_shape_family(user_request, images=None):
        specs = plan_structural_specs(
            str(user_request or ""),
            images=images or [],
            capabilities=default_capability_model(),
        )
        if not specs:
            return "furniture_boxy_panel"
        return str((specs[0] or {}).get("object_family") or "furniture_boxy_panel")

    def _select_structural_specs_for_create(self, user_request, images=None):
        cap_model = default_capability_model()
        raw_specs = plan_structural_specs(
            str(user_request or ""),
            images=images or [],
            capabilities=cap_model,
        )
        valid_specs = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                continue
            errors = validate_structural_spec(spec)
            if errors:
                continue
            valid_specs.append(spec)
        if not valid_specs:
            return []
        ranked = rank_structural_specs(valid_specs, str(user_request or ""), capabilities=cap_model)
        return [item.get("spec") for item in ranked if isinstance(item, dict) and isinstance(item.get("spec"), dict)]

    def _build_create_policy(self, user_request, images=None, structural_spec=None):
        spec = structural_spec if isinstance(structural_spec, dict) else {}
        if not spec:
            ranked_specs = self._select_structural_specs_for_create(user_request, images=images)
            if ranked_specs:
                spec = ranked_specs[0]
        if not spec:
            family = "furniture_boxy_panel"
            outcome = "blocked"
            allowed_primitives = []
            approx = "blocked"
            reason = "structural planner failed to produce a valid spec"
        else:
            family = str(spec.get("object_family") or "furniture_boxy_panel")
            outcome = str(spec.get("build_outcome_target") or "exact")
            allowed_primitives = [str(p).strip().lower() for p in (spec.get("allowed_primitives") or []) if str(p).strip()]
            approx = str((spec.get("approximation_policy") or {}).get("mode") or "none")
            reason = str(spec.get("blocked_reason") or "").strip()

        return {
            "family": family,
            "outcome": outcome,
            "allowed_primitives": allowed_primitives,
            "approximation_strategy": approx,
            "reason": reason,
        }

    @staticmethod
    def _create_family_prompt_addendum(policy, structural_spec=None):
        family = str((policy or {}).get("family") or "")
        outcome = str((policy or {}).get("outcome") or "exact")
        approx = str((policy or {}).get("approximation_strategy") or "none")
        reason = str((policy or {}).get("reason") or "").strip()
        spec = dict(structural_spec or {})

        common = (
            "Structural Build Spec is the primary contract for create mode.\n"
            f"Create family: {family}.\n"
            f"Target outcome: {outcome}.\n"
            f"Approximation strategy: {approx}.\n"
        )
        if reason:
            common += f"Capability note: {reason}.\n"
        if spec:
            common += "Structural Build Spec JSON:\n"
            common += json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
            common += "Synthesize CAD Plan from this spec. Preserve build_order and role intent.\n"
        return common

    @staticmethod
    def _plan_primitive_counts(plan):
        counts = {}
        for step in (plan.get("steps") or []):
            if not isinstance(step, dict):
                continue
            prim = str(step.get("primitive") or "").strip().lower()
            if not prim:
                continue
            counts[prim] = counts.get(prim, 0) + 1
        return counts

    def _score_create_candidate_for_family(self, plan, family):
        counts = self._plan_primitive_counts(plan)
        rect = counts.get("rect_extrude", 0)
        circle = counts.get("circle_extrude", 0)
        poly = counts.get("poly_extrude", 0)
        wedge = counts.get("wedge_extrude", 0)
        cut = counts.get("cut_extrude", 0)
        steps_n = len(plan.get("steps") or [])

        score = 0.0
        if family == "furniture_boxy_panel":
            score += 2.0 * rect + 1.0 * cut + 0.5 * circle
            score -= 1.5 * max(0, poly + wedge - rect - 1)
        elif family == "lowpoly_hard_surface_vehicle":
            score += 2.0 * (poly + wedge) + 1.2 * circle + 0.4 * rect + 0.8 * cut
            if rect > (poly + wedge + 2):
                score -= 3.0
        elif family == "rotational_bodies":
            score += 2.2 * circle + 1.0 * poly + 0.4 * cut
            if rect > circle + 2:
                score -= 2.5
        elif family == "profile_driven_symmetric":
            score += 1.8 * poly + 1.2 * rect + 0.9 * cut + 0.5 * circle

        if steps_n < 3:
            score -= 2.0
        if steps_n > 18:
            score -= float(steps_n - 18) * 0.6
        return score

    def _candidate_allowed_by_create_policy(self, plan, policy):
        cap_model = default_capability_model()
        allowed = set((policy or {}).get("allowed_primitives") or [])
        for step in (plan.get("steps") or []):
            if not isinstance(step, dict):
                return False
            primitive = str(step.get("primitive") or "").strip().lower()
            operation = str(step.get("operation") or "").strip()
            if primitive not in allowed:
                return False
            required = capability_required_for_primitive(primitive, operation)
            if required == "unsupported":
                return False
            if not cap_model.supports(required):
                return False
        return True

    @staticmethod
    def _attach_create_policy_metadata(plan, policy, structural_spec=None):
        if not isinstance(plan, dict):
            return plan
        global_cfg = plan.get("global")
        if not isinstance(global_cfg, dict):
            global_cfg = {}
        p = dict(policy or {})
        global_cfg["shape_family"] = p.get("family", "")
        global_cfg["build_outcome_target"] = p.get("outcome", "exact")
        global_cfg["approximation_policy"] = {
            "strategy": p.get("approximation_strategy", ""),
            "reason": p.get("reason", ""),
        }
        if isinstance(structural_spec, dict):
            global_cfg["structural_spec"] = {
                "object_family": structural_spec.get("object_family"),
                "build_outcome_target": structural_spec.get("build_outcome_target"),
                "construction_strategy": structural_spec.get("construction_strategy"),
                "build_order": list(structural_spec.get("build_order") or []),
                "main_roles": [
                    (part or {}).get("role")
                    for part in (structural_spec.get("main_masses") or [])
                    if isinstance(part, dict)
                ],
            }
        plan["global"] = global_cfg
        return plan

    def _prepare_create_candidates(self, candidates, policy, structural_spec=None):
        family = str((policy or {}).get("family") or "")
        cap_model = default_capability_model()
        prepared = []
        for plan in candidates:
            if not isinstance(plan, dict):
                continue
            if not self._candidate_allowed_by_create_policy(plan, policy):
                continue
            annotated = self._attach_create_policy_metadata(plan, policy, structural_spec=structural_spec)
            family_score = self._score_create_candidate_for_family(annotated, family)
            structural_score = 0.0
            structural_breakdown = None
            if isinstance(structural_spec, dict):
                ranked = rank_dsl_candidates([annotated], structural_spec, capabilities=cap_model)
                if ranked:
                    structural_score = float((ranked[0] or {}).get("score", 0.0))
                    structural_breakdown = (ranked[0] or {}).get("breakdown")
            combined = family_score + (3.0 * structural_score)
            if isinstance(structural_breakdown, dict):
                annotated.setdefault("global", {})
                if isinstance(annotated["global"], dict):
                    annotated["global"]["structural_rank"] = structural_breakdown
            prepared.append((combined, annotated))
        prepared.sort(key=lambda row: row[0], reverse=True)
        return [p for _, p in prepared]

    def _create_policy_failure_reason(self):
        policy = self.last_create_policy if isinstance(self.last_create_policy, dict) else {}
        if not policy:
            return ""
        if str(policy.get("outcome", "")) == "blocked":
            return str(policy.get("reason", "")).strip()
        return ""

    @staticmethod
    def _structural_roles_summary(spec):
        if not isinstance(spec, dict):
            return {"expected_count": 0, "main": [], "supporting": [], "secondary": []}
        main = [str((p or {}).get("role") or "").strip() for p in (spec.get("main_masses") or []) if isinstance(p, dict)]
        supporting = [str((p or {}).get("role") or "").strip() for p in (spec.get("supporting_parts") or []) if isinstance(p, dict)]
        secondary = [str((p or {}).get("role") or "").strip() for p in (spec.get("secondary_parts") or []) if isinstance(p, dict)]
        main = [r for r in main if r]
        supporting = [r for r in supporting if r]
        secondary = [r for r in secondary if r]
        return {
            "expected_count": len(main) + len(supporting) + len(secondary),
            "main": main,
            "supporting": supporting,
            "secondary": secondary,
        }

    def _print_create_policy_summary(self, best_plan=None):
        policy = self.last_create_policy if isinstance(self.last_create_policy, dict) else {}
        spec = self.last_structural_spec if isinstance(self.last_structural_spec, dict) else {}
        if not policy:
            return
        print(
            "  Create policy: "
            f"family={policy.get('family', '')} "
            f"outcome={policy.get('outcome', '')} "
            f"strategy={policy.get('approximation_strategy', '')}"
        )
        grammar = ""
        if isinstance(best_plan, dict):
            grammar = str(((best_plan.get("global") or {}).get("synthesis_grammar")) or "").strip()
        if not grammar and spec:
            grammar = str(spec.get("construction_strategy") or "").strip()
        if grammar:
            print(f"  Synthesis grammar: {grammar}")

        roles = self._structural_roles_summary(spec)
        if roles["expected_count"] > 0:
            print(
                "  Roles expected: "
                f"{roles['expected_count']} "
                f"(main={len(roles['main'])}, supporting={len(roles['supporting'])}, secondary={len(roles['secondary'])})"
            )
        if policy.get("reason"):
            print(f"  Policy reason: {policy.get('reason')}")

        if isinstance(best_plan, dict):
            global_cfg = best_plan.get("global") if isinstance(best_plan.get("global"), dict) else {}
            role_cov = global_cfg.get("structural_role_coverage") if isinstance(global_cfg.get("structural_role_coverage"), dict) else {}
            rank = global_cfg.get("structural_rank") if isinstance(global_cfg.get("structural_rank"), dict) else {}
            safety = global_cfg.get("create_safety") if isinstance(global_cfg.get("create_safety"), dict) else {}
            coverage_ratio = role_cov.get("coverage_ratio")
            if coverage_ratio is None:
                coverage_ratio = rank.get("structural_coverage")
            if coverage_ratio is not None:
                role_to_step = rank.get("role_to_step_coherence")
                attachment = rank.get("attachment_plausibility")
                details = []
                if role_to_step is not None:
                    details.append(f"role_to_step={float(role_to_step):.2f}")
                if attachment is not None:
                    details.append(f"attachment={float(attachment):.2f}")
                if safety:
                    details.append(
                        "safety="
                        + str(safety.get("mode", ""))
                        + f"(pruned={int(float(safety.get('risky_pruned_count', 0) or 0))},"
                        + f"kept={int(float(safety.get('risky_retained_count', 0) or 0))})"
                    )
                details_s = (" " + " ".join(details)) if details else ""
                print(f"  Role coverage summary: coverage={float(coverage_ratio):.2f}{details_s}")

    def _print_iterative_create_summary(self, loop_trace):
        rows = [r for r in (loop_trace or []) if isinstance(r, dict)]
        if not rows:
            return
        print(f"  Iterative create loop: {len(rows)} cycles")
        for row in rows[:10]:
            it = int(row.get("iteration", 0) or 0)
            action = str(row.get("action", "") or "")
            phase = str(row.get("phase", "") or "")
            roles = ",".join(row.get("selected_roles", []) or [])
            reason = str(row.get("reason", "") or "")
            ids = ",".join(row.get("executed_step_ids", []) or [])
            msg = f"    - iter={it} action={action}"
            if phase:
                msg += f" phase={phase}"
            if roles:
                msg += f" roles=[{roles}]"
            if ids:
                msg += f" steps=[{ids}]"
            if reason:
                msg += f" reason={reason}"
            print(msg)

    def _sanitize_cad_plan(self, raw_plan, user_request, mode="create"):
        if not isinstance(raw_plan, dict):
            return None
        session = str(raw_plan.get("session", "")).strip() or f"dsl_{int(time.time())}"
        allowed_primitives = {str(p).strip().lower() for p in CAD_DSL_PRIMITIVES}
        allowed_operations = {
            "NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation"
        }
        allowed_modes = {"create", "edit"}

        in_steps = raw_plan.get("steps", [])
        if not isinstance(in_steps, list):
            return None

        clean_steps = []
        used_ids = set()

        for i, s in enumerate(in_steps):
            if not isinstance(s, dict):
                continue
            step = dict(s)
            sid = str(step.get("id", "")).strip() or f"step_{i+1}"
            if sid in used_ids:
                sid = f"{sid}_{i+1}"
            used_ids.add(sid)

            primitive = str(step.get("primitive", "rect_extrude")).strip().lower()
            is_known_primitive = primitive in allowed_primitives

            op = str(step.get("op", "ensure")).strip().lower()
            if op not in ("ensure", "create", "update"):
                op = "ensure"

            plane_base, plane_off = self._parse_plane_and_offset(step.get("plane", "XY@0"))
            plane = f"{plane_base}@{round(plane_off, 3)}" if abs(plane_off) > 1e-9 else plane_base

            operation = str(step.get("operation", "NewBodyFeatureOperation")).strip()
            if primitive == "cut_extrude":
                operation = "CutFeatureOperation"
            if operation not in allowed_operations:
                operation = "NewBodyFeatureOperation"

            distance = abs(self._to_float(step.get("distance", 1.0), 1.0))
            if distance < 0.2:
                distance = 0.2

            profile = step.get("profile")
            if not isinstance(profile, dict):
                profile = {}

            ptype = str(profile.get("type", "")).strip().lower()
            if primitive in ("rect_extrude", "cut_extrude"):
                if ptype not in ("", "rect"):
                    ptype = "rect"
                profile = {
                    "type": "rect",
                    "cx": round(self._to_float(profile.get("cx", 0.0), 0.0), 3),
                    "cy": round(self._to_float(profile.get("cy", 0.0), 0.0), 3),
                    "w": round(max(0.2, abs(self._to_float(profile.get("w", 10.0), 10.0))), 3),
                    "h": round(max(0.2, abs(self._to_float(profile.get("h", 10.0), 10.0))), 3),
                }
            elif primitive == "circle_extrude":
                profile = {
                    "type": "circle",
                    "cx": round(self._to_float(profile.get("cx", 0.0), 0.0), 3),
                    "cy": round(self._to_float(profile.get("cy", 0.0), 0.0), 3),
                    "radius": round(max(0.1, abs(self._to_float(profile.get("radius", 2.0), 2.0))), 3),
                }
            elif primitive in ("poly_extrude", "wedge_extrude"):
                pts = profile.get("pts", []) if isinstance(profile.get("pts", []), list) else []
                norm_pts = []
                for p in pts:
                    if not isinstance(p, dict):
                        continue
                    if "x" not in p or "y" not in p:
                        continue
                    norm_pts.append(
                        {"x": round(self._to_float(p.get("x"), 0.0), 3), "y": round(self._to_float(p.get("y"), 0.0), 3)}
                    )
                if len(norm_pts) < 3:
                    norm_pts = [
                        {"x": -2.0, "y": -2.0},
                        {"x": 2.0, "y": -2.0},
                        {"x": 2.0, "y": 2.0},
                        {"x": -2.0, "y": 2.0},
                    ]
                profile = {"type": "poly", "pts": norm_pts}
            else:
                # Keep advanced/unknown primitive payload as-is for honest validator rejection.
                if not isinstance(profile, dict):
                    profile = {}

            clean_step = {
                "id": sid,
                "op": op,
                "primitive": primitive if is_known_primitive else str(step.get("primitive", "")).strip().lower(),
                "plane": plane,
                "profile": profile,
                "distance": round(distance, 3),
                "operation": operation,
            }
            if isinstance(step.get("names"), dict):
                clean_step["names"] = dict(step.get("names") or {})
            if isinstance(step.get("selectors"), dict):
                clean_step["selectors"] = dict(step.get("selectors") or {})
            if primitive == "loft":
                clean_step["loft"] = dict(step.get("loft") or {})
            elif primitive == "sweep":
                clean_step["sweep"] = dict(step.get("sweep") or {})
            elif primitive == "fillet":
                clean_step["fillet"] = dict(step.get("fillet") or {})
            elif primitive == "chamfer":
                clean_step["chamfer"] = dict(step.get("chamfer") or {})
            elif primitive == "revolve":
                clean_step["revolve"] = dict(step.get("revolve") or {})

            clean_steps.append(clean_step)

        if not clean_steps:
            return None

        budget_raw = raw_plan.get("budget", {}) if isinstance(raw_plan.get("budget"), dict) else {}
        max_steps = int(max(1, min(40, self._to_float(budget_raw.get("max_steps", 25), 25))))
        max_parts = int(max(1, min(40, self._to_float(budget_raw.get("max_parts", 18), 18))))
        requested_mode = str(mode or "create").strip().lower()
        if requested_mode not in allowed_modes:
            requested_mode = "create"
        plan = {
            "units": "cm",
            "session": session,
            "mode": requested_mode,
            "budget": {"max_steps": max_steps, "max_parts": max_parts},
            "global": raw_plan.get("global", {"source": "cad_dsl_text_planner"}),
            "steps": clean_steps[:max_steps],
        }
        if not isinstance(plan["global"], dict):
            plan["global"] = {"source": "cad_dsl_text_planner"}
        plan["global"].setdefault("request", str(user_request or "")[:240])
        return plan

    def _resolve_create_structural_context(self, user_request, images=None):
        specs = self._select_structural_specs_for_create(user_request, images=images)
        if not specs:
            self.last_structural_spec = None
            self.last_create_policy = {
                "family": "",
                "outcome": "blocked",
                "allowed_primitives": [],
                "approximation_strategy": "blocked",
                "reason": "structural planner produced no valid spec candidates",
            }
            return self.last_create_policy, []

        best_spec = specs[0]
        self.last_structural_spec = dict(best_spec)
        create_policy = self._build_create_policy(user_request, images=images, structural_spec=best_spec)
        self.last_create_policy = dict(create_policy)
        return create_policy, specs

    def _synthesize_candidates_from_structural_specs(self, user_request, mode, specs, images=None):
        req = str(user_request or "").strip()
        if not req or not specs:
            return []
        n = max(1, min(8, int(DSL_CANDIDATE_COUNT)))
        cap_model = default_capability_model()
        use_specs = [s for s in specs[:2] if isinstance(s, dict)]
        if not use_specs:
            return []

        # Primary path: deterministic family grammar synthesis.
        synthesized = []
        per_spec = max(1, n // len(use_specs))
        for spec in use_specs:
            fam_candidates = synthesize_dsl_candidates_from_structural_spec(
                spec,
                user_request=req,
                candidate_count=per_spec,
                mode=mode,
                capabilities=cap_model,
            )
            for plan in fam_candidates:
                plan = self._sanitize_cad_plan(plan, req, mode=mode)
                if isinstance(plan, dict):
                    plan.setdefault("global", {})
                    if isinstance(plan["global"], dict):
                        plan["global"]["structural_spec_family"] = spec.get("object_family", "")
                        plan["global"].setdefault("source", "structural_spec_family_grammar")
                    synthesized.append(plan)

        # Optional augmentation: one LLM candidate per top spec, kept only if capability-safe.
        llm_augmented = []
        for spec in use_specs:
            create_policy = self._build_create_policy(req, images=images, structural_spec=spec)
            addendum = self._create_family_prompt_addendum(create_policy, structural_spec=spec)
            request_text = (
                f"User request: {req}\n"
                f"Mode: {mode}\n"
                + addendum
                + "Output one complete CAD Plan JSON."
            )
            try:
                if images:
                    raw = self._call_llm_with_images(CAD_DSL_VISUAL_PLANNER_PROMPT, request_text, images)
                else:
                    raw = self._call_llm(CAD_DSL_PLANNER_PROMPT, request_text)
                payload = self._parse_json(raw)
                plan = self._sanitize_cad_plan(payload, req, mode=mode)
                if isinstance(plan, dict):
                    if self._candidate_allowed_by_create_policy(plan, create_policy):
                        plan.setdefault("global", {})
                        if isinstance(plan["global"], dict):
                            plan["global"]["structural_spec_family"] = spec.get("object_family", "")
                            plan["global"].setdefault("source", "structural_spec_llm_augmented")
                        llm_augmented.append(plan)
            except Exception:
                pass

        # Grammar plans are primary; LLM candidates are supplemental.
        return synthesized + llm_augmented

    def _plan_text_to_cad_candidates(self, user_request, mode="create"):
        """Generate CAD DSL candidates from structural build specs (create) or direct LLM (edit)."""
        req = str(user_request or "").strip()
        if not req:
            return []
        mode_norm = str(mode or "create").strip().lower()
        if mode_norm == "create":
            create_policy, specs = self._resolve_create_structural_context(req, images=None)
            if create_policy.get("outcome") == "blocked":
                return []
            candidates = self._synthesize_candidates_from_structural_specs(req, mode, specs, images=None)
            return self._prepare_create_candidates(candidates, create_policy, structural_spec=specs[0] if specs else None)

        self.last_create_policy = None
        self.last_structural_spec = None
        n = max(1, min(8, int(DSL_CANDIDATE_COUNT)))
        candidates = []
        request_text = (
            f"User request: {req}\n"
            f"Mode: {mode}\n"
            "Output one complete CAD Plan JSON."
        )
        for _ in range(n):
            try:
                raw = self._call_llm(CAD_DSL_PLANNER_PROMPT, request_text)
                payload = self._parse_json(raw)
                plan = self._sanitize_cad_plan(payload, req, mode=mode)
                if plan is not None:
                    candidates.append(plan)
            except Exception:
                continue
        return candidates

    def _plan_images_to_cad_candidates(self, user_request, images, mode="create"):
        req = str(user_request or "").strip()
        if not req or not images:
            return []
        mode_norm = str(mode or "create").strip().lower()
        if mode_norm == "create":
            create_policy, specs = self._resolve_create_structural_context(req, images=images)
            if create_policy.get("outcome") == "blocked":
                return []
            candidates = self._synthesize_candidates_from_structural_specs(req, mode, specs, images=images)
            return self._prepare_create_candidates(candidates, create_policy, structural_spec=specs[0] if specs else None)

        self.last_create_policy = None
        self.last_structural_spec = None
        n = max(1, min(8, int(DSL_CANDIDATE_COUNT)))
        candidates = []
        request_text = (
            f"User request: {req}\n"
            f"Mode: {mode}\n"
            "Build one physically plausible CAD plan from these images.\n"
            + "Output one complete CAD Plan JSON."
        )
        for _ in range(n):
            try:
                raw = self._call_llm_with_images(CAD_DSL_VISUAL_PLANNER_PROMPT, request_text, images)
                payload = self._parse_json(raw)
                plan = self._sanitize_cad_plan(payload, req, mode=mode)
                if plan is not None:
                    candidates.append(plan)
            except Exception:
                continue
        return candidates

    def _legacy_bboxes_to_cad_plan(self, parts, user_request, mode="create"):
        try:
            structural = structural_from_legacy_bboxes(parts or [])
            planner = LegacyBboxCadPlanner()
            raw_plan = planner.plan(structural, session=f"legacy_{int(time.time())}", mode=mode)
            plan = self._sanitize_cad_plan(raw_plan, user_request, mode=mode)
            if isinstance(plan, dict):
                plan.setdefault("global", {})
                if isinstance(plan["global"], dict):
                    plan["global"]["source"] = "legacy_bbox_fallback"
            return plan
        except Exception:
            return None

    @staticmethod
    def _print_cad_trace(result, max_events=16):
        if not result or not getattr(result, "trace", None):
            return
        print("  Trace:")
        for ev in list(result.trace)[:max_events]:
            path = " -> ".join(getattr(ev, "backend_path", []) or [])
            reason = getattr(ev, "reason", "") or ""
            line = (
                f"    - step={getattr(ev, 'step_id', '')} "
                f"primitive={getattr(ev, 'primitive', '')} "
                f"action={getattr(ev, 'action', '')}"
            )
            if path:
                line += f" backend={path}"
            if reason:
                line += f" reason={reason}"
            print(line)

    def _execute_cad_dsl_best(self, candidates, mode="create", previous_plan=None, user_request=""):
        if not candidates:
            return False, None, None

        cap_model = default_capability_model()
        mode_norm = str(mode or "create").strip().lower()
        self.last_iterative_loop_trace = []
        self.last_iterative_state_snapshots = []
        if mode_norm == "edit":
            valid_candidates = []
            for p in candidates:
                vr = validate_plan(p, capabilities=cap_model, allow_unsupported=False)
                if vr.valid:
                    valid_candidates.append(p)
            if not valid_candidates:
                print("  CAD DSL planner: no valid edit candidates.")
                return False, None, None
            best_plan = valid_candidates[0]
            best_idx = candidates.index(best_plan)
            result = run_cad_operator(
                self.fusion,
                best_plan,
                dry_run=False,
                step_delay=STEP_DELAY,
                edit_mode=True,
                previous_plan=previous_plan,
                capabilities=cap_model,
            )
        else:
            structural_spec = self.last_structural_spec if isinstance(self.last_structural_spec, dict) else {}
            request_text = str(user_request or self.last_request or "").strip()
            if structural_spec:
                agent = IterativeCreateAgent(
                    self.fusion,
                    capabilities=cap_model,
                    step_delay=STEP_DELAY,
                )
                outcome = agent.run(
                    user_request=request_text,
                    structural_spec=structural_spec,
                    candidates=candidates,
                )
                best_idx = outcome.best_index
                best_plan = outcome.best_plan
                result = outcome.result
                self.last_iterative_loop_trace = list(outcome.loop_trace or [])
                self.last_iterative_state_snapshots = list(outcome.state_snapshots or [])
                print("  Create execution mode: iterative tool-driven CAD agent")
                self._print_iterative_create_summary(self.last_iterative_loop_trace)
            else:
                # Fallback only when structural context is unavailable.
                try:
                    self.fusion.clear()
                    self.fusion.refresh()
                except Exception:
                    pass
                best_idx, best_plan, result = run_best_of_n_create(
                    self.fusion,
                    candidates,
                    dry_run=False,
                    step_delay=STEP_DELAY,
                    pre_render_fix_fn=pre_render_fix,
                    call_llm=self._call_llm,
                    capabilities=cap_model,
                )
                if best_idx < 0:
                    print("  CAD DSL planner: no valid candidates.")
                    return False, None, result

        if result.success:
            print(f"  CAD DSL execution OK: {result.message}")
            print(f"  Candidate selected: {best_idx + 1}/{len(candidates)}")
            self._print_cad_trace(result)
            return True, best_plan, result
        print(f"  CAD DSL execution failed: {result.message}")
        self._print_cad_trace(result)
        return False, best_plan, result

    def _revise_model_parts(self, current_parts, user_request):
        """Given current bbox list and user request to extend/modify, return full updated parts list."""
        parts_summary = "\n".join(
            f"  {p['name']}: x_min={p['x_min']}, x_max={p['x_max']}, "
            f"y_min={p['y_min']}, y_max={p['y_max']}, z_min={p['z_min']}, z_max={p['z_max']}"
            for p in current_parts
        )
        format_hint = (
            '{"parts": [\n'
            '  {"name": "Part A", "x_min": -20, "x_max": 20, '
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
            '  ...\n'
            ']}'
        )
        prompt = EXTEND_REVISION_PROMPT.replace("{format_hint}", format_hint)
        request_text = (
            f"Current model parts (cm):\n{parts_summary}\n\n"
            f"User request: {user_request}\n\n"
            "Return the COMPLETE updated parts list (JSON with key 'parts')."
        )
        raw = self._call_llm(prompt, request_text)
        out = self._parse_json(raw)
        parts = out.get("parts", [])
        if not parts:
            return []
        parts = self._sanitize_parts(parts, normalize_frame=False)
        return self._limit_parts_for_stability(parts, max_parts=max(12, len(current_parts) + 4))

    def _detail_model_parts(self, current_parts, user_request, images=None, max_parts=None):
        """Add functional details on top of current model while preserving global shape."""
        if not current_parts:
            return []
        parts_summary = "\n".join(
            f"  {p['name']}: x_min={p['x_min']}, x_max={p['x_max']}, "
            f"y_min={p['y_min']}, y_max={p['y_max']}, z_min={p['z_min']}, z_max={p['z_max']}"
            for p in current_parts
        )
        format_hint = (
            '{"parts": [\n'
            '  {"name": "Part A", "x_min": -20, "x_max": 20, '
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
            '  ...\n'
            ']}'
        )
        prompt = DETAILING_PROMPT.replace("{format_hint}", format_hint)
        request_text = (
            f"Current model parts (cm):\n{parts_summary}\n\n"
            f"Context/request: {user_request}\n\n"
            "Add missing functional details while preserving silhouette and connectivity. "
            "Return the COMPLETE updated parts list."
        )
        if images:
            raw = self._call_llm_with_images(prompt, request_text, images)
        else:
            raw = self._call_llm(prompt, request_text)
        out = self._parse_json(raw)
        parts = out.get("parts", [])
        if not parts:
            return []
        parts = self._sanitize_parts(parts, normalize_frame=False)
        cap = max_parts if isinstance(max_parts, int) and max_parts > 0 else max(16, len(current_parts) + 8)
        return self._limit_parts_for_stability(parts, max_parts=cap)

    @staticmethod
    def _parse_plane_and_offset(plane):
        s = str(plane or "XY").strip().upper()
        if "@" in s:
            base, raw = s.split("@", 1)
            base = base.strip()
            try:
                off = float(raw)
            except Exception:
                off = 0.0
        else:
            base, off = s, 0.0
        if base not in ("XY", "XZ", "YZ"):
            base = "XY"
            off = 0.0
        return base, off

    def _steps_to_bboxes(self, steps):
        out = []
        for i, st in enumerate(steps):
            if str(st.get("action", "")).lower() != "build":
                continue
            shape = str(st.get("shape", "rect")).lower()
            plane_base, plane_off = self._parse_plane_and_offset(st.get("plane", "XY"))
            repeat = int(max(1, min(32, self._to_float(st.get("repeat"), 1))))
            dx_rep = self._to_float(st.get("dx"), 0.0)
            dy_rep = self._to_float(st.get("dy"), 0.0)
            dz_rep = self._to_float(st.get("dz"), 0.0)
            dist = abs(self._to_float(st.get("distance"), 0.0))
            if dist <= 0.1:
                continue
            name = str(st.get("description", f"detail_{i+1}")).strip() or f"detail_{i+1}"
            base_cx = self._to_float(st.get("cx"), 0.0)
            base_cy = self._to_float(st.get("cy"), 0.0)
            for r_idx in range(repeat):
                off = plane_off + dz_rep * r_idx
                off1 = off + dist
                cx = base_cx + dx_rep * r_idx
                cy = base_cy + dy_rep * r_idx
                item_name = f"{name}_{r_idx+1}" if repeat > 1 else name
                if shape in ("circle", "ring", "gear", "polygon"):
                    if shape == "circle":
                        rad = abs(self._to_float(st.get("radius"), 0.0))
                    elif shape == "polygon":
                        rad = abs(self._to_float(st.get("radius"), 0.0))
                    else:
                        rad = abs(self._to_float(st.get("outer_radius"), 0.0))
                    if rad <= 0.1:
                        continue
                    if plane_base == "XY":
                        out.append({
                            "name": item_name,
                            "x_min": cx - rad, "x_max": cx + rad,
                            "y_min": cy - rad, "y_max": cy + rad,
                            "z_min": min(off, off1), "z_max": max(off, off1),
                        })
                    elif plane_base == "XZ":
                        out.append({
                            "name": item_name,
                            "x_min": cx - rad, "x_max": cx + rad,
                            "y_min": min(off, off1), "y_max": max(off, off1),
                            "z_min": cy - rad, "z_max": cy + rad,
                        })
                    else:  # YZ
                        out.append({
                            "name": item_name,
                            "x_min": min(off, off1), "x_max": max(off, off1),
                            "y_min": cx - rad, "y_max": cx + rad,
                            "z_min": cy - rad, "z_max": cy + rad,
                        })
                else:
                    w = abs(self._to_float(st.get("w"), 0.0))
                    h = abs(self._to_float(st.get("h"), 0.0))
                    if w <= 0.1 or h <= 0.1:
                        continue
                    if plane_base == "XY":
                        out.append({
                            "name": item_name,
                            "x_min": cx - w / 2.0, "x_max": cx + w / 2.0,
                            "y_min": cy - h / 2.0, "y_max": cy + h / 2.0,
                            "z_min": min(off, off1), "z_max": max(off, off1),
                        })
                    elif plane_base == "XZ":
                        out.append({
                            "name": item_name,
                            "x_min": cx - w / 2.0, "x_max": cx + w / 2.0,
                            "y_min": min(off, off1), "y_max": max(off, off1),
                            "z_min": cy - h / 2.0, "z_max": cy + h / 2.0,
                        })
                    else:  # YZ
                        out.append({
                            "name": item_name,
                            "x_min": min(off, off1), "x_max": max(off, off1),
                            "y_min": cx - w / 2.0, "y_max": cx + w / 2.0,
                            "z_min": cy - h / 2.0, "z_max": cy + h / 2.0,
                        })
        return out

    def _summarize_reconstruction_json(self, json_path):
        summary = {
            "extrude_total": 0,
            "new_body": 0,
            "join": 0,
            "cut": 0,
            "sketch_circle_count": 0,
            "sketch_line_count": 0,
        }
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return summary
        entities = data.get("entities", {}) or {}
        for ent in entities.values():
            if not isinstance(ent, dict):
                continue
            t = str(ent.get("type", ""))
            if t == "ExtrudeFeature":
                summary["extrude_total"] += 1
                op = str(ent.get("operation", ""))
                if op == "NewBodyFeatureOperation":
                    summary["new_body"] += 1
                elif op == "JoinFeatureOperation":
                    summary["join"] += 1
                elif op == "CutFeatureOperation":
                    summary["cut"] += 1
            if t == "Sketch":
                curves = ent.get("curves", {}) or {}
                for c in curves.values():
                    ct = str((c or {}).get("type", ""))
                    if "Circle" in ct:
                        summary["sketch_circle_count"] += 1
                    elif "Line" in ct:
                        summary["sketch_line_count"] += 1
        return summary

    def _dataset_action_hints(self, query, top_k=2):
        hints = []
        local_top_k = max(top_k, 4) if self._is_track_query(query) else top_k
        for d in self._find_reference_designs(query, top_k=local_top_k):
            src = str(d.get("source_file", "")).strip()
            rec_summary = {}
            p = self.find_reconstruction_json(src) if src else None
            if p is not None:
                rec_summary = self._summarize_reconstruction_json(p)
            hints.append({
                "description": d.get("description", ""),
                "keywords": d.get("keywords", [])[:8],
                "source_file": src,
                "reconstruction_summary": rec_summary,
            })
        return hints

    def _iter_assembly_json_files(self, max_files=200):
        if not self.assembly_root.exists():
            return []
        files = []
        try:
            for p in self.assembly_root.rglob("assembly.json"):
                if p.is_file():
                    files.append(p)
                    if len(files) >= max_files:
                        break
        except Exception:
            return []
        return files

    def _summarize_assembly_json(self, assembly_json):
        key = str(assembly_json)
        if key in self._assembly_summary_cache:
            return self._assembly_summary_cache[key]
        summary = {
            "file": key,
            "num_occurrences": 0,
            "num_components": 0,
            "num_bodies": 0,
            "num_joints": 0,
            "num_as_built_joints": 0,
            "joint_types": {},
            "num_contacts": 0,
            "num_holes": 0,
            "hole_types_top": [],
        }
        try:
            with open(assembly_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            occ = data.get("occurrences", {}) or {}
            comps = data.get("components", {}) or {}
            bodies = data.get("bodies", {}) or {}
            joints = data.get("joints", {}) or {}
            as_built = data.get("as_built_joints", {}) or {}
            contacts = data.get("contacts", []) or []
            holes = data.get("holes", []) or []
            summary["num_occurrences"] = len(occ)
            summary["num_components"] = len(comps)
            summary["num_bodies"] = len(bodies)
            summary["num_joints"] = len(joints)
            summary["num_as_built_joints"] = len(as_built)
            summary["num_contacts"] = len(contacts)
            summary["num_holes"] = len(holes)

            jt = {}
            for j in list(joints.values()) + list(as_built.values()):
                if not isinstance(j, dict):
                    continue
                motion = j.get("joint_motion", {}) or {}
                jtype = str(motion.get("joint_type", "Unknown")).strip()
                jt[jtype] = jt.get(jtype, 0) + 1
            summary["joint_types"] = jt

            ht = {}
            for h in holes:
                if not isinstance(h, dict):
                    continue
                htype = str(h.get("type", "Unknown")).strip()
                ht[htype] = ht.get(htype, 0) + 1
            top_holes = sorted(ht.items(), key=lambda kv: kv[1], reverse=True)[:6]
            summary["hole_types_top"] = [{"type": k, "count": v} for k, v in top_holes]
        except Exception:
            pass
        self._assembly_summary_cache[key] = summary
        return summary

    def _infer_joint_signature(self, query, current_parts=None):
        text = (query or "").strip().lower()
        part_names = " ".join(
            str((p or {}).get("name", "")).strip().lower()
            for p in (current_parts or [])
        )
        hay = f"{text} {part_names}".strip()

        desired = set()
        wants_holes = False
        wants_contacts = False
        wants_symmetry = False

        if any(k in hay for k in ("tank", "track", "wheel", "sprocket", "idler", "vehicle", "car", "robot")):
            desired.update(("RevoluteJointType", "CylindricalJointType"))
            wants_holes = True
            wants_contacts = True
            wants_symmetry = True
        if any(k in hay for k in ("caterpillar", "crawler", "РіСѓСЃРµРЅРёС†Р°", "РіСѓСЃРµРЅРёС†", "С‚СЂР°Рє")):
            desired.update(("RevoluteJointType", "CylindricalJointType"))
            wants_holes = True
            wants_contacts = True
            wants_symmetry = True
        if any(k in hay for k in ("gear", "shaft", "axle", "bearing", "pulley", "rotor", "motor")):
            desired.update(("RevoluteJointType", "CylindricalJointType"))
            wants_holes = True
        if any(k in hay for k in ("door", "hinge", "lid", "flap")):
            desired.add("RevoluteJointType")
            wants_holes = True
        if any(k in hay for k in ("slider", "rail", "linear", "carriage")):
            desired.add("SliderJointType")
        if any(k in hay for k in ("ball", "socket", "gimbal")):
            desired.add("BallJointType")
        if any(k in hay for k in ("planar", "plate", "panel")):
            desired.add("PlanarJointType")
        if any(k in hay for k in ("pin", "slot")):
            desired.add("PinSlotJointType")

        # Fallback to generic mechanical prior if no explicit signature inferred.
        if not desired:
            desired.update(("RevoluteJointType", "RigidJointType"))

        return {
            "desired_joint_types": sorted(desired),
            "wants_holes": wants_holes,
            "wants_contacts": wants_contacts,
            "wants_symmetry": wants_symmetry,
        }

    @staticmethod
    def _score_assembly_signature_match(summary, signature):
        jt = summary.get("joint_types", {}) or {}
        desired = set(signature.get("desired_joint_types", []) or [])
        score = 0.0
        for d in desired:
            score += 18.0 if jt.get(d, 0) > 0 else -6.0
        if signature.get("wants_holes", False):
            score += min(18.0, 1.6 * float(summary.get("num_holes", 0)))
        if signature.get("wants_contacts", False):
            score += min(16.0, 0.8 * float(summary.get("num_contacts", 0)))
        if signature.get("wants_symmetry", False):
            rev = float(jt.get("RevoluteJointType", 0))
            cyl = float(jt.get("CylindricalJointType", 0))
            score += min(12.0, (rev + cyl) * 1.2)
        return score

    def _assembly_priors_for_query(self, query, top_k=3, current_parts=None):
        files = self._iter_assembly_json_files(max_files=400)
        if not files:
            return []
        signature = self._infer_joint_signature(query, current_parts=current_parts)
        # Heuristic ranking: prefer mechanically rich assemblies.
        scored = []
        for p in files:
            s = self._summarize_assembly_json(p)
            richness = (
                3.0 * s.get("num_joints", 0) +
                2.0 * s.get("num_as_built_joints", 0) +
                1.0 * s.get("num_contacts", 0) +
                0.8 * s.get("num_holes", 0)
            )
            sig_score = self._score_assembly_signature_match(s, signature)
            total = 0.55 * richness + 0.45 * sig_score
            row = dict(s)
            row["signature"] = signature
            row["signature_score"] = round(sig_score, 2)
            row["retrieval_score"] = round(total, 2)
            scored.append((total, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max(1, top_k)]]

    def _sanitize_program_steps(self, steps, max_steps=None):
        if not isinstance(steps, list):
            return []
        cap = max_steps if isinstance(max_steps, int) and max_steps > 0 else PROGRAM_DETAIL_MAX_STEPS
        out = []
        cut_keywords = ("cut", "hole", "slot", "port", "vent", "notch", "window", "opening")
        for st in steps:
            if not isinstance(st, dict):
                continue
            action = str(st.get("action", "")).strip().lower()
            if action not in ("build", "refresh"):
                continue
            if action == "refresh":
                out.append({"action": "refresh"})
            else:
                shape = str(st.get("shape", "rect")).strip().lower()
                if shape not in ("rect", "circle", "polygon", "ring", "gear"):
                    continue
                step = {
                    "action": "build",
                    "description": str(st.get("description", "detail")).strip() or "detail",
                    "plane": str(st.get("plane", "XY")).strip() or "XY",
                    "shape": shape,
                    "cx": round(self._to_float(st.get("cx"), 0.0), 1),
                    "cy": round(self._to_float(st.get("cy"), 0.0), 1),
                    "distance": round(self._to_float(st.get("distance"), 0.0), 1),
                    "operation": str(st.get("operation", "NewBodyFeatureOperation")).strip() or "NewBodyFeatureOperation",
                }
                if step["distance"] <= 0.1:
                    continue
                base_plane, off = self._parse_plane_and_offset(step["plane"])
                step["plane"] = f"{base_plane}@{round(off, 3)}" if abs(off) > 1e-9 else base_plane
                repeat = int(max(1, min(32, self._to_float(st.get("repeat"), 1))))
                step["repeat"] = repeat
                step["dx"] = round(self._to_float(st.get("dx"), 0.0), 1)
                step["dy"] = round(self._to_float(st.get("dy"), 0.0), 1)
                step["dz"] = round(self._to_float(st.get("dz"), 0.0), 1)
                if shape == "rect":
                    step["w"] = round(abs(self._to_float(st.get("w"), 0.0)), 1)
                    step["h"] = round(abs(self._to_float(st.get("h"), 0.0)), 1)
                    if step["w"] <= 0.1 or step["h"] <= 0.1:
                        continue
                elif shape == "circle":
                    step["radius"] = round(abs(self._to_float(st.get("radius"), 0.0)), 1)
                    if step["radius"] <= 0.1:
                        continue
                elif shape == "polygon":
                    step["radius"] = round(abs(self._to_float(st.get("radius"), 0.0)), 1)
                    step["sides"] = int(max(3, min(96, self._to_float(st.get("sides"), 6))))
                    step["rotation_deg"] = round(self._to_float(st.get("rotation_deg"), 0.0), 1)
                    if step["radius"] <= 0.1:
                        continue
                elif shape == "ring":
                    step["outer_radius"] = round(abs(self._to_float(st.get("outer_radius"), 0.0)), 1)
                    step["inner_radius"] = round(abs(self._to_float(st.get("inner_radius"), 0.0)), 1)
                    if step["outer_radius"] <= 0.2 or step["inner_radius"] <= 0.1:
                        continue
                    if step["inner_radius"] >= step["outer_radius"]:
                        step["inner_radius"] = round(max(0.1, step["outer_radius"] * 0.7), 1)
                elif shape == "gear":
                    step["outer_radius"] = round(abs(self._to_float(st.get("outer_radius"), 0.0)), 1)
                    step["inner_radius"] = round(abs(self._to_float(st.get("inner_radius"), 0.0)), 1)
                    step["tooth_count"] = int(max(6, min(80, self._to_float(st.get("tooth_count"), 16))))
                    step["rotation_deg"] = round(self._to_float(st.get("rotation_deg"), 0.0), 1)
                    if step["outer_radius"] <= 0.3 or step["inner_radius"] <= 0.1:
                        continue
                    if step["inner_radius"] >= step["outer_radius"]:
                        step["inner_radius"] = round(max(0.1, step["outer_radius"] * 0.75), 1)
                if step["operation"] not in ("NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation"):
                    step["operation"] = "NewBodyFeatureOperation"
                # Stabilize: convert unsafe cut operations to additive ops unless
                # description clearly indicates a cutout-like feature.
                desc_lower = step["description"].strip().lower()
                if step["operation"] == "CutFeatureOperation":
                    if not any(k in desc_lower for k in cut_keywords):
                        step["operation"] = "JoinFeatureOperation"
                out.append(step)
            if len(out) >= cap:
                break
        return out

    @staticmethod
    def _parts_bounds(parts):
        if not parts:
            return {
                "x_min": -100.0, "x_max": 100.0,
                "y_min": 0.0, "y_max": 100.0,
                "z_min": 0.0, "z_max": 100.0,
            }
        return {
            "x_min": min(p["x_min"] for p in parts),
            "x_max": max(p["x_max"] for p in parts),
            "y_min": min(p["y_min"] for p in parts),
            "y_max": max(p["y_max"] for p in parts),
            "z_min": min(p["z_min"] for p in parts),
            "z_max": max(p["z_max"] for p in parts),
        }

    def _constrain_program_steps_to_bounds(self, steps, bounds, margin_ratio=0.18):
        if not steps:
            return []
        bx0, bx1 = bounds["x_min"], bounds["x_max"]
        by0, by1 = bounds["y_min"], bounds["y_max"]
        bz0, bz1 = bounds["z_min"], bounds["z_max"]
        bw = max(1.0, bx1 - bx0)
        bd = max(1.0, by1 - by0)
        bh = max(1.0, bz1 - bz0)
        mx = bw * margin_ratio
        my = bd * margin_ratio
        mz = bh * margin_ratio
        min_x, max_x = bx0 - mx, bx1 + mx
        min_y, max_y = by0 - my, by1 + my
        min_z, max_z = max(0.0, bz0 - mz), bz1 + mz

        def infer_plane_offset(base, explicit, desc, repeat, dz):
            if explicit:
                return None, dz
            inset_x = max(0.8, 0.04 * bw)
            inset_y = max(0.8, 0.04 * bd)
            d = (desc or "").lower()
            if base == "YZ":
                left = "left" in d
                right = "right" in d
                side_like = any(k in d for k in ("side", "track", "wheel", "armor", "panel", "pipe", "exhaust"))
                if left:
                    return bx0 + inset_x, dz
                if right:
                    return bx1 - inset_x, dz
                if side_like and repeat == 2 and abs(dz) < 1e-6:
                    span = max(1.0, (bx1 - inset_x) - (bx0 + inset_x))
                    return bx0 + inset_x, span
                if side_like:
                    return bx1 - inset_x, dz
            if base == "XZ":
                rear = any(k in d for k in ("rear", "back", "engine", "exhaust"))
                front = any(k in d for k in ("front", "nose", "gun", "barrel"))
                if rear:
                    return by0 + inset_y, dz
                if front:
                    return by1 - inset_y, dz
            return None, dz

        out = []
        for st in steps:
            if st.get("action") != "build":
                out.append(st)
                continue
            s = dict(st)
            plane_raw = str(s.get("plane", "XY")).strip().upper()
            explicit_off = "@" in plane_raw
            base, off = self._parse_plane_and_offset(plane_raw)
            repeat = int(max(1, min(32, self._to_float(s.get("repeat"), 1))))
            dz = self._to_float(s.get("dz"), 0.0)
            inferred_off, inferred_dz = infer_plane_offset(
                base=base,
                explicit=explicit_off,
                desc=str(s.get("description", "")),
                repeat=repeat,
                dz=dz,
            )
            if inferred_off is not None:
                off = inferred_off
                s["dz"] = round(inferred_dz, 1)

            # Clamp sketch coordinates in the local coordinates of each plane.
            cx = self._to_float(s.get("cx"), 0.0)
            cy = self._to_float(s.get("cy"), 0.0)
            if base == "XY":
                cx = max(min_x, min(max_x, cx))   # X
                cy = max(min_y, min(max_y, cy))   # Y
            elif base == "XZ":
                cx = max(min_x, min(max_x, cx))   # X
                cy = max(min_z, min(max_z, cy))   # Z
            else:  # YZ
                cx = max(min_y, min(max_y, cx))   # Y
                cy = max(min_z, min(max_z, cy))   # Z
            s["cx"] = round(cx, 1)
            s["cy"] = round(cy, 1)
            if base == "XY":
                off = max(min_z, min(max_z, off))
            elif base == "XZ":
                off = max(min_y, min(max_y, off))
            else:  # YZ
                off = max(min_x, min(max_x, off))
            s["plane"] = f"{base}@{round(off, 3)}" if abs(off) > 1e-9 else base
            out.append(s)
        return out

    def _plan_detail_program_steps(self, current_parts, user_request, images=None):
        if not current_parts:
            return []
        parts_summary = "\n".join(
            f"  {p['name']}: x_min={p['x_min']}, x_max={p['x_max']}, "
            f"y_min={p['y_min']}, y_max={p['y_max']}, z_min={p['z_min']}, z_max={p['z_max']}"
            for p in current_parts
        )
        hints = self._dataset_action_hints(user_request, top_k=2)
        assembly_priors = self._assembly_priors_for_query(
            user_request, top_k=3, current_parts=current_parts
        )
        bounds = self._parts_bounds(current_parts)
        request_text = (
            f"Current model parts (cm):\n{parts_summary}\n\n"
            f"Detail request: {user_request}\n\n"
            f"Current bounds (cm): {json.dumps(bounds, ensure_ascii=True)}\n\n"
            f"{ASSEMBLY_MECHANICAL_PRIORS}\n\n"
            f"Dataset action hints:\n{json.dumps(hints, ensure_ascii=True)}\n\n"
            f"Assembly dataset priors:\n{json.dumps(assembly_priors, ensure_ascii=True)}\n\n"
            "Plan only additional steps to add functional details on top of current model."
        )
        raw = (
            self._call_llm_with_images(PROGRAM_DETAIL_PLANNER_PROMPT, request_text, images)
            if images else
            self._call_llm(PROGRAM_DETAIL_PLANNER_PROMPT, request_text)
        )
        payload = self._parse_json(raw)
        local_cap = PROGRAM_DETAIL_MAX_STEPS
        if not images:
            local_cap = min(local_cap, 8)
        steps = self._sanitize_program_steps(payload.get("steps", []), max_steps=local_cap)
        return self._constrain_program_steps_to_bounds(steps, bounds)

    def _evaluate_detail_needs(self, current_parts, user_request, images=None):
        if not current_parts:
            return {
                "should_add_details": False,
                "confidence": 0.0,
                "detail_request": "",
                "missing_detail_groups": [],
            }
        parts_summary = "\n".join(
            f"  {p['name']}: x_min={p['x_min']}, x_max={p['x_max']}, "
            f"y_min={p['y_min']}, y_max={p['y_max']}, z_min={p['z_min']}, z_max={p['z_max']}"
            for p in current_parts
        )
        text = (
            f"User context: {user_request}\n\n"
            f"Current parts:\n{parts_summary}\n\n"
            "Evaluate whether additional functional/mechanical details should be added."
        )
        try:
            raw = (
                self._call_llm_with_images(DETAIL_EVALUATOR_PROMPT, text, images)
                if images else
                self._call_llm(DETAIL_EVALUATOR_PROMPT, text)
            )
            out = self._parse_json(raw)
        except Exception:
            return {
                "should_add_details": False,
                "confidence": 0.0,
                "detail_request": "",
                "missing_detail_groups": [],
            }
        conf = max(0.0, min(1.0, self._to_float(out.get("confidence"), 0.0)))
        should = bool(out.get("should_add_details")) and conf >= AUTO_DETAIL_MIN_CONFIDENCE
        return {
            "should_add_details": should,
            "confidence": conf,
            "detail_request": str(out.get("detail_request", "")).strip(),
            "missing_detail_groups": out.get("missing_detail_groups", []) or [],
            "notes": out.get("notes", []) or [],
        }

    def _auto_detail_program(self, current_parts, user_request, images=None, rounds=None):
        if not current_parts:
            return list(current_parts)
        max_rounds = rounds if isinstance(rounds, int) and rounds > 0 else max(1, AUTO_DETAIL_PROGRAM_ROUNDS)
        if not images:
            # Text-only mode is less grounded; keep detailing conservative.
            max_rounds = min(max_rounds, 1)
        parts = list(current_parts)
        for r in range(max_rounds):
            assessment = self._evaluate_detail_needs(parts, user_request, images=images)
            if not assessment.get("should_add_details", False):
                if r == 0:
                    print("  [AutoDetail] Evaluator: current detail level is sufficient.")
                break
            req = assessment.get("detail_request", "").strip() or user_request
            print(f"  [AutoDetail] Round {r+1}: planning details (confidence={assessment.get('confidence', 0.0):.2f})...")
            try:
                steps = self._plan_detail_program_steps(parts, req, images=images)
            except Exception:
                steps = []
            if not steps:
                print("  [AutoDetail] Planner returned no detail steps.")
                break
            print(f"  [AutoDetail] Executing {len(steps)} detail steps...")
            ok, total = self.execute_plan(steps)
            if total > 0 and ok / total < 0.7:
                print("  [AutoDetail] Too many failed detail steps; stopping auto detail for stability.")
                break
            added = self._steps_to_bboxes(steps)
            if not added:
                break
            merged = parts + added
            merged = self._sanitize_parts(merged, normalize_frame=False)
            parts = self._limit_parts_for_stability(merged, max_parts=DETAILING_MAX_PARTS)
        return parts

    # -- screenshot -----------------------------------------------------------

    def take_screenshot(self):
        tmp = os.path.join(tempfile.gettempdir(), "fusion360_feedback.png")
        r = self.fusion.screenshot(tmp, SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT)
        if r is None or r.status_code != 200:
            return None
        with open(tmp, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _detect_media_type(path):
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            return "image/jpeg"
        if ext == ".webp":
            return "image/webp"
        if ext == ".gif":
            return "image/gif"
        return "image/png"

    @staticmethod
    def _clean_stem_for_dedupe(path):
        stem = path.stem.lower().strip()
        stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
        return stem

    @staticmethod
    def _infer_view_label_from_name(path):
        s = path.stem.lower()
        if "front" in s:
            return "front"
        if "back" in s or "rear" in s:
            return "back"
        if "left" in s:
            return "left"
        if "right" in s:
            return "right"
        if "top" in s or "up" in s:
            return "top"
        if "bottom" in s or "down" in s:
            return "bottom"
        if "side" in s:
            return "side"
        return "unknown"

    def _load_images_from_paths(self, paths):
        images = []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    images.append({
                        "data": base64.b64encode(f.read()).decode("utf-8"),
                        "media_type": self._detect_media_type(p),
                        "label": self._infer_view_label_from_name(p),
                        "name": p.name,
                    })
            except Exception:
                continue
        return self._ensure_view_labels(images)

    def _gather_image_paths(self, folder_path, max_images=8):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif")
        files = []
        for pat in exts:
            files.extend(folder_path.glob(pat))
        files = sorted([p for p in files if p.is_file()])
        if not files:
            return files

        # Deduplicate common copies like "name (2).jpg".
        unique = {}
        for p in files:
            key = self._clean_stem_for_dedupe(p)
            if key not in unique:
                unique[key] = p
        files = list(unique.values())

        # Prefer canonical multi-view order when labels are available.
        priority = {"front": 0, "right": 1, "back": 2, "left": 3, "top": 4, "bottom": 5, "side": 6, "unknown": 7}
        files.sort(key=lambda p: (priority.get(self._infer_view_label_from_name(p), 7), p.name.lower()))
        if len(files) > max_images:
            files = files[:max_images]
        return files

    # -- JSON parsing ---------------------------------------------------------

    @staticmethod
    def _parse_json(raw):
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        return json.loads(text)

    @staticmethod
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    def _ensure_view_labels(self, images):
        unknown = [img for img in images if str(img.get("label", "unknown")) == "unknown"]
        if not unknown:
            return images
        if len(images) == 1:
            return images
        print("  Some view labels are unknown. Labeling improves multiview stability.")
        allowed = {"front", "back", "left", "right", "top", "bottom", "side", "unknown"}
        for img in unknown:
            name = str(img.get("name", "view")).strip() or "view"
            ans = input(
                f"    Label for '{name}' [front/back/left/right/top/bottom/side/unknown, Enter=unknown]: "
            ).strip().lower()
            if ans in allowed:
                img["label"] = ans
        return images

    @staticmethod
    def _boxes_connected(a, b, eps=0.8):
        ox = min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"])
        oy = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
        oz = min(a["z_max"], b["z_max"]) - max(a["z_min"], b["z_min"])
        if ox > 0 and oy > 0 and oz > 0:
            return True
        near_x = ox >= -eps and oy > 0 and oz > 0
        near_y = oy >= -eps and ox > 0 and oz > 0
        near_z = oz >= -eps and ox > 0 and oy > 0
        return near_x or near_y or near_z

    def _score_parts(self, parts):
        if not parts:
            return -10_000.0
        score = 0.0
        # Penalize too many parts for demo stability.
        score -= max(0, len(parts) - 12) * 8.0
        # Penalize too-few parts for likely complex requests.
        if len(parts) < 3:
            score -= 120.0
        # Penalize negative Z and very thin/degenerate boxes.
        for p in parts:
            if p["z_min"] < -0.1:
                score -= 100.0
            dx = p["x_max"] - p["x_min"]
            dy = p["y_max"] - p["y_min"]
            dz = p["z_max"] - p["z_min"]
            if min(dx, dy, dz) < 0.2:
                score -= 40.0
            # Very extreme aspect boxes are usually decomposition artifacts.
            dims = sorted([max(0.01, dx), max(0.01, dy), max(0.01, dz)])
            if dims[-1] / dims[0] > 18.0:
                score -= 20.0
        # Connectivity score.
        n = len(parts)
        visited = set([0])
        stack = [0]
        while stack:
            i = stack.pop()
            for j in range(n):
                if j in visited:
                    continue
                if self._boxes_connected(parts[i], parts[j]):
                    visited.add(j)
                    stack.append(j)
        disconnected = n - len(visited)
        score -= disconnected * 120.0
        # Penalize highly-overlapping duplicates.
        dup_penalty = 0.0
        for i in range(n):
            a = parts[i]
            av = max(0.01, self._part_volume(a))
            for j in range(i + 1, n):
                b = parts[j]
                ix = max(0.0, min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"]))
                iy = max(0.0, min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"]))
                iz = max(0.0, min(a["z_max"], b["z_max"]) - max(a["z_min"], b["z_min"]))
                iv = ix * iy * iz
                if iv <= 0.0:
                    continue
                bv = max(0.01, self._part_volume(b))
                overlap_small = iv / min(av, bv)
                if overlap_small > 0.88:
                    dup_penalty += 12.0
        score -= min(120.0, dup_penalty)
        # Reward floor contact for supporting parts.
        floor_touch = sum(1 for p in parts if abs(p["z_min"]) <= 0.8)
        score += min(4, floor_touch) * 8.0
        return score

    def _repair_parts_with_audit(self, user_request, parts):
        if not parts:
            return []
        base = self._sanitize_parts(parts)
        if not base:
            return []
        base_score = self._score_parts(base)
        # Skip costly repair when candidate is already strong.
        if base_score >= -8.0 and len(base) >= 6:
            return base

        format_hint = (
            '{"parts": [\n'
            '  {"name": "Part A", "x_min": -20, "x_max": 20, '
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
            '  ...\n'
            ']}'
        )
        request_text = (
            f'User request: "{user_request}"\n\n'
            "Current candidate parts:\n"
            f"{json.dumps({'parts': base}, ensure_ascii=True)}\n\n"
            "Improve structural plausibility, connectivity, and orientation.\n"
            "You may split grouped parts, add missing structural supports, or remove redundant overlaps.\n"
            "Return full corrected parts JSON."
        )
        try:
            prompt = GEOMETRY_AUDIT_PROMPT.replace("{format_hint}", format_hint)
            raw = self._call_llm(prompt, request_text)
            payload = self._parse_json(raw)
            repaired = self._sanitize_parts(payload.get("parts", []))
            if not repaired:
                return base
            repaired_score = self._score_parts(repaired)
            return repaired if repaired_score >= base_score else base
        except Exception:
            return base

    @staticmethod
    def _part_volume(p):
        return (
            max(0.0, (p["x_max"] - p["x_min"])) *
            max(0.0, (p["y_max"] - p["y_min"])) *
            max(0.0, (p["z_max"] - p["z_min"]))
        )

    @staticmethod
    def _is_floor_support_part(p):
        return abs(p.get("z_min", 0.0)) <= 0.8

    @staticmethod
    def _is_structural_name(name):
        n = str(name or "").strip().lower()
        if not n:
            return False
        keywords = (
            "base", "leg", "support", "frame", "seat", "top", "tabletop", "back",
            "panel", "post", "upright", "column", "stand", "rail", "handle",
        )
        return any(k in n for k in keywords)

    @staticmethod
    def _is_suspicious_cantilever(p):
        dx = max(0.0, p["x_max"] - p["x_min"])
        dy = max(0.0, p["y_max"] - p["y_min"])
        dz = max(0.0, p["z_max"] - p["z_min"])
        if dz <= 0.0:
            return True
        long_h = max(dx, dy)
        short_h = min(dx, dy)
        very_long = long_h >= 3.8 * max(0.2, dz)
        very_thin = short_h <= 0.45 * max(0.2, long_h)
        floating = p["z_min"] > 1.5
        return very_long and very_thin and floating

    def _limit_parts_for_stability(self, parts, max_parts=12):
        if len(parts) <= max_parts:
            return parts

        def rank_key(p):
            is_structural = 1 if self._is_structural_name(p.get("name")) else 0
            touches_floor = 1 if self._is_floor_support_part(p) else 0
            suspicious = 1 if self._is_suspicious_cantilever(p) else 0
            vol = self._part_volume(p)
            # Keep required/structural/floor parts before generic large pieces.
            return (is_structural, touches_floor, -suspicious, vol)

        ranked = sorted(parts, key=rank_key, reverse=True)
        kept = ranked[:max_parts]
        kept.sort(key=lambda a: (
            round(a["z_min"], 3), round(a["y_min"], 3), round(a["x_min"], 3), a["name"]
        ))
        return kept

    # -- deformation layer (tilt/splay via segments, no Fusion server changes) ---

    TILT_ANGLE_CLAMP = (0.0, 12.0)
    SLENDER_RATIO = 3.0
    DEFORM_SEGMENTS_MIN, DEFORM_SEGMENTS_MAX = 2, 4

    @staticmethod
    def _is_slender(part, ratio=3.0):
        """True if part is tall and thin (posts, rails, backrest uprights)."""
        h = part["z_max"] - part["z_min"]
        wx = part["x_max"] - part["x_min"]
        wy = part["y_max"] - part["y_min"]
        if h <= 0 or min(wx, wy) <= 0:
            return False
        return h / min(wx, wy) >= ratio

    def _select_deformation_candidates(self, parts, apply_to, feat=None):
        """Return indices of parts that match apply_to (geometry-based). Excludes floor-touching for posts; filters rear/front for tilt."""
        if apply_to not in ("slender_posts", "slender_rails", "legs", "handles"):
            return []

        y0 = min(p["y_min"] for p in parts)
        y1 = max(p["y_max"] for p in parts)
        depth = max(0.1, y1 - y0)

        direction = ""
        primary_axis = ""
        if isinstance(feat, dict):
            direction = (feat.get("direction") or "").lower().strip()
            primary_axis = (feat.get("primary_axis") or "").lower().strip()

        idxs = []
        for i, p in enumerate(parts):
            if not self._is_slender(p, self.SLENDER_RATIO):
                continue

            if apply_to in ("slender_posts", "handles", "slender_rails"):
                if p["z_min"] <= 0.8:
                    continue

            if primary_axis == "x" and direction in ("backward", "forward"):
                cy = 0.5 * (p["y_min"] + p["y_max"])
                if direction == "backward":
                    if cy > y0 + depth * 0.60:
                        continue
                if direction == "forward":
                    if cy < y0 + depth * 0.40:
                        continue

            idxs.append(i)
        return idxs

    def _tilt_part_to_segments(self, part, angle_deg, primary_axis, direction, segments=None):
        """Split one part into segments with constant thickness; shift at z_mid per segment (step approximation, no thickening)."""
        angle_deg = max(self.TILT_ANGLE_CLAMP[0], min(self.TILT_ANGLE_CLAMP[1], angle_deg))
        if abs(angle_deg) < 0.5:
            return [part]

        z_min, z_max = part["z_min"], part["z_max"]
        height = z_max - z_min
        if height <= 0:
            return [part]

        primary_axis = (primary_axis or "x").lower().strip()
        direction = (direction or "backward").lower().strip()
        shift_axis = "y" if primary_axis == "x" else "x"
        sign = -1.0 if direction == "backward" else 1.0
        tan_angle = math.tan(math.radians(angle_deg)) * sign
        delta_top = tan_angle * height

        wx = part["x_max"] - part["x_min"]
        wy = part["y_max"] - part["y_min"]
        thickness = wy if shift_axis == "y" else wx
        thickness = max(0.2, thickness)

        if segments is None:
            n = int(math.ceil(abs(delta_top) / (thickness * 0.6))) if thickness > 0 else self.DEFORM_SEGMENTS_MAX
        else:
            n = int(segments)
        n = max(self.DEFORM_SEGMENTS_MIN, min(self.DEFORM_SEGMENTS_MAX, n))

        base_name = part.get("name", "part")
        out = []
        for i in range(n):
            z0 = z_min + (i / n) * height
            z1 = z_min + ((i + 1) / n) * height
            z_mid = 0.5 * (z0 + z1)
            shift_mid = tan_angle * (z_mid - z_min)
            seg = dict(part)
            seg["z_min"], seg["z_max"] = round(z0, 1), round(z1, 1)
            if shift_axis == "y":
                seg["y_min"] = round(part["y_min"] + shift_mid, 1)
                seg["y_max"] = round(part["y_max"] + shift_mid, 1)
            else:
                seg["x_min"] = round(part["x_min"] + shift_mid, 1)
                seg["x_max"] = round(part["x_max"] + shift_mid, 1)
            seg["name"] = f"{base_name}_seg_{i+1}"
            out.append(seg)
        return out

    def apply_deformation_layer(self, parts, features):
        """Apply tilt/splay features by splitting selected slender parts into segments. Deterministic, no Fusion changes."""
        if not features or not parts:
            return list(parts)
        # Which part indices to replace with segments, and with which feature (first matching)
        to_deform = {}
        for feat in features:
            if not isinstance(feat, dict):
                continue
            ftype = (feat.get("type") or "").strip().lower()
            apply_to = (feat.get("apply_to") or "").strip().lower()
            if ftype != "tilt":
                continue
            indices = self._select_deformation_candidates(parts, apply_to, feat=feat)
            for idx in indices:
                if idx not in to_deform:
                    to_deform[idx] = feat
        out = []
        for i, p in enumerate(parts):
            if i in to_deform:
                feat = to_deform[i]
                mag = feat.get("magnitude") or {}
                if isinstance(mag, dict) and mag.get("type") == "angle_deg":
                    angle = self._to_float(mag.get("value"), 6.0)
                else:
                    angle = self._to_float(mag if isinstance(mag, (int, float)) else 6.0, 6.0)
                angle = max(self.TILT_ANGLE_CLAMP[0], min(self.TILT_ANGLE_CLAMP[1], angle))
                primary_axis = (feat.get("primary_axis") or "x").strip().lower()
                direction = (feat.get("direction") or "backward").strip().lower()
                segs = self._tilt_part_to_segments(p, angle, primary_axis, direction, segments=None)
                out.extend(segs)
            else:
                out.append(dict(p))
        out.sort(key=lambda a: (
            round(a["z_min"], 3), round(a["y_min"], 3), round(a["x_min"], 3), a["name"]
        ))
        return out

    def _sanitize_parts(self, parts, target_dims=None, symmetry="none", normalize_frame=True):
        """Normalize generated boxes for stable, universal reconstruction."""
        clean = []
        for i, p in enumerate(parts):
            name = str(p.get("name", f"Part {i+1}")).strip() or f"Part {i+1}"
            raw_plane = str(p.get("plane", "")).strip()
            plane = ""
            if raw_plane:
                plane, _ = self._parse_plane_and_offset(raw_plane)
            x0 = self._to_float(p.get("x_min"))
            x1 = self._to_float(p.get("x_max"))
            y0 = self._to_float(p.get("y_min"))
            y1 = self._to_float(p.get("y_max"))
            z0 = self._to_float(p.get("z_min"))
            z1 = self._to_float(p.get("z_max"))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            if z0 > z1:
                z0, z1 = z1, z0
            if (x1 - x0) < 0.2 or (y1 - y0) < 0.2 or (z1 - z0) < 0.2:
                continue
            item = {
                "name": name,
                "x_min": x0, "x_max": x1,
                "y_min": y0, "y_max": y1,
                "z_min": z0, "z_max": z1,
            }
            if plane:
                item["plane"] = plane
            clean.append(item)
        if not clean:
            return []

        # Frame normalization is useful for first-pass generation, but must be
        # disabled for incremental detailing to avoid drifting global coordinates.
        if normalize_frame:
            x_min = min(p["x_min"] for p in clean)
            x_max = max(p["x_max"] for p in clean)
            y_min_global = min(p["y_min"] for p in clean)
            z_min_global = min(p["z_min"] for p in clean)
            x_center = (x_min + x_max) / 2.0
            for p in clean:
                p["x_min"] -= x_center
                p["x_max"] -= x_center
                p["y_min"] -= y_min_global
                p["y_max"] -= y_min_global
                p["z_min"] -= z_min_global
                p["z_max"] -= z_min_global
                if p["z_min"] < 0:
                    p["z_min"] = 0.0

        # Scale to photo-estimated outer dimensions when available.
        if isinstance(target_dims, dict):
            tw = self._to_float(target_dims.get("width"), 0.0)
            td = self._to_float(target_dims.get("depth"), 0.0)
            th = self._to_float(target_dims.get("height"), 0.0)
            cur_w = max(p["x_max"] for p in clean) - min(p["x_min"] for p in clean)
            cur_d = max(p["y_max"] for p in clean) - min(p["y_min"] for p in clean)
            cur_h = max(p["z_max"] for p in clean) - min(p["z_min"] for p in clean)
            sx = max(0.4, min(2.5, tw / cur_w)) if tw > 1.0 and cur_w > 0.1 else 1.0
            sy = max(0.4, min(2.5, td / cur_d)) if td > 1.0 and cur_d > 0.1 else 1.0
            sz = max(0.4, min(2.5, th / cur_h)) if th > 1.0 and cur_h > 0.1 else 1.0
            mean_s = (sx + sy + sz) / 3.0
            sx = max(mean_s * 0.75, min(mean_s * 1.35, sx))
            sy = max(mean_s * 0.75, min(mean_s * 1.35, sy))
            sz = max(mean_s * 0.75, min(mean_s * 1.35, sz))
            for p in clean:
                p["x_min"] *= sx
                p["x_max"] *= sx
                p["y_min"] *= sy
                p["y_max"] *= sy
                p["z_min"] *= sz
                p["z_max"] *= sz

        # If symmetry around X is expected, keep central volumes centered.
        if symmetry in ("approx_x", "approx_xy"):
            for p in clean:
                cx = (p["x_min"] + p["x_max"]) / 2.0
                if abs(cx) < 1.5:
                    w = p["x_max"] - p["x_min"]
                    p["x_min"] = -w / 2.0
                    p["x_max"] = w / 2.0

        # Final safety: enforce floor and valid extents.
        for p in clean:
            if p["z_min"] < 0:
                p["z_min"] = 0.0
            if p["x_min"] > p["x_max"]:
                p["x_min"], p["x_max"] = p["x_max"], p["x_min"]
            if p["y_min"] > p["y_max"]:
                p["y_min"], p["y_max"] = p["y_max"], p["y_min"]
            if p["z_min"] > p["z_max"]:
                p["z_min"], p["z_max"] = p["z_max"], p["z_min"]

        # Stable ordering and rounded coordinates.
        clean.sort(key=lambda a: (
            round(a["z_min"], 3), round(a["y_min"], 3), round(a["x_min"], 3), a["name"]
        ))
        for p in clean:
            for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
                p[k] = round(p[k], 1)
        return clean

    # -- shape helpers --------------------------------------------------------

    def _draw_rect(self, sketch_name, cx, cy, w, h):
        hw, hh = w / 2, h / 2
        pts = [
            {"x": cx - hw, "y": cy - hh},
            {"x": cx + hw, "y": cy - hh},
            {"x": cx + hw, "y": cy + hh},
            {"x": cx - hw, "y": cy + hh},
        ]
        for pt in pts:
            self.fusion.add_point(sketch_name, pt)
        r = self.fusion.close_profile(sketch_name)
        return self._extract_profile(r)

    def _draw_circle(self, sketch_name, cx, cy, radius):
        r = self.fusion.add_circle(sketch_name, {"x": cx, "y": cy}, radius)
        return self._extract_profile(r)

    def _draw_polygon(self, sketch_name, cx, cy, radius, sides, rotation_deg=0.0):
        n = int(max(3, min(96, sides)))
        rad = max(0.1, float(radius))
        rot = math.radians(float(rotation_deg))
        pts = []
        for i in range(n):
            a = rot + (2.0 * math.pi * i / n)
            pts.append({"x": cx + rad * math.cos(a), "y": cy + rad * math.sin(a)})
        for pt in pts:
            self.fusion.add_point(sketch_name, pt)
        r = self.fusion.close_profile(sketch_name)
        return self._extract_profile(r)

    def _draw_gear(self, sketch_name, cx, cy, outer_radius, inner_radius, tooth_count, rotation_deg=0.0):
        t = int(max(6, min(80, tooth_count)))
        ro = max(0.2, float(outer_radius))
        ri = max(0.1, min(ro * 0.95, float(inner_radius)))
        rot = math.radians(float(rotation_deg))
        pts = []
        for i in range(t * 2):
            a = rot + (2.0 * math.pi * i / (t * 2))
            rr = ro if (i % 2 == 0) else ri
            pts.append({"x": cx + rr * math.cos(a), "y": cy + rr * math.sin(a)})
        for pt in pts:
            self.fusion.add_point(sketch_name, pt)
        r = self.fusion.close_profile(sketch_name)
        return self._extract_profile(r)

    def _draw_ring(self, sketch_name, cx, cy, outer_radius, inner_radius):
        ro = max(0.2, float(outer_radius))
        ri = max(0.1, min(ro * 0.95, float(inner_radius)))
        self.fusion.add_circle(sketch_name, {"x": cx, "y": cy}, ro)
        r = self.fusion.add_circle(sketch_name, {"x": cx, "y": cy}, ri)
        target_area = math.pi * (ro * ro - ri * ri)
        return self._extract_profile(r, target_area=target_area)

    @staticmethod
    def _extract_profile(response, target_area=None):
        if response.status_code != 200:
            return None
        profiles = response.json().get("data", {}).get("profiles", {})
        if not profiles:
            return None
        if target_area is not None:
            best_id = None
            best_err = float("inf")
            for pid, pobj in profiles.items():
                try:
                    area = float((pobj or {}).get("properties", {}).get("area", 0.0))
                except Exception:
                    area = 0.0
                err = abs(area - target_area)
                if err < best_err:
                    best_err = err
                    best_id = pid
            if best_id is not None:
                return best_id
        return next(iter(profiles))

    # -- step execution -------------------------------------------------------

    def execute_step(self, step):
        action = step.get("action")

        if action == "clear":
            print("    [clear] Clearing document")
            r = self.fusion.clear()
            return r.status_code == 200

        if action == "refresh":
            print("    [view]  Fitting camera")
            self.fusion.refresh()
            return True

        if action == "build":
            desc = step.get("description", "")
            plane = step.get("plane", "XY")
            shape = step.get("shape", "rect")
            distance = step.get("distance", 5)
            operation = step.get("operation", "NewBodyFeatureOperation")
            repeat = int(max(1, min(32, self._to_float(step.get("repeat"), 1))))
            dx_rep = self._to_float(step.get("dx"), 0.0)
            dy_rep = self._to_float(step.get("dy"), 0.0)
            dz_rep = self._to_float(step.get("dz"), 0.0)

            if shape == "circle":
                dims = f"r={step.get('radius')}  d={distance}"
            elif shape == "polygon":
                dims = f"r={step.get('radius')}  n={step.get('sides')}  d={distance}"
            elif shape in ("ring", "gear"):
                dims = f"ro={step.get('outer_radius')}  ri={step.get('inner_radius')}  d={distance}"
            else:
                dims = f"w={step.get('w')}  h={step.get('h')}  d={distance}"
            print(f"    [build] {desc}  |  {plane}  cx={step.get('cx')}  cy={step.get('cy')}  {dims}  repeat={repeat}")

            base_cx = self._to_float(step.get("cx"), 0.0)
            base_cy = self._to_float(step.get("cy"), 0.0)
            base_plane_name, base_plane_off = self._parse_plane_and_offset(plane)
            base_plane = f"{base_plane_name}@{round(base_plane_off, 3)}" if abs(base_plane_off) > 1e-9 else base_plane_name

            for idx in range(repeat):
                cur_plane = base_plane
                if abs(dz_rep) > 1e-6:
                    off = base_plane_off + dz_rep * idx
                    cur_plane = f"{base_plane_name}@{round(off, 3)}"

                r = self.fusion.add_sketch(cur_plane)
                if r.status_code != 200:
                    print(f"            Error: add_sketch -> {r.json().get('message', '')}")
                    return False
                sketch_name = r.json()["data"]["sketch_name"]

                cx = base_cx + dx_rep * idx
                cy = base_cy + dy_rep * idx

                if shape == "rect":
                    profile_id = self._draw_rect(
                        sketch_name,
                        cx, cy,
                        step.get("w", 10), step.get("h", 10),
                    )
                elif shape == "circle":
                    profile_id = self._draw_circle(
                        sketch_name,
                        cx, cy,
                        step.get("radius", 5),
                    )
                elif shape == "polygon":
                    profile_id = self._draw_polygon(
                        sketch_name,
                        cx, cy,
                        step.get("radius", 5),
                        step.get("sides", 6),
                        step.get("rotation_deg", 0),
                    )
                elif shape == "ring":
                    profile_id = self._draw_ring(
                        sketch_name,
                        cx, cy,
                        step.get("outer_radius", 6),
                        step.get("inner_radius", 4),
                    )
                elif shape == "gear":
                    profile_id = self._draw_gear(
                        sketch_name,
                        cx, cy,
                        step.get("outer_radius", 6),
                        step.get("inner_radius", 4),
                        step.get("tooth_count", 16),
                        step.get("rotation_deg", 0),
                    )
                else:
                    print(f"            Unknown shape: {shape}")
                    return False

                if profile_id is None:
                    print("            Error: profile not created")
                    return False

                r = self.fusion.add_extrude(sketch_name, profile_id, distance, operation)
                if r.status_code != 200:
                    msg = ""
                    try:
                        msg = r.json().get("message", "")
                    except Exception:
                        msg = ""
                    # Robust fallback: if Cut/Intersect has no valid target body,
                    # retry as Join to avoid dropping the full detail step.
                    if operation in ("CutFeatureOperation", "IntersectFeatureOperation") and "No target body found" in str(msg):
                        r2 = self.fusion.add_extrude(sketch_name, profile_id, distance, "JoinFeatureOperation")
                        if r2.status_code == 200:
                            continue
                    print(f"            Error: extrude -> {msg}")
                    return False

            time.sleep(STEP_DELAY)
            return True

        print(f"    [?]     Unknown action: {action}")
        return False

    def execute_plan(self, steps):
        # First action: fit camera so the whole build is visible (plan may start with refresh, then clear).
        self.fusion.refresh()
        total = len(steps)
        ok = 0
        for i, step in enumerate(steps, 1):
            print(f"  Step {i}/{total}")
            if self.execute_step(step):
                ok += 1
                if step.get("action") == "clear":
                    self.fusion.refresh()
        self.fusion.refresh()
        print(f"\n  Done: {ok}/{total} steps completed successfully.")
        return ok, total

    def _interactive_user_refine(self, base_request, current_parts, images=None):
        """Ask user to accept or provide manual corrections, then rebuild."""
        parts = current_parts
        while True:
            ans = input("  Accept this result? [y/n]: ").strip().lower()
            if ans in ("y", "yes"):
                print("  Accepted.\n")
                return parts
            if ans not in ("n", "no"):
                print("  Please answer 'y' or 'n'.")
                continue
            feedback = input("  What should be changed? ").strip()
            if not feedback:
                print("  No feedback entered.")
                continue

            parts_summary = "\n".join(
                f"  {p['name']}: X[{p['x_min']},{p['x_max']}] "
                f"Y[{p['y_min']},{p['y_max']}] Z[{p['z_min']},{p['z_max']}]"
                for p in parts
            )
            correction_request = (
                f'Original request: "{base_request}"\n'
                f"User correction request: {feedback}\n\n"
                f"Current parts (bounding boxes):\n{parts_summary}\n\n"
                "Return a corrected full parts list that improves spatial layout and proportions. "
                "Keep coordinate conventions and use stable rectangular solids."
            )
            try:
                format_hint = (
                    '{"parts": [\n'
                    '  {"name": "Part A", "x_min": -20, "x_max": 20, '
                    '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
                    '  ...\n'
                    ']}'
                )
                prompt = CORRECTION_PROMPT.replace("{format_hint}", format_hint)
                if images:
                    raw = self._call_llm_with_images(
                        prompt, correction_request, images
                    )
                else:
                    raw = self._call_llm(prompt, correction_request)
                result = self._parse_json(raw)
                new_parts = self._sanitize_parts(result.get("parts", []), normalize_frame=False)
                new_parts = self._limit_parts_for_stability(new_parts, max_parts=max(12, len(parts)))
            except Exception as e:
                print(f"  Correction failed: {e}")
                continue

            if not new_parts:
                print("  Model returned no corrected parts.")
                continue

            print(f"  Rebuilding with {len(new_parts)} corrected parts...\n")
            self.execute_plan(self.encode(new_parts))
            parts = new_parts

    # -- visual feedback loop -------------------------------------------------

    def review_and_fix(self, user_request, original_parts):
        """Review the model visually. If VLM is not satisfied, re-run the
        architect with the critique, then rebuild through the deterministic encoder."""

        for iteration in range(1, MAX_FEEDBACK_ITERATIONS + 1):
            print(f"\n  --- Visual review (iteration {iteration}/{MAX_FEEDBACK_ITERATIONS}) ---")
            print("  Taking screenshot...")

            image_b64 = self.take_screenshot()
            if image_b64 is None:
                print("  Failed to take screenshot. Skipping review.")
                return original_parts

            parts_summary = "\n".join(
                f"  {p['name']}: X[{p['x_min']},{p['x_max']}] "
                f"Y[{p['y_min']},{p['y_max']}] Z[{p['z_min']},{p['z_max']}]"
                for p in original_parts
            )
            review_text = REVIEW_PROMPT.format(
                user_request=user_request,
                original_parts=parts_summary,
            )
            print("  Sending screenshot to model for evaluation...")

            try:
                raw = self._call_llm_with_image(
                    "You are a 3D model reviewer. Respond only with JSON.",
                    review_text,
                    image_b64,
                )
                review = self._parse_json(raw)
            except (json.JSONDecodeError, ValueError):
                print("  Model returned invalid JSON. Skipping review.")
                return original_parts
            except Exception as e:
                print(f"  Review error: {e}")
                return original_parts

            satisfied = review.get("satisfied", True)
            comment = review.get("comment", "")

            print(f"  Assessment: {comment}")

            if satisfied:
                print("  Model approved the result!")
                return original_parts

            # Re-run architect with the critique вЂ" VLM never touches coordinates
            print("  Re-running architect with feedback...\n")
            correction_request = (
                f'Original request: "{user_request}"\n'
                f"The model was built but a visual review found problems:\n"
                f"{comment}\n\n"
                f"Previous parts (bounding boxes):\n{parts_summary}\n\n"
                f"Please produce a CORRECTED set of parts that fixes these issues. "
                f"Keep the same coordinate conventions. Return the full parts list."
            )

            try:
                format_hint = (
                    '{"parts": [\n'
                    '  {"name": "Part A", "x_min": -20, "x_max": 20, '
                    '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3, "plane": "YZ"},\n'
                    '  ...\n'
                    ']}'
                )
                prompt = CORRECTION_PROMPT.replace("{format_hint}", format_hint)

                raw = self._call_llm(prompt, correction_request)
                result = self._parse_json(raw)
                new_parts = self._sanitize_parts(result.get("parts", []))
                new_parts = self._limit_parts_for_stability(new_parts, max_parts=max(12, len(original_parts)))
            except Exception as e:
                print(f"  Architect correction failed: {e}")
                return original_parts

            if not new_parts:
                print("  Architect returned no parts. Keeping current model.")
                return original_parts

            print(f"  Corrected architect produced {len(new_parts)} parts:")
            for p in new_parts:
                print(f"    - {p['name']:30s}  X[{p['x_min']:7.1f},{p['x_max']:7.1f}]"
                      f"  Y[{p['y_min']:7.1f},{p['y_max']:7.1f}]"
                      f"  Z[{p['z_min']:7.1f},{p['z_max']:7.1f}]")

            steps = encode_bboxes_to_plan(new_parts, include_clear=True)
            print("\n  Rebuilding in Fusion 360...\n")
            self.execute_plan(steps)
            original_parts = new_parts

        print(f"\n  Iteration limit reached ({MAX_FEEDBACK_ITERATIONS}).")
        return original_parts

    # -- main loop ------------------------------------------------------------

    def run(self):
        sep = "=" * 64
        print(sep)
        print("  Fusion 360 AI Assistant  (CAD DSL-first pipeline)")
        print(f"  Provider: {self.provider}  |  Model: {self.model}")
        if USE_TOOL_DRIVEN_AGENT:
            print("  Pipeline: Tool-Driven Agent (LLM calls CAD tools directly via tool_use)")
        elif USE_CAD_DSL_PLANNER:
            print("  Pipeline: Structural Spec -> family grammar DSL -> iterative tool-driven execute/inspect loop -> Fusion 360")
        else:
            print("  Pipeline: Architect LLM -> Deterministic Encoder -> Fusion 360")
        print("  Describe what you want to build in plain text. To extend current model: e.g. 'add armrests', 'make the back higher'.")
        print(f"  Visual review: {'ON' if self.review_enabled else 'OFF'}")
        print(f"  Tool-driven agent: {'ON' if USE_TOOL_DRIVEN_AGENT else 'OFF'}")
        print(f"  CAD DSL planner: {'ON' if USE_CAD_DSL_PLANNER else 'OFF (legacy bbox mode)'}")
        print(f"  Legacy bbox fallback for vision: {'ON' if USE_LEGACY_BBOX_FALLBACK else 'OFF'}")
        print("  Commands: exit | clear | detach | relaunch | review on/off/status | save <name[.ext]> | ping | help")
        print(sep)

        if not self.check_connection():
            print("\n  Fusion 360 Gym is not responding!")
            print("  Open Fusion 360 and run the add-in (Add-ins -> Run).\n")
            return
        print("  Fusion 360 Gym: connected\n")

        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            command_input = user_input[1:].strip() if user_input.startswith("/") else user_input
            user_lower = command_input.lower()

            if user_lower in ("exit", "quit", "q"):
                print("Exiting.")
                break
            if user_lower in ("help", "?"):
                print("\n  Commands:")
                print("    - exit / quit / q")
                print("    - clear")
                print("    - detach                (release Fusion UI; run add-in again to continue)")
                print("    - relaunch              (launch new Fusion instance and auto-reconnect)")
                print("    - review on | review off | review status")
                print("    - save <name>            (defaults to .f3d)")
                print("    - save <name.step>")
                print("    - save <name.smt>")
                print("    - save <name.obj>")
                print("    - save <name.stl>")
                print("    - ping")
                print("    - build model <id>      (build exact indexed design, no LLM)")
                print("    - build model <id> step (step-by-step replay for dataset model)")
                print("    - build from image <path>  (vision: build from photo; needs vision-capable model)")
                print("    - build from photo <path>")
                print("    - build from folder <path> (multi-view: all photos are one object)")
                print("    - detail <what to add/refine> (refine current model with functional details)")
                print("    - detail program <what to add/refine> (plan and execute detail actions)")
                print("  Default path is CAD DSL iterative create agent (spec -> stepwise plan/inspect/execute).")
                print("  To extend current model: e.g. 'add armrests', 'make the back higher' (no command).")
                print()
                continue
            if user_lower == "ping":
                if self.ensure_connection():
                    print("  Fusion 360 Gym: connected\n")
                continue
            if user_lower == "detach":
                if not self.ensure_connection():
                    continue
                try:
                    r = self.fusion.detach()
                    if r is not None and r.status_code == 200:
                        print("  Server detached. Fusion UI is free now.")
                        print("  To continue using this script: in Fusion run add-in again, then type 'ping'.\n")
                    else:
                        print("  Detach command sent, but server returned non-200.\n")
                except Exception as e:
                    print(f"  Detach failed: {e}\n")
                continue
            if user_lower == "relaunch":
                self.relaunch_gym()
                continue
            if user_lower.startswith("review "):
                mode = user_lower.split(" ", 1)[1].strip()
                if mode == "on":
                    self.review_enabled = True
                    print("  Visual review: ON\n")
                elif mode == "off":
                    self.review_enabled = False
                    print("  Visual review: OFF\n")
                elif mode == "status":
                    print(f"  Visual review is {'ON' if self.review_enabled else 'OFF'}\n")
                else:
                    print("  Usage: review on | review off | review status\n")
                continue
            if user_lower == "review":
                print(f"  Visual review is {'ON' if self.review_enabled else 'OFF'}\n")
                continue
            if user_lower.startswith("save "):
                if not self.ensure_connection():
                    continue
                name_raw = user_input.split(" ", 1)[1].strip().strip("\"'")
                if not name_raw:
                    print("  Usage: save <name[.f3d|.step|.smt|.obj|.stl]>\n")
                    continue
                out_path = Path(name_raw)
                if out_path.suffix == "":
                    out_path = out_path.with_suffix(".f3d")
                if not out_path.is_absolute():
                    out_path = Path.cwd() / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                suffix = out_path.suffix.lower()
                try:
                    if suffix in (".f3d", ".step", ".smt"):
                        r = self.fusion.brep(out_path)
                    elif suffix in (".obj", ".stl"):
                        r = self.fusion.mesh(out_path)
                    else:
                        print("  Unsupported extension. Use .f3d/.step/.smt/.obj/.stl\n")
                        continue
                    if r is not None and r.status_code == 200:
                        print(f"  Saved: {out_path}\n")
                    else:
                        print("  Save failed (server returned non-200).\n")
                except Exception as e:
                    print(f"  Save failed: {e}\n")
                continue
            if user_lower == "clear":
                if not self.ensure_connection():
                    continue
                self.fusion.clear()
                self.fusion.refresh()
                self.last_parts = []
                self.last_cad_plan = None
                self.last_request = ""
                self.last_images = []
                print("  Model cleared.\n")
                continue

            if (user_lower == "detail" or user_lower.startswith("detail ")) and not user_lower.startswith("detail program"):
                if not self.ensure_connection():
                    continue
                if not self.last_parts:
                    print("  No current model parts. Build something first.\n")
                    continue
                detail_req = command_input[len("detail "):].strip() if user_lower.startswith("detail ") else ""
                if not detail_req:
                    detail_req = f"Refine details for: {self.last_request or 'current object'}"
                print("\n  [Detail] Refining current model with functional details...\n")
                try:
                    detailed = self._detail_model_parts(
                        self.last_parts,
                        detail_req,
                        images=self.last_images if self.last_images else None,
                        max_parts=DETAILING_MAX_PARTS,
                    )
                except json.JSONDecodeError:
                    print("  Error: model returned invalid JSON for detail pass.\n")
                    continue
                except Exception as e:
                    print(f"  Detailing failed: {e}\n")
                    continue
                if not detailed:
                    print("  Detail pass returned no parts.\n")
                    continue
                print(f"  Detail pass produced {len(detailed)} parts.\n")
                self.execute_plan(self.encode(detailed))
                self.last_parts = self._interactive_user_refine(
                    f"detail {detail_req}",
                    detailed,
                    images=self.last_images if self.last_images else None,
                )
                self.last_cad_plan = None
                self.last_request = f"detail {detail_req}"
                print("  Done.\n")
                continue

            if user_lower == "detail program" or user_lower.startswith("detail program "):
                if not self.ensure_connection():
                    continue
                if not self.last_parts:
                    print("  No current model parts. Build something first.\n")
                    continue
                req = command_input[len("detail program "):].strip() if user_lower.startswith("detail program ") else ""
                if not req:
                    req = f"Add functional details for: {self.last_request or 'current object'}"
                print("\n  [Detail Program] Planning detail actions...\n")
                try:
                    steps = self._plan_detail_program_steps(
                        self.last_parts,
                        req,
                        images=self.last_images if self.last_images else None,
                    )
                except json.JSONDecodeError:
                    print("  Error: model returned invalid JSON for detail program.\n")
                    continue
                except Exception as e:
                    print(f"  Detail program planning failed: {e}\n")
                    continue
                if not steps:
                    print("  Planner returned no executable detail steps.\n")
                    continue
                print(f"  Planned {len(steps)} detail steps.\n")
                self.execute_plan(steps)
                added_parts = self._steps_to_bboxes(steps)
                if added_parts:
                    merged = list(self.last_parts) + added_parts
                    merged = self._sanitize_parts(merged, normalize_frame=False)
                    self.last_parts = self._limit_parts_for_stability(merged, max_parts=DETAILING_MAX_PARTS)
                self.last_cad_plan = None
                self.last_request = f"detail program {req}"
                print("  Done.\n")
                continue

            # Build from folder (multi-view vision model)
            if user_lower.startswith("build from folder "):
                if not self.ensure_connection():
                    continue
                folder_str = command_input[len("build from folder "):].strip().strip("\"'")
                if not folder_str:
                    print("  Usage: build from folder <path>\n")
                    continue
                folder = Path(folder_str)
                if not folder.is_absolute():
                    folder = Path.cwd() / folder
                if not folder.exists() or not folder.is_dir():
                    print(f"  Folder not found: {folder}\n")
                    continue
                img_files = self._gather_image_paths(folder, max_images=8)
                if not img_files:
                    print("  No image files found (.jpg/.jpeg/.png/.webp/.gif).\n")
                    continue
                images = self._load_images_from_paths(img_files)
                if not images:
                    print("  Failed to read images from folder.\n")
                    continue
                if USE_TOOL_DRIVEN_AGENT:
                    print(f"\n  [Tool-Driven Agent] Building from folder ({len(images)} views)...\n")
                    req = f"Build CAD model from folder: {folder}"
                    try:
                        ok, _ = self._run_tool_driven_agent(req, images=images)
                    except Exception as ex:
                        print(f"  Tool agent error: {ex}\n")
                        ok = False
                    if ok:
                        self.last_parts = []
                        self.last_cad_plan = None
                        self.last_request = req
                        self.last_images = images
                        print("  Done.\n")
                    else:
                        print("  Tool agent did not complete successfully.\n")
                    continue
                if USE_CAD_DSL_PLANNER:
                    print(f"\n  [CAD DSL] Vision planning from folder ({len(images)} views)...\n")
                    req = f"Build CAD model from folder: {folder}"
                    candidates = self._plan_images_to_cad_candidates(req, images, mode="create")
                    if not candidates and USE_LEGACY_BBOX_FALLBACK:
                        print("  CAD DSL image planner returned no plan. Trying explicit legacy bbox fallback...\n")
                        try:
                            result = self.decompose_from_images(images)
                            parts = result.get("parts", [])
                        except Exception:
                            parts = []
                        legacy_plan = self._legacy_bboxes_to_cad_plan(parts, req, mode="create") if parts else None
                        if legacy_plan is not None:
                            candidates = [legacy_plan]
                    if not candidates:
                        reason = self._create_policy_failure_reason()
                        if reason:
                            print(f"  Create policy blocked: {reason}")
                        print("  CAD DSL planner produced no valid plans for this folder.\n")
                        continue
                    print(f"  CAD DSL candidates: {len(candidates)}")
                    self._print_create_policy_summary()
                    ok, best_plan, _ = self._execute_cad_dsl_best(
                        candidates,
                        mode="create",
                        user_request=req,
                    )
                    if not ok:
                        print("  Build failed in CAD DSL mode.\n")
                        continue
                    self._print_create_policy_summary(best_plan=best_plan)
                    self.last_cad_plan = best_plan
                    self.last_parts = []
                    self.last_request = f"build from folder {folder}"
                    self.last_images = images
                    print("  Done.\n")
                    continue
                print(f"\n  [Step A] Architect (multi-view): using {len(images)} images from folder...\n")
                print("  View order sent to model:")
                for i, img in enumerate(images, 1):
                    print(f"    {i}. {img.get('label', 'unknown'):7s}  {img.get('name', '')}")
                print()
                try:
                    result = self.decompose_from_images(images)
                    parts = result.get("parts", [])
                except json.JSONDecodeError:
                    print("  Error: model returned invalid JSON. Try another folder.\n")
                    continue
                except Exception as e:
                    print(f"  LLM error: {e}\n")
                    continue
                if not parts:
                    print("  Error: no parts from multi-view photos.\n")
                    continue
                if ENABLE_DETAILING_PASS:
                    print("  [Step A2] Detail pass: enriching functional geometry...\n")
                    try:
                        detailed = self._detail_model_parts(
                            parts,
                            f"Object context: multi-view build from folder {folder}",
                            images=images,
                            max_parts=DETAILING_MAX_PARTS,
                        )
                        if detailed:
                            parts = detailed
                    except Exception:
                        pass
                print(f"  Architect produced {len(parts)} parts from folder:")
                for p in parts:
                    print(f"    - {p['name']:30s}  X[{p['x_min']:7.1f},{p['x_max']:7.1f}]"
                          f"  Y[{p['y_min']:7.1f},{p['y_max']:7.1f}]  Z[{p['z_min']:7.1f},{p['z_max']:7.1f}]")
                print()
                print("  [Step B] Encoder: converting to build commands...\n")
                steps = self.encode(parts)
                build_steps = [s for s in steps if s.get("action") == "build"]
                if build_steps:
                    print("  Build plan:")
                    for s in build_steps:
                        print("    - " + self._format_build_step_line(s))
                    print()
                print("  [Execute] Building in Fusion 360...\n")
                self.execute_plan(steps)
                if ENABLE_AUTO_DETAIL_PROGRAM:
                    print("  [AutoDetail] Running automatic detail loop...\n")
                    parts = self._auto_detail_program(
                        parts,
                        f"build from folder {folder}",
                        images=images,
                        rounds=AUTO_DETAIL_PROGRAM_ROUNDS,
                    )
                self.last_parts = self._interactive_user_refine(
                    f"build from folder {folder}",
                    parts,
                    images=images,
                )
                self.last_cad_plan = None
                self.last_request = f"build from folder {folder}"
                self.last_images = images
                print("  Done.\n")
                continue

            # Build from image (single-view vision model)
            if user_lower.startswith("build from image ") or user_lower.startswith("build from photo "):
                if not self.ensure_connection():
                    continue
                prefix = "build from image " if user_lower.startswith("build from image ") else "build from photo "
                path_str = command_input[len(prefix):].strip().strip("\"'")
                if not path_str:
                    print("  Usage: build from image <path>   or   build from photo <path>\n")
                    continue
                img_path = Path(path_str)
                if not img_path.is_absolute():
                    img_path = Path.cwd() / img_path
                if not img_path.exists():
                    print(f"  File not found: {img_path}\n")
                    continue
                images = self._load_images_from_paths([img_path])
                if not images:
                    print("  Could not read image.\n")
                    continue
                if USE_TOOL_DRIVEN_AGENT:
                    print(f"\n  [Tool-Driven Agent] Building from image...\n")
                    req = f"Build CAD model from image: {path_str}"
                    try:
                        ok, _ = self._run_tool_driven_agent(req, images=images)
                    except Exception as ex:
                        print(f"  Tool agent error: {ex}\n")
                        ok = False
                    if ok:
                        self.last_parts = []
                        self.last_cad_plan = None
                        self.last_request = req
                        self.last_images = images
                        print("  Done.\n")
                    else:
                        print("  Tool agent did not complete successfully.\n")
                    continue
                if USE_CAD_DSL_PLANNER:
                    print("\n  [CAD DSL] Vision planning from image...\n")
                    req = f"Build CAD model from image: {path_str}"
                    candidates = self._plan_images_to_cad_candidates(req, images, mode="create")
                    if not candidates and USE_LEGACY_BBOX_FALLBACK:
                        print("  CAD DSL image planner returned no plan. Trying explicit legacy bbox fallback...\n")
                        try:
                            result = self.decompose_from_images(images)
                            parts = result.get("parts", [])
                        except Exception:
                            parts = []
                        legacy_plan = self._legacy_bboxes_to_cad_plan(parts, req, mode="create") if parts else None
                        if legacy_plan is not None:
                            candidates = [legacy_plan]
                    if not candidates:
                        reason = self._create_policy_failure_reason()
                        if reason:
                            print(f"  Create policy blocked: {reason}")
                        print("  CAD DSL planner produced no valid plans for this image.\n")
                        continue
                    print(f"  CAD DSL candidates: {len(candidates)}")
                    self._print_create_policy_summary()
                    ok, best_plan, _ = self._execute_cad_dsl_best(
                        candidates,
                        mode="create",
                        user_request=req,
                    )
                    if not ok:
                        print("  Build failed in CAD DSL mode.\n")
                        continue
                    self._print_create_policy_summary(best_plan=best_plan)
                    self.last_cad_plan = best_plan
                    self.last_parts = []
                    self.last_request = f"build from image {path_str}"
                    self.last_images = images
                    print("  Done.\n")
                    continue
                print(f"\n  [Step A] Architect (vision): decomposing object from image...\n")
                try:
                    result = self.decompose_from_images(images)
                    parts = result.get("parts", [])
                except json.JSONDecodeError:
                    print("  Error: model returned invalid JSON. Try another image or text.\n")
                    continue
                except Exception as e:
                    print(f"  LLM error: {e}\n")
                    continue
                if not parts:
                    print("  Error: no parts from image. Try another photo or describe in text.\n")
                    continue
                if ENABLE_DETAILING_PASS:
                    print("  [Step A2] Detail pass: enriching functional geometry...\n")
                    try:
                        detailed = self._detail_model_parts(
                            parts,
                            f"Object context: build from image {path_str}",
                            images=images,
                            max_parts=DETAILING_MAX_PARTS,
                        )
                        if detailed:
                            parts = detailed
                    except Exception:
                        pass
                print(f"  Architect produced {len(parts)} parts from image:")
                for p in parts:
                    print(f"    - {p['name']:30s}  X[{p['x_min']:7.1f},{p['x_max']:7.1f}]"
                          f"  Y[{p['y_min']:7.1f},{p['y_max']:7.1f}]  Z[{p['z_min']:7.1f},{p['z_max']:7.1f}]")
                print()
                print("  [Step B] Encoder: converting to build commands...\n")
                steps = self.encode(parts)
                build_steps = [s for s in steps if s.get("action") == "build"]
                if build_steps:
                    print("  Build plan:")
                    for s in build_steps:
                        print("    - " + self._format_build_step_line(s))
                    print()
                print("  [Execute] Building in Fusion 360...\n")
                self.execute_plan(steps)
                if ENABLE_AUTO_DETAIL_PROGRAM:
                    print("  [AutoDetail] Running automatic detail loop...\n")
                    parts = self._auto_detail_program(
                        parts,
                        f"build from image {path_str}",
                        images=images,
                        rounds=AUTO_DETAIL_PROGRAM_ROUNDS,
                    )
                self.last_parts = self._interactive_user_refine(
                    f"build from image {path_str}",
                    parts,
                    images=images,
                )
                self.last_cad_plan = None
                self.last_request = f"build from image {path_str}"
                self.last_images = images
                print("  Done.\n")
                continue

            # Deterministic model build from index (no LLM)
            if user_lower.startswith("build model "):
                if not self.ensure_connection():
                    continue
                model_tokens = command_input.split()
                if len(model_tokens) < 3:
                    print("  Usage: build model <id> [step]\n")
                    continue
                model_id = model_tokens[2].strip()
                stepwise = len(model_tokens) >= 4 and model_tokens[3].lower() in ("step", "stepwise", "timeline")
                if not model_id:
                    print("  Usage: build model <id>\n")
                    continue
                design = self._find_exact_model_any(model_id)
                if design is None:
                    print(f"  Model '{model_id}' not found in current index.")
                    print("  Tip: set DESIGN_INDEX_FILE / TRACK_DESIGN_INDEX_FILE in .env and restart.\n")
                    continue

                parts = design.get("parts", [])
                if not parts:
                    print(f"  Model '{model_id}' has no parts in index.\n")
                    continue

                source_file = design.get("source_file", "")
                json_path = self.find_reconstruction_json(source_file)

                if json_path is not None:
                    # Preferred path for dataset models: exact replay from original JSON.
                    mode = "stepwise replay" if stepwise else "full reconstruct"
                    print(f"\n  [Indexed Model] Reconstructing '{model_id}' from dataset JSON ({mode}) ...")
                    print(f"  Source: {json_path}\n")
                    try:
                        if stepwise:
                            ok = self.reconstruct_stepwise(json_path, only_extrudes=True)
                            if not ok:
                                continue
                        else:
                            self.fusion.clear()
                            r = self.fusion.reconstruct(str(json_path))
                            if r is None or r.status_code != 200:
                                msg = ""
                                try:
                                    msg = r.json().get("message", "") if r is not None else ""
                                except Exception:
                                    pass
                                print(f"  Reconstruct failed. {msg}\n")
                                continue
                            self.fusion.refresh()
                            print("  Reconstruct completed.\n")
                    except Exception as e:
                        print(f"  Reconstruct failed: {e}\n")
                        continue
                    # No bbox-based review for reconstructed CAD sequences.
                    self.last_parts = []
                    self.last_cad_plan = None
                    self.last_request = f"build model {model_id}"
                    self.last_images = []
                    continue

                # Fallback: bbox replay (works for simple curated box-like models only).
                print(f"\n  [Indexed Model] Building '{model_id}' from bbox index ...\n")
                print(f"  Loaded {len(parts)} parts from index.")
                steps = self.encode(parts)

                build_steps = [s for s in steps if s.get("action") == "build"]
                print("  Build plan:")
                for s in build_steps:
                    print("    - " + self._format_build_step_line(s))
                print()

                print("  [Execute] Building in Fusion 360...\n")
                self.execute_plan(steps)
                self.last_parts = self._interactive_user_refine(f"build model {model_id}", parts)
                self.last_cad_plan = None
                self.last_request = f"build model {model_id}"
                self.last_images = []
                print()
                continue

            if not self.ensure_connection():
                continue

            # --- Tool-driven agent mode (free-form LLM tool_use) ---
            if USE_TOOL_DRIVEN_AGENT:
                print(f"\n  [Tool-Driven Agent] Building from request...\n")
                try:
                    ok, agent_result = self._run_tool_driven_agent(command_input)
                except Exception as ex:
                    print(f"  Tool agent error: {ex}\n")
                    ok = False
                if ok:
                    self.last_parts = []
                    self.last_cad_plan = None
                    self.last_request = command_input
                    self.last_images = []
                    print("  Done.\n")
                else:
                    print("  Tool agent did not complete successfully.\n")
                continue

            if USE_CAD_DSL_PLANNER:
                has_existing_plan = bool(isinstance(self.last_cad_plan, dict) and self.last_cad_plan.get("steps"))
                intent = self._intent_extend_or_new(command_input, has_existing_plan)
                mode = "edit" if intent == "extend" and has_existing_plan else "create"
                print(f"\n  [CAD DSL] Planning candidate plans from request (mode={mode})...\n")
                candidates = self._plan_text_to_cad_candidates(command_input, mode=mode)
                if not candidates:
                    if mode == "create":
                        reason = self._create_policy_failure_reason()
                        if reason:
                            print(f"  Create policy blocked: {reason}")
                    print("  CAD DSL planner produced no valid JSON plans. Try rephrasing request.\n")
                    continue
                print(f"  CAD DSL candidates: {len(candidates)}")
                if mode == "create":
                    self._print_create_policy_summary()
                ok, best_plan, _ = self._execute_cad_dsl_best(
                    candidates,
                    mode=mode,
                    previous_plan=self.last_cad_plan if mode == "edit" else None,
                    user_request=command_input,
                )
                if not ok:
                    print("  Build failed in CAD DSL mode.\n")
                    continue
                if mode == "create":
                    self._print_create_policy_summary(best_plan=best_plan)
                # In DSL-first mode bbox refinement is intentionally disabled.
                self.last_cad_plan = best_plan
                self.last_parts = []
                self.last_request = command_input
                self.last_images = []
                print()
                continue

            parts = []
            intent = self._intent_extend_or_new(command_input, bool(self.last_parts))
            if intent == "extend" and self.last_parts:
                # User wants to add to or modify the current model вЂ" revise, don't decompose from scratch
                print("\n  [Extend] Modifying current model per your request...\n")
                try:
                    parts = self._revise_model_parts(self.last_parts, command_input)
                except json.JSONDecodeError:
                    print("  Error: LLM returned invalid JSON. Falling back to building new.\n")
                    parts = []
                except Exception as e:
                    print(f"  Revision error: {e}. Falling back to building new.\n")
                    parts = []
                if not parts:
                    print("  Retrying as new build...\n")
                    intent = "new"

            if intent == "new" or not parts:
                # --- Step A: Architect LLM decomposes into bounding boxes ---
                print("\n  [Step A] Architect: decomposing object into 3D blocks...\n")
                try:
                    result = self.decompose_object(command_input)
                    parts = result.get("parts", [])
                except json.JSONDecodeError:
                    print("  Error: LLM returned invalid JSON. Try again.\n")
                    continue
                except Exception as e:
                    print(f"  LLM error: {e}\n")
                    continue

            if not parts:
                print("  Error: no parts generated. Try again.\n")
                continue

            print(f"  {'Revised' if intent == 'extend' else 'Architect'} produced {len(parts)} parts:")
            for p in parts:
                print(f"    - {p['name']:30s}  X[{p['x_min']:7.1f},{p['x_max']:7.1f}]"
                      f"  Y[{p['y_min']:7.1f},{p['y_max']:7.1f}]"
                      f"  Z[{p['z_min']:7.1f},{p['z_max']:7.1f}]")
            print()

            # --- Step B: Deterministic encoder в†’ Gym commands ---
            print("  [Step B] Encoder: converting bounding boxes to build commands...\n")
            steps = self.encode(parts)

            build_steps = [s for s in steps if s.get("action") == "build"]
            if build_steps:
                print("  Build plan:")
                for s in build_steps:
                    print("    - " + self._format_build_step_line(s))
                print()

            # --- Execute in Fusion 360 ---
            print("  [Execute] Building in Fusion 360...\n")
            self.execute_plan(steps)
            if ENABLE_AUTO_DETAIL_PROGRAM and self._should_auto_detail(command_input):
                print("  [AutoDetail] Running automatic detail loop...\n")
                parts = self._auto_detail_program(
                    parts,
                    command_input,
                    images=None,
                    rounds=AUTO_DETAIL_PROGRAM_ROUNDS,
                )
            elif ENABLE_AUTO_DETAIL_PROGRAM:
                print("  [AutoDetail] Skipped (no explicit detail intent).\n")
            self.last_parts = self._interactive_user_refine(command_input, parts)
            self.last_cad_plan = None
            self.last_request = command_input
            self.last_images = []
            print()


def main():
    provider, llm_client = create_llm_client()

    assistant = FusionAIAssistant(
        provider=provider,
        llm_client=llm_client,
        fusion_host=HOST_NAME,
        fusion_port=PORT_NUMBER,
    )
    assistant.run()


if __name__ == "__main__":
    main()
