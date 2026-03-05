"""
Fusion 360 AI Assistant — two-step pipeline with deterministic encoder.

Architecture:
  1. Architect LLM: text → 3D bounding boxes (chain-of-thought decomposition)
  2. Deterministic encoder: bounding boxes → Fusion 360 Gym JSON commands
  3. Fusion 360 Gym: execute commands → live 3D model
  4. Visual feedback: screenshot → VLM review → fix loop

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

from fusion360gym_client import Fusion360GymClient

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
# Cap number of designs loaded into memory (0 = no cap). Use when index is huge.
DESIGN_INDEX_MAX_ENTRIES = int(os.environ.get("DESIGN_INDEX_MAX_ENTRIES", "0"))
RECON_DATASET_ROOT = os.environ.get("RECON_DATASET_ROOT", r"e:\Work\Fusion360\datasets\r1.0.1")
SERVER_LAUNCH_PATH = os.path.join(os.path.dirname(__file__), "..", "server", "launch.py")
STEP_REPLAY_DELAY = float(os.environ.get("STEP_REPLAY_DELAY", "0.8"))

BEST_OF_N_CANDIDATES = int(os.environ.get("BEST_OF_N_CANDIDATES", "3"))
MULTIVIEW_BEST_OF_N_CANDIDATES = int(
    os.environ.get("MULTIVIEW_BEST_OF_N_CANDIDATES", "6")
)
MULTIVIEW_MIN_CONFIRM_VIEWS = int(os.environ.get("MULTIVIEW_MIN_CONFIRM_VIEWS", "2"))

# ---------------------------------------------------------------------------
# Architect prompt — LLM decomposes objects into 3D bounding boxes
# ---------------------------------------------------------------------------

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
- build (rect or circle extrude only)

Build step schema:
{
  "action": "build",
  "description": "name",
  "plane": "XY" or "XY@<z_offset_cm>",
  "shape": "rect" or "circle",
  "cx": number,
  "cy": number,
  "w": number,      // required for rect
  "h": number,      // required for rect
  "radius": number, // required for circle
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
- Keep dimensions realistic relative to existing model bounds.
- If uncertain, output fewer safer steps.

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

EXTEND_REVISION_PROMPT = r"""The user has an EXISTING 3D model in the CAD program and wants to EXTEND or MODIFY it (add parts, change dimensions, add armrests, make back higher, etc.).

You receive:
1) The current model as a list of axis-aligned bounding boxes (name, x_min, x_max, y_min, y_max, z_min, z_max in cm).
2) The user's request describing what to add or change.

Your task: Output the COMPLETE updated parts list as a single JSON object with key "parts".
- You MAY add new parts (e.g. armrests, new back slats).
- You MAY remove or merge parts if the user asks.
- You MAY change dimensions/positions of existing parts.
- Keep the same coordinate system: floor Z=0, center X≈0, back Y≈0. Stay connected and physically plausible.
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
or build commands — just explain the problems clearly.
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
- tilt: use primary_axis "x" for backward/forward lean (YZ plane), "y" for left/right; direction "backward"|"forward"|"outward"; magnitude angle_deg 0–12; apply_to "slender_posts"|"slender_rails"|"legs"|"handles".
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
- tilt: primary_axis "x" for backward/forward (YZ), "y" for left/right; direction "backward"|"forward"|"outward"; magnitude angle_deg 0–12; apply_to "slender_posts"|"slender_rails"|"legs"|"handles".
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
# Design Index — few-shot retrieval from curated / dataset examples
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
# Deterministic encoder — bounding boxes → Fusion 360 Gym JSON plan
# ---------------------------------------------------------------------------

def encode_bboxes_to_plan(parts, include_clear=True, include_refresh=False):
    """Convert a list of 3D bounding boxes into Fusion 360 Gym build steps.

    Each part is a dict with: name, x_min, x_max, y_min, y_max, z_min, z_max.
    The encoder uses XY offset planes (XY@z_min) and extrudes +Z by box height.
    When include_clear is True, the first step is refresh (fit camera) so the view is usable during build.
    """
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


def _encode_single_bbox(part):
    """Convert one absolute 3D bbox into a stable XY@offset build step."""
    name = part.get("name", "unnamed")
    x0, x1 = part["x_min"], part["x_max"]
    y0, y1 = part["y_min"], part["y_max"]
    z0, z1 = part["z_min"], part["z_max"]

    wx = abs(x1 - x0)
    wy = abs(y1 - y0)
    wz = abs(z1 - z0)

    if wx < 0.01 or wy < 0.01 or wz < 0.01:
        return None

    # Stable default for demo mode:
    # sketch on XY plane offset to z_min, then extrude +Z by height.
    return _make_step(
        name=name,
        plane=f"XY@{round(z0, 3)}",
        cx=(x0 + x1) / 2,
        cy=(y0 + y1) / 2,
        w=wx,
        h=wy,
        distance=wz,
    )


def _make_step(name, plane, cx, cy, w, h, distance):
    return {
        "action": "build",
        "description": name,
        "plane": plane,
        "shape": "rect",
        "cx": round(cx, 1),
        "cy": round(cy, 1),
        "w": round(w, 1),
        "h": round(h, 1),
        "distance": round(distance, 1),
        "operation": "NewBodyFeatureOperation",
    }


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
        self.review_enabled = ENABLE_VISUAL_REVIEW
        self.last_parts = []
        self.last_request = ""
        self.last_images = []
        self.recon_root = Path(RECON_DATASET_ROOT)
        self._recon_file_cache = {}

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

    # -- Step A: Architect — decompose into bounding boxes --------------------

    def decompose_object(self, user_request):
        similar = self.index.find_similar(user_request)
        few_shot = self.index.format_few_shot(similar)

        if similar:
            names = [d["description"] for d in similar]
            print(f"  Found {len(similar)} reference design(s): {', '.join(names)}")

        format_hint = (
            '{"parts": [\n'
            '  {"name": "Part A", "x_min": -20, "x_max": 20, '
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
            '  ...\n'
            ']}'
        )

        prompt = ARCHITECT_PROMPT.replace("{format_hint}", format_hint)
        prompt = prompt.replace("{few_shot_section}", few_shot if few_shot else "")

        request = f'Decompose this object into 3D blocks: "{user_request}"'
        best_payload = None
        best_score = float("-inf")
        attempts = max(1, BEST_OF_N_CANDIDATES)
        for _ in range(attempts):
            raw = self._call_llm(prompt, request)
            payload = self._parse_json(raw)
            parts = self._sanitize_parts(payload.get("parts", []))
            score = self._score_parts(parts)
            if score > best_score:
                best_score = score
                best_payload = {"parts": parts}
        return best_payload or {"parts": []}

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
                '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
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
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
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
        candidates = self.index.find_similar(query, top_k=1)
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

    # -- Step B: Deterministic encoder — bboxes → Gym commands ----------------

    @staticmethod
    def encode(parts, include_clear=True):
        return encode_bboxes_to_plan(parts, include_clear)

    # -- intent: extend existing vs build new ---------------------------------

    @staticmethod
    def _intent_extend_or_new(user_input, has_existing_parts):
        """Classify request: 'extend' = modify/add to current model; 'new' = build from scratch."""
        if not has_existing_parts:
            return "new"
        text = (user_input or "").strip().lower()
        extend_cues = (
            "add ", "добавь", "extend", "to existing", "to current", "to the current",
            "also add", "make the ", "change the ", "modify", "higher", "longer", "wider",
            "дострой", "дополни", "ещё ", "подлокотник", "ручки", "на текущ", "к текущ",
            "в текущ", "текущую ", "текущий ", "эту модель", "этот ", "существующ",
            "на этом", "к этому", "armrest", "backrest", "спинку", "сиденье выше",
        )
        new_cues = (
            "from scratch", "с нуля", "заново", "new ", "another ", "different ",
            "новый ", "другой ", "build a new", "create a new", "start over",
        )
        for c in new_cues:
            if c in text:
                return "new"
        for c in extend_cues:
            if c in text:
                return "extend"
        return "extend"

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
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
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
        parts = self._sanitize_parts(parts)
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
            '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
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
        parts = self._sanitize_parts(parts)
        cap = max_parts if isinstance(max_parts, int) and max_parts > 0 else max(16, len(current_parts) + 8)
        return self._limit_parts_for_stability(parts, max_parts=cap)

    @staticmethod
    def _parse_xy_plane_offset(plane):
        s = str(plane or "XY").strip().upper()
        if s == "XY":
            return 0.0
        if s.startswith("XY@"):
            try:
                return float(s.split("@", 1)[1])
            except Exception:
                return 0.0
        return 0.0

    def _steps_to_bboxes(self, steps):
        out = []
        for i, st in enumerate(steps):
            if str(st.get("action", "")).lower() != "build":
                continue
            shape = str(st.get("shape", "rect")).lower()
            base_z0 = self._parse_xy_plane_offset(st.get("plane", "XY"))
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
                z0 = base_z0 + dz_rep * r_idx
                z1 = z0 + dist
                cx = base_cx + dx_rep * r_idx
                cy = base_cy + dy_rep * r_idx
                item_name = f"{name}_{r_idx+1}" if repeat > 1 else name
                if shape == "circle":
                    rad = abs(self._to_float(st.get("radius"), 0.0))
                    if rad <= 0.1:
                        continue
                    out.append({
                        "name": item_name,
                        "x_min": cx - rad, "x_max": cx + rad,
                        "y_min": cy - rad, "y_max": cy + rad,
                        "z_min": min(z0, z1), "z_max": max(z0, z1),
                    })
                else:
                    w = abs(self._to_float(st.get("w"), 0.0))
                    h = abs(self._to_float(st.get("h"), 0.0))
                    if w <= 0.1 or h <= 0.1:
                        continue
                    out.append({
                        "name": item_name,
                        "x_min": cx - w / 2.0, "x_max": cx + w / 2.0,
                        "y_min": cy - h / 2.0, "y_max": cy + h / 2.0,
                        "z_min": min(z0, z1), "z_max": max(z0, z1),
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
        for d in self.index.find_similar(query, top_k=top_k):
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

    def _sanitize_program_steps(self, steps, max_steps=None):
        if not isinstance(steps, list):
            return []
        cap = max_steps if isinstance(max_steps, int) and max_steps > 0 else PROGRAM_DETAIL_MAX_STEPS
        out = []
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
                if shape not in ("rect", "circle"):
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
                else:
                    step["radius"] = round(abs(self._to_float(st.get("radius"), 0.0)), 1)
                    if step["radius"] <= 0.1:
                        continue
                if step["operation"] not in ("NewBodyFeatureOperation", "JoinFeatureOperation", "CutFeatureOperation"):
                    step["operation"] = "NewBodyFeatureOperation"
                out.append(step)
            if len(out) >= cap:
                break
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
        request_text = (
            f"Current model parts (cm):\n{parts_summary}\n\n"
            f"Detail request: {user_request}\n\n"
            f"Dataset action hints:\n{json.dumps(hints, ensure_ascii=True)}\n\n"
            "Plan only additional steps to add functional details on top of current model."
        )
        raw = (
            self._call_llm_with_images(PROGRAM_DETAIL_PLANNER_PROMPT, request_text, images)
            if images else
            self._call_llm(PROGRAM_DETAIL_PLANNER_PROMPT, request_text)
        )
        payload = self._parse_json(raw)
        return self._sanitize_program_steps(payload.get("steps", []), max_steps=PROGRAM_DETAIL_MAX_STEPS)

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
            self.execute_plan(steps)
            added = self._steps_to_bboxes(steps)
            if not added:
                break
            merged = parts + added
            merged = self._sanitize_parts(merged)
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
        # Penalize negative Z and very thin/degenerate boxes.
        for p in parts:
            if p["z_min"] < -0.1:
                score -= 100.0
            dx = p["x_max"] - p["x_min"]
            dy = p["y_max"] - p["y_min"]
            dz = p["z_max"] - p["z_min"]
            if min(dx, dy, dz) < 0.2:
                score -= 40.0
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
        # Reward floor contact for supporting parts.
        floor_touch = sum(1 for p in parts if abs(p["z_min"]) <= 0.8)
        score += min(4, floor_touch) * 8.0
        return score

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

    def _sanitize_parts(self, parts, target_dims=None, symmetry="none"):
        """Normalize generated boxes for stable, universal reconstruction."""
        clean = []
        for i, p in enumerate(parts):
            name = str(p.get("name", f"Part {i+1}")).strip() or f"Part {i+1}"
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
            clean.append({
                "name": name,
                "x_min": x0, "x_max": x1,
                "y_min": y0, "y_max": y1,
                "z_min": z0, "z_max": z1,
            })
        if not clean:
            return []

        # Global translation normalization: center in X, anchor floor to Z=0, back to Y=0.
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

    @staticmethod
    def _extract_profile(response):
        if response.status_code != 200:
            return None
        profiles = response.json().get("data", {}).get("profiles", {})
        if not profiles:
            return None
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

            print(f"    [build] {desc}  |  {plane}  cx={step.get('cx')}"
                  f"  cy={step.get('cy')}  w={step.get('w')}"
                  f"  h={step.get('h')}  d={distance}"
                  f"  repeat={repeat}")

            base_cx = self._to_float(step.get("cx"), 0.0)
            base_cy = self._to_float(step.get("cy"), 0.0)
            base_plane = str(plane)

            for idx in range(repeat):
                cur_plane = base_plane
                if abs(dz_rep) > 1e-6:
                    z0 = self._parse_xy_plane_offset(base_plane) + dz_rep * idx
                    cur_plane = f"XY@{round(z0, 3)}"

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
                else:
                    print(f"            Unknown shape: {shape}")
                    return False

                if profile_id is None:
                    print("            Error: profile not created")
                    return False

                r = self.fusion.add_extrude(sketch_name, profile_id, distance, operation)
                if r.status_code != 200:
                    print(f"            Error: extrude -> {r.json().get('message', '')}")
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
                    '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
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
                new_parts = self._sanitize_parts(result.get("parts", []))
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

            # Re-run architect with the critique — VLM never touches coordinates
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
                    '"y_min": 0, "y_max": 40, "z_min": 0, "z_max": 3},\n'
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
        print("  Fusion 360 AI Assistant  (two-step pipeline)")
        print(f"  Provider: {self.provider}  |  Model: {self.model}")
        print("  Pipeline: Architect LLM → Deterministic Encoder → Fusion 360")
        print("  Describe what you want to build in plain text. To extend current model: e.g. 'add armrests', 'make the back higher'.")
        print(f"  Visual review: {'ON' if self.review_enabled else 'OFF'}")
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
                    merged = self._sanitize_parts(merged)
                    self.last_parts = self._limit_parts_for_stability(merged, max_parts=DETAILING_MAX_PARTS)
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
                        print(f"    - {s['description']:30s}  {s['plane']}  cx={s['cx']:6.1f}  cy={s['cy']:6.1f}  "
                              f"w={s['w']:6.1f}  h={s['h']:6.1f}  d={s['distance']:7.1f}")
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
                        print(f"    - {s['description']:30s}  {s['plane']}  cx={s['cx']:6.1f}  cy={s['cy']:6.1f}  "
                              f"w={s['w']:6.1f}  h={s['h']:6.1f}  d={s['distance']:7.1f}")
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
                design = self.index.find_exact_model(model_id)
                if design is None:
                    print(f"  Model '{model_id}' not found in current index.")
                    print("  Tip: set DESIGN_INDEX_FILE in .env and restart.\n")
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
                    print(f"    - {s['description']:30s}  "
                          f"{s['plane']}  cx={s['cx']:6.1f}  cy={s['cy']:6.1f}  "
                          f"w={s['w']:6.1f}  h={s['h']:6.1f}  d={s['distance']:7.1f}")
                print()

                print("  [Execute] Building in Fusion 360...\n")
                self.execute_plan(steps)
                self.last_parts = self._interactive_user_refine(f"build model {model_id}", parts)
                self.last_request = f"build model {model_id}"
                self.last_images = []
                print()
                continue

            if not self.ensure_connection():
                continue

            intent = self._intent_extend_or_new(command_input, bool(self.last_parts))
            if intent == "extend" and self.last_parts:
                # User wants to add to or modify the current model — revise, don't decompose from scratch
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

            # --- Step B: Deterministic encoder → Gym commands ---
            print("  [Step B] Encoder: converting bounding boxes to build commands...\n")
            steps = self.encode(parts)

            build_steps = [s for s in steps if s.get("action") == "build"]
            if build_steps:
                print("  Build plan:")
                for s in build_steps:
                    print(f"    - {s['description']:30s}  "
                          f"{s['plane']}  cx={s['cx']:6.1f}  cy={s['cy']:6.1f}  "
                          f"w={s['w']:6.1f}  h={s['h']:6.1f}  d={s['distance']:7.1f}")
                print()

            # --- Execute in Fusion 360 ---
            print("  [Execute] Building in Fusion 360...\n")
            self.execute_plan(steps)
            if ENABLE_AUTO_DETAIL_PROGRAM:
                print("  [AutoDetail] Running automatic detail loop...\n")
                parts = self._auto_detail_program(
                    parts,
                    command_input,
                    images=None,
                    rounds=AUTO_DETAIL_PROGRAM_ROUNDS,
                )
            self.last_parts = self._interactive_user_refine(command_input, parts)
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
