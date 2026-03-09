"""
Tool-driven CAD agent — Claude directly calls Fusion CAD tools.

This replaces the template/grammar pipeline with free-form LLM reasoning.
Claude sees the available CAD tools, reasons about geometry, and calls
tools iteratively to build the requested object.

Architecture:
  User request
    -> Claude sees tool definitions
    -> Claude calls create_sketch / add_rectangle / extrude / get_bodies / ...
    -> Backend executes in Fusion 360
    -> Claude sees result, decides next step
    -> Repeat until done
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cad_backend import FusionCadBackend
from .cad_capabilities import default_capability_model
from .cad_tool_use import ToolDispatcher, get_tool_definitions


# ---------------------------------------------------------------------------
# System prompt for the tool-driven agent
# ---------------------------------------------------------------------------

TOOL_AGENT_SYSTEM_PROMPT = r"""You are a CAD engineer building 3D objects in Fusion 360.

You have direct access to CAD tools. Use them to build what the user requests.

## CRITICAL RULE: One part per step
Each step should build exactly ONE part:
1. Call `create_sketch` and WAIT for the result to get the sketch_name.
2. In the NEXT step, use the returned sketch_name to call `add_rectangle`/`add_circle`/`add_polygon`.
3. In the NEXT step, call `extrude` using the same sketch_name.

You CAN batch independent tools (e.g., `get_model_state` + `screenshot`), but NEVER batch
tools that depend on each other's results. The sketch_name returned by create_sketch is
needed by add_rectangle and extrude — you must wait for it.

Exception: `clear` can be called alone or with other independent tools.

## Coordinate system
- Units: centimeters (cm)
- X: left/right (center at X=0)
- Y: front/back (back at Y=0, front at +Y)
- Z: up/down (floor at Z=0, up is +Z)

## ONLY use XY plane — NEVER use XZ or YZ
- ALWAYS use `XY` or `XY@<offset>` as the sketch plane.
- The XY plane is the only plane with predictable coordinate mapping:
  - Sketch X → World X
  - Sketch Y → World Y
  - Plane offset → World Z (height)
  - Extrude goes along +Z
- XZ and YZ planes have inverted coordinate mappings in Fusion 360 that cause
  parts to appear at negative Z (below the floor). NEVER use them.
- To build vertical walls or side features, use XY plane at the correct Z offset
  and shape the profile in X/Y.

## Positioning rules (CRITICAL)
- **XY@Z sketch**: Drawing at (cx, cy) means world position (cx, cy, Z). Extrude distance D creates geometry from Z to Z+D.
- **ALL geometry must have Z >= 0**. Nothing below the floor.
- **To stack parts**: If body A ends at z_max=10, next sketch at "XY@10" (or "XY@9.5" for Join overlap).
- **Legs/supports**: Sketch at "XY@0", extrude upward. E.g., 40cm tall legs → extrude(40).
- **Tabletop on legs**: If legs end at Z=40, sketch at "XY@39.5" (overlap), extrude 3.5cm → top at Z=43.
- **Symmetric parts**: Mirror X or Y coordinates. E.g., legs at (cx=-20, cy=-15) and (cx=20, cy=-15).
- **Never guess coordinates** — calculate from dimensions and existing geometry.

## Planning before building
Before calling any tool, plan the entire object decomposition in a text block:
1. List ALL parts with their roles (e.g. "hull", "turret", "leg_front_left")
2. For each part, calculate exact coordinates: start Z, end Z, center X/Y, width, height
3. Determine build order: base/largest part first, then attached parts
4. Verify that all Z values are >= 0

## Workflow
1. Call `clear` to reset the model.
2. Call `create_sketch("XY@0")` for the base part.
3. Add geometry to the sketch (add_rectangle, add_circle, etc.).
4. Extrude with NewBodyFeatureOperation.
5. Call `get_model_state` + `screenshot` to verify the base body.
6. Build next part: create_sketch → add geometry → extrude.
7. After every 2-3 parts, call `screenshot` to verify progress.
8. Continue until done. Take a final `screenshot`.

## Screenshot schedule (MANDATORY)
- Take a `screenshot` after the FIRST extrude to verify the base body.
- Take a `screenshot` after every 2-3 new parts to check positioning.
- Take a final `screenshot` when done.
- You should take at least 3 screenshots during a typical build.
- Each screenshot auto-fits the camera to show the entire model.
- If you see parts below the floor (negative Z) in a screenshot, something is wrong — fix it.

## JoinFeatureOperation overlap rules
- The new extrusion MUST volumetrically intersect with an existing body by at least 0.5cm.
- For a part on TOP: sketch plane = target z_max - 0.5
- For a part BESIDE: the extrude profile should overlap the existing body footprint by ≥0.5cm.
- If Join fails, check coordinates and retry with corrected overlap.

## Sketch rules
- Each sketch has exactly ONE profile. Create a new sketch for each separate extrusion.
- profile_id is typically "profile_0".
- add_rectangle/add_circle/add_polygon automatically close the profile. Do NOT call close_profile after them.
- Only call close_profile when building manual shapes with add_line.

## Quality guidelines
- Build 5-10 major parts for most objects. Add meaningful detail.
- Use realistic proportions and real-world scale in cm.
- Keep ALL geometry above Z=0 (floor level).
- Center the object around X=0, Y=0 when possible.

## Example: Building a simple table (height=40cm, top=50x35cm, legs=3x3cm)

Plan:
- 4 legs: 3x3cm cross-section, Z=0 to Z=40, at corners of a 44x29cm rectangle
- 1 tabletop: 50x35cm, 3cm thick, Z=39.5 to Z=43 (0.5cm overlap with legs)
- All Z >= 0 ✓

Step 1: clear
  clear()

Step 2: create sketch for leg 1
  create_sketch("XY@0") → returns sketch_name

Step 3: draw leg 1 profile
  add_rectangle(sketch_name, cx=-22, cy=-13, w=3, h=3)

Step 4: extrude leg 1
  extrude(sketch_name, distance=40, operation=NewBodyFeatureOperation)

Step 5: verify base
  get_model_state() + screenshot()

Step 6-8: repeat for legs 2-4 (each: create_sketch → add_rectangle → extrude)

Step 9: verify all legs
  screenshot()

Step 10-12: build tabletop
  create_sketch("XY@39.5") → add_rectangle(cx=0,cy=0,w=50,h=35) → extrude(3.5, JoinFeatureOperation)

Step 13: final verification
  get_model_state() + screenshot()
"""


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class ToolAgentResult:
    """Result of tool-driven agent execution."""
    ok: bool
    steps_executed: int = 0
    iterations: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    error: str = ""
    conversation: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool-driven agent
# ---------------------------------------------------------------------------

class ToolDrivenAgent:
    """
    Agent that lets Claude directly call Fusion CAD tools.

    Instead of template -> grammar -> compile -> execute,
    this agent gives Claude the tool definitions and lets it
    reason about geometry and call tools iteratively.
    """

    def __init__(
        self,
        backend: FusionCadBackend,
        llm_call: Callable,
        *,
        model: str = "",
        max_iterations: int = 50,
        step_delay: float = 0.3,
        verbose: bool = True,
    ):
        self.backend = backend
        self.llm_call = llm_call
        self.model = model
        self.max_iterations = max_iterations
        self.step_delay = step_delay
        self.verbose = verbose
        self.dispatcher = ToolDispatcher(backend)
        self.tools = get_tool_definitions()

    def run(self, user_request: str, images: Optional[List[Dict[str, Any]]] = None) -> ToolAgentResult:
        """
        Run the tool-driven agent for a user request.

        Args:
            user_request: What the user wants to build
            images: Optional reference images

        Returns:
            ToolAgentResult with execution details
        """
        result = ToolAgentResult(ok=False)

        # Build initial message
        user_content = self._build_user_content(user_request, images)
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]

        tool_calls_made = 0

        for iteration in range(1, self.max_iterations + 1):
            result.iterations = iteration

            if self.verbose:
                print(f"    - iter={iteration}", end="", flush=True)

            # Call LLM with tools
            response = self._call_llm_with_tools(messages)
            if response is None:
                result.error = "LLM call failed"
                if self.verbose:
                    print(f" error: LLM call failed")
                break

            # Extract stop_reason and content
            stop_reason = self._get_stop_reason(response)
            content_blocks = self._get_content_blocks(response)

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": content_blocks})

            # Check if done (no more tool calls)
            tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

            if not tool_use_blocks:
                # Claude is done — extract any text response
                text_blocks = [b for b in content_blocks if b.get("type") == "text"]
                if text_blocks and self.verbose:
                    print(f" done")
                    for tb in text_blocks:
                        text = tb.get("text", "").strip()
                        if text:
                            print(f"    Agent: {text[:200]}")
                result.ok = True
                break

            # Execute tool calls
            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                tool_id = block.get("id", "")

                # Execute
                dispatch_result = self.dispatcher.dispatch(tool_name, tool_input)
                tool_calls_made += 1

                # Log
                is_ok = dispatch_result.get("ok", False)
                action = "ok" if is_ok else "error"
                if self.verbose:
                    print(f" {tool_name}={action}", end="", flush=True)

                result.tool_calls.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "input": tool_input,
                    "ok": is_ok,
                })

                # Build tool result for Claude
                # Screenshot returns image — send as multimodal content
                if tool_name == "screenshot" and is_ok and "image_base64" in dispatch_result:
                    tool_result_content = [
                        {"type": "text", "text": "Screenshot captured (camera fitted to model):"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": dispatch_result["image_base64"],
                            },
                        },
                    ]
                else:
                    tool_result_content = self._format_tool_result(dispatch_result)

                # Z-validation after extrude: check for negative Z
                if tool_name == "extrude" and is_ok:
                    z_warning = self._check_negative_z()
                    if z_warning:
                        if self.verbose:
                            print(f" Z-WARNING", end="", flush=True)
                        # Append warning to the tool result
                        if isinstance(tool_result_content, str):
                            tool_result_content = tool_result_content + "\n\n" + z_warning
                        elif isinstance(tool_result_content, list):
                            tool_result_content.append({"type": "text", "text": z_warning})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_result_content,
                })

                if self.step_delay > 0:
                    time.sleep(self.step_delay)

            if self.verbose:
                print()

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

            result.steps_executed = tool_calls_made

        # Get final model state (include bodies separately)
        try:
            state_result = self.backend.get_model_state()
            if state_result.ok:
                result.final_state = self.dispatcher._extract_data(state_result)
            # Also fetch bodies for accurate reporting
            bodies_result = self.backend.get_bodies()
            if bodies_result.ok:
                bodies_data = self.dispatcher._extract_data(bodies_result)
                if result.final_state is None:
                    result.final_state = {}
                if isinstance(result.final_state, dict):
                    result.final_state["bodies"] = bodies_data
        except Exception:
            pass

        if result.iterations >= self.max_iterations and not result.ok:
            result.error = "max iterations reached"
            result.ok = True  # partial success

        result.conversation = messages
        return result

    def _build_user_content(
        self, user_request: str, images: Optional[List[Dict[str, Any]]]
    ) -> Any:
        """Build the user message content, optionally with images."""
        if not images:
            return user_request

        # Multi-modal: images + text
        content: List[Dict[str, Any]] = []
        for idx, img in enumerate(images, 1):
            label = str(img.get("label", f"view_{idx}")).strip()
            content.append({"type": "text", "text": f"[Reference image {idx}] {label}"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": img["data"],
                },
            })
        content.append({"type": "text", "text": user_request})
        return content

    def _call_llm_with_tools(self, messages: List[Dict[str, Any]]) -> Any:
        """Call the LLM with tool definitions."""
        try:
            response = self.llm_call(
                model=self.model,
                system=TOOL_AGENT_SYSTEM_PROMPT,
                messages=messages,
                tools=self.tools,
                max_tokens=8192,
                temperature=0.1,
            )
            return response
        except Exception as ex:
            if self.verbose:
                print(f" LLM error: {ex}")
            return None

    @staticmethod
    def _get_stop_reason(response: Any) -> str:
        """Extract stop_reason from API response."""
        if hasattr(response, "stop_reason"):
            return str(response.stop_reason or "")
        if isinstance(response, dict):
            return str(response.get("stop_reason", ""))
        return ""

    @staticmethod
    def _get_content_blocks(response: Any) -> List[Dict[str, Any]]:
        """Extract content blocks from API response."""
        if hasattr(response, "content"):
            blocks = []
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                elif isinstance(block, dict):
                    blocks.append(block)
            return blocks
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return []

    def _check_negative_z(self) -> str:
        """Check model bounding box for negative Z values after extrude."""
        try:
            bbox_result = self.backend.query_bounding_box()
            if not bbox_result.ok:
                return ""
            data = self.dispatcher._extract_data(bbox_result)
            if isinstance(data, dict):
                z_min = None
                # Try different response formats
                if "z_min" in data:
                    z_min = data["z_min"]
                elif "min" in data and isinstance(data["min"], dict):
                    z_min = data["min"].get("z")
                elif "min_point" in data and isinstance(data["min_point"], dict):
                    z_min = data["min_point"].get("z")
                if z_min is not None and z_min < -0.01:
                    return (
                        f"⚠ WARNING: Model has geometry below floor level! "
                        f"Bounding box z_min = {z_min:.2f} cm. "
                        f"All geometry should be at Z >= 0. "
                        f"This usually means you used an XZ or YZ plane, or placed a sketch at a negative Z offset. "
                        f"Use ONLY 'XY' or 'XY@<positive_offset>' planes. "
                        f"Consider clearing and rebuilding the affected part at the correct Z position."
                    )
        except Exception:
            pass
        return ""

    @staticmethod
    def _format_tool_result(dispatch_result: Dict[str, Any]) -> str:
        """Format tool dispatch result as string for Claude."""
        if dispatch_result.get("ok"):
            data = dispatch_result.get("result", "ok")
            if isinstance(data, (dict, list)):
                try:
                    return json.dumps(data, default=str, indent=None)[:4000]
                except Exception:
                    return str(data)[:4000]
            return str(data)[:4000]
        return f"ERROR: {dispatch_result.get('error', 'unknown error')}"
