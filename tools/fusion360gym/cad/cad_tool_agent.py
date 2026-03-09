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

## Plane coordinate mapping (MEMORIZE THIS)

| Plane      | Sketch X → | Sketch Y → | Extrude → | Typical use                    |
|------------|-----------|-----------|-----------|--------------------------------|
| `XY@Z_off` | World X   | World Y   | +Z (up)   | Bodies, floors, tabletops      |
| `XZ@Y_off` | World X   | World Z   | +Y (front)| Wheels, front/back panels      |
| `YZ@X_off` | World Y   | World Z   | +X (right)| Side panels, side features     |

**Default to XY plane** for most geometry. Use XZ or YZ only when you need sideways extrusion
(e.g., wheels, axles, side panels).

### Key rules for XZ and YZ planes:
- On XZ plane: sketch Y maps to World Z. **Keep sketch Y >= 0** to stay above the floor.
- On YZ plane: sketch Y maps to World Z. **Keep sketch Y >= 0** to stay above the floor.
- The plane offset sets position along the extrude axis (Y for XZ, X for YZ).
- After extruding on XZ or YZ, call `get_model_state` to verify coordinates are correct.

### Wheel example (XZ plane):
A car wheel at rear-left, radius=8cm, thickness=5cm, axle at Y=-30, Z=8:
  create_sketch("XZ@-30") → add_circle(center_x=0, center_y=8, radius=8) → extrude(5, Join)
  This places wheel center at world (0, -30, 8), extending from Y=-30 to Y=-25.

## Positioning rules (CRITICAL)
- **ALL geometry must have Z >= 0**. Nothing below the floor.
- **XY@Z sketch**: Drawing at (cx, cy) → world (cx, cy, Z). Extrude D → body from Z to Z+D.
- **XZ@Y sketch**: Drawing at (cx, cy) → world (cx, Y, cy). Extrude D → body from Y to Y+D.
- **YZ@X sketch**: Drawing at (cx, cy) → world (X, cx, cy). Extrude D → body from X to X+D.
- **To stack vertically**: If body A has z_max=10, sketch at "XY@10" (or "XY@9.5" for Join overlap).
- **Legs/supports**: Sketch at "XY@0", extrude upward. E.g., 40cm tall legs → extrude(40).
- **Tabletop on legs**: If legs end at Z=40, sketch at "XY@39.5" (overlap), extrude 3.5cm.
- **Symmetric parts**: Mirror X or Y coordinates. E.g., legs at (cx=-20, cy=-15) and (cx=20, cy=-15).
- **Never guess coordinates** — calculate from dimensions and existing geometry.
- After every extrude, the system checks for negative Z. If you see a Z-WARNING, fix the part.

## Planning before building
Before calling any tool, plan the entire object decomposition in a text block:
1. List ALL parts with their roles (e.g. "hull", "turret", "leg_front_left")
2. For each part, calculate exact coordinates: start Z, end Z, center X/Y, width, height
3. Determine build order: base/largest part first, then attached parts
4. Verify that all Z values are >= 0

## NEVER RESTART — fix forward
- Call `clear` ONCE at the very beginning. After that, NEVER call `clear` again.
- If something looks wrong in a screenshot, fix it by adding corrective geometry.
- Restarting wastes iterations. You have a limited budget — use it to build, not to redo.
- The system will BLOCK additional clear calls after the first extrude.

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

## JoinFeatureOperation overlap rules (prevents floating parts)
- The new extrusion MUST volumetrically intersect with an existing body by at least 0.5cm.
- For a part on TOP: sketch plane = target z_max - 0.5
- For a part BESIDE: the sketch profile must overlap the existing body's footprint in that plane by ≥0.5cm.
- For a wheel on side of body: the extrude range must overlap the body's Y or X range by ≥0.5cm.
- If Join fails or part appears detached, call `get_bodies` to check body count.
  If body count increased unexpectedly, the Join failed → fix overlap and retry.
- **Common mistake**: Placing a part entirely outside the existing body. Always verify that
  at least 0.5cm of the new extrusion overlaps with existing geometry in all 3 axes.

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

## Example 1: Table (XY plane only)

Plan:
- 4 legs: 3x3cm, Z=0→40, at corners of 44x29cm rectangle
- 1 tabletop: 50x35cm, 3cm thick, Z=39.5→43 (0.5cm overlap)
- All Z >= 0 ✓

Steps: clear → create_sketch("XY@0") → add_rectangle(cx=-22,cy=-13,w=3,h=3) → extrude(40, NewBody)
→ screenshot + get_model_state → [repeat for 3 more legs] → screenshot
→ create_sketch("XY@39.5") → add_rectangle(cx=0,cy=0,w=50,h=35) → extrude(3.5, Join)
→ screenshot + get_model_state

## Example 2: Car with wheels (XY + XZ planes)

Plan:
- Body: 180×80cm, 30cm tall, Z=10→40 (raised for wheel clearance). XY plane.
- Cabin: 80×70cm, 25cm tall, Z=39.5→64.5. XY plane, Join.
- 4 wheels: radius=15cm, thickness=8cm, centers at Z=15. XZ plane.
  - Front-left:  XZ@-36, cx=-70, cy=15 (cy=Z=15), extrude 8 toward +Y. Join.
  - Front-right: XZ@36, cx=-70, cy=15, extrude -8 toward -Y. Or XZ@28, extrude 8.
  - Rear-left:   XZ@-36, cx=70, cy=15, extrude 8. Join.
  - Rear-right:  XZ@28, cx=70, cy=15, extrude 8. Join.
- All Z >= 0 ✓ (wheel centers at Z=15, bottom at Z=0)

Key: wheels use XZ plane so circles extrude sideways (along Y), not upward.
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
        has_extruded = False  # Track whether first extrude happened (for clear-blocking)

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

                # Block clear after first extrude to prevent restart loops
                if tool_name == "clear" and has_extruded:
                    dispatch_result = {
                        "ok": False,
                        "error": (
                            "BLOCKED: clear is not allowed after building has started. "
                            f"You have {self.max_iterations - iteration} iterations remaining. "
                            "Fix issues by adding corrective geometry, not by restarting."
                        ),
                    }
                    if self.verbose:
                        print(f" {tool_name}=BLOCKED", end="", flush=True)
                else:
                    # Execute
                    dispatch_result = self.dispatcher.dispatch(tool_name, tool_input)

                tool_calls_made += 1

                # Track first successful extrude
                is_ok = dispatch_result.get("ok", False)
                if tool_name == "extrude" and is_ok:
                    has_extruded = True

                # Log
                action = "ok" if is_ok else "error"
                if self.verbose and tool_name != "clear":
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
                        f"This usually means you used an XZ or YZ plane incorrectly, "
                        f"or placed a sketch at a negative Z offset. "
                        f"Do NOT restart — continue building and compensate with corrective geometry."
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
