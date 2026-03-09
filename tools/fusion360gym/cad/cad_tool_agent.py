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

## Coordinate system
- Units: centimeters (cm)
- X: left/right (center at X=0)
- Y: front/back (back at Y=0, front at +Y)
- Z: up (floor at Z=0)
- Plane XY extrudes along +Z
- Plane XZ extrudes along +Y (sketch Y maps to -Z in world)
- Plane YZ extrudes along +X

## Workflow
1. Start by calling `clear` to reset the model (unless extending existing work).
2. Think about the object decomposition: what are the main masses?
3. Build the foundation/base body first using `create_sketch` + `add_rectangle`/`add_polygon`/`add_circle` + `extrude` with NewBodyFeatureOperation.
4. Add attached parts using JoinFeatureOperation (requires volumetric overlap with existing body).
5. Add cut features using CutFeatureOperation for holes/slots.
6. Use `get_bodies` or `get_model_state` to inspect what you've built.
7. Continue until the object is complete.

## Rules
- Keep geometry physically connected and plausible.
- Floor is at Z=0. No negative Z.
- Use realistic proportions.
- For JoinFeatureOperation: the new extrusion MUST physically overlap with an existing body (at least 0.5cm overlap).
- When building a profile, always close it before extruding.
- profile_id is typically "profile_0" for the first (and usually only) profile in a sketch.
- Each sketch can only have one profile. Create separate sketches for separate extrusions.
- Prefer simpler geometry with fewer steps over complex detail.
- If you're unsure about exact dimensions, use reasonable estimates.

## Sketch workflow
For each solid body you want to create:
1. create_sketch(plane) — creates a new sketch
2. add_rectangle/add_circle/add_polygon — draw the profile shape
3. extrude(sketch_name, profile_id, distance, operation) — create the 3D body

## Example: Building a simple table
1. create_sketch("XY@0") → sketch for legs
2. add_rectangle(sketch, cx=-20, cy=-15, w=3, h=3) → leg profile
3. extrude(sketch, "profile_0", 40, "NewBodyFeatureOperation") → first leg
4. create_sketch("XY@0") → new sketch for second leg
5. add_rectangle(sketch, cx=20, cy=-15, w=3, h=3)
6. extrude(sketch, "profile_0", 40, "NewBodyFeatureOperation") → second leg
7. create_sketch("XY@40") → sketch for tabletop at z=40
8. add_rectangle(sketch, cx=0, cy=0, w=50, h=35)
9. extrude(sketch, "profile_0", 3, "JoinFeatureOperation") → tabletop merged with legs
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
        max_iterations: int = 30,
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
                action = "ok" if dispatch_result.get("ok") else "error"
                if self.verbose:
                    print(f" {tool_name}={action}", end="", flush=True)

                result.tool_calls.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "input": tool_input,
                    "ok": dispatch_result.get("ok", False),
                })

                # Build tool result for Claude
                tool_result_content = self._format_tool_result(dispatch_result)
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

        # Get final model state
        try:
            state_result = self.backend.get_model_state()
            if state_result.ok:
                result.final_state = self.dispatcher._extract_data(state_result)
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
                max_tokens=4096,
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
