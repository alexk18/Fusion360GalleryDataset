"""Tests for tool-driven agent: tool definitions, dispatcher, and agent loop."""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cad.cad_tool_use import get_tool_definitions, ToolDispatcher, CAD_TOOLS
from cad.cad_tool_agent import ToolDrivenAgent, ToolAgentResult, TOOL_AGENT_SYSTEM_PROMPT
from cad.cad_backend import FusionCadBackend, BackendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_backend():
    """Create a mock FusionCadBackend that returns ok for everything."""
    backend = MagicMock(spec=FusionCadBackend)
    backend.create_sketch.return_value = BackendResult(ok=True, response={"data": {"sketch_name": "Sketch1"}})
    backend.add_point.return_value = BackendResult(ok=True, response={"data": "ok"})
    backend.add_line.return_value = BackendResult(ok=True, response={"data": "ok"})
    backend.add_circle.return_value = BackendResult(ok=True, response={"data": "ok"})
    backend.close_profile.return_value = BackendResult(ok=True, response={"data": "ok"})
    backend.extrude.return_value = BackendResult(ok=True, response={"data": {"feature_name": "Extrude1"}})
    backend.get_model_state.return_value = BackendResult(ok=True, response={"data": {"bodies": ["Body1"]}})
    backend.get_bodies.return_value = BackendResult(ok=True, response={"data": ["Body1"]})
    backend.get_features.return_value = BackendResult(ok=True, response={"data": []})
    backend.get_body_bbox.return_value = BackendResult(ok=True, response={"data": {}})
    backend.get_connected_components.return_value = BackendResult(ok=True, response={"data": []})
    backend.get_body_relations.return_value = BackendResult(ok=True, response={"data": []})
    backend.query_bounding_box.return_value = BackendResult(ok=True, response={"data": {}})
    backend.screenshot.return_value = BackendResult(ok=True, response={"data": "ok"})
    backend._call.return_value = {"data": "ok"}
    return backend


# ---------------------------------------------------------------------------
# Tool definitions tests
# ---------------------------------------------------------------------------

class TestToolDefinitions(unittest.TestCase):
    def test_tool_count(self):
        tools = get_tool_definitions()
        self.assertEqual(len(tools), 16)

    def test_all_tools_have_required_fields(self):
        for tool in CAD_TOOLS:
            self.assertIn("name", tool, f"Tool missing 'name': {tool}")
            self.assertIn("description", tool, f"Tool {tool.get('name')} missing 'description'")
            self.assertIn("input_schema", tool, f"Tool {tool.get('name')} missing 'input_schema'")

    def test_tool_names_unique(self):
        names = [t["name"] for t in CAD_TOOLS]
        self.assertEqual(len(names), len(set(names)), "Duplicate tool names found")

    def test_essential_tools_present(self):
        names = {t["name"] for t in CAD_TOOLS}
        essential = {"create_sketch", "extrude", "add_rectangle", "add_circle", "clear", "get_bodies"}
        self.assertTrue(essential.issubset(names), f"Missing: {essential - names}")

    def test_extrude_operations_enum(self):
        extrude_tool = next(t for t in CAD_TOOLS if t["name"] == "extrude")
        ops = extrude_tool["input_schema"]["properties"]["operation"]["enum"]
        self.assertIn("NewBodyFeatureOperation", ops)
        self.assertIn("JoinFeatureOperation", ops)
        self.assertIn("CutFeatureOperation", ops)


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestToolDispatcher(unittest.TestCase):
    def setUp(self):
        self.backend = make_mock_backend()
        self.dispatcher = ToolDispatcher(self.backend)

    def test_dispatch_create_sketch(self):
        result = self.dispatcher.dispatch("create_sketch", {"sketch_plane": "XY"})
        self.assertTrue(result["ok"])
        self.backend.create_sketch.assert_called_once_with(sketch_plane="XY", sketch_name=None)

    def test_dispatch_add_rectangle(self):
        result = self.dispatcher.dispatch("add_rectangle", {
            "sketch_name": "Sketch1", "cx": 0, "cy": 0, "width": 10, "height": 5,
        })
        self.assertTrue(result["ok"])
        # Should call add_point 4 times + close_profile
        self.assertEqual(self.backend.add_point.call_count, 4)
        self.backend.close_profile.assert_called_once_with("Sketch1")

    def test_dispatch_extrude(self):
        result = self.dispatcher.dispatch("extrude", {
            "sketch_name": "Sketch1", "distance": 10, "operation": "NewBodyFeatureOperation",
        })
        self.assertTrue(result["ok"])
        self.backend.extrude.assert_called_once_with(
            sketch_name="Sketch1", profile_id="profile_0",
            distance=10, operation="NewBodyFeatureOperation", feature_name=None,
        )

    def test_dispatch_unknown_tool(self):
        result = self.dispatcher.dispatch("nonexistent_tool", {})
        self.assertFalse(result["ok"])
        self.assertIn("Unknown tool", result["error"])

    def test_dispatch_add_circle(self):
        result = self.dispatcher.dispatch("add_circle", {
            "sketch_name": "Sketch1", "center": {"x": 0, "y": 0}, "radius": 5,
        })
        self.assertTrue(result["ok"])
        self.backend.add_circle.assert_called_once()

    def test_dispatch_add_polygon(self):
        result = self.dispatcher.dispatch("add_polygon", {
            "sketch_name": "Sketch1",
            "points": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 5, "y": 8}],
        })
        self.assertTrue(result["ok"])
        self.assertEqual(self.backend.add_point.call_count, 3)
        self.backend.close_profile.assert_called_once()

    def test_dispatch_get_bodies(self):
        result = self.dispatcher.dispatch("get_bodies", {})
        self.assertTrue(result["ok"])
        self.backend.get_bodies.assert_called_once()

    def test_dispatch_clear(self):
        result = self.dispatcher.dispatch("clear", {})
        self.assertTrue(result["ok"])
        self.backend._call.assert_called_once_with("clear", {})

    def test_dispatch_error_propagation(self):
        self.backend.create_sketch.return_value = BackendResult(ok=False, reason="test error")
        result = self.dispatcher.dispatch("create_sketch", {"sketch_plane": "XY"})
        self.assertFalse(result["ok"])
        self.assertIn("test error", result["error"])

    def test_dispatch_exception_handling(self):
        self.backend.create_sketch.side_effect = ValueError("boom")
        result = self.dispatcher.dispatch("create_sketch", {"sketch_plane": "XY"})
        self.assertFalse(result["ok"])
        self.assertIn("ValueError", result["error"])


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestToolDrivenAgent(unittest.TestCase):
    def _make_llm_response(self, content_blocks, stop_reason="end_turn"):
        """Create a mock Anthropic-style response."""
        resp = MagicMock()
        resp.stop_reason = stop_reason

        blocks = []
        for block in content_blocks:
            b = MagicMock()
            b.type = block["type"]
            if block["type"] == "text":
                b.text = block["text"]
            elif block["type"] == "tool_use":
                b.id = block["id"]
                b.name = block["name"]
                b.input = block["input"]
            blocks.append(b)
        resp.content = blocks
        return resp

    def test_agent_simple_build(self):
        """Agent builds a box: clear -> create_sketch -> add_rectangle -> extrude -> done."""
        backend = make_mock_backend()
        call_count = [0]

        def mock_llm_call(*, model, system, messages, tools, max_tokens, temperature):
            call_count[0] += 1
            if call_count[0] == 1:
                # First: clear
                return self._make_llm_response([
                    {"type": "tool_use", "id": "call_1", "name": "clear", "input": {}},
                ], stop_reason="tool_use")
            elif call_count[0] == 2:
                # Second: create_sketch
                return self._make_llm_response([
                    {"type": "tool_use", "id": "call_2", "name": "create_sketch",
                     "input": {"sketch_plane": "XY"}},
                ], stop_reason="tool_use")
            elif call_count[0] == 3:
                # Third: add_rectangle
                return self._make_llm_response([
                    {"type": "tool_use", "id": "call_3", "name": "add_rectangle",
                     "input": {"sketch_name": "Sketch1", "cx": 0, "cy": 0, "width": 10, "height": 10}},
                ], stop_reason="tool_use")
            elif call_count[0] == 4:
                # Fourth: extrude
                return self._make_llm_response([
                    {"type": "tool_use", "id": "call_4", "name": "extrude",
                     "input": {"sketch_name": "Sketch1", "distance": 5,
                               "operation": "NewBodyFeatureOperation"}},
                ], stop_reason="tool_use")
            else:
                # Done
                return self._make_llm_response([
                    {"type": "text", "text": "Built a 10x10x5 box."},
                ], stop_reason="end_turn")

        agent = ToolDrivenAgent(
            backend=backend, llm_call=mock_llm_call,
            model="test-model", max_iterations=10, step_delay=0, verbose=False,
        )
        result = agent.run("Build a simple box")
        self.assertTrue(result.ok)
        self.assertEqual(result.steps_executed, 4)
        self.assertEqual(result.iterations, 5)
        self.assertEqual(len(result.tool_calls), 4)

    def test_agent_max_iterations(self):
        """Agent stops after max iterations."""
        backend = make_mock_backend()

        def mock_llm_call(*, model, system, messages, tools, max_tokens, temperature):
            return self._make_llm_response([
                {"type": "tool_use", "id": "call_x", "name": "get_bodies", "input": {}},
            ], stop_reason="tool_use")

        agent = ToolDrivenAgent(
            backend=backend, llm_call=mock_llm_call,
            model="test-model", max_iterations=3, step_delay=0, verbose=False,
        )
        result = agent.run("Build something forever")
        self.assertTrue(result.ok)  # partial success
        self.assertEqual(result.iterations, 3)
        self.assertIn("max iterations", result.error)

    def test_agent_llm_failure(self):
        """Agent handles LLM call failure."""
        backend = make_mock_backend()

        def mock_llm_call(*, model, system, messages, tools, max_tokens, temperature):
            raise ConnectionError("API down")

        agent = ToolDrivenAgent(
            backend=backend, llm_call=mock_llm_call,
            model="test-model", max_iterations=5, step_delay=0, verbose=False,
        )
        result = agent.run("Build a chair")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "LLM call failed")

    def test_agent_with_images(self):
        """Agent handles image input."""
        backend = make_mock_backend()
        call_count = [0]

        def mock_llm_call(*, model, system, messages, tools, max_tokens, temperature):
            call_count[0] += 1
            # Verify first message has image content
            if call_count[0] == 1:
                user_msg = messages[0]
                assert isinstance(user_msg["content"], list)
                assert any(b.get("type") == "image" for b in user_msg["content"])
            return self._make_llm_response([
                {"type": "text", "text": "Done."},
            ])

        agent = ToolDrivenAgent(
            backend=backend, llm_call=mock_llm_call,
            model="test-model", max_iterations=5, step_delay=0, verbose=False,
        )
        images = [{"data": "base64data==", "media_type": "image/png", "label": "front"}]
        result = agent.run("Build from this image", images=images)
        self.assertTrue(result.ok)

    def test_system_prompt_content(self):
        """System prompt includes key instructions."""
        self.assertIn("Fusion 360", TOOL_AGENT_SYSTEM_PROMPT)
        self.assertIn("create_sketch", TOOL_AGENT_SYSTEM_PROMPT)
        self.assertIn("extrude", TOOL_AGENT_SYSTEM_PROMPT)
        self.assertIn("NewBodyFeatureOperation", TOOL_AGENT_SYSTEM_PROMPT)
        self.assertIn("JoinFeatureOperation", TOOL_AGENT_SYSTEM_PROMPT)
        self.assertIn("CutFeatureOperation", TOOL_AGENT_SYSTEM_PROMPT)

    def test_agent_result_dataclass(self):
        """ToolAgentResult has correct defaults."""
        r = ToolAgentResult(ok=False)
        self.assertFalse(r.ok)
        self.assertEqual(r.steps_executed, 0)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(r.tool_calls, [])
        self.assertIsNone(r.final_state)
        self.assertEqual(r.error, "")
        self.assertEqual(r.conversation, [])


if __name__ == "__main__":
    unittest.main()
