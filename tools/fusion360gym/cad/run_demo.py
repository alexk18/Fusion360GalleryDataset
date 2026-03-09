"""
Run a demo scenario with the tool-driven agent.

Usage:
    python -m cad.run_demo bearing_housing
    python -m cad.run_demo flanged_pipe
    python -m cad.run_demo motor_mount
    python -m cad.run_demo --list

Requires:
    - ANTHROPIC_API_KEY environment variable
    - Fusion 360 running with fusion360gym server on port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anthropic

# Add client dir to path
_client_dir = os.path.join(os.path.dirname(__file__), "..", "client")
if _client_dir not in sys.path:
    sys.path.append(_client_dir)

from fusion360gym_client import Fusion360GymClient
from .cad_backend import FusionCadBackend
from .cad_tool_agent import ToolDrivenAgent
from .demo_scenarios import SCENARIOS, list_scenarios


def make_llm_call(client: anthropic.Anthropic):
    """Create an LLM call function compatible with ToolDrivenAgent."""
    def llm_call(*, model: str, system: str, messages, tools, max_tokens: int, temperature: float):
        return client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return llm_call


def main():
    parser = argparse.ArgumentParser(description="Run a CAD demo scenario")
    parser.add_argument("scenario", nargs="?", help="Scenario name")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--max-iter", type=int, default=50, help="Max iterations")
    parser.add_argument("--host", default="localhost", help="Fusion server host")
    parser.add_argument("--port", type=int, default=8080, help="Fusion server port")
    parser.add_argument("--save-log", type=str, help="Save conversation log to file")
    args = parser.parse_args()

    if args.list:
        print("Available demo scenarios:")
        for name, info in SCENARIOS.items():
            print(f"  {name:20s} — {info['description']}")
        return

    if not args.scenario:
        parser.error("Specify a scenario name or --list")

    if args.scenario not in SCENARIOS:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {', '.join(list_scenarios())}")
        sys.exit(1)

    scenario = SCENARIOS[args.scenario]
    print(f"=== Demo: {scenario['name']} ===")
    print(f"Description: {scenario['description']}")
    print(f"Expected iterations: ~{scenario['expected_iterations']}")
    print(f"Key features: {', '.join(scenario['key_features'])}")
    print()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Create backend and agent
    fusion_client = Fusion360GymClient(f"http://{args.host}:{args.port}")
    backend = FusionCadBackend(fusion_client)
    client = anthropic.Anthropic(api_key=api_key)

    agent = ToolDrivenAgent(
        backend=backend,
        llm_call=make_llm_call(client),
        model=args.model,
        max_iterations=args.max_iter,
        step_delay=0.3,
        verbose=True,
    )

    # Run
    print(f"Starting build (max {args.max_iter} iterations)...")
    print("-" * 60)
    t0 = time.time()

    result = agent.run(scenario["prompt"])

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"Completed in {elapsed:.1f}s, {result.iterations} iterations, {result.steps_executed} tool calls")
    print(f"Result: {'OK' if result.ok else 'FAILED'}")
    if result.error:
        print(f"Error: {result.error}")

    # Report final state
    if result.final_state:
        bodies = result.final_state.get("bodies", [])
        if isinstance(bodies, list):
            print(f"Bodies: {len(bodies)}")
        elif isinstance(bodies, dict):
            print(f"Bodies: {bodies}")

    # Count operations
    tool_summary = {}
    for tc in result.tool_calls:
        name = tc["tool"]
        tool_summary[name] = tool_summary.get(name, 0) + 1
    if tool_summary:
        print(f"Tool usage: {json.dumps(tool_summary, indent=None)}")

    # Save log
    if args.save_log:
        log_data = {
            "scenario": args.scenario,
            "model": args.model,
            "ok": result.ok,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "final_state": result.final_state,
            "elapsed_seconds": elapsed,
        }
        with open(args.save_log, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"Log saved to {args.save_log}")


if __name__ == "__main__":
    main()
