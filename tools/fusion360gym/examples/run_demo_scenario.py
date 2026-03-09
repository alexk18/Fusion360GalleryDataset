"""
Run a demo scenario with the tool-driven CAD agent.

Usage (from examples/ folder):

  # Pre-built scenarios (with explicit build instructions):
  python run_demo_scenario.py bearing_housing
  python run_demo_scenario.py flanged_pipe
  python run_demo_scenario.py motor_mount

  # With reference image (model analyzes photo and follows instructions):
  python run_demo_scenario.py bearing_housing --image photo_original.png

  # Free-form build from image only (no pre-built instructions):
  python run_demo_scenario.py --image bracket.jpg --prompt "Build this bracket"

  # From dataset assembly (uses assembly.png as reference):
  python run_demo_scenario.py --dataset 49902_81b1dde9 --prompt "Build this object"

  # List available scenarios:
  python run_demo_scenario.py --list

Requires:
    - .env file with ANTHROPIC_API_KEY
    - Fusion 360 running with fusion360gym server on port 8080
"""

import base64
import sys
import os
import json
import time
from pathlib import Path

# ---- path setup (same as ai_assistant.py) ----
FUSION_DIR = os.path.join(os.path.dirname(__file__), "..")
if FUSION_DIR not in sys.path:
    sys.path.insert(0, FUSION_DIR)
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")
if CLIENT_DIR not in sys.path:
    sys.path.append(CLIENT_DIR)

from dotenv import load_dotenv
load_dotenv(override=True)

import anthropic
from fusion360gym_client import Fusion360GymClient
from cad.cad_tool_agent import ToolDrivenAgent
from cad.cad_backend import FusionCadBackend
from cad.demo_scenarios import SCENARIOS, list_scenarios


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def detect_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def load_image(file_path: str, label: str = "") -> dict:
    """Load a single image file and return Claude-compatible image dict."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    if not label:
        label = p.stem.replace("_", " ")
    return {
        "data": data,
        "media_type": detect_media_type(p),
        "label": label,
        "name": p.name,
    }


def load_images_from_args(image_paths: list, dataset_id: str = "") -> list:
    """Load images from file paths and/or dataset assembly."""
    images = []

    # Load explicit image files
    for path in image_paths:
        try:
            images.append(load_image(path, label="reference"))
            print(f"  Loaded image: {path}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # Load dataset assembly image
    if dataset_id:
        dataset_root = os.environ.get(
            "ASSEMBLY_DATASET_ROOT",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "assembly"),
        )
        assembly_png = os.path.join(dataset_root, dataset_id, "assembly.png")
        try:
            images.append(load_image(assembly_png, label="dataset reference"))
            print(f"  Loaded dataset image: {assembly_png}")
        except FileNotFoundError:
            print(f"  WARNING: Dataset image not found: {assembly_png}")

    return images if images else None


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Simple arg parser (no argparse to keep it lightweight)."""
    args = {
        "scenario": None,
        "images": [],
        "dataset": "",
        "prompt": "",
        "list": False,
    }

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--list":
            args["list"] = True
        elif arg == "--image" and i + 1 < len(sys.argv):
            i += 1
            args["images"].append(sys.argv[i])
        elif arg == "--dataset" and i + 1 < len(sys.argv):
            i += 1
            args["dataset"] = sys.argv[i]
        elif arg == "--prompt" and i + 1 < len(sys.argv):
            i += 1
            args["prompt"] = sys.argv[i]
        elif arg in ("-h", "--help"):
            print(__doc__)
            print(f"Available scenarios: {', '.join(list_scenarios())}")
            sys.exit(0)
        elif not arg.startswith("--") and args["scenario"] is None:
            args["scenario"] = arg
        i += 1

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args["list"]:
        print("Available demo scenarios:\n")
        for name, info in SCENARIOS.items():
            print(f"  {name}")
            print(f"    {info['description']}")
            print(f"    Features: {', '.join(info['key_features'])}")
            print()
        sys.exit(0)

    # Determine prompt
    scenario = None
    prompt = args["prompt"]
    scenario_name = args["scenario"] or "custom"

    if args["scenario"] and args["scenario"] in SCENARIOS:
        scenario = SCENARIOS[args["scenario"]]
        prompt = scenario["prompt"]
    elif args["scenario"] and args["scenario"] not in SCENARIOS and not prompt:
        # Scenario name not found and no --prompt — maybe it's a typo
        print(f"Unknown scenario: {args['scenario']}")
        print(f"Available: {', '.join(list_scenarios())}")
        print(f"\nOr use: python run_demo_scenario.py --image photo.png --prompt \"Build this\"")
        sys.exit(1)

    if not prompt and not args["images"] and not args["dataset"]:
        print("ERROR: Specify a scenario name, --prompt, or --image")
        print(f"\nAvailable scenarios: {', '.join(list_scenarios())}")
        print(f"Or: python run_demo_scenario.py --image photo.png --prompt \"Build this object\"")
        sys.exit(1)

    if not prompt:
        prompt = "Build this object. Analyze the reference image carefully and recreate it using available CAD tools."

    # ---- config from .env ----
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    max_iter = int(os.environ.get("TOOL_AGENT_MAX_ITERATIONS", "50"))
    host = os.environ.get("FUSION_HOST", "127.0.0.1")
    port = int(os.environ.get("FUSION_PORT", "8080"))

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    # ---- load images ----
    images = load_images_from_args(args["images"], args["dataset"])

    # ---- print info ----
    print("=" * 60)
    if scenario:
        print(f"  Demo: {scenario['name']}")
        print(f"  Expected iterations: ~{scenario['expected_iterations']}")
    else:
        print(f"  Demo: Custom build")
        print(f"  Prompt: {prompt[:80]}...")
    print(f"  Model: {model}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Server: {host}:{port}")
    print(f"  Reference images: {len(images) if images else 0}")
    print("=" * 60)

    if scenario:
        print(f"\nFeatures to build:")
        for feat in scenario["key_features"]:
            print(f"  - {feat}")
    print()

    # ---- create agent ----
    anthropic_client = anthropic.Anthropic(api_key=api_key)

    def llm_call(*, model, system, messages, tools, max_tokens=8192, temperature=0.1):
        return anthropic_client.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    fusion_client = Fusion360GymClient(f"http://{host}:{port}")
    backend = FusionCadBackend(fusion_client)
    agent = ToolDrivenAgent(
        backend=backend,
        llm_call=llm_call,
        model=model,
        max_iterations=max_iter,
        step_delay=0.3,
        verbose=True,
    )

    # ---- run ----
    print("Starting build...\n")
    t0 = time.time()
    result = agent.run(prompt, images=images)
    elapsed = time.time() - t0

    # ---- report ----
    print("\n" + "=" * 60)
    print(f"  Result: {'OK' if result.ok else 'FAILED'}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Iterations: {result.iterations}/{max_iter}")
    print(f"  Tool calls: {result.steps_executed}")
    if result.error:
        print(f"  Error: {result.error}")

    # body count
    if result.final_state and isinstance(result.final_state, dict):
        bodies = result.final_state.get("bodies", [])
        if isinstance(bodies, list):
            expected = scenario["expected_body_count"] if scenario else "?"
            print(f"  Bodies: {len(bodies)} (expected: {expected})")

    # tool usage summary
    tool_summary = {}
    for tc in result.tool_calls:
        name = tc["tool"]
        tool_summary[name] = tool_summary.get(name, 0) + 1
    if tool_summary:
        print(f"  Tool usage: {json.dumps(tool_summary)}")

    print("=" * 60)

    # ---- save log ----
    log_file = f"demo_{scenario_name}_{int(time.time())}.json"
    log_data = {
        "scenario": scenario_name,
        "model": model,
        "ok": result.ok,
        "iterations": result.iterations,
        "steps_executed": result.steps_executed,
        "elapsed_seconds": round(elapsed, 1),
        "tool_calls": result.tool_calls,
        "final_state": result.final_state,
        "images_used": len(images) if images else 0,
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"\nLog saved: {log_file}")


if __name__ == "__main__":
    main()
