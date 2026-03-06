"""
Apply a CAD DSL Patch to a Plan (in-place or return new plan).

Patch ops: replace, add, remove.
Path format: "steps[<id>].distance", "steps[<id>].profile.pts", "steps", etc.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional


def _parse_path(path: str) -> List[str]:
    """Split path into segments: steps[id] -> ['steps', id], steps[0].distance -> ['steps', 0, 'distance']."""
    if not path or not path.strip():
        return []
    parts = []
    rest = path.strip()
    while rest:
        if rest.startswith("."):
            rest = rest[1:].lstrip()
            continue
        m = re.match(r"^([^.\[\]]+)(.*)$", rest)
        if not m:
            break
        key, rest = m.group(1), m.group(2)
        if not key:
            break
        if key.isdigit():
            parts.append(int(key))
        else:
            parts.append(key)
        if rest.startswith("["):
            m2 = re.match(r"^\[([^\]]+)\](.*)$", rest)
            if m2:
                idx = m2.group(1).strip()
                rest = m2.group(2)
                if idx.isdigit():
                    parts.append(int(idx))
                else:
                    parts.append(idx)
    return parts


def _get_at(data: Dict | List, parts: List) -> Any:
    cur = data
    for p in parts:
        if p is None:
            break
        if isinstance(cur, list):
            cur = cur[int(p)] if isinstance(p, int) else cur[p]
        else:
            cur = cur[p]
    return cur


def _set_at(data: Dict | List, parts: List, value: Any) -> None:
    if not parts:
        return
    cur = data
    for i, p in enumerate(parts[:-1]):
        if isinstance(cur, list):
            idx = int(parts[i]) if isinstance(parts[i], int) else parts[i]
            cur = cur[idx]
        else:
            cur = cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def _ensure_path(data: Dict | List, parts: List) -> None:
    """Ensure path exists; create dicts/lists as needed."""
    cur = data
    for i, p in enumerate(parts):
        if isinstance(cur, list):
            idx = int(p) if isinstance(p, int) else p
            if idx >= len(cur):
                cur.extend([None] * (idx - len(cur) + 1))
            cur = cur[idx]
        else:
            if p not in cur:
                # Next part decides list vs dict
                next_p = parts[i + 1] if i + 1 < len(parts) else None
                if isinstance(next_p, int):
                    cur[p] = []
                else:
                    cur[p] = {}
            cur = cur[p]


def apply_patch(plan_dict: Dict[str, Any], patch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a Patch to a Plan. Returns a new plan dict (does not mutate input).
    Patch format: {"patches": [{"op": "replace"|"add"|"remove", "path": "...", "value": ...}], "intent": "..."}
    """
    plan = copy.deepcopy(plan_dict)
    patches = (patch_dict or {}).get("patches") or []
    steps = plan.get("steps") or []
    steps_by_id = {s.get("id"): i for i, s in enumerate(steps) if s.get("id")}

    for p in patches:
        op = (p.get("op") or "").strip().lower()
        path = (p.get("path") or "").strip()
        value = p.get("value")
        if not path and op != "add":
            continue
        segs = _parse_path(path)

        if op == "replace":
            if not segs:
                continue
            if segs[0] == "steps" and len(segs) >= 2:
                step_id = segs[1]
                if step_id in steps_by_id:
                    idx = steps_by_id[step_id]
                    if len(segs) == 2:
                        steps[idx] = value
                    else:
                        _set_at(plan, ["steps", idx] + list(segs[2:]), value)
            else:
                _ensure_path(plan, segs[:-1])
                _set_at(plan, segs, value)

        elif op == "add":
            if path == "steps" or (segs and segs[0] == "steps" and len(segs) == 1):
                if isinstance(value, dict) and value.get("id"):
                    steps.append(value)
                    steps_by_id[value["id"]] = len(steps) - 1
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, dict) and v.get("id"):
                            steps.append(v)
                            steps_by_id[v["id"]] = len(steps) - 1
            elif segs and segs[0] == "steps" and len(segs) >= 2:
                step_id = segs[1]
                if step_id in steps_by_id:
                    idx = steps_by_id[step_id]
                    _ensure_path(plan["steps"][idx], segs[2:-1])
                    cur = plan["steps"][idx]
                    for s in segs[2:-1]:
                        cur = cur[s]
                    last = segs[-1]
                    if isinstance(cur.get(last), list):
                        cur[last].append(value)
                    else:
                        cur[last] = value

        elif op == "remove":
            if segs and segs[0] == "steps" and len(segs) == 2:
                step_id = segs[1]
                if step_id in steps_by_id:
                    idx = steps_by_id[step_id]
                    steps.pop(idx)
                    steps_by_id = {s.get("id"): i for i, s in enumerate(steps) if s.get("id")}

    plan["steps"] = steps
    return plan
