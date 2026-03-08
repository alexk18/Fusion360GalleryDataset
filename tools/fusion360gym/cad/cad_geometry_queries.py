"""
Geometry/state query helpers for iterative create-mode reasoning.

This module centralizes:
- confidence/source labeling for introspection signals
- lightweight bbox/spatial relation utilities
- projection helpers between step-space and body-space signals
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


SOURCE_EXACT_FUSION_API = "exact_fusion_api"
SOURCE_DERIVED_EXACT = "derived_exact"
SOURCE_HEURISTIC_ESTIMATE = "heuristic_estimate"


def default_confidence_for_source(source_kind: str) -> float:
    sk = str(source_kind or "").strip().lower()
    if sk == SOURCE_EXACT_FUSION_API:
        return 0.99
    if sk == SOURCE_DERIVED_EXACT:
        return 0.85
    return 0.55


def make_signal(
    data: Any,
    *,
    source_kind: str,
    note: str = "",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    conf = default_confidence_for_source(source_kind) if confidence is None else float(confidence)
    conf = max(0.0, min(1.0, conf))
    return {
        "data": data,
        "source_kind": str(source_kind or SOURCE_HEURISTIC_ESTIMATE),
        "confidence": conf,
        "note": str(note or ""),
    }


def signal_data(value: Any, default: Any = None) -> Any:
    if isinstance(value, dict) and "data" in value:
        return value.get("data")
    if value is None:
        return default
    return value


def signal_source(value: Any, default: str = SOURCE_HEURISTIC_ESTIMATE) -> str:
    if isinstance(value, dict) and "source_kind" in value:
        return str(value.get("source_kind") or default)
    return str(default)


def signal_confidence(value: Any) -> float:
    if isinstance(value, dict) and "confidence" in value:
        try:
            return float(value.get("confidence"))
        except Exception:
            return default_confidence_for_source(signal_source(value))
    return default_confidence_for_source(signal_source(value))


def unwrap_payload(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "json"):
        try:
            body = response.json()
            if isinstance(body, dict):
                return body
        except Exception:
            return {}
    return {}


def extract_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload if isinstance(payload, dict) else {}


def parse_plane(plane: Any) -> Tuple[str, float]:
    s = str(plane or "").strip().upper()
    if "@" in s:
        base, off = s.split("@", 1)
        try:
            return base.strip(), float(off.strip())
        except Exception:
            return base.strip(), 0.0
    if s in ("XY", "XZ", "YZ"):
        return s, 0.0
    return "XY", 0.0


def axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def bbox_volume(b: Dict[str, float]) -> float:
    return max(0.0, b["x_max"] - b["x_min"]) * max(0.0, b["y_max"] - b["y_min"]) * max(0.0, b["z_max"] - b["z_min"])


def bbox_intersection(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, float, float]:
    ox = axis_overlap(a["x_min"], a["x_max"], b["x_min"], b["x_max"])
    oy = axis_overlap(a["y_min"], a["y_max"], b["y_min"], b["y_max"])
    oz = axis_overlap(a["z_min"], a["z_max"], b["z_min"], b["z_max"])
    return ox, oy, oz


def bbox_gap(a: Dict[str, float], b: Dict[str, float]) -> float:
    gx = max(0.0, max(a["x_min"], b["x_min"]) - min(a["x_max"], b["x_max"]))
    gy = max(0.0, max(a["y_min"], b["y_min"]) - min(a["y_max"], b["y_max"]))
    gz = max(0.0, max(a["z_min"], b["z_min"]) - min(a["z_max"], b["z_max"]))
    return max(gx, gy, gz)


def bbox_touch_or_overlap(a: Dict[str, float], b: Dict[str, float], tol: float = 0.15) -> bool:
    ox, oy, oz = bbox_intersection(a, b)
    if ox > 0.0 and oy > 0.0 and oz > 0.0:
        return True
    near_xy = ox > 0.0 and oy > 0.0 and abs(a["z_max"] - b["z_min"]) <= tol
    near_yz = oy > 0.0 and oz > 0.0 and abs(a["x_max"] - b["x_min"]) <= tol
    near_xz = ox > 0.0 and oz > 0.0 and abs(a["y_max"] - b["y_min"]) <= tol
    return bool(near_xy or near_yz or near_xz)


def coerce_bbox(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None

    if "min_point" in value and "max_point" in value:
        mn = value.get("min_point") if isinstance(value.get("min_point"), dict) else {}
        mx = value.get("max_point") if isinstance(value.get("max_point"), dict) else {}
        try:
            return {
                "x_min": float(mn.get("x", 0.0) or 0.0),
                "y_min": float(mn.get("y", 0.0) or 0.0),
                "z_min": float(mn.get("z", 0.0) or 0.0),
                "x_max": float(mx.get("x", 0.0) or 0.0),
                "y_max": float(mx.get("y", 0.0) or 0.0),
                "z_max": float(mx.get("z", 0.0) or 0.0),
            }
        except Exception:
            return None

    if "min" in value and "max" in value:
        mn = value.get("min") if isinstance(value.get("min"), dict) else {}
        mx = value.get("max") if isinstance(value.get("max"), dict) else {}
        try:
            return {
                "x_min": float(mn.get("x", 0.0) or 0.0),
                "y_min": float(mn.get("y", 0.0) or 0.0),
                "z_min": float(mn.get("z", 0.0) or 0.0),
                "x_max": float(mx.get("x", 0.0) or 0.0),
                "y_max": float(mx.get("y", 0.0) or 0.0),
                "z_max": float(mx.get("z", 0.0) or 0.0),
            }
        except Exception:
            return None

    keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    if all(k in value for k in keys):
        try:
            return {
                "x_min": float(value["x_min"]),
                "x_max": float(value["x_max"]),
                "y_min": float(value["y_min"]),
                "y_max": float(value["y_max"]),
                "z_min": float(value["z_min"]),
                "z_max": float(value["z_max"]),
            }
        except Exception:
            return None
    return None


def step_bounds(step: Dict[str, Any]) -> Optional[Dict[str, float]]:
    profile = step.get("profile") if isinstance(step.get("profile"), dict) else {}
    ptype = str(profile.get("type") or "").strip().lower()
    if ptype == "rect":
        cx = float(profile.get("cx", 0.0) or 0.0)
        cy = float(profile.get("cy", 0.0) or 0.0)
        w = abs(float(profile.get("w", 0.0) or 0.0))
        h = abs(float(profile.get("h", 0.0) or 0.0))
        p0, p1, q0, q1 = (cx - w * 0.5, cx + w * 0.5, cy - h * 0.5, cy + h * 0.5)
    elif ptype == "circle":
        cx = float(profile.get("cx", 0.0) or 0.0)
        cy = float(profile.get("cy", 0.0) or 0.0)
        r = abs(float(profile.get("radius", 0.0) or 0.0))
        p0, p1, q0, q1 = (cx - r, cx + r, cy - r, cy + r)
    elif ptype == "poly":
        pts = [p for p in (profile.get("pts") or []) if isinstance(p, dict)]
        if not pts:
            return None
        xs = [float(p.get("x", 0.0) or 0.0) for p in pts]
        ys = [float(p.get("y", 0.0) or 0.0) for p in pts]
        p0, p1, q0, q1 = (min(xs), max(xs), min(ys), max(ys))
    else:
        return None

    base, off = parse_plane(step.get("plane"))
    d = abs(float(step.get("distance", 0.0) or 0.0))
    if base == "XY":
        return {"x_min": p0, "x_max": p1, "y_min": q0, "y_max": q1, "z_min": off, "z_max": off + d}
    if base == "XZ":
        # In this Fusion setup, XZ plane sketch Y maps to -Z in world coords.
        # Negate z so heuristic bounds match actual Fusion body positions.
        return {"x_min": p0, "x_max": p1, "y_min": off, "y_max": off + d, "z_min": min(-q0, -q1), "z_max": max(-q0, -q1)}
    if base == "YZ":
        return {"x_min": off, "x_max": off + d, "y_min": p0, "y_max": p1, "z_min": q0, "z_max": q1}
    return None


def build_step_bounds_map(steps: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("id") or "").strip()
        if not sid:
            continue
        b = step_bounds(step)
        if b is not None:
            out[sid] = b
    return out


def compute_relations_from_bboxes(
    bounds_by_id: Dict[str, Dict[str, float]],
    *,
    source_kind: str = SOURCE_HEURISTIC_ESTIMATE,
    contact_tol: float = 0.15,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    keys = list(bounds_by_id.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ba = bounds_by_id[a]
            bb = bounds_by_id[b]
            ox, oy, oz = bbox_intersection(ba, bb)
            overlap_vol = ox * oy * oz
            relation = "separate"
            touches = False
            intersects = overlap_vol > 0.0
            if intersects:
                relation = "intersects"
            else:
                touches = bbox_touch_or_overlap(ba, bb, tol=contact_tol)
                if touches:
                    relation = "touches_estimate"
            out.append(
                {
                    "a": a,
                    "b": b,
                    "relation": relation,
                    "intersects": bool(intersects),
                    "touches": bool(touches),
                    "overlap_volume_approx": round(overlap_vol, 6),
                    "gap_estimate": round(bbox_gap(ba, bb), 6),
                    "source_kind": str(source_kind or SOURCE_HEURISTIC_ESTIMATE),
                }
            )
    return out


def compute_connected_components(
    ids: Iterable[str],
    relations: Iterable[Dict[str, Any]],
) -> List[List[str]]:
    nodes = [str(x) for x in ids if str(x)]
    if not nodes:
        return []
    edges: Dict[str, List[str]] = {n: [] for n in nodes}
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        a = str(rel.get("a") or "")
        b = str(rel.get("b") or "")
        if not a or not b or a not in edges or b not in edges:
            continue
        relation = str(rel.get("relation") or "").strip().lower()
        if relation not in ("intersects", "touches", "touches_estimate", "contacts_estimate"):
            continue
        edges[a].append(b)
        edges[b].append(a)

    visited = set()
    components: List[List[str]] = []
    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        comp: List[str] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in edges[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c))
    return components


def grounded_ids_from_bboxes(bounds_by_id: Dict[str, Dict[str, float]], z_tol: float = 0.15) -> List[str]:
    return sorted([sid for sid, b in bounds_by_id.items() if b["z_min"] <= z_tol])


def floating_ids_from_components(components: List[List[str]], grounded_ids: Iterable[str]) -> List[str]:
    grounded = set(str(x) for x in grounded_ids if str(x))
    floating: List[str] = []
    for comp in components:
        comp_ids = [str(x) for x in comp if str(x)]
        if not comp_ids:
            continue
        if any(cid in grounded for cid in comp_ids):
            continue
        floating.extend(comp_ids)
    return sorted(set(floating))


def _bbox_overlap_score(a: Dict[str, float], b: Dict[str, float]) -> float:
    ox, oy, oz = bbox_intersection(a, b)
    inter = ox * oy * oz
    if inter <= 0.0:
        return 0.0
    va = max(1e-9, bbox_volume(a))
    vb = max(1e-9, bbox_volume(b))
    return float(inter / min(va, vb))


def map_steps_to_bodies(
    step_bboxes: Dict[str, Dict[str, float]],
    body_bboxes: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    Greedy map step id -> nearest/most-overlapping body id.
    """
    out: Dict[str, str] = {}
    if not step_bboxes or not body_bboxes:
        return out
    used_bodies = set()
    for sid, sb in step_bboxes.items():
        best_id = ""
        best_score = -1.0
        for bid, bb in body_bboxes.items():
            if bid in used_bodies:
                continue
            score = _bbox_overlap_score(sb, bb)
            if score > best_score:
                best_score = score
                best_id = bid
        if not best_id:
            # Fallback: minimum bbox gap match.
            for bid, bb in body_bboxes.items():
                if bid in used_bodies:
                    continue
                score = -bbox_gap(sb, bb)
                if score > best_score:
                    best_score = score
                    best_id = bid
        if best_id:
            out[sid] = best_id
            used_bodies.add(best_id)
    return out


def normalize_body_relations(relations: Any) -> List[Dict[str, Any]]:
    rows = relations if isinstance(relations, list) else []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        a = str(item.get("a") or item.get("body_a") or "")
        b = str(item.get("b") or item.get("body_b") or "")
        if not a or not b or a == b:
            continue
        relation = str(item.get("relation") or "").strip().lower()
        if not relation:
            relation = "separate"
        out.append(
            {
                "a": a,
                "b": b,
                "relation": relation,
                "intersects": bool(item.get("intersects", relation == "intersects_exact")),
                "touches": bool(item.get("touches", "touch" in relation or "contact" in relation)),
                "overlap_volume": float(item.get("overlap_volume", item.get("overlap_volume_approx", 0.0)) or 0.0),
                "source_kind": str(item.get("source_kind") or SOURCE_HEURISTIC_ESTIMATE),
            }
        )
    return out


def project_body_relations_to_steps(
    body_relations: List[Dict[str, Any]],
    step_to_body: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not body_relations or not step_to_body:
        return []
    body_to_step = {b: s for s, b in step_to_body.items()}
    out: List[Dict[str, Any]] = []
    for rel in body_relations:
        if not isinstance(rel, dict):
            continue
        a_body = str(rel.get("a") or "")
        b_body = str(rel.get("b") or "")
        a_step = body_to_step.get(a_body, "")
        b_step = body_to_step.get(b_body, "")
        if not a_step or not b_step:
            continue
        out.append(
            {
                "a": a_step,
                "b": b_step,
                "relation": str(rel.get("relation") or "separate"),
                "intersects": bool(rel.get("intersects")),
                "touches": bool(rel.get("touches")),
                "overlap_volume": float(rel.get("overlap_volume", 0.0) or 0.0),
                "source_kind": str(rel.get("source_kind") or SOURCE_DERIVED_EXACT),
            }
        )
    return out
