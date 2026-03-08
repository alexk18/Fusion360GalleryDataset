"""
Runtime state introspection utilities for iterative create-mode CAD agent loops.

Layered strategy:
- exact_fusion_api where command surface can return direct Fusion state
- derived_exact for deterministic projections from exact data
- heuristic_estimate fallback when exact runtime geometry is unavailable
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .cad_backend import FusionCadBackend
from .cad_geometry_queries import (
    SOURCE_DERIVED_EXACT,
    SOURCE_EXACT_FUSION_API,
    SOURCE_HEURISTIC_ESTIMATE,
    build_step_bounds_map,
    coerce_bbox,
    compute_connected_components,
    compute_relations_from_bboxes,
    extract_data,
    floating_ids_from_components,
    grounded_ids_from_bboxes,
    make_signal,
    map_steps_to_bodies,
    normalize_body_relations,
    project_body_relations_to_steps,
    signal_data,
    unwrap_payload,
)


def _query_backend(backend: FusionCadBackend, method_name: str) -> Tuple[bool, Dict[str, Any], str]:
    fn = getattr(backend, method_name, None)
    if not callable(fn):
        return False, {}, f"backend has no method '{method_name}'"
    try:
        result = fn()
    except Exception as ex:
        return False, {}, f"{method_name} raised: {ex}"
    if not getattr(result, "ok", False):
        return False, {}, str(getattr(result, "reason", "") or f"{method_name} failed")
    payload = extract_data(unwrap_payload(getattr(result, "response", None)))
    return True, payload, ""


def _source_from_payload(payload: Dict[str, Any], default_source: str) -> str:
    if not isinstance(payload, dict):
        return default_source
    src = str(payload.get("source_kind") or payload.get("source") or "").strip().lower()
    if src in (SOURCE_EXACT_FUSION_API, SOURCE_DERIVED_EXACT, SOURCE_HEURISTIC_ESTIMATE):
        return src
    if src in ("exact", "fusion_api", "exact_fusion"):
        return SOURCE_EXACT_FUSION_API
    if src in ("derived", "derived_exact"):
        return SOURCE_DERIVED_EXACT
    if src in ("heuristic", "estimate"):
        return SOURCE_HEURISTIC_ESTIMATE
    return default_source


def _extract_bodies(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload.get("bodies"), list):
        return [b for b in payload.get("bodies", []) if isinstance(b, dict)]
    if isinstance(payload.get("parts"), list):
        return [b for b in payload.get("parts", []) if isinstance(b, dict)]
    return []


def _extract_body_bbox_map(payload: Dict[str, Any], bodies: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    raw_map = payload.get("body_bbox")
    if isinstance(raw_map, dict):
        for bid, bbox in raw_map.items():
            cb = coerce_bbox(bbox)
            if cb is not None:
                out[str(bid)] = cb
    elif isinstance(payload.get("bounding_box"), dict):
        cb = coerce_bbox(payload.get("bounding_box"))
        if cb is not None:
            out["model"] = cb

    for b in bodies:
        bid = str(b.get("id") or b.get("name") or "")
        if not bid:
            continue
        if bid in out:
            continue
        cb = coerce_bbox(b.get("bbox"))
        if cb is not None:
            out[bid] = cb
    return out


def _extract_feature_bbox_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    fb = payload.get("feature_bbox")
    if isinstance(fb, dict):
        for fid, bbox in fb.items():
            cb = coerce_bbox(bbox)
            if cb is not None:
                out[str(fid)] = cb
    return out


def _extract_components(payload: Dict[str, Any]) -> List[List[str]]:
    rows = payload.get("components")
    if not isinstance(rows, list):
        return []
    out: List[List[str]] = []
    for item in rows:
        if isinstance(item, list):
            comp = [str(x) for x in item if str(x)]
            if comp:
                out.append(comp)
    return out


def _extract_overlaps(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("overlaps")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _extract_feature_body_relations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("relations")
    if not isinstance(rows, list):
        rows = payload.get("feature_body_relations")
    if not isinstance(rows, list):
        return []
    return [dict(r) for r in rows if isinstance(r, dict)]


def _extract_active_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = payload.get("active_context")
    if isinstance(ctx, dict):
        return dict(ctx)
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _project_components_to_steps(components: List[List[str]], step_to_body: Dict[str, str]) -> List[List[str]]:
    if not components or not step_to_body:
        return []
    body_to_step = {b: s for s, b in step_to_body.items()}
    out: List[List[str]] = []
    for comp in components:
        step_comp = sorted({body_to_step.get(str(x), "") for x in comp if body_to_step.get(str(x), "")})
        if step_comp:
            out.append(step_comp)
    return out


def inspect_runtime_state(
    backend: FusionCadBackend,
    *,
    executed_steps: Optional[List[Dict[str, Any]]] = None,
    planned_steps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    executed = [s for s in (executed_steps or []) if isinstance(s, dict)]
    planned = [s for s in (planned_steps or []) if isinstance(s, dict)]
    executed_ids = sorted([str(s.get("id") or "") for s in executed if str(s.get("id") or "")])

    sources: Dict[str, Dict[str, Any]] = {}
    state: Dict[str, Any] = {}

    def record(name: str, data: Any, source_kind: str, note: str = "") -> None:
        sig = make_signal(data, source_kind=source_kind, note=note)
        sources[name] = {
            "source_kind": sig["source_kind"],
            "confidence": sig["confidence"],
            "note": sig["note"],
        }
        state[name] = signal_data(sig)

    # ------------------------------------------------------------------
    # Core queries (exact first)
    # ------------------------------------------------------------------
    ok, tools_payload, tools_reason = _query_backend(backend, "list_tools")
    if ok:
        record(
            "list_tools",
            tools_payload.get("tools", []),
            _source_from_payload(tools_payload, SOURCE_EXACT_FUSION_API),
            "server tool registry",
        )
    else:
        record("list_tools", [], SOURCE_HEURISTIC_ESTIMATE, tools_reason)

    caps = {}
    if hasattr(backend, "capabilities") and getattr(backend, "capabilities", None):
        for name, cap in (backend.capabilities.capabilities or {}).items():
            caps[name] = {
                "status": getattr(cap, "status", "unsupported"),
                "reason": getattr(cap, "reason", ""),
            }
    record("list_capabilities", caps, SOURCE_DERIVED_EXACT, "capability model reflection")

    ok, model_payload, model_reason = _query_backend(backend, "get_model_state")
    if ok:
        record(
            "get_model_state",
            model_payload,
            _source_from_payload(model_payload, SOURCE_EXACT_FUSION_API),
            "model state summary",
        )
    else:
        record("get_model_state", {}, SOURCE_HEURISTIC_ESTIMATE, model_reason)

    ok, features_payload, features_reason = _query_backend(backend, "get_features")
    features: List[Dict[str, Any]] = []
    if ok:
        if isinstance(features_payload.get("features"), list):
            features = [f for f in features_payload.get("features", []) if isinstance(f, dict)]
        elif isinstance(features_payload.get("extrude_features"), list):
            features = [f for f in features_payload.get("extrude_features", []) if isinstance(f, dict)]
        record(
            "get_features",
            features,
            _source_from_payload(features_payload, SOURCE_EXACT_FUSION_API),
            "feature list",
        )
    else:
        record("get_features", [], SOURCE_HEURISTIC_ESTIMATE, features_reason)

    ok, sketches_payload, sketches_reason = _query_backend(backend, "get_sketches")
    sketches: List[Dict[str, Any]] = []
    if ok:
        if isinstance(sketches_payload.get("sketches"), list):
            sketches = [s for s in sketches_payload.get("sketches", []) if isinstance(s, dict)]
        record(
            "get_sketches",
            sketches,
            _source_from_payload(sketches_payload, SOURCE_EXACT_FUSION_API),
            "sketch list",
        )
    else:
        record("get_sketches", [], SOURCE_HEURISTIC_ESTIMATE, sketches_reason)

    ok, bodies_payload, bodies_reason = _query_backend(backend, "get_bodies")
    bodies = _extract_bodies(bodies_payload) if ok else []
    if not bodies and features:
        # Legacy fallback: feature-proxy body list.
        bodies = [
            {
                "id": str(f.get("name") or f"body_{i+1}"),
                "name": str(f.get("name") or f"body_{i+1}"),
                "source": "feature_proxy",
            }
            for i, f in enumerate(features)
        ]
    if ok and bodies:
        record(
            "get_bodies",
            bodies,
            _source_from_payload(bodies_payload, SOURCE_EXACT_FUSION_API),
            "body list",
        )
    else:
        record("get_bodies", bodies, SOURCE_HEURISTIC_ESTIMATE, bodies_reason or "feature-proxy fallback")
    record("get_parts", list(state.get("get_bodies", [])), sources["get_bodies"]["source_kind"], "alias of bodies")

    ok, body_bbox_payload, body_bbox_reason = _query_backend(backend, "get_body_bbox")
    body_bbox_map = _extract_body_bbox_map(body_bbox_payload if ok else {}, bodies)
    if body_bbox_map:
        record(
            "get_body_bbox",
            body_bbox_map,
            _source_from_payload(body_bbox_payload, SOURCE_EXACT_FUSION_API) if ok else SOURCE_HEURISTIC_ESTIMATE,
            "body bbox map",
        )
    else:
        record("get_body_bbox", {}, SOURCE_HEURISTIC_ESTIMATE, body_bbox_reason or "no body bbox available")

    ok, feature_bbox_payload, feature_bbox_reason = _query_backend(backend, "get_feature_bbox")
    feature_bbox_map = _extract_feature_bbox_map(feature_bbox_payload if ok else {})

    # ------------------------------------------------------------------
    # Fallback step-space geometry signals
    # ------------------------------------------------------------------
    step_bbox_map = build_step_bounds_map(executed)
    fallback_relations = compute_relations_from_bboxes(step_bbox_map, source_kind=SOURCE_HEURISTIC_ESTIMATE, contact_tol=0.75)
    fallback_components = compute_connected_components(step_bbox_map.keys(), fallback_relations)
    fallback_overlaps = [r for r in fallback_relations if str(r.get("relation") or "") == "intersects"]
    grounded_step_ids = grounded_ids_from_bboxes(step_bbox_map)
    floating_step_ids = floating_ids_from_components(fallback_components, grounded_step_ids)

    if not feature_bbox_map:
        feature_bbox_map = dict(step_bbox_map)
        record(
            "get_feature_bbox",
            feature_bbox_map,
            SOURCE_HEURISTIC_ESTIMATE,
            feature_bbox_reason or "fallback from executed-step bounds",
        )
    else:
        record(
            "get_feature_bbox",
            feature_bbox_map,
            _source_from_payload(feature_bbox_payload, SOURCE_EXACT_FUSION_API) if ok else SOURCE_DERIVED_EXACT,
            "feature bbox from runtime",
        )

    # Exact/derived body relations, then step projection.
    ok, body_rel_payload, body_rel_reason = _query_backend(backend, "get_body_relations")
    body_relations = normalize_body_relations((body_rel_payload or {}).get("relations") if ok else [])
    if body_relations:
        body_rel_source = _source_from_payload(body_rel_payload, SOURCE_DERIVED_EXACT)
        record("get_body_to_body_relations", body_relations, body_rel_source, "runtime body relation summary")
    else:
        body_relations = [dict(r) for r in fallback_relations]
        record(
            "get_body_to_body_relations",
            body_relations,
            SOURCE_HEURISTIC_ESTIMATE,
            body_rel_reason or "fallback from executed-step bbox relations",
        )

    step_to_body = map_steps_to_bodies(step_bbox_map, body_bbox_map)
    projected_step_relations = project_body_relations_to_steps(body_relations, step_to_body)
    if projected_step_relations:
        step_relations = projected_step_relations
        step_rel_source = SOURCE_DERIVED_EXACT
        step_rel_note = "projected from body relations via bbox mapping"
    else:
        step_relations = [dict(r) for r in fallback_relations]
        step_rel_source = SOURCE_HEURISTIC_ESTIMATE
        step_rel_note = "fallback from step bbox relations"
    record("get_step_relations", step_relations, step_rel_source, step_rel_note)

    ok, feature_body_payload, feature_body_reason = _query_backend(backend, "get_feature_body_relations")
    feature_body_relations = _extract_feature_body_relations(feature_body_payload if ok else {})
    if feature_body_relations:
        record(
            "get_feature_to_body_relations",
            feature_body_relations,
            _source_from_payload(feature_body_payload, SOURCE_EXACT_FUSION_API),
            "runtime feature-body mapping",
        )
    else:
        feature_body_relations = [{"step_id": sid, "body_id": bid} for sid, bid in step_to_body.items()]
        record(
            "get_feature_to_body_relations",
            feature_body_relations,
            SOURCE_DERIVED_EXACT if feature_body_relations else SOURCE_HEURISTIC_ESTIMATE,
            feature_body_reason or "derived from step-body bbox mapping",
        )

    ok, components_payload, components_reason = _query_backend(backend, "get_connected_components")
    components = _extract_components(components_payload if ok else {})
    projected_components = _project_components_to_steps(components, step_to_body)
    if projected_components:
        components = projected_components
    if not components:
        components = fallback_components
        record(
            "get_connected_components",
            components,
            SOURCE_HEURISTIC_ESTIMATE,
            components_reason or "fallback from step relations",
        )
    else:
        record(
            "get_connected_components",
            components,
            SOURCE_DERIVED_EXACT if projected_components else _source_from_payload(components_payload, SOURCE_DERIVED_EXACT),
            "connected components over runtime relation graph",
        )

    ok, overlaps_payload, overlaps_reason = _query_backend(backend, "get_overlaps")
    overlaps = _extract_overlaps(overlaps_payload if ok else {})
    if not overlaps:
        overlaps = fallback_overlaps
        record("get_overlaps", overlaps, SOURCE_HEURISTIC_ESTIMATE, overlaps_reason or "fallback from step bbox overlaps")
    else:
        record(
            "get_overlaps",
            overlaps,
            _source_from_payload(overlaps_payload, SOURCE_DERIVED_EXACT),
            "runtime overlap/intersection summary",
        )

    ok, faces_payload, faces_reason = _query_backend(backend, "get_faces")
    faces_data = {}
    if ok:
        faces_data = dict(faces_payload.get("faces") or faces_payload)
        record("get_faces", faces_data, _source_from_payload(faces_data, SOURCE_EXACT_FUSION_API), "face summary")
    else:
        faces_data = {"count_estimate": max(0, 6 * len(step_bbox_map)), "source_kind": SOURCE_HEURISTIC_ESTIMATE}
        record("get_faces", faces_data, SOURCE_HEURISTIC_ESTIMATE, faces_reason or "heuristic face estimate")

    ok, edges_payload, edges_reason = _query_backend(backend, "get_edges")
    edges_data = {}
    if ok:
        edges_data = dict(edges_payload.get("edges") or edges_payload)
        record("get_edges", edges_data, _source_from_payload(edges_data, SOURCE_EXACT_FUSION_API), "edge summary")
    else:
        edges_data = {"count_estimate": max(0, 12 * len(step_bbox_map)), "source_kind": SOURCE_HEURISTIC_ESTIMATE}
        record("get_edges", edges_data, SOURCE_HEURISTIC_ESTIMATE, edges_reason or "heuristic edge estimate")

    ok, context_payload, context_reason = _query_backend(backend, "get_active_construction_context")
    active_context = _extract_active_context(context_payload if ok else {})
    if not active_context:
        active_context = {}
    active_context.setdefault("last_executed_step_id", executed_ids[-1] if executed_ids else "")
    next_id = next((str(s.get("id") or "") for s in planned if str(s.get("id") or "") not in executed_ids), "")
    active_context.setdefault("next_candidate_step_id", next_id)
    active_context["grounded_step_ids"] = grounded_step_ids
    active_context["floating_step_ids"] = floating_step_ids
    if ok:
        record(
            "get_active_construction_context",
            active_context,
            _source_from_payload(context_payload, SOURCE_DERIVED_EXACT),
            "active construction context",
        )
    else:
        record(
            "get_active_construction_context",
            active_context,
            SOURCE_HEURISTIC_ESTIMATE,
            context_reason or "fallback context",
        )

    # Backward-compatible aliases.
    state["get_body_relations"] = list(state.get("get_body_to_body_relations", []))
    sources["get_body_relations"] = dict(sources.get("get_body_to_body_relations", {}))

    state["executed_step_ids"] = executed_ids
    state["introspection_sources"] = sources
    state["state_signal_summary"] = {
        "exact_queries": sorted([k for k, v in sources.items() if str(v.get("source_kind")) == SOURCE_EXACT_FUSION_API]),
        "derived_queries": sorted([k for k, v in sources.items() if str(v.get("source_kind")) == SOURCE_DERIVED_EXACT]),
        "heuristic_queries": sorted([k for k, v in sources.items() if str(v.get("source_kind")) == SOURCE_HEURISTIC_ESTIMATE]),
    }
    return state

