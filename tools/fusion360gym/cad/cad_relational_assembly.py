"""
Relational assembly reasoning for create-mode CAD planning.

The graph is family-aware but generic: parts, dependencies, support/attachment,
symmetry and grounding requirements are represented explicitly for iterative
build decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _roles_from_spec(spec: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    def bucket(name: str) -> List[str]:
        out: List[str] = []
        for p in (spec.get(name) or []):
            if not isinstance(p, dict):
                continue
            role = str(p.get("role") or "").strip()
            if role:
                out.append(role)
        return out

    main = bucket("main_masses")
    supporting = bucket("supporting_parts")
    secondary = bucket("secondary_parts")
    return main, supporting, secondary


def _find_step_id_for_role(steps: List[Dict[str, Any]], role: str) -> str:
    rr = _norm(role)
    tokens = [t for t in rr.replace("-", "_").split("_") if t]
    for step in steps:
        sid = _norm((step or {}).get("id"))
        if not sid:
            continue
        if rr == sid or rr in sid:
            return str(step.get("id") or "")
        sid_tokens = [t for t in sid.replace("-", "_").split("_") if t]
        overlap = sum(1 for t in tokens if t in sid_tokens)
        if overlap >= max(1, min(2, len(tokens))):
            return str(step.get("id") or "")
    return ""


def _add_relation(relations: List[Dict[str, Any]], relation_type: str, src: str, dst: str, note: str = "") -> None:
    if not src or not dst:
        return
    relations.append(
        {
            "type": relation_type,
            "src": src,
            "dst": dst,
            "note": note,
        }
    )


def build_relational_assembly_graph(spec: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    family = _norm(spec.get("object_family"))
    main, supporting, secondary = _roles_from_spec(spec)
    all_roles = list(dict.fromkeys(main + supporting + secondary))
    build_order = [r for r in (spec.get("build_order") or []) if isinstance(r, str) and r.strip()]
    if not build_order:
        build_order = list(all_roles)

    role_to_step = {role: _find_step_id_for_role(steps, role) for role in all_roles}
    nodes: List[Dict[str, Any]] = []
    for role in all_roles:
        role_kind = "secondary"
        if role in main:
            role_kind = "main"
        elif role in supporting:
            role_kind = "supporting"
        nodes.append(
            {
                "role": role,
                "step_id": role_to_step.get(role, ""),
                "kind": role_kind,
                "grounded": ("support" in _norm(role) or "base" in _norm(role)),
                "load_bearing": role in main or role in supporting,
            }
        )

    relations: List[Dict[str, Any]] = []
    dependencies: Dict[str, Set[str]] = {role: set() for role in all_roles}

    # Generic dependency chain by declared build order.
    for i in range(1, len(build_order)):
        prev_role = build_order[i - 1]
        cur_role = build_order[i]
        if prev_role in dependencies and cur_role in dependencies:
            dependencies[cur_role].add(prev_role)
            _add_relation(relations, "dependent_part", cur_role, prev_role, "build_order")

    # Generic symmetry links for left/right role pairs.
    role_set = set(all_roles)
    for role in list(all_roles):
        rn = _norm(role)
        if rn.startswith("left_"):
            pair = "right_" + role[len("left_") :]
            if pair in role_set:
                _add_relation(relations, "mirrored_with", role, pair, "x-symmetry prior")

    # Family-aware relation priors.
    if family == "lowpoly_hard_surface_vehicle":
        if "upper_hull" in role_set and "lower_hull" in role_set:
            dependencies["upper_hull"].add("lower_hull")
            _add_relation(relations, "parent", "upper_hull", "lower_hull", "upper sits on lower hull")
            _add_relation(relations, "must_be_above", "upper_hull", "lower_hull")
        if "turret" in role_set and "upper_hull" in role_set:
            dependencies["turret"].add("upper_hull")
            _add_relation(relations, "parent", "turret", "upper_hull")
            _add_relation(relations, "must_be_above", "turret", "upper_hull")
            _add_relation(relations, "must_be_centered_on", "turret", "upper_hull")
        if "gun" in role_set and "turret" in role_set:
            dependencies["gun"].add("turret")
            _add_relation(relations, "attachment_type", "gun", "turret", "forward_projecting")
            _add_relation(relations, "must_be_in_front_of", "gun", "turret")
        for track_role in ("left_track_module", "right_track_module"):
            if track_role in role_set and "lower_hull" in role_set:
                dependencies[track_role].add("lower_hull")
                _add_relation(relations, "support_relation", track_role, "lower_hull", "running gear support")
                _add_relation(relations, "must_touch", track_role, "lower_hull")

    elif family == "furniture_boxy_panel":
        for support_role in ("left_support", "right_support"):
            if support_role in role_set:
                _add_relation(relations, "grounded", support_role, support_role, "support contacts floor")
                if "base_panel" in role_set:
                    dependencies["base_panel"].add(support_role)
                    _add_relation(relations, "support_relation", "base_panel", support_role, "panel supported by leg")
                    _add_relation(relations, "must_be_above", "base_panel", support_role)
        if "back_panel" in role_set and "base_panel" in role_set:
            dependencies["back_panel"].add("base_panel")
            _add_relation(relations, "attachment_type", "back_panel", "base_panel", "back-to-base attachment")
            _add_relation(relations, "must_touch", "back_panel", "base_panel")
        if "top_panel" in role_set and "base_panel" in role_set:
            dependencies["top_panel"].add("base_panel")
            _add_relation(relations, "must_be_above", "top_panel", "base_panel")

    elif family == "profile_driven_symmetric":
        if "left_support" in role_set and "primary_profile_body" in role_set:
            dependencies["left_support"].add("primary_profile_body")
            _add_relation(relations, "attachment_type", "left_support", "primary_profile_body", "profile-attached")
        if "right_support" in role_set and "primary_profile_body" in role_set:
            dependencies["right_support"].add("primary_profile_body")
            _add_relation(relations, "attachment_type", "right_support", "primary_profile_body", "profile-attached")

    elif family == "rotational_bodies":
        if "neck_or_top" in role_set and "radial_core" in role_set:
            dependencies["neck_or_top"].add("radial_core")
            _add_relation(relations, "must_be_above", "neck_or_top", "radial_core")
            _add_relation(relations, "must_be_centered_on", "neck_or_top", "radial_core")
        if "base_ring" in role_set and "radial_core" in role_set:
            dependencies["base_ring"].add("radial_core")
            _add_relation(relations, "must_touch", "base_ring", "radial_core")

    return {
        "family": family,
        "nodes": nodes,
        "relations": relations,
        "build_order": build_order,
        "dependencies": {k: sorted(v) for k, v in dependencies.items()},
        "required_roles": sorted(set(main + supporting)),
        "role_to_step": role_to_step,
    }


def _signal_rows(state_snapshot: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    raw = state_snapshot.get(key)
    if isinstance(raw, dict) and "data" in raw:
        raw = raw.get("data")
    if isinstance(raw, dict):
        for sub in ("relations", "overlaps", "items"):
            if isinstance(raw.get(sub), list):
                raw = raw.get(sub)
                break
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, dict)]


def _signal_components(state_snapshot: Dict[str, Any]) -> List[List[str]]:
    raw = state_snapshot.get("get_connected_components")
    if isinstance(raw, dict) and "data" in raw:
        raw = raw.get("data")
    if isinstance(raw, dict):
        raw = raw.get("components")
    if not isinstance(raw, list):
        return []
    out: List[List[str]] = []
    for comp in raw:
        if isinstance(comp, list):
            rows = [str(x) for x in comp if str(x)]
            if rows:
                out.append(rows)
    return out


def _active_context(state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    raw = state_snapshot.get("get_active_construction_context")
    if isinstance(raw, dict) and "data" in raw:
        raw = raw.get("data")
    if isinstance(raw, dict) and isinstance(raw.get("active_context"), dict):
        return dict(raw.get("active_context") or {})
    return raw if isinstance(raw, dict) else {}


def _relation_pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a or ""), str(b or ""))))


def _relation_is_attach_like(relation: str) -> bool:
    r = _norm(relation)
    if not r:
        return False
    return (
        "intersect" in r
        or "touch" in r
        or "contact" in r
        or "overlap" in r
    )


def evaluate_relational_graph(graph: Dict[str, Any], state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    role_to_step = graph.get("role_to_step") or {}
    executed = set(state_snapshot.get("executed_step_ids") or [])
    built_roles = sorted([role for role, sid in role_to_step.items() if sid and sid in executed])
    built_set = set(built_roles)
    required_roles = set(graph.get("required_roles") or [])

    # Connectivity + relation evidence from introspection layer.
    components = _signal_components(state_snapshot)
    disconnected_count = max(0, len(components) - 1) if executed else 0

    step_rel_rows = _signal_rows(state_snapshot, "get_step_relations")
    if not step_rel_rows:
        step_rel_rows = _signal_rows(state_snapshot, "get_overlaps")
    relation_pairs = {}
    for row in step_rel_rows:
        a = str(row.get("a") or row.get("src") or "")
        b = str(row.get("b") or row.get("dst") or "")
        if not a or not b or a == b:
            continue
        relation = str(row.get("relation") or row.get("type") or "")
        if _relation_is_attach_like(relation):
            relation_pairs[_relation_pair_key(a, b)] = relation

    active_ctx = _active_context(state_snapshot)
    floating_ids = sorted(set(str(x) for x in (active_ctx.get("floating_step_ids") or []) if str(x)))
    grounded_ids = sorted(set(str(x) for x in (active_ctx.get("grounded_step_ids") or []) if str(x)))

    unmet_dependencies = []
    for role, deps in (graph.get("dependencies") or {}).items():
        for dep in deps:
            if role in built_set and dep not in built_set:
                unmet_dependencies.append({"role": role, "missing_dependency": dep})

    # Attachment plausibility for built dependency pairs.
    # If runtime relation evidence is absent, keep neutral prior (do not force failure).
    attachment_checks_total = 0
    attachment_checks_passed = 0
    unattached_pairs = []
    if relation_pairs:
        for role, deps in (graph.get("dependencies") or {}).items():
            if role not in built_set:
                continue
            role_sid = str(role_to_step.get(role) or "")
            if not role_sid:
                continue
            for dep in deps:
                if dep not in built_set:
                    continue
                dep_sid = str(role_to_step.get(dep) or "")
                if not dep_sid:
                    continue
                attachment_checks_total += 1
                if _relation_pair_key(role_sid, dep_sid) in relation_pairs:
                    attachment_checks_passed += 1
                else:
                    unattached_pairs.append({"role": role, "dependency": dep, "step_id": role_sid, "dep_step_id": dep_sid})
        attachment_plausibility = float(attachment_checks_passed) / float(max(1, attachment_checks_total))
    else:
        # No relation evidence at all.  If there are built dependency pairs
        # that *should* have attachment evidence, use a skeptical prior
        # instead of optimistic neutral -- absence of evidence is not evidence
        # of attachment.
        expected_attach_pairs = 0
        for role, deps in (graph.get("dependencies") or {}).items():
            if role not in built_set:
                continue
            for dep in deps:
                if dep in built_set:
                    expected_attach_pairs += 1
        attachment_plausibility = 0.3 if expected_attach_pairs > 0 else 1.0

    required_built_ratio = (len(required_roles.intersection(built_set)) / max(1, len(required_roles))) if required_roles else 1.0
    connectivity_factor = 1.0 / float(1 + disconnected_count)
    floating_factor = 1.0 / float(1 + len(floating_ids))
    raw_score = (
        (0.45 * required_built_ratio)
        + (0.25 * attachment_plausibility)
        + (0.20 * connectivity_factor)
        + (0.10 * floating_factor)
    )
    if unmet_dependencies:
        raw_score = max(0.0, raw_score - 0.15 * min(3, len(unmet_dependencies)))
    if unattached_pairs:
        raw_score = max(0.0, raw_score - 0.10 * min(3, len(unattached_pairs)))
    # Disconnection is a hard structural defect: apply multiplicative penalty
    # so that "all roles built but disconnected" cannot produce a high score.
    if disconnected_count > 0:
        score = raw_score * connectivity_factor
    else:
        score = raw_score

    # Lightweight confidence summary from introspection layer.
    src = state_snapshot.get("introspection_sources") if isinstance(state_snapshot.get("introspection_sources"), dict) else {}
    conf_keys = ("get_connected_components", "get_step_relations", "get_active_construction_context")
    conf_values = []
    for key in conf_keys:
        meta = src.get(key)
        if isinstance(meta, dict):
            try:
                conf_values.append(float(meta.get("confidence", 0.0) or 0.0))
            except Exception:
                pass
    avg_conf = (sum(conf_values) / float(max(1, len(conf_values)))) if conf_values else 0.0

    return {
        "built_roles": built_roles,
        "required_roles_total": len(required_roles),
        "required_roles_built": len(required_roles.intersection(built_set)),
        "required_built_ratio": round(required_built_ratio, 6),
        "disconnected_components": disconnected_count,
        "floating_parts_count": len(floating_ids),
        "floating_step_ids": floating_ids,
        "grounded_step_ids": grounded_ids,
        "attachment_checks_total": attachment_checks_total,
        "attachment_checks_passed": attachment_checks_passed,
        "attachment_plausibility": round(attachment_plausibility, 6),
        "unattached_pairs": unattached_pairs,
        "step_relation_evidence_count": len(relation_pairs),
        "unmet_dependencies": unmet_dependencies,
        "state_signal_confidence": round(avg_conf, 6),
        "assembly_score": round(score, 6),
    }
