"""
Next-step planner for iterative create-mode execution.

Step selection is driven by *constructive assembly logic*, not role coverage.

Priority order (general, not family-specific):
  1. Grounded base / load-bearing masses with no unbuilt dependencies
  2. Support scaffold that reduces floating or disconnected count
  3. Closure fragments that attach to an existing connected component
  4. Main dependent parts whose parent attachment context is already built
  5. Secondary / detail parts (only when assembly is structurally stable)

A candidate step is *admissible* only when:
  - all its declared dependencies are built,
  - its parent / attachment context exists in the current assembly,
  - building it will not leave the assembly in a worse structural state
    than skipping it and picking a closure-enabling step instead.

When the assembly is *unstable* (disconnected > 0, floating > 0, or
attachment evidence below threshold), the planner enters closure-first
mode: only steps that can reduce instability are considered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _step_role(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    role = str(meta.get("role") or "").strip()
    if role:
        return role
    return str(step.get("id") or "").strip()


def _risk_level(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    risk = _norm(meta.get("risk_level"))
    if risk:
        return risk
    primitive = _norm(step.get("primitive"))
    operation = str(step.get("operation") or "").strip()
    if primitive == "cut_extrude" or operation in ("CutFeatureOperation", "IntersectFeatureOperation"):
        return "risky_detail"
    return "core"


def _detail_level(step: Dict[str, Any]) -> str:
    meta = step.get("meta") if isinstance(step.get("meta"), dict) else {}
    detail = _norm(meta.get("detail_level"))
    if detail:
        return detail
    if _risk_level(step) == "risky_detail":
        return "secondary"
    return "core"


def _dependencies_satisfied(role: str, graph: Dict[str, Any], built_roles: Set[str]) -> bool:
    deps = set((graph.get("dependencies") or {}).get(role, []) or [])
    return deps.issubset(built_roles)


# ---------------------------------------------------------------------------
# Structural classification helpers (general, not object-specific)
# ---------------------------------------------------------------------------

def _node_map(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """role -> node dict from graph."""
    out: Dict[str, Dict[str, Any]] = {}
    for n in (graph.get("nodes") or []):
        if isinstance(n, dict):
            r = str(n.get("role") or "").strip()
            if r:
                out[r] = n
    return out


def _is_grounded_role(role: str, node: Dict[str, Any]) -> bool:
    """A role that contacts the ground plane or acts as the foundation."""
    return bool(node.get("grounded"))


def _is_load_bearing(node: Dict[str, Any]) -> bool:
    return bool(node.get("load_bearing"))


def _parent_roles(role: str, graph: Dict[str, Any]) -> Set[str]:
    """Direct dependency parents of *role* in the assembly graph."""
    return set((graph.get("dependencies") or {}).get(role, []) or [])


def _has_attachment_context(role: str, graph: Dict[str, Any], built_roles: Set[str],
                            relation_evidence: Dict[Tuple[str, str], str]) -> bool:
    """True when every parent dependency of *role* is built AND
    at least one parent shows attachment evidence (touch / overlap),
    OR when the role has no parents (root / grounded)."""
    parents = _parent_roles(role, graph)
    if not parents:
        return True  # root role — no attachment context needed
    if not parents.issubset(built_roles):
        return False  # parents not even built yet
    # Parents are built.  If we have relation evidence at all, require
    # at least one parent pair to show attachment.  If no evidence
    # system is active (empty evidence dict), accept based on deps alone.
    if not relation_evidence:
        return True  # no evidence system — fall back to dependency check
    role_to_step = graph.get("role_to_step") or {}
    role_sid = str(role_to_step.get(role) or "")
    for p in parents:
        p_sid = str(role_to_step.get(p) or "")
        if not role_sid or not p_sid:
            continue
        key = tuple(sorted((role_sid, p_sid)))
        if key in relation_evidence:
            return True
    # Parents built but none show attachment evidence — context is weak.
    return False


def _role_structural_priority(
    role: str,
    graph: Dict[str, Any],
    nodes: Dict[str, Dict[str, Any]],
    built_roles: Set[str],
    floating_step_ids: Set[str],
    disconnected_components: int,
    role_to_step: Dict[str, str],
) -> int:
    """Return a numeric priority tier (lower = higher priority).

    Tier 0 — grounded base / load-bearing with no deps (foundation)
    Tier 1 — load-bearing role whose deps are all built (scaffold)
    Tier 2 — non-load-bearing core role whose deps are built (dependent mass)
    Tier 3 — secondary / detail role
    Tier 4 — role whose deps are NOT satisfied (inadmissible)
    """
    node = nodes.get(role, {})
    deps = _parent_roles(role, graph)
    deps_met = deps.issubset(built_roles)

    if not deps_met:
        return 4  # inadmissible

    grounded = _is_grounded_role(role, node)
    load_bearing = _is_load_bearing(node)

    if grounded and load_bearing and not deps:
        return 0  # foundation
    if load_bearing:
        return 1  # scaffold
    kind = str(node.get("kind") or "").strip().lower()
    if kind in ("main", "supporting"):
        return 1
    if kind == "secondary":
        return 3
    return 2  # default core dependent


def _would_reduce_instability(
    role: str,
    role_to_step: Dict[str, str],
    floating_step_ids: Set[str],
    graph: Dict[str, Any],
    built_roles: Set[str],
) -> bool:
    """Heuristic: would building *role* plausibly reduce floating / disconnected?

    True when the role's step id is currently floating, OR when it has
    a built dependency (meaning it can potentially bridge / attach to an
    existing component).
    """
    sid = str(role_to_step.get(role) or "")
    if sid and sid in floating_step_ids:
        return True
    parents = _parent_roles(role, graph)
    if parents and parents.issubset(built_roles):
        return True
    return False


# ---------------------------------------------------------------------------
# Relation evidence extraction (reusable from assembly eval snapshot)
# ---------------------------------------------------------------------------

def _extract_relation_evidence(state_snapshot: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """Build {(step_a, step_b): relation} from state snapshot signals."""
    pairs: Dict[Tuple[str, str], str] = {}
    for key in ("get_step_relations", "get_overlaps"):
        raw = state_snapshot.get(key)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw.get("data")
        if isinstance(raw, dict):
            for sub in ("relations", "overlaps", "items"):
                if isinstance(raw.get(sub), list):
                    raw = raw.get(sub)
                    break
        if not isinstance(raw, list):
            continue
        for row in raw:
            if not isinstance(row, dict):
                continue
            a = str(row.get("a") or row.get("src") or "")
            b = str(row.get("b") or row.get("dst") or "")
            rel = str(row.get("relation") or row.get("type") or "").lower()
            if a and b and a != b and ("touch" in rel or "intersect" in rel or "contact" in rel or "overlap" in rel):
                pairs[tuple(sorted((a, b)))] = rel
    return pairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def choose_next_fragment(
    *,
    plan_steps: List[Dict[str, Any]],
    graph: Dict[str, Any],
    state_snapshot: Dict[str, Any],
    priors: Dict[str, Any],
    max_steps_per_iteration: int = 2,
) -> Dict[str, Any]:
    executed_ids = set(state_snapshot.get("executed_step_ids") or [])
    role_to_step = graph.get("role_to_step") or {}
    built_roles = {role for role, sid in role_to_step.items() if sid and sid in executed_ids}
    required_roles = set(graph.get("required_roles") or [])
    core_complete = required_roles.issubset(built_roles) if required_roles else False

    # Assembly state signals.
    assembly_eval = state_snapshot.get("assembly_eval") if isinstance(state_snapshot.get("assembly_eval"), dict) else {}
    disconnected_components = int(assembly_eval.get("disconnected_components", 0) or 0)
    floating_parts = int(assembly_eval.get("floating_parts_count", 0) or 0)
    attachment_plausibility = float(assembly_eval.get("attachment_plausibility", 1.0) or 0.0)
    floating_step_ids = set(str(x) for x in (assembly_eval.get("floating_step_ids") or []) if str(x))

    assembly_unstable = disconnected_components > 0 or floating_parts > 0 or attachment_plausibility < 0.5

    # Relation evidence for attachment-context checks.
    relation_evidence = _extract_relation_evidence(state_snapshot)

    # Graph metadata.
    nodes = _node_map(graph)

    # Base ordering from priors, then graph build_order as fallback.
    ordering = [r for r in (priors.get("ordering") or []) if isinstance(r, str) and r.strip()]
    if not ordering:
        ordering = [r for r in (graph.get("build_order") or []) if isinstance(r, str) and r.strip()]

    # ------------------------------------------------------------------
    # Score every unbuilt role by structural priority
    # ------------------------------------------------------------------
    step_by_id = {
        str((s or {}).get("id") or ""): s
        for s in plan_steps
        if isinstance(s, dict) and str((s or {}).get("id") or "").strip()
    }

    candidates: List[Dict[str, Any]] = []  # [{role, step, tier, order_idx, reduces_instability}]
    for idx, role in enumerate(ordering):
        sid = str(role_to_step.get(role) or "")
        if not sid or sid in executed_ids:
            continue
        step = step_by_id.get(sid)
        if not isinstance(step, dict):
            continue

        tier = _role_structural_priority(
            role, graph, nodes, built_roles, floating_step_ids,
            disconnected_components, role_to_step,
        )
        if tier >= 4:
            continue  # deps not met — inadmissible

        risk = _risk_level(step)
        detail = _detail_level(step)
        is_detail = risk == "risky_detail" or detail == "secondary"

        # Promote detail to tier 3 if not already.
        if is_detail and tier < 3:
            tier = 3

        reduces = _would_reduce_instability(
            role, role_to_step, floating_step_ids, graph, built_roles,
        )

        # --- Attachment-context gate (general rule) ---
        # A dependent role (has parents) must not be built when its parent
        # attachment context is absent AND the assembly is already unstable.
        has_context = _has_attachment_context(role, graph, built_roles, relation_evidence)
        if not has_context and assembly_unstable and not _is_grounded_role(role, nodes.get(role, {})):
            # Demote: treat as inadmissible while unstable without context.
            continue

        candidates.append({
            "role": role,
            "step": step,
            "tier": tier,
            "order_idx": idx,
            "is_detail": is_detail,
            "reduces_instability": reduces,
        })

    # ------------------------------------------------------------------
    # Sort candidates by constructive assembly priority
    # ------------------------------------------------------------------
    # Primary: tier (lower = build first)
    # Secondary: prefer roles that reduce instability
    # Tertiary: prior ordering (lower idx = earlier in declared order)
    candidates.sort(key=lambda c: (
        c["tier"],
        0 if c["reduces_instability"] else 1,
        c["order_idx"],
    ))

    # ------------------------------------------------------------------
    # Apply assembly-state-aware filters
    # ------------------------------------------------------------------
    selected_steps: List[Dict[str, Any]] = []
    selected_roles: List[str] = []
    reason = "no suitable role"

    for cand in candidates:
        role = cand["role"]
        step = cand["step"]
        is_detail = cand["is_detail"]
        tier = cand["tier"]

        # Rule: do not build detail/secondary until core roles are done.
        if not core_complete and is_detail:
            reason = f"defer non-core role={role} until primary masses complete"
            continue

        # Rule: when assembly is unstable, only accept steps that could
        # reduce instability (closure-first mode).
        if assembly_unstable and not cand["reduces_instability"] and is_detail:
            reason = f"defer detail role={role} while assembly unstable"
            continue

        # Rule: when assembly is unstable AND core is complete,
        # block ALL detail steps — only closure/support steps allowed.
        if core_complete and assembly_unstable and is_detail:
            reason = f"defer detail role={role} while assembly stability is low"
            continue

        selected_steps.append(step)
        selected_roles.append(role)
        reason = f"selected role={role} (tier={tier})"
        if assembly_unstable:
            reason += " [closure-priority]"
        if len(selected_steps) >= max(1, int(max_steps_per_iteration)):
            break

    # Fallback: if no role matched but core is complete and stable,
    # allow one safe secondary step.
    if not selected_steps and core_complete and not assembly_unstable:
        for cand in candidates:
            if _risk_level(cand["step"]) == "risky_detail":
                continue
            selected_steps.append(cand["step"])
            selected_roles.append(cand["role"])
            reason = f"fallback secondary role={cand['role']}"
            break

    return {
        "selected_roles": selected_roles,
        "steps": selected_steps,
        "phase": "core" if not core_complete else "secondary",
        "reason": reason,
        "core_complete_before_selection": core_complete,
        "stability_guard_active": assembly_unstable,
    }
