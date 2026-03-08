"""
Iterative tool-driven CAD create agent loop.

Primary create architecture:
request -> structural spec -> candidate DSL prior -> (loop)
    inspect runtime state -> choose next subgoal -> execute fragment ->
    inspect state -> update relational world model -> continue
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cad_backend import FusionCadBackend
from .cad_capabilities import CapabilityModel, default_capability_model
from .cad_dataset_priors import build_dataset_priors
from .cad_executor import ExecutionResult, ExecutionTraceEvent
from .cad_next_step_planner import choose_next_fragment
from .cad_operator import run_cad_operator
from .cad_relational_assembly import build_relational_assembly_graph, evaluate_relational_graph
from .cad_state_introspection import inspect_runtime_state
from .cad_structural_ranker import rank_dsl_candidates
from .cad_validate import validate_plan


@dataclass
class IterativeCreateOutcome:
    success: bool
    best_index: int
    best_plan: Dict[str, Any] = field(default_factory=dict)
    result: ExecutionResult = field(
        default_factory=lambda: ExecutionResult(success=False, steps_ok=0, steps_total=0, message="")
    )
    loop_trace: List[Dict[str, Any]] = field(default_factory=list)
    state_snapshots: List[Dict[str, Any]] = field(default_factory=list)


class IterativeCreateAgent:
    def __init__(
        self,
        client: Any,
        *,
        capabilities: Optional[CapabilityModel] = None,
        step_delay: float = 0.4,
        inspect_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        execute_fn: Optional[Callable[[Dict[str, Any]], ExecutionResult]] = None,
    ):
        self.client = client
        self.capabilities = capabilities or default_capability_model()
        self.step_delay = float(step_delay)
        self.backend = FusionCadBackend(client, capabilities=self.capabilities)
        self.inspect_fn = inspect_fn
        self.execute_fn = execute_fn

    def _execute_fragment(self, fragment_plan: Dict[str, Any]) -> ExecutionResult:
        if self.execute_fn is not None:
            return self.execute_fn(fragment_plan)
        return run_cad_operator(
            self.client,
            fragment_plan,
            dry_run=False,
            step_delay=self.step_delay,
            edit_mode=False,
            capabilities=self.capabilities,
        )

    def _inspect_state(self, executed_steps: List[Dict[str, Any]], planned_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.inspect_fn is not None:
            return self.inspect_fn(executed_steps=executed_steps, planned_steps=planned_steps)
        return inspect_runtime_state(
            self.backend,
            executed_steps=executed_steps,
            planned_steps=planned_steps,
        )

    @staticmethod
    def _merge_registries(reg_a: Dict[str, Dict[str, Any]], reg_b: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        merged = dict(reg_a or {})
        for key, value in (reg_b or {}).items():
            if isinstance(value, dict):
                merged[key] = dict(value)
        return merged

    def run(
        self,
        *,
        user_request: str,
        structural_spec: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        max_iterations: int = 32,
    ) -> IterativeCreateOutcome:
        if not candidates:
            result = ExecutionResult(success=False, steps_ok=0, steps_total=0, message="No create candidates")
            return IterativeCreateOutcome(False, -1, {}, result, [], [])

        spec = dict(structural_spec or {})
        family = str(spec.get("object_family") or "").strip().lower()
        ranked = rank_dsl_candidates(candidates, spec, capabilities=self.capabilities) if spec else []
        if ranked:
            best_plan = dict((ranked[0] or {}).get("plan") or {})
            best_index = candidates.index(best_plan) if best_plan in candidates else 0
        else:
            best_plan = dict(candidates[0] or {})
            best_index = 0

        plan_steps = [s for s in (best_plan.get("steps") or []) if isinstance(s, dict)]
        if not plan_steps:
            result = ExecutionResult(success=False, steps_ok=0, steps_total=0, message="Selected candidate has no steps")
            return IterativeCreateOutcome(False, best_index, best_plan, result, [], [])

        graph = build_relational_assembly_graph(spec, plan_steps)
        priors = build_dataset_priors(
            family,
            user_request=user_request,
            structural_build_order=list(spec.get("build_order") or []),
        )

        executed_steps: List[Dict[str, Any]] = []
        executed_ids = set()
        loop_trace: List[Dict[str, Any]] = []
        state_snapshots: List[Dict[str, Any]] = []
        merged_registry: Dict[str, Dict[str, Any]] = {}
        merged_exec_trace: List[ExecutionTraceEvent] = []
        total_steps_target = len(plan_steps)
        last_assembly_eval: Dict[str, Any] = {}

        # Create-mode deterministic reset: clear once before iterative loop.
        try:
            if hasattr(self.client, "clear"):
                self.client.clear()
            if hasattr(self.client, "refresh"):
                self.client.refresh()
        except Exception:
            pass

        max_loops = max(1, min(int(max_iterations), max(4, total_steps_target * 2)))
        failure_message = ""
        hard_failure = False

        for iteration in range(1, max_loops + 1):
            state = self._inspect_state(executed_steps, plan_steps)
            state["executed_step_ids"] = sorted(executed_ids)
            assembly_eval = evaluate_relational_graph(graph, state)
            state["assembly_eval"] = assembly_eval
            state["snapshot_phase"] = "pre_step"
            state_snapshots.append(state)
            last_assembly_eval = dict(assembly_eval or {})

            next_fragment = choose_next_fragment(
                plan_steps=plan_steps,
                graph=graph,
                state_snapshot=state,
                priors=priors,
                max_steps_per_iteration=1,
            )
            fragment_steps = [s for s in (next_fragment.get("steps") or []) if isinstance(s, dict)]
            selected_roles = list(next_fragment.get("selected_roles") or [])
            phase = str(next_fragment.get("phase") or "core")

            if not fragment_steps:
                loop_trace.append(
                    {
                        "iteration": iteration,
                        "action": "stop",
                        "phase": phase,
                        "reason": next_fragment.get("reason", "no next fragment"),
                        "selected_roles": selected_roles,
                    }
                )
                break

            fragment_plan = {
                "units": best_plan.get("units", "cm"),
                "session": str(best_plan.get("session") or "iterative_session"),
                "mode": "create",
                "budget": dict(best_plan.get("budget") or {}),
                "global": copy.deepcopy(best_plan.get("global") or {}),
                "steps": copy.deepcopy(fragment_steps),
            }
            fragment_plan.setdefault("global", {})
            if isinstance(fragment_plan["global"], dict):
                fragment_plan["global"]["iterative_fragment"] = {
                    "iteration": iteration,
                    "phase": phase,
                    "selected_roles": selected_roles,
                    "reason": next_fragment.get("reason", ""),
                }

            vr = validate_plan(fragment_plan, capabilities=self.capabilities, allow_unsupported=False)
            if not vr.valid:
                msg = "; ".join(vr.errors[:3]) if vr.errors else "fragment validation failed"
                if phase != "core":
                    loop_trace.append(
                        {
                            "iteration": iteration,
                            "action": "skip_fragment",
                            "phase": phase,
                            "selected_roles": selected_roles,
                            "reason": msg,
                        }
                    )
                    continue
                failure_message = msg
                hard_failure = True
                loop_trace.append(
                    {
                        "iteration": iteration,
                        "action": "failed_fragment",
                        "phase": phase,
                        "selected_roles": selected_roles,
                        "reason": msg,
                    }
                )
                break

            fragment_result = self._execute_fragment(fragment_plan)
            merged_exec_trace.extend(list(fragment_result.trace or []))
            merged_registry = self._merge_registries(merged_registry, fragment_result.registry or {})
            if not fragment_result.success:
                if phase != "core":
                    loop_trace.append(
                        {
                            "iteration": iteration,
                            "action": "skip_failed_optional",
                            "phase": phase,
                            "selected_roles": selected_roles,
                            "reason": fragment_result.message,
                        }
                    )
                    continue
                failure_message = fragment_result.message
                hard_failure = True
                loop_trace.append(
                    {
                        "iteration": iteration,
                        "action": "failed_core",
                        "phase": phase,
                        "selected_roles": selected_roles,
                        "reason": fragment_result.message,
                    }
                )
                break

            new_ids = []
            for step in fragment_steps:
                sid = str(step.get("id") or "")
                if sid and sid not in executed_ids:
                    executed_ids.add(sid)
                    executed_steps.append(step)
                    new_ids.append(sid)
            loop_trace.append(
                {
                    "iteration": iteration,
                    "action": "created",
                    "phase": phase,
                    "selected_roles": selected_roles,
                    "executed_step_ids": new_ids,
                }
            )

            # Refresh runtime state after execution to drive state-aware stop conditions.
            post_state = self._inspect_state(executed_steps, plan_steps)
            post_state["executed_step_ids"] = sorted(executed_ids)
            post_eval = evaluate_relational_graph(graph, post_state)
            post_state["assembly_eval"] = post_eval
            post_state["snapshot_phase"] = "post_step"
            state_snapshots.append(post_state)
            last_assembly_eval = dict(post_eval or {})
            loop_trace.append(
                {
                    "iteration": iteration,
                    "action": "post_step_eval",
                    "phase": phase,
                    "disconnected_components": int(post_eval.get("disconnected_components", 0) or 0),
                    "floating_parts_count": int(post_eval.get("floating_parts_count", 0) or 0),
                    "attachment_plausibility": float(post_eval.get("attachment_plausibility", 0.0) or 0.0),
                }
            )

            # --- Structural stop condition ---
            # Success requires BOTH role completion AND structural plausibility.
            req = set(graph.get("required_roles") or [])
            built_roles = {
                role
                for role, sid in (graph.get("role_to_step") or {}).items()
                if sid and sid in executed_ids
            }
            post_disconnected = int(post_eval.get("disconnected_components", 0) or 0)
            post_floating = int(post_eval.get("floating_parts_count", 0) or 0)
            post_attachment = float(post_eval.get("attachment_plausibility", 1.0) or 0.0)
            connectivity_ok = post_disconnected <= 0
            floating_ok = post_floating <= 0
            attachment_ok = post_attachment >= 0.5
            structurally_plausible = connectivity_ok and floating_ok and attachment_ok

            if req and req.issubset(built_roles) and structurally_plausible:
                loop_trace.append(
                    {
                        "iteration": iteration,
                        "action": "stop_required_complete",
                        "phase": phase,
                        "required_roles_built": len(req),
                        "connectivity_ok": connectivity_ok,
                        "floating_ok": floating_ok,
                        "attachment_ok": attachment_ok,
                    }
                )
                break

            if req and req.issubset(built_roles) and not structurally_plausible:
                loop_trace.append(
                    {
                        "iteration": iteration,
                        "action": "continue_closure",
                        "phase": "closure",
                        "reason": "required roles built but assembly structurally unstable",
                        "disconnected_components": post_disconnected,
                        "floating_parts_count": post_floating,
                        "attachment_plausibility": post_attachment,
                    }
                )

        final_plan = copy.deepcopy(best_plan)
        final_plan["steps"] = copy.deepcopy(executed_steps)
        final_plan.setdefault("global", {})
        if isinstance(final_plan["global"], dict):
            final_plan["global"]["iterative_agent"] = {
                "mode": "tool_driven_create",
                "family": family,
                "iterations": len(loop_trace),
                "priors_source": priors.get("source", ""),
                "priors_strategy": priors.get("strategy", ""),
                "executed_step_count": len(executed_steps),
                "target_step_count": total_steps_target,
            }

        final_disconnected = int(last_assembly_eval.get("disconnected_components", 0) or 0)
        final_floating = int(last_assembly_eval.get("floating_parts_count", 0) or 0)
        final_attachment = float(last_assembly_eval.get("attachment_plausibility", 1.0) or 0.0)
        runtime_plausible = final_disconnected <= 0 and final_floating <= 0 and final_attachment >= 0.5
        success = bool(executed_steps) and not hard_failure and runtime_plausible
        if not success and not failure_message:
            if not executed_steps:
                failure_message = "iterative create produced no executable core steps"
            elif not runtime_plausible:
                failure_message = (
                    "iterative create stopped with unresolved runtime assembly issues: "
                    f"disconnected={final_disconnected}, floating={final_floating}, "
                    f"attachment={round(final_attachment, 4)}"
                )
        if success:
            message = f"Iterative create completed: {len(executed_steps)}/{total_steps_target} steps"
        else:
            message = failure_message

        result = ExecutionResult(
            success=success,
            steps_ok=len(executed_steps),
            steps_total=total_steps_target,
            message=message,
            registry=merged_registry,
            trace=merged_exec_trace,
        )
        return IterativeCreateOutcome(success, best_index, final_plan, result, loop_trace, state_snapshots)
