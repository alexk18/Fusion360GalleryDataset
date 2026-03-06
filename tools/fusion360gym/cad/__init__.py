# Universal CAD DSL for Fusion360Gym LLM-as-CAD-operator.
# Plan (create/edit) and Patch (edit-only) formats; validation; compiler; executor;
# inspector; fixer; operator (best-of-N, post-render fix).

from .cad_dsl import (
    PLAN_SCHEMA_VERSION,
    Plan,
    Step,
    Patch,
    default_budget,
    make_step_names,
    EXAMPLE_PLAN_LOWPOLY_TANK,
    EXAMPLE_PLAN_LOWPOLY_PLANE,
)
from .cad_patch import apply_patch
from .cad_validate import validate_plan, ValidationResult
from .cad_compiler import compile_plan, CompiledStep
from .cad_executor import CadExecutor, ExecutionResult

__all__ = [
    "PLAN_SCHEMA_VERSION",
    "Plan",
    "Step",
    "Patch",
    "default_budget",
    "make_step_names",
    "EXAMPLE_PLAN_LOWPOLY_TANK",
    "EXAMPLE_PLAN_LOWPOLY_PLANE",
    "apply_patch",
    "validate_plan",
    "ValidationResult",
    "compile_plan",
    "CompiledStep",
    "CadExecutor",
    "ExecutionResult",
]
