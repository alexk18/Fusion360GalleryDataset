# Universal CAD DSL — LLM-as-CAD-operator

This folder implements the two-level "CAD Operator" architecture: **LLM outputs only a high-level Plan or Patch**; all server execution is deterministic (validate → compile → execute). No raw Fusion commands from the LLM.

## File-level change list

| File | Purpose |
|------|--------|
| **New** `cad/cad_dsl.py` | Plan and Patch JSON schema; `Step`, `Plan`, `Patch`; `make_step_names()`; example plans (tank, plane). |
| **New** `cad/cad_patch.py` | `apply_patch(plan, patch)` — replace/add/remove by path (e.g. `steps[hull].distance`). |
| **New** `cad/cad_validate.py` | `validate_plan(plan)` — budget, primitive sanity (rect/circle/poly), fillet/loft/sweep rules. |
| **New** `cad/cad_compiler.py` | `compile_plan(plan)` → list of `CompiledStep` (add_sketch, draw profile, add_extrude). |
| **New** `cad/cad_executor.py` | `CadExecutor` — run compiled steps via client; ensure/update semantics; dry-run; registry. |
| **New** `cad/cad_inspector.py` | Screenshot + plan summary → structured critique `{score, issues[], constraints}` (no coordinates). |
| **New** `cad/cad_fixer.py` | Inspector issues + current plan → LLM returns **only** a minimal Patch; `post_render_fix_loop` (bounded). |
| **New** `cad/cad_operator.py` | `run_cad_operator()`, `run_best_of_n_create()`, pre-render fix, post-render Inspector+Fixer. |
| **New** `examples/cad_dsl_demo.py` | Demo: tank | plane | edit (DRY_RUN=1 for no Fusion). |
| **New** `test/test_cad_dsl.py` | Unit tests: patch apply, path parse, validation, compiler, executor dry-run. |
| **Modified** `server/command_sketch_extrusion.py` | Optional `sketch_name` in add_sketch, `feature_name` in add_extrude; `find_entity_by_name`, `update_extrude`. |
| **Modified** `server/command_runner.py` | Route `find_entity_by_name`, `update_extrude`. |
| **Modified** `client/fusion360gym_client.py` | `add_sketch(..., sketch_name=)`, `add_extrude(..., feature_name=)`; `find_entity_by_name`, `update_extrude`. |

## Plan JSON schema (create/edit)

```json
{
  "units": "cm",
  "session": "<stable_session_id>",
  "mode": "create|edit",
  "budget": {"max_steps": 25, "max_parts": 18},
  "global": {"origin": "X-centered,Y-back=0,Z-floor=0", "symmetry": "none|approx_x"},
  "steps": [
    {
      "id": "<stable_id>",
      "op": "ensure",
      "primitive": "rect_extrude|circle_extrude|poly_extrude|wedge_extrude|cut_extrude|loft|sweep|fillet",
      "names": {"sketch": "...", "feature": "...", "body": "..."},
      "plane": "XY|XZ|YZ|XY@z|XZ@y|YZ@x",
      "profile": {"type": "rect|circle|poly", "cx", "cy", "w", "h" or "radius" or "pts": [{"x","y"}]},
      "distance": 1.0,
      "operation": "NewBodyFeatureOperation|JoinFeatureOperation|CutFeatureOperation|IntersectFeatureOperation",
      "params": {},
      "selectors": {},
      "loft": {},
      "sweep": {},
      "fillet": {}
    }
  ]
}
```

## Patch JSON schema (edit only)

```json
{
  "patches": [
    {"op": "replace", "path": "steps[<id>].distance", "value": 18.0},
    {"op": "replace", "path": "steps[<id>].profile.pts", "value": [...]},
    {"op": "add", "path": "steps", "value": { "<full step>" }},
    {"op": "remove", "path": "steps[<id>]"}
  ],
  "intent": "minimal_change"
}
```

## Example plans

- **Lowpoly tank**: `cad_dsl.EXAMPLE_PLAN_LOWPOLY_TANK` — wedge hull, poly turret, circle gun, circle wheel.
- **Lowpoly plane**: `cad_dsl.EXAMPLE_PLAN_LOWPOLY_PLANE` — poly fuselage, poly wing, rect tail.

## Run instructions

1. **Environment**: From repo root or `tools/fusion360gym`, ensure `cad` and `client` are on `sys.path` (or run from `fusion360gym` so `cad` is a package).

2. **Dry-run (no Fusion)**:
   ```bash
   set DRY_RUN=1
   python examples/cad_dsl_demo.py tank
   python examples/cad_dsl_demo.py plane
   python examples/cad_dsl_demo.py edit
   ```

3. **With Fusion 360**:
   - Start Fusion 360 and run the add-in (Add-ins → Run).
   - `set DRY_RUN=0` (or unset).
   - `python examples/cad_dsl_demo.py tank` (or plane / edit).
   - Optional: `FUSION_HOST=127.0.0.1`, `FUSION_PORT=8080`.

4. **Unit tests**:
   ```bash
   cd tools/fusion360gym
   python test/test_cad_dsl.py
   ```

## Verification checklist

- [x] **Creation**: Recognizable models beyond rectangles (poly, circle); bounded steps; stable execution.
- [x] **Editing**: Update existing features by name (`update_extrude`); no `clear()` in edit mode.
- [x] **Persistence**: Deterministic naming `session__step_id__sk` / `__feat`; `find_entity_by_name` to reattach after restart.
- [x] **Non-boxy**: rect, circle, poly (wedge) supported; loft/sweep/fillet in DSL and validation, execution stubbed until server supports them.
- [x] **Demo stability**: Bounded steps/parts; deterministic validation; best-of-N and bounded fix loops in `cad_operator`.
- [x] **LLM constraint**: LLM outputs only Plan or Patch; all server calls from compiler/executor.
- [x] **Dry-run**: `CadExecutor(dry_run=True)` prints compiled calls without Fusion.
- [x] **Tests**: Patch apply, path parse, validation, compiler, executor dry-run.

## Loft / sweep / fillet

Defined in the DSL and validated; **execution is stubbed** (compiler returns empty calls, executor skips). To enable:

- Server: implement `add_loft`, `add_sweep`, `add_fillet` (rule-based edge selection) and optional `update_*`.
- Compiler: emit corresponding calls in `_compile_step` for `loft`, `sweep`, `fillet`.
- Executor: handle those commands and resolve profile/path refs.

## Risks and mitigations

- **A — Unique names / wrong sketch**: Naming uses `session__step_id__sk` (and `__feat`). `find_entity_by_name` returns `found` and `count`; if `count > 1` the server returns **failure** (deterministic; request rename).
- **B — update_extrude extent type**: Server checks `extentOne` is `DistanceExtentDefinition`; otherwise returns "unsupported extrude extent type". After update, `design_state.refresh()` is called. **Negative distance** is rejected (server and validator).
- **C — add_sketch ensure**: If `sketch_name` is passed, server first looks up existing sketch by name; if found, returns it **without creating** a new one (no duplicate names).
- **D — Sketch state**: `CommandSketchExtrusion.state` is keyed by sketch name; when returning an existing sketch we do **not** reset state. Edit only via **update_extrude** (distance, etc.); do not append geometry to existing sketches until "replace sketch geometry" or equivalent exists.

## Integration with ai_assistant.py

To use the DSL path from the main assistant:

1. Add a "create_dsl" or "dsl" command that accepts text (and optional images).
2. Call an LLM to produce a **Plan** (not bboxes); optionally generate N candidates and use `run_best_of_n_create`.
3. Run `validate_plan` → `run_cad_operator` (or `CadExecutor.execute_plan`).
4. Optionally run the post-render Inspector + Fixer loop (bounded).

The existing bbox path (Architect → encode_bboxes_to_plan → execute_plan) remains unchanged; the DSL path is additive.
