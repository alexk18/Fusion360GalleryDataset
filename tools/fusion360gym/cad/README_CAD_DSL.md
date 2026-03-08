# Fusion CAD DSL Backend Contract

This module set implements an MCP-style CAD backend flow:

`LLM/agent -> CAD Plan or Patch DSL -> validator -> compiler -> backend tools -> executor -> inspector/fixer`

The primary system contract is CAD DSL operations, not bounding boxes and not raw Fusion commands.

## Current State

- LLM-facing contract is Plan/Patch DSL (`cad_dsl.py`, `cad_patch.py`).
- Capability-aware validation is enforced (`cad_validate.py`, `cad_capabilities.py`).
- Execution uses backend tool surface (`cad_backend.py`) and stateful executor (`cad_executor.py`).
- Edit mode supports real in-place `update_extrude` for distance change.
- Unsupported edits are deterministically classified and do not silently pass.
- Unsupported primitives do not report success in normal execution.
- Dry-run can report unsupported primitives as stubs without building geometry.

## Backend Tool Surface

Main backend methods (`FusionCadBackend`):

- `list_tools`
- `create_sketch`
- `add_line`
- `add_arc`
- `add_circle`
- `close_profile`
- `extrude`
- `cut`
- `find_entity_by_name`
- `list_entities`
- `list_features`
- `get_model_state`
- `get_features`
- `get_sketches`
- `get_bodies`
- `get_parts`
- `get_body_bbox`
- `get_feature_bbox`
- `get_faces`
- `get_edges`
- `get_body_relations`
- `get_feature_body_relations`
- `get_connected_components`
- `get_overlaps`
- `get_active_construction_context`
- `screenshot`
- `query_bounding_box`
- `update_extrude`

Compatibility helper used by compiler profile emission:

- `add_point` (legacy helper, not primary external contract)

Declared extension points (currently unsupported):

- `fillet`
- `chamfer`
- `revolve`
- `loft`
- `sweep`
- future: edge/face queries, suppress/delete, sketch parameter updates

## Capability Table

Source: `cad_capabilities.default_capability_model()`

Create:
- `sketch_rect`: supported
- `sketch_circle`: supported
- `sketch_poly`: supported
- `sketch_arc`: unsupported
- `profile_close`: supported
- `extrude_new_body`: supported
- `extrude_cut`: supported

Edit:
- `update_extrude_distance`: supported
- `sketch_edit_existing`: partial
- `suppress_feature`: unsupported
- `delete_feature`: unsupported

Query:
- `list_tools`: supported
- `find_entity_by_name`: supported
- `list_entities`: supported
- `list_features`: supported
- `get_features`: supported
- `get_model_state`: supported
- `get_sketches`: supported
- `get_bodies`: partial (exact when runtime command is available)
- `get_parts`: partial
- `get_body_bbox`: partial
- `get_feature_bbox`: partial
- `get_faces`: partial
- `get_edges`: partial
- `get_body_relations`: partial
- `get_feature_body_relations`: partial
- `get_connected_components`: partial
- `get_overlaps`: partial
- `active_construction_context`: supported
- `model_state_summary`: supported
- `screenshot`: supported
- `bbox_query`: supported
- `edge_query`: unsupported
- `face_query`: unsupported

Advanced primitives:
- `fillet`: unsupported
- `chamfer`: unsupported
- `revolve`: unsupported
- `loft`: unsupported
- `sweep`: unsupported

## Supported Create Operations

- `rect_extrude`
- `circle_extrude`
- `poly_extrude`
- `wedge_extrude`
- `cut_extrude`

Validator enforces primitive/profile consistency:
- `rect_extrude` / `cut_extrude` -> `profile.type=rect`
- `circle_extrude` -> `profile.type=circle`
- `poly_extrude` / `wedge_extrude` -> `profile.type=poly`

## Supported In-Place Edit Operations

- Existing extrude distance update:
  - detect existing feature by stable name
  - call `update_extrude(feature_name, new_distance)`
  - update registry and trace with action `updated`
- Operator path supports deterministic edit diff checks when `previous_plan` is provided
  (`run_cad_operator(..., edit_mode=True, previous_plan=...)`).

No `clear()` is used in edit mode.

## Recreate-Required Operations

Detected by `classify_edit_change(step_before, step_after)` in `cad_executor.py`.

Current deterministic recreate-required cases:

- plane change
- operation change
- profile geometry change (rect/circle/poly)
- feature-name change

Current behavior for recreate-required:

- controlled recreate pipeline is triggered with a deterministic suffix strategy
- strategy: rebuild from changed step to plan tail (no `clear()`)
- precheck: if suffix feature names already exist and delete/suppress is unavailable, fail explicitly
- if suffix features are absent, suffix is rebuilt deterministically
- no silent skip
- no fake success

## Unsupported Operations

Normal execution (non dry-run) fails for:

- `loft`
- `sweep`
- `fillet`
- `chamfer`
- `revolve`

Unsupported primitive result is explicit (`action=unsupported`, `success=False`).

## Arcs Status

Arcs are not supported in current execution path.

- Poly profile with `profile.arcs` is rejected by validator.
- Compiler emits straight segments only.
- Arc-related fields are never silently ignored.

## Dry-Run Semantics

- Dry-run executes deterministic compile/executor flow without Fusion geometry mutation.
- Unsupported primitives can be emitted as stubs:
  - trace action: `unsupported`
  - reason includes `dry-run stub; no geometry built`
- This is not treated as built geometry.

## No Fake Success Policy

- Unsupported primitives cannot pass as successful build in normal mode.
- Non-compilable steps (unknown/malformed primitives that compiler cannot emit) fail execution explicitly.
- Unsupported in-place edits cannot pass as successful update.
- Duplicate entity-name lookup errors propagate as failures.
- Capability rejection reports explicit reason.

## add_sketch Contract

`add_sketch` with an existing `sketch_name` uses existing sketch as authoritative.

- Existing sketch is returned.
- Requested `sketch_plane` is informational and does not remap existing sketch plane.
- Edit path should update feature via `update_extrude`; not append profile geometry blindly.

## Observability Contract

`ExecutionResult.trace` records:

- compiled step id
- primitive
- action: `created | updated | recreate-required | unsupported | failed`
- backend command path
- reason (validation/capability/update failure details)

Post-render patch application is logged by operator trace with backend path:

- `post_render_patch_application`

## Text-to-CAD and Image-to-CAD Convergence

Integration contract (prepared architecture):

`images/text -> interpretation -> structural representation -> CAD planning -> validation -> execution`

Module: `cad_multimodal.py`

- `text_to_cad_plan(...)`
- `image_to_cad_plan(...)`
- `StructuralObjectRepresentation`
- `CadPlanner` interface

Legacy adapter:

- `structural_from_legacy_bboxes(...)`
- marked as fallback bridge only

Important: text and image paths are expected to converge into one CAD DSL execution path.

`examples/ai_assistant.py` now uses CAD DSL as the default for:
- text requests
- `build from image ...`
- `build from folder ...`
- create-mode now enforces a Structural Build Spec intermediate stage:
  - `text/image/folder -> shape family classification -> structural build spec -> family grammar DSL synthesis -> validate -> execute`
  - structural spec is a required contract for create synthesis and ranking
  - generic one-shot DSL synthesis is no longer primary for create mode
- create execution now runs in iterative tool-driven mode (`cad_agent_loop.py`) as primary path:
  - `input -> structural spec -> ranked DSL candidate prior -> inspect runtime state -> choose next safe fragment -> validate -> execute -> inspect -> continue`
  - loop uses runtime introspection (`cad_state_introspection.py`) and relational assembly graph (`cad_relational_assembly.py`)
  - planning uses dataset-informed generalized priors (`cad_dataset_priors.py`) for ordering/primitive preferences (no instance replay)
  - primary massing/core roles are executed before risky detail/cut operations
  - non-essential risky detail can be deferred/skipped in first-pass create without reporting fake failure
- family grammar synthesis (`cad_family_grammar.py`, `cad_dsl_synthesizer.py`) enforces:
  - role-aware step generation
  - family-specific primitive mix
  - construction-order constraints
  - silhouette/support/attachment-oriented templates
  - create execution safety layer:
    - per-step classification: `core | secondary | risky_detail`
    - primary-first candidate generation (risky detail pruned)
    - optional detail candidate only when risky booleans pass safety checks
    - non-essential cut/intersect details are demoted/pruned in first-pass create mode
- structural build families:
  - `furniture_boxy_panel`
  - `lowpoly_hard_surface_vehicle`
  - `profile_driven_symmetric`
  - `rotational_bodies`
  - `unsupported_smooth_freeform`
- structural outcomes:
  - `exact`
  - `lowpoly_approx`
  - `blocked`
- iterative state introspection surface includes:
  - `list_tools`, `list_capabilities`
  - `get_model_state`, `get_features`, `get_sketches`, `get_bodies`, `get_parts`
  - `get_faces`, `get_edges` (exact summaries when Fusion query is available; explicit heuristic fallback otherwise)
  - `get_body_bbox`, `get_feature_bbox`
  - `get_body_to_body_relations`, `get_feature_to_body_relations`, `get_step_relations`
  - `get_connected_components`, `get_overlaps`, `get_active_construction_context`
  - `introspection_sources` per-query confidence/source labels:
    - `exact_fusion_api`
    - `derived_exact`
    - `heuristic_estimate`
  - no heuristic signal is reported as exact geometry
- relational assembly evaluation now consumes runtime state signals (components, overlaps/contacts, feature-body mapping, floating/grounded context) to score:
  - attachment plausibility
  - disconnected components
  - floating parts
  - dependency satisfaction
- iterative stop condition is runtime-state-aware:
  - required roles built
  - connectedness acceptable
  - floating parts resolved
  - attachment plausibility above threshold
- rotational requests (e.g. bottle/vase) with unavailable `revolve` are explicit `lowpoly_approx` with stepped-radial policy
- smooth/freeform requests are explicit `blocked` and are not silently downgraded to box junk
- DSL candidate ranking in create mode uses structural plausibility metrics in addition to validator pass:
  - family fit
  - capability fit
  - structural coverage
  - role-to-step coherence
  - primitive appropriateness
  - attachment plausibility
  - build-order coherence
  - silhouette consistency
  - support plausibility
  - robust-primary preference (`risky_detail_ratio`, `create_safety_score`)
- advanced primitives are not silently downgraded to basic extrudes during plan sanitization;
  unsupported ones are rejected by capability-aware validation in normal mode
- create-only shape-family policy:
  - `furniture_boxy_panel`
  - `rotational_bodies` (explicit lowpoly approximation when revolve is unavailable)
  - `lowpoly_hard_surface_vehicle`
  - `profile_driven_symmetric`
  - `unsupported_smooth_freeform` (explicit blocked outcome)
- create outcomes are explicit:
  - `exact`
  - `lowpoly_approx`
  - `blocked`

Legacy bbox bridge in assistant is explicit opt-in:
- `USE_LEGACY_BBOX_FALLBACK=1`
- used only when vision CAD planner returns no valid plan

## Create Benchmark Harness

Modules:
- `cad/cad_benchmark.py`
- `cad/cad_quality_metrics.py`

Default benchmark suite includes 12 representative prompts:
- lowpoly tank
- armored vehicle
- simple military truck
- bedside table
- simple chair with straight frame
- bed with headboard
- lowpoly airplane
- lowpoly boat
- bottle
- vase
- mug
- smooth human face sculpture (blocked)

Programmatic usage:

```python
from cad.cad_benchmark import run_create_benchmark, default_create_benchmark_cases

out = run_create_benchmark(
    client,
    cases=default_create_benchmark_cases(),
    candidate_count=3,
    max_iterations=32,
    step_delay=0.0,
)
print(out["metrics"])
```

Benchmark case output fields include:
- `input_id`, `prompt`
- `family`, `subtype`
- `outcome_target` (`exact | lowpoly_approx | blocked`)
- `build_status` / `build_success`
- `iteration_count`
- `roles_completed`, `roles_total`
- `disconnected_components_count`
- `floating_parts_count`
- `risky_steps_executed`, `risky_steps_skipped`
- `final_trace_summary`
- `screenshot_path` (when available)

Quality aggregates:
- `build_success_rate`
- `blocked_honesty_rate`
- `disconnected_parts_metric`
- `floating_parts_metric`
- `primary_role_completion`
- `silhouette_plausibility_proxy`
- `average_iterations_to_stop`
- `risky_detail_execution_rate`

## Legacy BBox Path

BBox-based encoding may remain as temporary fallback in legacy scripts.
It is not the primary internal truth for the CAD backend contract.

## Test Coverage Notes

`test/test_cad_dsl.py` covers:

- supported create path
- in-place extrude update path
- no silent skip in edit
- deterministic unsupported/recreate-required detection
- unsupported primitive behavior in normal run
- dry-run unsupported stub behavior
- add_sketch contract behavior
- duplicate-name error propagation
- capability-aware validation rejection
- arcs rejection
- registry update on create and update
- no clear in edit mode
- documentation contract presence checks
