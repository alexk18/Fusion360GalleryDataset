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

- `create_sketch`
- `add_line`
- `add_arc`
- `add_circle`
- `close_profile`
- `extrude`
- `cut`
- `find_entity_by_name`
- `list_features`
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
- `find_entity_by_name`: supported
- `list_features`: supported
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

## Supported In-Place Edit Operations

- Existing extrude distance update:
  - detect existing feature by stable name
  - call `update_extrude(feature_name, new_distance)`
  - update registry and trace with action `updated`

No `clear()` is used in edit mode.

## Recreate-Required Operations

Detected by `classify_edit_change(step_before, step_after)` in `cad_executor.py`.

Current deterministic recreate-required cases:

- plane change
- operation change
- profile geometry change (rect/circle/poly)
- feature-name change

Current behavior for recreate-required:

- fail-fast with action `recreate-required`
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
