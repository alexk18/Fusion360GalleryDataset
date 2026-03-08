# Fusion360Gym Agent Base Context

## Project goal

The goal is to build a **build-first iterative CAD agent** for Fusion360Gym, conceptually closer to `onshape-mcp`, but implemented within the Fusion360Gym / Fusion architecture.

The target system is NOT:
- one-shot full-plan generation,
- bbox-centric generation,
- object-specific scripted recipes,
- edit-first CAD automation.

The target system IS:
- tool-driven,
- iterative,
- state-aware,
- structurally grounded,
- relationally aware,
- honest about runtime limits.

The intended loop is:

`inspect -> choose next CAD step/fragment -> validate -> execute -> inspect -> continue`

DSL remains the executable contract:

`plan -> validate -> compile -> execute`

But DSL is not the only source of reasoning. Main intelligence should come from:
- runtime state introspection,
- relational assembly reasoning,
- structural roles,
- family/subtype priors,
- dataset-informed priors,
- iterative next-step planning.

---

## External architectural reference

Reference repo for architectural direction:
- https://github.com/clarsbyte/onshape-mcp

Important:
The goal is NOT to copy that repository literally.
The goal is to approximate its logic:
- rich CAD tool surface,
- state queries,
- iterative action-selection,
- geometry-aware reasoning,
- less one-shot hallucination.

---

## Architecture evolution so far

### Stage 1 — bbox / one-shot path
Initial path was roughly:
- text/image -> LLM -> bbox-like decomposition / rough geometry -> deterministic encoder -> execute

Problems:
- hallucinations,
- poor resemblance,
- box-junk geometry,
- weak constructive reasoning,
- weak image-to-CAD behavior.

### Stage 2 — CAD DSL
Execution moved to:
- `plan -> validate -> compile -> execute`

Correct direction, but still insufficient for quality.

### Stage 3 — Structural Build Spec
Added create-time intermediate contract with fields like:
- object family,
- outcome target,
- approximation policy,
- main masses,
- supporting parts,
- secondary parts,
- silhouette constraints,
- build order,
- allowed primitives,
- required capabilities.

This improved create planning but remained too one-shot.

### Stage 4 — family/subtype grammars
Added family-aware/subtype-aware create grammars for:
- `furniture_boxy_panel`
- `lowpoly_hard_surface_vehicle`
- `profile_driven_symmetric`
- `rotational_bodies`
- `unsupported_smooth_freeform`

Subtypes added included examples like:
- tank
- armored vehicle
- truck-like vehicle
- cabinet / bedside table
- chair_frame
- bed_frame_headboard
- lowpoly_aircraft
- lowpoly_boat
- bottle / vase / mug

This improved role coverage but not assembly closure.

### Stage 5 — create safety layer
Introduced:
- `core`
- `secondary`
- `risky_detail`

First-pass create became primary-first, and risky booleans/details were pruned or deferred.

### Stage 6 — iterative CAD agent
Introduced iterative create loop:
- inspect -> choose next fragment -> validate -> execute -> inspect

Modules added included:
- `cad_agent_loop.py`
- `cad_state_introspection.py`
- `cad_relational_assembly.py`
- `cad_dataset_priors.py`
- `cad_next_step_planner.py`

### Stage 7 — geometry/state query layer
Introduced layered introspection:
- `exact_fusion_api`
- `derived_exact`
- `heuristic_estimate`

Query surface includes commands such as:
- `list_capabilities`
- `list_tools`
- `get_model_state`
- `get_features`
- `get_sketches`
- `get_bodies`
- `get_parts`
- `get_body_bbox`
- `get_feature_bbox`
- `get_connected_components`
- `get_overlaps`
- `get_faces`
- `get_edges`
- `get_body_relations`
- `get_feature_body_relations`
- `get_active_construction_context`

### Stage 8 — live Fusion validation
Live validation was run in a real Fusion session.

Confirmed live:
- query surface works,
- blocked honesty works,
- rotational lowpoly approximation cases (`bottle`, `vase`, `mug`) succeed,
- execution pipeline is live and stable.

Confirmed live bottlenecks:
1. exact-family create cases reach completed roles but remain spatially disconnected;
2. furniture family fails to choose an executable bootstrap/core fragment;
3. there is a routing bug: `bed_headboard` misroutes to `unsupported_smooth_freeform`.

---

## Confirmed live benchmark findings

### Live smoke summary
Representative live results:
- `tank_lowpoly`: fail, roles complete, disconnected remains > 0
- `bedside_table`: fail, no executable core steps
- `bottle`: success
- `smooth_human_face_sculpture`: correctly blocked

### Full benchmark summary
Live suite size: 12 cases

Observed:
- success = 3
- fail = 7
- blocked = 2

Successful:
- bottle
- vase
- mug

Blocked honestly:
- smooth/freeform case(s)

Failing:
- tank-like exact-family cases
- furniture bootstrap cases
- airplane/boat exact-family cases

---

## Current confirmed bottlenecks

These are the currently confirmed real bottlenecks and should be treated as ground truth unless new live evidence disproves them.

1. **Assembly closure reasoning is too weak**
Roles can be considered built while the object remains disconnected.

2. **Attachment evidence is too weak**
Attachment plausibility can remain neutral/acceptable even when there is little or no actual evidence.

3. **Furniture bootstrap planner is too weak**
Furniture cases can stop immediately because every role waits for dependencies and no executable first scaffold is chosen.

4. **Routing correctness has bugs**
At least:
- `bed_headboard` -> `unsupported_smooth_freeform` is wrong.

---

## Project invariants

These should not be broken.

1. Build-first, not edit-first.
2. No object-specific hardcoded scripts per object instance.
3. No fake support for unsupported runtime primitives.
4. Smooth/freeform cases must stay honestly blocked when unsupported.
5. Rotational bodies without revolve must remain honest lowpoly approximations.
6. DSL remains the executable contract.
7. Iterative agent loop remains the main create path.
8. Runtime evidence matters more than optimistic summaries.
9. Live Fusion behavior is the final arbiter, not only unit tests.
10. Minimal targeted patches are preferred over large speculative rewrites.

---

## What is NOT wanted

Do NOT optimize for:
- edit path,
- controlled recreate,
- delete/suppress,
- new advanced primitives like loft/sweep/revolve/fillet/chamfer unless explicitly requested,
- object-specific hacks,
- prompt-only “fixes”,
- architectural rewrites without live evidence.

---

## Current priority

Current priority is to improve live create performance for supported exact families by fixing:
1. assembly closure,
2. attachment evidence,
3. furniture bootstrap fragment selection,
4. routing correctness,

while preserving:
- bottle/vase/mug success,
- blocked honesty,
- iterative CAD agent architecture.

---

## Expected working style for an agent

The agent should:
1. inspect current code and runtime artifacts,
2. identify root causes,
3. apply minimal targeted patches,
4. run tests,
5. run live validation where required,
6. report evidence-based results,
7. separate:
   - confirmed live behavior,
   - test-only behavior,
   - not yet confirmed behavior.

---

## Standard report structure

When completing a task, prefer structured outputs like:

1. ROOT CAUSE ANALYSIS
2. EXECUTION PLAN
3. FILES CHANGED
4. SUMMARY OF CHANGES
5. TESTS
6. LIVE BEFORE / AFTER SUMMARY
7. REMAINING GAPS
8. DEMO TRACE