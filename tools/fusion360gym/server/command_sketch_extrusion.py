"""

Sketch Extrusion Reconstruction

"""

import adsk.core
import adsk.fusion
import os
import sys
import importlib
import math

from .command_base import CommandBase

# Add the common folder to sys.path
COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "common"))
if COMMON_DIR not in sys.path:
    sys.path.append(COMMON_DIR)
import name
import match
import deserialize
import serialize
importlib.reload(match)


class CommandSketchExtrusion(CommandBase):

    def add_sketch(self, data):
        """Add a sketch or return existing one by name (ensure semantics). Optional sketch_name for durable identification.
        Contract: When sketch_name is given and a sketch with that name already exists, the existing sketch is returned
        and sketch_plane is informational only — the existing sketch's plane is not changed. Caller must not assume
        they can build a new profile by appending add_point/add_line (edit only via update_extrude)."""
        if data is None or "sketch_plane" not in data:
            return self.runner.return_failure("sketch_plane not specified")
        sketch_name_requested = None
        if isinstance(data.get("sketch_name"), str) and data["sketch_name"].strip():
            sketch_name_requested = data["sketch_name"].strip()
        comp = self.design_state.reconstruction.component
        sketches = comp.sketches
        if sketch_name_requested:
            existing = match.sketch_by_name(sketch_name_requested, sketches)
            if existing is not None:
                sketch_uuid = name.get_uuid(existing)
                if sketch_uuid is None:
                    sketch_uuid = name.set_uuid(existing)
                # Risk D: We do not reset self.state for this sketch; caller must not assume
                # they can build a new profile by appending add_point/add_line (edit only via update_extrude).
                return self.runner.return_success({
                    "sketch_id": sketch_uuid,
                    "sketch_name": existing.name
                })
        sketch_plane = self.__resolve_sketch_plane(data["sketch_plane"])
        if sketch_plane is None:
            return self.runner.return_failure("sketch_plane could not be found")
        sketch = sketches.addWithoutEdges(sketch_plane)
        if sketch_name_requested:
            try:
                sketch.name = sketch_name_requested
            except Exception:
                pass
        sketch_uuid = name.set_uuid(sketch)
        return self.runner.return_success({
            "sketch_id": sketch_uuid,
            "sketch_name": sketch.name
        })

    def __resolve_sketch_plane(self, sketch_plane_data):
        """Resolve base and offset construction planes (e.g. 'XY@12.5')."""
        component = self.design_state.reconstruction.component
        base_map = {
            "xy": component.xYConstructionPlane,
            "xz": component.xZConstructionPlane,
            "yz": component.yZConstructionPlane,
        }
        if isinstance(sketch_plane_data, str) and "@" in sketch_plane_data:
            base_name, offset_raw = sketch_plane_data.split("@", 1)
            base = base_map.get(base_name.strip().lower())
            if base is None:
                return None
            try:
                offset = float(offset_raw.strip())
            except Exception:
                return None
            planes = component.constructionPlanes
            plane_input = planes.createInput()
            offset_value = adsk.core.ValueInput.createByReal(offset)
            plane_input.setByOffset(base, offset_value)
            return planes.add(plane_input)
        if isinstance(sketch_plane_data, str):
            direct = base_map.get(sketch_plane_data.strip().lower())
            if direct is not None:
                return direct
        return match.sketch_plane(sketch_plane_data)

    def add_point(self, data):
        """Add a point to create a new sequential line in the given sketch"""
        if (data is None or "sketch_name" not in data or
                "pt" not in data):
            return self.runner.return_failure("add_point data not specified")
        sketch = match.sketch_by_name(
            data["sketch_name"],
            sketches=self.design_state.reconstruction.component.sketches
        )
        if sketch is None:
            return self.runner.return_failure("sketch not found")
        sketch_uuid = name.get_uuid(sketch)
        # If this is the first point, store it and return
        if sketch.name not in self.state:
            self.__init_sketch_state(sketch.name, data["pt"], data["pt"])
            profile_data = serialize.sketch_profiles(sketch.profiles)
            return self.runner.return_success({
                "sketch_id": sketch_uuid,
                "sketch_name": sketch.name,
                "profiles": profile_data
            })
        state = self.state[sketch.name]
        transform = data["transform"] if "transform" in data else None
        return self.__add_line(
            sketch,
            sketch_uuid,
            state["last_pt"],
            data["pt"],
            transform
        )

    def add_line(self, data):
        """Add a line to an existing sketch"""
        if (data is None or "sketch_name" not in data or
                "pt1" not in data or "pt2" not in data):
            return self.runner.return_failure("add_line data not specified")
        sketch = match.sketch_by_name(
            data["sketch_name"],
            sketches=self.design_state.reconstruction.component.sketches
        )
        if sketch is None:
            return self.runner.return_failure("sketch not found")
        sketch_uuid = name.get_uuid(sketch)
        transform = data["transform"] if "transform" in data else None
        return self.__add_line(
            sketch,
            sketch_uuid,
            data["pt1"],
            data["pt2"],
            transform
        )

    def add_arc(self, data):
        """Add an arc to an existing sketch"""
        if (data is None or "sketch_name" not in data or
                "pt1" not in data or "pt2" not in data or
                "angle" not in data):
            return self.runner.return_failure("add_arc data not specified")
        sketch = match.sketch_by_name(
            data["sketch_name"],
            sketches=self.design_state.reconstruction.component.sketches
        )
        if sketch is None:
            return self.runner.return_failure("sketch not found")
        sketch_uuid = name.get_uuid(sketch)
        transform = data["transform"] if "transform" in data else None
        return self.__add_arc(
            sketch,
            sketch_uuid,
            data["pt1"],
            data["pt2"],
            data["angle"],
            transform
        )

    def add_circle(self, data):
        """Add a circle to an existing sketch"""
        if (data is None or "sketch_name" not in data or
                "pt" not in data or "radius" not in data):
            return self.runner.return_failure("add_circle data not specified")
        sketch = match.sketch_by_name(
            data["sketch_name"],
            sketches=self.design_state.reconstruction.component.sketches
        )
        if sketch is None:
            return self.runner.return_failure("sketch not found")
        sketch_uuid = name.get_uuid(sketch)
        transform = data["transform"] if "transform" in data else None
        return self.__add_circle(
            sketch,
            sketch_uuid,
            data["pt"],
            data["radius"],
            transform
        )

    def close_profile(self, data):
        """Close the current set of lines to create one or more profiles
           by joining the first point to the last"""
        if data is None or "sketch_name" not in data:
            return self.runner.return_failure("close_profile data not specified")
        sketch = match.sketch_by_name(
            data["sketch_name"],
            sketches=self.design_state.reconstruction.component.sketches
        )
        if sketch is None:
            return self.runner.return_failure("sketch not found")
        sketch_uuid = name.get_uuid(sketch)
        if sketch.name not in self.state:
            return self.runner.return_failure("sketch state not found")
        state = self.state[sketch.name]
        # We need at least 4 points (2 lines with 2 points each)
        if state["pt_count"] < 4:
            return self.runner.return_failure("sketch has too few points")
        if state["last_pt"] is None or state["first_pt"] is None:
            return self.runner.return_failure("sketch end points invalid")
        transform = state["transform"]
        return self.__add_line(
            sketch,
            sketch_uuid,
            state["last_pt"],
            state["first_pt"],
            transform
        )

    def add_extrude(self, data):
        """Add an extrude feature from a sketch. Optional feature_name for durable identification."""
        if (data is None or "sketch_name" not in data or
                "profile_id" not in data or "distance" not in data or
                "operation" not in data):
            return self.runner.return_failure("add_extrude data not specified")
        sketch = match.sketch_by_name(
            data["sketch_name"],
            sketches=self.design_state.reconstruction.component.sketches
        )
        if sketch is None:
            return self.runner.return_failure("extrude sketch not found")
        profile = match.sketch_profile_by_id(data["profile_id"], [sketch])
        if profile is None:
            return self.runner.return_failure("extrude sketch profile not found")
        operation = self.__get_extrude_operation(data["operation"])
        if operation is None:
            return self.runner.return_failure("extrude operation not found")

        # Make the extrude
        extrudes = self.design_state.reconstruction.component.features.extrudeFeatures
        extrude_input = extrudes.createInput(profile, operation)
        distance = adsk.core.ValueInput.createByReal(data["distance"])
        extent_distance = adsk.fusion.DistanceExtentDefinition.create(distance)
        extrude_input.setOneSideExtent(extent_distance, adsk.fusion.ExtentDirections.PositiveExtentDirection)
        extrude = extrudes.add(extrude_input)
        if isinstance(data.get("feature_name"), str) and data["feature_name"].strip():
            try:
                extrude.name = data["feature_name"].strip()
            except Exception:
                pass
        # Serialize the data and return
        return self.return_extrude_data(extrude)

    def __add_line(self, sketch, sketch_uuid, pt1, pt2, transform=None):
        start_point = deserialize.point3d(pt1)
        end_point = deserialize.point3d(pt2)
        if transform is not None:
            if isinstance(transform, str):
                # Transform world coords to sketch coords
                if transform.lower() == "world":
                    start_point = sketch.modelToSketchSpace(start_point)
                    end_point = sketch.modelToSketchSpace(end_point)
            elif isinstance(transform, dict):
                # For mapping Fusion exported data back correctly
                xform = deserialize.matrix3d(transform)
                sketch_transform = sketch.transform
                sketch_transform.invert()
                xform.transformBy(sketch_transform)
                start_point.transformBy(xform)
                end_point.transformBy(xform)

        line = sketch.sketchCurves.sketchLines.addByTwoPoints(start_point, end_point)
        curve_uuid = name.set_uuid(line)
        name.set_uuids_for_sketch(sketch)
        profile_data = serialize.sketch_profiles(sketch.profiles)
        if sketch.name not in self.state:
            self.__init_sketch_state(sketch.name, pt1, pt2, transform=transform)
        else:
            self.__inc_sketch_state(sketch.name, pt2, transform=transform)
        return self.runner.return_success({
            "sketch_id": sketch_uuid,
            "sketch_name": sketch.name,
            "curve_id": curve_uuid,
            "profiles": profile_data
        })

    def __add_arc(self, sketch, sketch_uuid, pt1, pt2, angle_degrees, transform=None):
        start_point = deserialize.point3d(pt1)
        center_point = deserialize.point3d(pt2)
        angle_radians = math.radians(angle_degrees)
        if transform is not None:
            if isinstance(transform, str):
                # Transform world coords to sketch coords
                if transform.lower() == "world":
                    start_point = sketch.modelToSketchSpace(start_point)
                    center_point = sketch.modelToSketchSpace(center_point)
            elif isinstance(transform, dict):
                # For mapping Fusion exported data back correctly
                xform = deserialize.matrix3d(transform)
                sketch_transform = sketch.transform
                sketch_transform.invert()
                xform.transformBy(sketch_transform)
                start_point.transformBy(xform)
                center_point.transformBy(xform)

        arc = sketch.sketchCurves.sketchArcs.addByCenterStartSweep(
            center_point,
            start_point,
            angle_radians
        )
        end_point = serialize.point3d(arc.endSketchPoint.geometry)
        curve_uuid = name.set_uuid(arc)
        name.set_uuids_for_sketch(sketch)
        profile_data = serialize.sketch_profiles(sketch.profiles)
        if sketch.name not in self.state:
            self.__init_sketch_state(sketch.name, pt1, end_point, transform=transform)
        else:
            self.__inc_sketch_state(sketch.name, end_point, transform=transform)
        return self.runner.return_success({
            "sketch_id": sketch_uuid,
            "sketch_name": sketch.name,
            "curve_id": curve_uuid,
            "profiles": profile_data
        })

    def __add_circle(self, sketch, sketch_uuid, pt1, radius, transform=None):
        center_point = deserialize.point3d(pt1)
        if transform is not None:
            if isinstance(transform, str):
                # Transform world coords to sketch coords
                if transform.lower() == "world":
                    center_point = sketch.modelToSketchSpace(center_point)
            elif isinstance(transform, dict):
                # For mapping Fusion exported data back correctly
                xform = deserialize.matrix3d(transform)
                sketch_transform = sketch.transform
                sketch_transform.invert()
                xform.transformBy(sketch_transform)
                center_point.transformBy(xform)

        circle = sketch.sketchCurves.sketchCircles.addByCenterRadius(
            center_point,
            radius
        )
        curve_uuid = name.set_uuid(circle)
        name.set_uuids_for_sketch(sketch)
        profile_data = serialize.sketch_profiles(sketch.profiles)
        return self.runner.return_success({
            "sketch_id": sketch_uuid,
            "sketch_name": sketch.name,
            "curve_id": curve_uuid,
            "profiles": profile_data
        })

    def __get_extrude_operation(self, operation):
        """Return an appropriate extrude operation"""
        # Check that the operation is going to work
        body_count = self.design_state.reconstruction.bRepBodies.count
        # If there are no other bodies, we have to make a new body
        if body_count == 0:
            operation = "NewBodyFeatureOperation"
        return deserialize.feature_operations(operation)

    def __init_sketch_state(self, sketch_name, first_pt=None, last_pt=None,
                            pt_count=0, transform=None):
        """Initialize the sketch state"""
        self.state[sketch_name] = {
            "first_pt": first_pt,
            "last_pt": last_pt,
            "pt_count": pt_count,
            "transform": None
        }

    def __inc_sketch_state(self, sketch_name, last_pt, transform=None):
        """Increment the sketch state with the latest point"""
        state = self.state[sketch_name]
        state["last_pt"] = last_pt
        # Increment by 2 as we are adding a curve
        state["pt_count"] += 2
        state["transform"] = transform

    def find_entity_by_name(self, data):
        """Find an entity (Sketch, ExtrudeFeature, etc.) by name for stateful editing.
        Returns found, count. If count > 1, treat as error (duplicate names)."""
        if data is None or "type" not in data or "name" not in data:
            return self.runner.return_failure("find_entity_by_name requires type and name")
        entity_type = (data.get("type") or "").strip()
        entity_name = (data.get("name") or "").strip()
        if not entity_name:
            return self.runner.return_success({"found": False, "count": 0})
        comp = self.design_state.reconstruction.component
        count = 0
        if entity_type == "Sketch":
            for i in range(comp.sketches.count):
                sk = comp.sketches.item(i)
                if getattr(sk, "name", None) == entity_name:
                    count += 1
            found = count > 0
        elif entity_type == "ExtrudeFeature":
            for i in range(comp.features.extrudeFeatures.count):
                feat = comp.features.extrudeFeatures.item(i)
                if getattr(feat, "name", None) == entity_name:
                    count += 1
            found = count > 0
        else:
            return self.runner.return_failure(f"Unknown entity type: {entity_type}")
        if count > 1:
            return self.runner.return_failure(
                f"Duplicate entity name: '{entity_name}' found {count} times (type={entity_type}). Rename to ensure uniqueness."
            )
        return self.runner.return_success({"found": found, "count": count})

    def update_extrude(self, data):
        """Update an extrude feature's distance by feature name (stateful edit).
        Only supports DistanceExtentDefinition; rejects negative distance."""
        if data is None or "feature_name" not in data or "distance" not in data:
            return self.runner.return_failure("update_extrude requires feature_name and distance")
        feature_name = (data.get("feature_name") or "").strip()
        distance = float(data.get("distance", 0))
        if not feature_name:
            return self.runner.return_failure("feature_name is empty")
        if distance < 0:
            return self.runner.return_failure(
                "update_extrude: negative distance not supported; use positive distance (extrusion direction is fixed)"
            )
        comp = self.design_state.reconstruction.component
        extrude_feature = None
        for i in range(comp.features.extrudeFeatures.count):
            feat = comp.features.extrudeFeatures.item(i)
            if getattr(feat, "name", None) == feature_name:
                extrude_feature = feat
                break
        if extrude_feature is None:
            return self.runner.return_failure(f"Extrude feature not found: {feature_name}")
        try:
            extent_one = extrude_feature.extentOne
            if extent_one is None:
                return self.runner.return_failure("Extrude feature has no extentOne")
            if not isinstance(extent_one, adsk.fusion.DistanceExtentDefinition):
                return self.runner.return_failure(
                    "update_extrude only supports distance-based extents; current extent type is unsupported"
                )
            extrude_feature.timelineObject.rollTo(True)
            distance_value = adsk.core.ValueInput.createByReal(distance)
            extent_def = adsk.fusion.DistanceExtentDefinition.create(distance_value)
            extrude_feature.extentOne = extent_def
            self.design_state.refresh()
            return self.return_extrude_data(extrude_feature)
        except Exception as ex:
            return self.runner.return_failure(f"update_extrude failed: {ex}")

    def _iter_collection(self, collection):
        if collection is None:
            return []
        try:
            return [item for item in collection]
        except Exception:
            pass
        try:
            count = int(getattr(collection, "count", 0) or 0)
            return [collection.item(i) for i in range(count)]
        except Exception:
            return []

    def _bbox3d_to_axes(self, bbox_data):
        if not isinstance(bbox_data, dict):
            return None
        mn = bbox_data.get("min_point") if isinstance(bbox_data.get("min_point"), dict) else {}
        mx = bbox_data.get("max_point") if isinstance(bbox_data.get("max_point"), dict) else {}
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

    def _axis_overlap(self, a0, a1, b0, b1):
        lo = max(min(a0, a1), min(b0, b1))
        hi = min(max(a0, a1), max(b0, b1))
        return max(0.0, hi - lo)

    def _bbox_gap(self, a, b):
        gx = max(0.0, max(a["x_min"], b["x_min"]) - min(a["x_max"], b["x_max"]))
        gy = max(0.0, max(a["y_min"], b["y_min"]) - min(a["y_max"], b["y_max"]))
        gz = max(0.0, max(a["z_min"], b["z_min"]) - min(a["z_max"], b["z_max"]))
        return max(gx, gy, gz)

    def _bbox_intersection_volume(self, a, b):
        ox = self._axis_overlap(a["x_min"], a["x_max"], b["x_min"], b["x_max"])
        oy = self._axis_overlap(a["y_min"], a["y_max"], b["y_min"], b["y_max"])
        oz = self._axis_overlap(a["z_min"], a["z_max"], b["z_min"], b["z_max"])
        return (ox * oy * oz), ox, oy, oz

    def _body_id(self, body, index):
        name = str(getattr(body, "name", "") or f"Body_{index+1}")
        return f"{name}#{index+1}"

    def _collect_bodies(self):
        comp = self.design_state.reconstruction.component
        body_records = []
        body_objects = {}
        bodies = self._iter_collection(comp.bRepBodies)
        for i, body in enumerate(bodies):
            bid = self._body_id(body, i)
            bbox = None
            bbox_axes = None
            try:
                bbox = serialize.bounding_box3d(body.boundingBox)
                bbox_axes = self._bbox3d_to_axes(bbox)
            except Exception:
                bbox = None
            volume = 0.0
            area = 0.0
            try:
                volume = float(getattr(body, "volume", 0.0) or 0.0)
            except Exception:
                volume = 0.0
            try:
                area = float(getattr(body, "area", 0.0) or 0.0)
            except Exception:
                area = 0.0
            body_records.append(
                {
                    "id": bid,
                    "name": str(getattr(body, "name", "") or f"Body_{i+1}"),
                    "temp_id": str(getattr(body, "tempId", "") or ""),
                    "revision_id": str(getattr(body, "revisionId", "") or ""),
                    "volume": volume,
                    "area": area,
                    "bbox": bbox,
                    "bbox_axes": bbox_axes,
                }
            )
            body_objects[bid] = body
        return body_records, body_objects

    def _collect_feature_body_relations(self):
        comp = self.design_state.reconstruction.component
        body_records, body_objects = self._collect_bodies()
        rev_to_body_id = {
            str(getattr(body_objects.get(r.get("id")), "revisionId", "") or ""): str(r.get("id"))
            for r in body_records
        }
        relations = []
        for i in range(comp.features.extrudeFeatures.count):
            feat = comp.features.extrudeFeatures.item(i)
            fname = str(getattr(feat, "name", "") or f"Extrude_{i+1}")
            feature_body_ids = []
            for body in self._iter_collection(getattr(feat, "bodies", None)):
                rid = str(getattr(body, "revisionId", "") or "")
                mapped = rev_to_body_id.get(rid, "")
                if mapped:
                    feature_body_ids.append(mapped)
            relations.append(
                {
                    "feature_name": fname,
                    "feature_type": "ExtrudeFeature",
                    "body_ids": sorted(set(feature_body_ids)),
                    "source_kind": "exact_fusion_api",
                }
            )
        return relations

    def _collect_body_relations(self, contact_tol=None):
        body_records, body_objects = self._collect_bodies()
        ids = [str(b.get("id") or "") for b in body_records if str(b.get("id") or "")]
        id_to_axes = {str(b.get("id")): b.get("bbox_axes") for b in body_records if isinstance(b.get("bbox_axes"), dict)}
        if contact_tol is None:
            tol = max(0.02, float(getattr(self.app, "pointTolerance", 0.01) or 0.01) * 10.0)
        else:
            tol = max(0.0, float(contact_tol))

        interference_pairs = {}
        try:
            design = self.design_state.design or adsk.fusion.Design.cast(self.app.activeProduct)
            if design is not None and len(ids) >= 2:
                coll = adsk.core.ObjectCollection.create()
                for bid in ids:
                    coll.add(body_objects[bid])
                input_data = design.createInterferenceInput(coll)
                results = design.analyzeInterference(input_data)
                rev_to_id = {
                    str(getattr(body_objects[bid], "revisionId", "") or ""): bid
                    for bid in ids
                }
                for res in self._iter_collection(results):
                    one = getattr(res, "entityOne", None)
                    two = getattr(res, "entityTwo", None)
                    if one is None or two is None:
                        continue
                    a = rev_to_id.get(str(getattr(one, "revisionId", "") or ""), "")
                    b = rev_to_id.get(str(getattr(two, "revisionId", "") or ""), "")
                    if not a or not b or a == b:
                        continue
                    key = tuple(sorted((a, b)))
                    inter_body = getattr(res, "interferenceBody", None)
                    vol = 0.0
                    try:
                        vol = float(getattr(inter_body, "volume", 0.0) or 0.0) if inter_body is not None else 0.0
                    except Exception:
                        vol = 0.0
                    interference_pairs[key] = max(float(interference_pairs.get(key, 0.0) or 0.0), vol)
        except Exception:
            interference_pairs = {}

        relations = []
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                a_axes = id_to_axes.get(a)
                b_axes = id_to_axes.get(b)
                if a_axes is None or b_axes is None:
                    continue
                key = tuple(sorted((a, b)))
                exact_overlap = key in interference_pairs
                overlap_volume_exact = float(interference_pairs.get(key, 0.0) or 0.0)
                bbox_overlap_vol, ox, oy, oz = self._bbox_intersection_volume(a_axes, b_axes)
                touches_estimate = False
                relation = "separate"
                source_kind = "heuristic_estimate"
                if exact_overlap:
                    relation = "intersects_exact"
                    source_kind = "exact_fusion_api"
                else:
                    near_xy = ox > 0.0 and oy > 0.0 and abs(a_axes["z_max"] - b_axes["z_min"]) <= tol
                    near_yz = oy > 0.0 and oz > 0.0 and abs(a_axes["x_max"] - b_axes["x_min"]) <= tol
                    near_xz = ox > 0.0 and oz > 0.0 and abs(a_axes["y_max"] - b_axes["y_min"]) <= tol
                    touches_estimate = bool(near_xy or near_yz or near_xz)
                    if touches_estimate:
                        relation = "contacts_estimate"
                relations.append(
                    {
                        "a": a,
                        "b": b,
                        "relation": relation,
                        "intersects": bool(exact_overlap),
                        "touches": bool(touches_estimate),
                        "overlap_volume": overlap_volume_exact,
                        "bbox_overlap_volume_estimate": round(float(bbox_overlap_vol), 6),
                        "gap_estimate": round(float(self._bbox_gap(a_axes, b_axes)), 6),
                        "source_kind": source_kind,
                    }
                )
        return body_records, relations

    def _connected_components(self, nodes, relations):
        graph = {n: [] for n in nodes}
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            relation = str(rel.get("relation") or "").strip().lower()
            if relation not in ("intersects_exact", "contacts_estimate"):
                continue
            a = str(rel.get("a") or "")
            b = str(rel.get("b") or "")
            if a in graph and b in graph:
                graph[a].append(b)
                graph[b].append(a)
        visited = set()
        comps = []
        for n in nodes:
            if n in visited:
                continue
            stack = [n]
            comp = []
            visited.add(n)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in graph[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            comps.append(sorted(comp))
        comps.sort(key=lambda c: (-len(c), c))
        return comps

    def list_features(self, data=None):
        """List sketch/extrude features for capability-aware inspection."""
        comp = self.design_state.reconstruction.component
        sketches = []
        for i in range(comp.sketches.count):
            sk = comp.sketches.item(i)
            sketches.append({"name": getattr(sk, "name", ""), "type": "Sketch"})
        extrudes = []
        for i in range(comp.features.extrudeFeatures.count):
            feat = comp.features.extrudeFeatures.item(i)
            extrudes.append({"name": getattr(feat, "name", ""), "type": "ExtrudeFeature"})
        return self.runner.return_success(
            {
                "sketches": sketches,
                "extrude_features": extrudes,
                "counts": {
                    "sketches": len(sketches),
                    "extrude_features": len(extrudes),
                },
            }
        )

    def query_bounding_box(self, data=None):
        """Return current reconstruction bounding box."""
        try:
            bbox = self.design_state.reconstruction.component.boundingBox
            return self.runner.return_success({"bounding_box": serialize.bounding_box3d(bbox)})
        except Exception as ex:
            return self.runner.return_failure(f"query_bounding_box failed: {ex}")

    def get_model_state(self, data=None):
        comp = self.design_state.reconstruction.component
        try:
            body_records, _ = self._collect_bodies()
            feature_rel = self._collect_feature_body_relations()
            context = self.get_active_construction_context({})[2]
            bbox = serialize.bounding_box3d(comp.boundingBox)
            return self.runner.return_success(
                {
                    "component_name": str(getattr(comp, "name", "") or ""),
                    "counts": {
                        "sketches": int(comp.sketches.count),
                        "extrude_features": int(comp.features.extrudeFeatures.count),
                        "bodies": len(body_records),
                    },
                    "bounding_box": bbox,
                    "feature_body_relations": feature_rel,
                    "active_context": context.get("active_context", context) if isinstance(context, dict) else {},
                    "source_kind": "exact_fusion_api",
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_model_state failed: {ex}")

    def get_features(self, data=None):
        comp = self.design_state.reconstruction.component
        features = []
        try:
            for i in range(comp.features.extrudeFeatures.count):
                feat = comp.features.extrudeFeatures.item(i)
                op_name = serialize.feature_operation(getattr(feat, "operation", None))
                body_count = len(self._iter_collection(getattr(feat, "bodies", None)))
                features.append(
                    {
                        "name": str(getattr(feat, "name", "") or f"Extrude_{i+1}"),
                        "type": "ExtrudeFeature",
                        "operation": op_name or "",
                        "body_count": int(body_count),
                    }
                )
            return self.runner.return_success({"features": features, "source_kind": "exact_fusion_api"})
        except Exception as ex:
            return self.runner.return_failure(f"get_features failed: {ex}")

    def get_sketches(self, data=None):
        comp = self.design_state.reconstruction.component
        sketches = []
        try:
            for i in range(comp.sketches.count):
                sk = comp.sketches.item(i)
                profile_count = int(getattr(sk.profiles, "count", 0) or 0)
                sketches.append(
                    {
                        "name": str(getattr(sk, "name", "") or f"Sketch_{i+1}"),
                        "type": "Sketch",
                        "profile_count": profile_count,
                    }
                )
            return self.runner.return_success({"sketches": sketches, "source_kind": "exact_fusion_api"})
        except Exception as ex:
            return self.runner.return_failure(f"get_sketches failed: {ex}")

    def get_bodies(self, data=None):
        try:
            body_records, _ = self._collect_bodies()
            rows = []
            for row in body_records:
                rows.append(
                    {
                        "id": row.get("id"),
                        "name": row.get("name"),
                        "volume": row.get("volume"),
                        "area": row.get("area"),
                        "bbox": row.get("bbox"),
                    }
                )
            return self.runner.return_success(
                {
                    "bodies": rows,
                    "counts": {"bodies": len(rows)},
                    "source_kind": "exact_fusion_api",
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_bodies failed: {ex}")

    def get_parts(self, data=None):
        try:
            code, msg, payload = self.get_bodies(data)
            if code != 200:
                return code, msg, payload
            return self.runner.return_success(
                {
                    "parts": list((payload or {}).get("bodies", [])),
                    "counts": dict((payload or {}).get("counts", {})),
                    "source_kind": "exact_fusion_api",
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_parts failed: {ex}")

    def get_body_bbox(self, data=None):
        data = data or {}
        body_name = str(data.get("body_name") or "").strip().lower()
        try:
            body_records, _ = self._collect_bodies()
            bbox_map = {}
            for row in body_records:
                rid = str(row.get("id") or "")
                rname = str(row.get("name") or "").strip().lower()
                if body_name and body_name not in (rid.lower(), rname):
                    continue
                if isinstance(row.get("bbox"), dict):
                    bbox_map[rid] = row.get("bbox")
            return self.runner.return_success({"body_bbox": bbox_map, "source_kind": "exact_fusion_api"})
        except Exception as ex:
            return self.runner.return_failure(f"get_body_bbox failed: {ex}")

    def get_feature_bbox(self, data=None):
        comp = self.design_state.reconstruction.component
        try:
            body_records, body_objects = self._collect_bodies()
            rev_to_body_bbox = {}
            for row in body_records:
                rid = str(row.get("revision_id") or "")
                if rid and isinstance(row.get("bbox"), dict):
                    rev_to_body_bbox[rid] = row.get("bbox")

            feature_bbox = {}
            for i in range(comp.features.extrudeFeatures.count):
                feat = comp.features.extrudeFeatures.item(i)
                fname = str(getattr(feat, "name", "") or f"Extrude_{i+1}")
                boxes = []
                for body in self._iter_collection(getattr(feat, "bodies", None)):
                    rid = str(getattr(body, "revisionId", "") or "")
                    bb = rev_to_body_bbox.get(rid)
                    if isinstance(bb, dict):
                        boxes.append(bb)
                if boxes:
                    mins = [self._bbox3d_to_axes(b) for b in boxes]
                    mins = [m for m in mins if isinstance(m, dict)]
                    if mins:
                        min_x = min(m["x_min"] for m in mins)
                        min_y = min(m["y_min"] for m in mins)
                        min_z = min(m["z_min"] for m in mins)
                        max_x = max(m["x_max"] for m in mins)
                        max_y = max(m["y_max"] for m in mins)
                        max_z = max(m["z_max"] for m in mins)
                        feature_bbox[fname] = {
                            "type": "BoundingBox3D",
                            "min_point": {"x": min_x, "y": min_y, "z": min_z},
                            "max_point": {"x": max_x, "y": max_y, "z": max_z},
                        }
            return self.runner.return_success({"feature_bbox": feature_bbox, "source_kind": "exact_fusion_api"})
        except Exception as ex:
            return self.runner.return_failure(f"get_feature_bbox failed: {ex}")

    def get_faces(self, data=None):
        try:
            body_records, body_objects = self._collect_bodies()
            per_body = {}
            total = 0
            for row in body_records:
                bid = str(row.get("id") or "")
                count = 0
                try:
                    count = int(getattr(body_objects[bid].faces, "count", 0) or 0)
                except Exception:
                    count = 0
                per_body[bid] = count
                total += count
            return self.runner.return_success(
                {
                    "faces": {
                        "count": int(total),
                        "per_body": per_body,
                        "source_kind": "exact_fusion_api",
                    }
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_faces failed: {ex}")

    def get_edges(self, data=None):
        try:
            body_records, body_objects = self._collect_bodies()
            per_body = {}
            total = 0
            for row in body_records:
                bid = str(row.get("id") or "")
                count = 0
                try:
                    count = int(getattr(body_objects[bid].edges, "count", 0) or 0)
                except Exception:
                    count = 0
                per_body[bid] = count
                total += count
            return self.runner.return_success(
                {
                    "edges": {
                        "count": int(total),
                        "per_body": per_body,
                        "source_kind": "exact_fusion_api",
                    }
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_edges failed: {ex}")

    def get_body_relations(self, data=None):
        data = data or {}
        try:
            tol = data.get("contact_tolerance", None)
            body_records, relations = self._collect_body_relations(contact_tol=tol)
            return self.runner.return_success(
                {
                    "relations": relations,
                    "body_count": len(body_records),
                    "source_kind": "derived_exact",
                    "notes": "intersections exact via analyzeInterference; contact uses bbox proximity estimate",
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_body_relations failed: {ex}")

    def get_feature_body_relations(self, data=None):
        try:
            rel = self._collect_feature_body_relations()
            return self.runner.return_success({"relations": rel, "source_kind": "exact_fusion_api"})
        except Exception as ex:
            return self.runner.return_failure(f"get_feature_body_relations failed: {ex}")

    def get_connected_components(self, data=None):
        data = data or {}
        try:
            tol = data.get("contact_tolerance", None)
            body_records, relations = self._collect_body_relations(contact_tol=tol)
            ids = [str(b.get("id") or "") for b in body_records if str(b.get("id") or "")]
            comps = self._connected_components(ids, relations)
            exact_edges = sum(1 for r in relations if str(r.get("relation") or "") == "intersects_exact")
            heuristic_edges = sum(1 for r in relations if str(r.get("relation") or "") == "contacts_estimate")
            source_kind = "derived_exact" if exact_edges > 0 else "heuristic_estimate"
            return self.runner.return_success(
                {
                    "components": comps,
                    "edge_mix": {"exact": exact_edges, "heuristic": heuristic_edges},
                    "source_kind": source_kind,
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_connected_components failed: {ex}")

    def get_overlaps(self, data=None):
        data = data or {}
        try:
            tol = data.get("contact_tolerance", None)
            _, relations = self._collect_body_relations(contact_tol=tol)
            overlaps = []
            for rel in relations:
                relation = str(rel.get("relation") or "")
                if relation in ("intersects_exact", "contacts_estimate"):
                    overlaps.append(dict(rel))
            return self.runner.return_success(
                {
                    "overlaps": overlaps,
                    "source_kind": "derived_exact",
                    "notes": "exact intersects + estimated contacts",
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_overlaps failed: {ex}")

    def get_active_construction_context(self, data=None):
        comp = self.design_state.reconstruction.component
        try:
            last_sketch = ""
            if comp.sketches.count > 0:
                last_sketch = str(getattr(comp.sketches.item(comp.sketches.count - 1), "name", "") or "")
            last_feature = ""
            if comp.features.extrudeFeatures.count > 0:
                last_feature = str(getattr(comp.features.extrudeFeatures.item(comp.features.extrudeFeatures.count - 1), "name", "") or "")
            timeline_count = 0
            try:
                timeline_count = int(getattr(self.design_state.design.timeline, "count", 0) or 0)
            except Exception:
                timeline_count = 0
            return self.runner.return_success(
                {
                    "active_context": {
                        "active_component": str(getattr(comp, "name", "") or ""),
                        "last_sketch": last_sketch,
                        "last_feature": last_feature,
                        "timeline_count": timeline_count,
                    },
                    "source_kind": "exact_fusion_api",
                }
            )
        except Exception as ex:
            return self.runner.return_failure(f"get_active_construction_context failed: {ex}")

    # ------------------------------------------------------------------
    # Fillet / Chamfer / Shell
    # ------------------------------------------------------------------

    def _find_body_by_name(self, body_name):
        """Find a BRepBody by name (or partial match). Returns (body, body_id) or (None, None)."""
        body_records, body_objects = self._collect_bodies()
        # Exact match on display name
        for rec in body_records:
            bid = str(rec.get("id") or "")
            obj = body_objects.get(bid)
            if obj is not None and str(getattr(obj, "name", "")) == body_name:
                return obj, bid
        # Fallback: first body if body_name is "Body1" or similar default
        if body_records and body_name.lower() in ("body1", "body 1", "body_1", "last"):
            bid = str(body_records[-1].get("id") or "")
            return body_objects.get(bid), bid
        return None, None

    def _collect_edges_for_body(self, body):
        """Collect edge info from a BRepBody. Returns list of edge dicts + ObjectCollection."""
        edge_infos = []
        edge_collection = adsk.core.ObjectCollection.create()
        try:
            for i in range(body.edges.count):
                edge = body.edges.item(i)
                edge_collection.add(edge)
                info = {"index": i}
                try:
                    sp = edge.startVertex.geometry if edge.startVertex else None
                    ep = edge.endVertex.geometry if edge.endVertex else None
                    if sp and ep:
                        info["start"] = {"x": round(sp.x, 3), "y": round(sp.y, 3), "z": round(sp.z, 3)}
                        info["end"] = {"x": round(ep.x, 3), "y": round(ep.y, 3), "z": round(ep.z, 3)}
                        info["midZ"] = round((sp.z + ep.z) / 2, 3)
                except Exception:
                    pass
                edge_infos.append(info)
        except Exception:
            pass
        return edge_infos, edge_collection

    def add_fillet(self, data):
        """Add fillet (rounded edges) to a body.
        Required: body_name (str), radius (float in cm).
        Optional: edge_indices (list of int) to fillet specific edges, otherwise all edges."""
        if data is None or "body_name" not in data or "radius" not in data:
            return self.runner.return_failure("add_fillet requires body_name and radius")
        body_name = str(data["body_name"])
        radius = float(data["radius"])
        if radius <= 0:
            return self.runner.return_failure("fillet radius must be > 0")

        body, bid = self._find_body_by_name(body_name)
        if body is None:
            return self.runner.return_failure(f"body '{body_name}' not found")

        edge_infos, all_edges = self._collect_edges_for_body(body)
        if all_edges.count == 0:
            return self.runner.return_failure("body has no edges")

        # Select specific edges or all
        edge_indices = data.get("edge_indices")
        edges_to_fillet = adsk.core.ObjectCollection.create()
        if edge_indices and isinstance(edge_indices, list):
            for idx in edge_indices:
                idx = int(idx)
                if 0 <= idx < body.edges.count:
                    edges_to_fillet.add(body.edges.item(idx))
            if edges_to_fillet.count == 0:
                return self.runner.return_failure("no valid edge indices provided")
        else:
            edges_to_fillet = all_edges

        try:
            comp = self.design_state.reconstruction.component
            fillets = comp.features.filletFeatures
            fillet_input = fillets.createInput()
            fillet_input.addConstantRadiusEdgeSet(
                edges_to_fillet,
                adsk.core.ValueInput.createByReal(radius),
                True  # isTangentChain
            )
            fillet = fillets.add(fillet_input)
            return self.runner.return_success({
                "feature_name": str(getattr(fillet, "name", "")),
                "body": bid,
                "radius": radius,
                "edges_filleted": edges_to_fillet.count,
            })
        except Exception as ex:
            return self.runner.return_failure(f"fillet failed: {ex}")

    def add_chamfer(self, data):
        """Add chamfer (beveled edges) to a body.
        Required: body_name (str), distance (float in cm).
        Optional: edge_indices (list of int) to chamfer specific edges, otherwise all edges."""
        if data is None or "body_name" not in data or "distance" not in data:
            return self.runner.return_failure("add_chamfer requires body_name and distance")
        body_name = str(data["body_name"])
        distance = float(data["distance"])
        if distance <= 0:
            return self.runner.return_failure("chamfer distance must be > 0")

        body, bid = self._find_body_by_name(body_name)
        if body is None:
            return self.runner.return_failure(f"body '{body_name}' not found")

        edge_infos, all_edges = self._collect_edges_for_body(body)
        if all_edges.count == 0:
            return self.runner.return_failure("body has no edges")

        edge_indices = data.get("edge_indices")
        edges_to_chamfer = adsk.core.ObjectCollection.create()
        if edge_indices and isinstance(edge_indices, list):
            for idx in edge_indices:
                idx = int(idx)
                if 0 <= idx < body.edges.count:
                    edges_to_chamfer.add(body.edges.item(idx))
            if edges_to_chamfer.count == 0:
                return self.runner.return_failure("no valid edge indices provided")
        else:
            edges_to_chamfer = all_edges

        try:
            comp = self.design_state.reconstruction.component
            chamfers = comp.features.chamferFeatures
            chamfer_input = chamfers.createInput(edges_to_chamfer, True)
            chamfer_input.setToEqualDistance(adsk.core.ValueInput.createByReal(distance))
            chamfer = chamfers.add(chamfer_input)
            return self.runner.return_success({
                "feature_name": str(getattr(chamfer, "name", "")),
                "body": bid,
                "distance": distance,
                "edges_chamfered": edges_to_chamfer.count,
            })
        except Exception as ex:
            return self.runner.return_failure(f"chamfer failed: {ex}")

    def add_shell(self, data):
        """Shell a body (hollow it out), removing specified face(s).
        Required: body_name (str), thickness (float in cm).
        Optional: remove_face ("top"|"bottom"|"none", default "top").
        "top" = face with highest avg Z; "bottom" = face with lowest avg Z."""
        if data is None or "body_name" not in data or "thickness" not in data:
            return self.runner.return_failure("add_shell requires body_name and thickness")
        body_name = str(data["body_name"])
        thickness = float(data["thickness"])
        if thickness <= 0:
            return self.runner.return_failure("shell thickness must be > 0")

        body, bid = self._find_body_by_name(body_name)
        if body is None:
            return self.runner.return_failure(f"body '{body_name}' not found")

        remove_which = str(data.get("remove_face", "top")).lower()

        # Find the face to remove
        faces_to_remove = adsk.core.ObjectCollection.create()
        if remove_which != "none" and body.faces.count > 0:
            best_face = None
            best_z = None
            for i in range(body.faces.count):
                face = body.faces.item(i)
                try:
                    # Compute average Z of face vertices
                    bbox = face.boundingBox
                    avg_z = (bbox.minPoint.z + bbox.maxPoint.z) / 2
                    if remove_which == "top":
                        if best_z is None or avg_z > best_z:
                            best_z = avg_z
                            best_face = face
                    elif remove_which == "bottom":
                        if best_z is None or avg_z < best_z:
                            best_z = avg_z
                            best_face = face
                except Exception:
                    continue
            if best_face is not None:
                faces_to_remove.add(best_face)

        try:
            comp = self.design_state.reconstruction.component
            shells = comp.features.shellFeatures
            shell_input = shells.createInput(faces_to_remove, False)
            shell_input.insideThickness = adsk.core.ValueInput.createByReal(thickness)
            shell = shells.add(shell_input)
            return self.runner.return_success({
                "feature_name": str(getattr(shell, "name", "")),
                "body": bid,
                "thickness": thickness,
                "faces_removed": faces_to_remove.count,
            })
        except Exception as ex:
            return self.runner.return_failure(f"shell failed: {ex}")

    def get_edges_by_body(self, data):
        """Get edge info for a specific body (indices + endpoint positions)."""
        if data is None or "body_name" not in data:
            return self.runner.return_failure("body_name required")
        body_name = str(data["body_name"])
        body, bid = self._find_body_by_name(body_name)
        if body is None:
            return self.runner.return_failure(f"body '{body_name}' not found")
        edge_infos, _ = self._collect_edges_for_body(body)
        return self.runner.return_success({
            "body": bid,
            "edge_count": len(edge_infos),
            "edges": edge_infos[:50],  # Limit to 50 edges to avoid huge responses
            "source_kind": "exact_fusion_api",
        })
