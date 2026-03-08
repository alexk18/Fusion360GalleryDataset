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
