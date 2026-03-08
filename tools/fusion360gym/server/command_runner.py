"""

Run incoming commands from the client

"""

import adsk.core
import traceback
import uuid

from .command_export import CommandExport
from .command_sketch_extrusion import CommandSketchExtrusion
from .command_face_extrusion import CommandFaceExtrusion
from .command_reconstruct import CommandReconstruct
from .design_state import DesignState
from .tool_registry import ToolRegistry, CommandSpec, CommandSchema


class CommandRunner():

    def __init__(self):
        self.logger = None
        self.app = adsk.core.Application.get()
        self.last_command = ""
        self.design_state = DesignState(self)
        self.export = CommandExport(self, self.design_state)
        self.sketch_extrusion = CommandSketchExtrusion(self, self.design_state)
        self.face_extrusion = CommandFaceExtrusion(self, self.design_state)
        self.reconstruct = CommandReconstruct(self, self.design_state)
        self.command_objects = [
            self.export,
            self.sketch_extrusion,
            self.face_extrusion,
            self.reconstruct
        ]
        self.registry = ToolRegistry()
        self.last_result_code = "ok"
        self.last_trace_id = ""
        self._register_commands()
        self.design_state.set_command_objects(self.command_objects)

    def set_logger(self, logger):
        """Set the logger in all command objects"""
        self.logger = logger
        self.design_state.set_logger(logger)
        for obj in self.command_objects:
            obj.set_logger(logger)

    def _register(self, name, description, handler, schema=None, validator=None, category="general"):
        self.registry.register(
            CommandSpec(
                name=name,
                description=description,
                handler=handler,
                schema=schema,
                validator=validator,
                category=category,
            )
        )

    def _validate_graph_payload(self, data):
        sequence = bool(data.get("sequence"))
        if sequence and "file" not in data:
            return "field 'file' is required when sequence=true"
        return None

    def _validate_actions_payload(self, data):
        actions = data.get("actions")
        if not isinstance(actions, list) or len(actions) == 0:
            return "field 'actions' must be a non-empty list"
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                return f"actions[{i}] must be an object"
            for key in ("start_face", "end_face", "operation"):
                if key not in action:
                    return f"actions[{i}] missing required field '{key}'"
                if not isinstance(action[key], str) or len(action[key].strip()) == 0:
                    return f"actions[{i}].{key} must be a non-empty string"
        return None

    def _register_commands(self):
        # Introspection
        self._register(
            "list_tools",
            "List available server tools and minimal payload contracts",
            lambda _data: self.list_tools(),
            category="system",
        )
        self._register("ping", "Ping server", lambda _data: self.ping(), category="system")
        self._register("refresh", "Refresh design state", lambda _data: self.design_state.refresh(), category="system")
        self._register("clear", "Clear design", lambda _data: self.design_state.clear(), category="system")

        # Reconstruction
        self._register(
            "reconstruct",
            "Reconstruct design from JSON payload",
            lambda data: self.reconstruct.reconstruct(data),
            schema=CommandSchema(required={}, root_type="dict"),
            category="reconstruction",
        )
        self._register(
            "reconstruct_stepwise",
            "Reconstruct incrementally with delay",
            lambda data: self.reconstruct.reconstruct_stepwise(data),
            schema=CommandSchema(required={}, optional={"json_data": "dict", "delay": "number"}, root_type="dict"),
            category="reconstruction",
        )
        self._register(
            "reconstruct_sketch",
            "Reconstruct a single sketch",
            lambda data: self.reconstruct.reconstruct_sketch(data),
            schema=CommandSchema(
                required={"sketch_data": "dict"},
                optional={
                    "sketch_plane": "plane",
                    "scale": "vector3d",
                    "translate": "vector3d",
                    "rotate": "vector3d",
                },
            ),
            category="reconstruction",
        )
        self._register(
            "reconstruct_profile",
            "Reconstruct one profile",
            lambda data: self.reconstruct.reconstruct_profile(data),
            schema=CommandSchema(
                required={"sketch_data": "dict", "sketch_name": "str", "profile_id": "str"},
                optional={"scale": "vector3d", "translate": "vector3d", "rotate": "vector3d"},
            ),
            category="reconstruction",
        )
        self._register(
            "reconstruct_curve",
            "Reconstruct one curve",
            lambda data: self.reconstruct.reconstruct_curve(data),
            schema=CommandSchema(
                required={"sketch_data": "dict", "sketch_name": "str", "curve_id": "str"},
                optional={"scale": "vector3d", "translate": "vector3d", "rotate": "vector3d"},
            ),
            category="reconstruction",
        )
        self._register(
            "reconstruct_curves",
            "Reconstruct all curves",
            lambda data: self.reconstruct.reconstruct_curves(data),
            schema=CommandSchema(
                required={"sketch_data": "dict", "sketch_name": "str"},
                optional={"scale": "vector3d", "translate": "vector3d", "rotate": "vector3d"},
            ),
            category="reconstruction",
        )

        # Export/query
        self._register(
            "mesh",
            "Export mesh (.obj/.stl)",
            lambda data: self.export.mesh(data),
            schema=CommandSchema(required={"file": "str"}),
            category="export",
        )
        self._register(
            "brep",
            "Export brep (.step/.smt/.f3d)",
            lambda data: self.export.brep(data),
            schema=CommandSchema(required={"file": "str"}),
            category="export",
        )
        self._register(
            "sketches",
            "Export sketches (.png/.dxf)",
            lambda data: self.export.sketches(data),
            schema=CommandSchema(required={"format": "str"}),
            category="export",
        )
        self._register(
            "screenshot",
            "Capture viewport screenshot",
            lambda data: self.export.screenshot(data),
            schema=CommandSchema(
                required={"file": "str"},
                optional={"width": "int", "height": "int", "fit_camera": "bool"},
            ),
            category="export",
        )
        self._register(
            "graph",
            "Export graph representation",
            lambda data: self.export.graph(data),
            schema=CommandSchema(
                required={"format": "str", "sequence": "bool", "labels": "bool"},
                optional={"file": "str"},
            ),
            validator=self._validate_graph_payload,
            category="export",
        )
        self._register("list_features", "List sketches/extrudes", lambda _data: self.sketch_extrusion.list_features({}), category="query")
        self._register("query_bounding_box", "Query current bounding box", lambda _data: self.sketch_extrusion.query_bounding_box({}), category="query")
        self._register("get_model_state", "Get model state summary", lambda _data: self.sketch_extrusion.get_model_state({}), category="query")
        self._register("get_features", "Get feature list", lambda _data: self.sketch_extrusion.get_features({}), category="query")
        self._register("get_sketches", "Get sketch list", lambda _data: self.sketch_extrusion.get_sketches({}), category="query")
        self._register("get_bodies", "Get body list", lambda _data: self.sketch_extrusion.get_bodies({}), category="query")
        self._register("get_parts", "Get part list", lambda _data: self.sketch_extrusion.get_parts({}), category="query")
        self._register(
            "get_body_bbox",
            "Get body bounding boxes",
            lambda data: self.sketch_extrusion.get_body_bbox(data or {}),
            schema=CommandSchema(required={}, optional={"body_name": "str"}, root_type="dict"),
            category="query",
        )
        self._register("get_feature_bbox", "Get feature bounding boxes", lambda _data: self.sketch_extrusion.get_feature_bbox({}), category="query")
        self._register("get_faces", "Get face summary", lambda _data: self.sketch_extrusion.get_faces({}), category="query")
        self._register("get_edges", "Get edge summary", lambda _data: self.sketch_extrusion.get_edges({}), category="query")
        self._register(
            "get_body_relations",
            "Get body-to-body relation summary",
            lambda data: self.sketch_extrusion.get_body_relations(data or {}),
            schema=CommandSchema(required={}, optional={"contact_tolerance": "number"}, root_type="dict"),
            category="query",
        )
        self._register("get_feature_body_relations", "Get feature-to-body relation summary", lambda _data: self.sketch_extrusion.get_feature_body_relations({}), category="query")
        self._register(
            "get_connected_components",
            "Get connected components from body relations",
            lambda data: self.sketch_extrusion.get_connected_components(data or {}),
            schema=CommandSchema(required={}, optional={"contact_tolerance": "number"}, root_type="dict"),
            category="query",
        )
        self._register(
            "get_overlaps",
            "Get overlaps/intersections/contact-like summary",
            lambda data: self.sketch_extrusion.get_overlaps(data or {}),
            schema=CommandSchema(required={}, optional={"contact_tolerance": "number"}, root_type="dict"),
            category="query",
        )
        self._register(
            "get_active_construction_context",
            "Get active construction context",
            lambda _data: self.sketch_extrusion.get_active_construction_context({}),
            category="query",
        )

        # Sketch/extrude editing
        self._register(
            "add_sketch",
            "Create or ensure sketch",
            lambda data: self.sketch_extrusion.add_sketch(data),
            schema=CommandSchema(required={"sketch_plane": "plane"}, optional={"sketch_name": "str"}),
            category="sketch",
        )
        self._register(
            "add_point",
            "Add point to sketch path",
            lambda data: self.sketch_extrusion.add_point(data),
            schema=CommandSchema(
                required={"sketch_name": "str", "pt": "point3d"},
                optional={"transform": ("dict", "str")},
            ),
            category="sketch",
        )
        self._register(
            "add_line",
            "Add line to sketch",
            lambda data: self.sketch_extrusion.add_line(data),
            schema=CommandSchema(
                required={"sketch_name": "str", "pt1": "point3d", "pt2": "point3d"},
                optional={"transform": ("dict", "str")},
            ),
            category="sketch",
        )
        self._register(
            "add_arc",
            "Add arc to sketch",
            lambda data: self.sketch_extrusion.add_arc(data),
            schema=CommandSchema(
                required={"sketch_name": "str", "pt1": "point3d", "pt2": "point3d", "angle": "number"},
                optional={"transform": ("dict", "str")},
            ),
            category="sketch",
        )
        self._register(
            "add_circle",
            "Add circle to sketch",
            lambda data: self.sketch_extrusion.add_circle(data),
            schema=CommandSchema(
                required={"sketch_name": "str", "pt": "point3d", "radius": "number"},
                optional={"transform": ("dict", "str")},
            ),
            category="sketch",
        )
        self._register(
            "close_profile",
            "Close sketch profile",
            lambda data: self.sketch_extrusion.close_profile(data),
            schema=CommandSchema(required={"sketch_name": "str"}),
            category="sketch",
        )
        self._register(
            "add_extrude",
            "Extrude sketch profile",
            lambda data: self.sketch_extrusion.add_extrude(data),
            schema=CommandSchema(
                required={
                    "sketch_name": "str",
                    "profile_id": "str",
                    "distance": "number",
                    "operation": "str",
                },
                optional={"feature_name": "str"},
            ),
            category="feature",
        )
        self._register(
            "find_entity_by_name",
            "Find sketch/feature by stable name",
            lambda data: self.sketch_extrusion.find_entity_by_name(data),
            schema=CommandSchema(required={"type": "str", "name": "str"}),
            category="query",
        )
        self._register(
            "update_extrude",
            "Update extrude distance by feature name",
            lambda data: self.sketch_extrusion.update_extrude(data),
            schema=CommandSchema(required={"feature_name": "str", "distance": "number"}),
            category="feature",
        )

        # Face extrusion flow
        self._register(
            "set_target",
            "Set target CAD body for face-extrusion flow",
            lambda data: self.face_extrusion.set_target(data),
            schema=CommandSchema(required={"file": "str", "file_data": "str"}),
            category="target",
        )
        self._register("revert_to_target", "Revert reconstruction to target", lambda _data: self.face_extrusion.revert_to_target(), category="target")
        self._register(
            "add_extrude_by_target_face",
            "Extrude between two target faces",
            lambda data: self.face_extrusion.add_extrude_by_target_face(data),
            schema=CommandSchema(required={"start_face": "str", "end_face": "str", "operation": "str"}),
            category="target",
        )
        self._register(
            "add_extrudes_by_target_face",
            "Run batch face extrusions",
            lambda data: self.face_extrusion.add_extrudes_by_target_face(data),
            schema=CommandSchema(required={"actions": "list"}, optional={"revert": "bool"}),
            validator=self._validate_actions_payload,
            category="target",
        )

    def run_command(self, command, data=None):
        """Run a command and route it to the right method"""
        try:
            self.last_trace_id = uuid.uuid4().hex[:16]
            self.last_result_code = "ok"
            self.last_command = command
            spec = self.registry.get(command)
            if spec is None:
                return self.return_failure("Unknown command", code="unknown_command")

            is_valid, validation_error = self.registry.validate(spec, data)
            if not is_valid:
                return self.return_failure(validation_error or "Invalid request payload", code="invalid_request")

            result = spec.handler(data)
            if result is None:
                return self.return_failure("Command handler returned no response", code="handler_contract_error")
            return result
        except Exception as ex:
            return self.return_exception(ex)
        finally:
            # Update the UI
            adsk.doEvents()

    def list_tools(self):
        return self.return_success({"tools": self.registry.describe()})

    def ping(self):
        """Ping for debugging"""
        return self.return_success()

    def get_last_meta(self):
        return {
            "trace_id": self.last_trace_id,
            "code": self.last_result_code,
        }

    def return_success(self, data=None, code="ok", message=None):
        self.last_result_code = code or "ok"
        if message is None:
            message = f"Success processing {self.last_command} command"
        return 200, message, data

    def return_failure(self, reason, code="command_failed"):
        self.last_result_code = code or "command_failed"
        message = f"Failed processing {self.last_command} command due to {reason}"
        return 500, message, None

    def return_exception(self, ex):
        self.last_result_code = "internal_exception"
        message = f"""Error processing {self.last_command} command\n
                        Exception of type {type(ex)} with args: {ex.args}\n
                        {traceback.format_exc()}"""
        return 500, message, None
