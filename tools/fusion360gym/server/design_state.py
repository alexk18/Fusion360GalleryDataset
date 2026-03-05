"""

Design State for the Fusion 360 Gym

"""

import adsk.core
import adsk.fusion


class _RootComponentAsReconstruction:
    """Wraps root Component to mimic Occurrence API for Part Design (single-component) documents."""

    def __init__(self, root_component):
        self._root = root_component

    @property
    def component(self):
        return self._root

    @property
    def bRepBodies(self):
        return self._root.bRepBodies

    def activate(self):
        # Component (root) has no activate() in Fusion API; only Occurrence does. No-op for Part Design.
        pass

    def deleteMe(self):
        # Part Design root cannot be deleted; remove all bodies instead
        for i in range(self._root.bRepBodies.count - 1, -1, -1):
            self._root.bRepBodies.item(i).deleteMe()


class DesignState():

    def __init__(self, runner):
        self.runner = runner
        self.app = adsk.core.Application.get()
        self.design = adsk.fusion.Design.cast(self.app.activeProduct)
        self.logger = None
        self.command_objects = None

        # Setup the target and reconstruction design components
        self.target = None
        self.reconstruction = None
        if self.design is not None:
            self.setup_reconstruction()
        else:
            raise RuntimeError("No active design. Open or create a design first, then run the add-in again.")

    def set_logger(self, logger):
        self.logger = logger

    def set_command_objects(self, command_objects):
        """Set a reference to the command objects"""
        self.command_objects = command_objects

    def refresh(self):
        """Refresh the active viewport and fit view to show all geometry"""
        try:
            self.app.activeViewport.fit()
        except Exception:
            pass
        self.app.activeViewport.refresh()
        return self.runner.return_success()

    def clear(self):
        """Clear (i.e. close) all open designs in Fusion"""
        for doc in self.app.documents:
            # Save without closing
            doc.close(False)
        if self.command_objects is not None:
            # Reset state in all command objects
            for obj in self.command_objects:
                obj.clear()
        self.target = None
        self.design = adsk.fusion.Design.cast(self.app.activeProduct)
        if self.design is not None:
            self.setup_reconstruction()
        else:
            self.reconstruction = None
        return self.runner.return_success()

    def setup_reconstruction(self):
        """Create the reconstruction component (or use root in Part Design)."""
        if self.design is None:
            raise RuntimeError("No active design. Open or create a design first.")
        try:
            self.reconstruction = self.design.rootComponent.occurrences.addNewComponent(
                adsk.core.Matrix3D.create()
            )
            self.reconstruction.activate()
            name = f"Reconstruction_{self.reconstruction.component.name}"
            self.reconstruction.component.name = name
        except RuntimeError as ex:
            if "Part Design documents can only contain one component" in str(ex):
                # Part Design: use root component as reconstruction (no extra component allowed)
                self.reconstruction = _RootComponentAsReconstruction(self.design.rootComponent)
                self.reconstruction.activate()
            else:
                raise
        adsk.doEvents()

    def clear_reconstruction(self):
        """Clear the reconstruction to an empty component"""
        self.reconstruction.deleteMe()
        self.setup_reconstruction()

    def set_target(self, file):
        """Set the target component from a smt or step file"""
        if file.suffix == ".step" or file.suffix == ".stp":
            import_options = self.app.importManager.createSTEPImportOptions(
                str(file.resolve())
            )
        else:
            import_options = self.app.importManager.createSMTImportOptions(
                str(file.resolve())
            )
        import_options.isViewFit = False
        imported_designs = self.app.importManager.importToTarget2(
            import_options,
            self.design.rootComponent
        )
        self.target = imported_designs[0]
        name = f"Target_{self.target.component.name}"
        self.target.component.name = name
        adsk.doEvents()
