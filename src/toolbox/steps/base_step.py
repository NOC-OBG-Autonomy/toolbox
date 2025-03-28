class BaseStep:
    def __init__(self, name, parameters=None, diagnostics=False):
        self.name = name
        self.parameters = parameters or {}
        self.diagnostics = diagnostics

    def run(self):
        """To be implemented by subclasses"""
        raise NotImplementedError(f"Step '{self.name}' must implement a run() method.")

    def generate_diagnostics(self):
        """Hook for diagnostics (optional)"""
        pass
