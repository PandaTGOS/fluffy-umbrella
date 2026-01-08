class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, name: str, model_ref: str, metadata: dict):
        self._models[name] = {
            "model_ref": model_ref,
            "metadata": metadata
        }

    def get(self, name: str):
        return self._models[name]
