from dataclasses import dataclass

@dataclass
class AdapterArtifact:
    base_model: str
    adapter_name: str
    path: str
    metrics: dict
