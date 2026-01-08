from dataclasses import dataclass
from datetime import datetime

@dataclass
class FineTuneArtifact:
    base_model: str
    adapter_name: str
    created_at: datetime
    dataset_hash: str
    metrics: dict
