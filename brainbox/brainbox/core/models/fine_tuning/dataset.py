from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FineTuneExample:
    input: str
    output: str
    metadata: Dict = None

class FineTuneDataset:
    def __init__(self, examples: List[FineTuneExample]):
        self.examples = examples

    def to_jsonl(self, path: str):
        import json
        with open(path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps({
                    "input": ex.input,
                    "output": ex.output
                }) + "\n")
